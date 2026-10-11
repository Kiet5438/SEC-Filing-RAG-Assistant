"""Tools the agent's executor can call.

- search_filings: hybrid search over the indexed filings, with metadata
  pre-filters (ticker, form, fiscal year/period, SEC Item, Part, text vs
  table) pushed down into Qdrant.
- list_available_filings: what is indexed, so filters use real values
  instead of guesses ("NVIDIA" vs "NVDA", a fiscal year that isn't there).
- list_sec_filings: what a company has filed with the SEC (read-only).
- index_filing: download and index one filing. The only tool that writes;
  its metadata marks it as requiring user approval, which the agent graph
  enforces before running it.

create_filing_tools() closes over the injected store and retriever, so the
tools hold no state of their own. Every tool returns (content, artifact):
content is the text the LLM reads; artifact is the structured result
(Documents for search, dicts otherwise) that code — e.g. citation
building — uses without re-parsing text. Expected failures (unknown ticker,
nothing matched, SEC errors) come back as explanatory content, not
exceptions, so the agent can correct course.
"""

from typing import Literal

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from src.agent.progress import emit_progress
from src.agent.state import passage_id
from src.ingestion import sec_loader
from src.ingestion.pipeline import index_filing as run_index_pipeline
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import FilingStore
from src.utils.logger import get_logger

logger = get_logger(__name__)

REQUIRES_APPROVAL = "requires_approval"  # tool metadata key the agent graph checks
PASSAGE_CHAR_LIMIT = 4000  # per passage in a tool result; the full text stays in the artifact

FilingType = Literal["10-K", "10-Q"]


class SearchFilingsInput(BaseModel):
    query: str = Field(
        description=(
            "What to look for, in English, phrased the way a 10-K/10-Q would word it "
            "(e.g. 'data center revenue growth drivers', 'unresolved staff comments')."
        )
    )
    ticker: str | None = Field(
        default=None,
        description="Restrict to one company by exact ticker, e.g. 'NVDA'. Use list_available_filings to see indexed tickers.",
    )
    filing_type: FilingType | None = Field(default=None, description="'10-K' (annual) or '10-Q' (quarterly).")
    fiscal_year: int | None = Field(
        default=None,
        description=(
            "The company's own fiscal-year label, e.g. 2026. Not the filing date's year: NVIDIA's fiscal 2026 "
            "ended January 2026, Alphabet's FY2025 10-K was filed in 2026. list_available_filings shows it."
        ),
    )
    fiscal_period: Literal["FY", "Q1", "Q2", "Q3"] | None = Field(
        default=None, description="'FY' for an annual report, 'Q1'-'Q3' for a quarterly one."
    )
    item_code: str | None = Field(
        default=None,
        description=(
            "SEC Item number. 10-K: '1' Business, '1A' Risk Factors, '1B' Unresolved Staff Comments, "
            "'1C' Cybersecurity, '2' Properties, '3' Legal Proceedings, '5' Market for Equity, '7' MD&A, "
            "'7A' Market Risk, '8' Financial Statements, '9A' Controls, '9B' Other Information. "
            "10-Q: Part I '1' Financial Statements, '2' MD&A, '3' Market Risk, '4' Controls; "
            "Part II '1' Legal Proceedings, '1A' Risk Factors, '5' Other Information."
        ),
    )
    part: Literal["I", "II"] | None = Field(
        default=None, description="10-Q Part, needed with item_code because Items 1-4 exist in both Parts."
    )
    content_type: Literal["text", "table"] | None = Field(
        default=None, description="'table' to search only tables (numbers), 'text' only prose."
    )


class ListSecFilingsInput(BaseModel):
    ticker: str = Field(description="Stock ticker, e.g. 'AAPL'.")
    filing_type: FilingType = Field(description="'10-K' or '10-Q'.")
    limit: int = Field(default=8, ge=1, le=20, description="How many of the most recent filings to list.")


class IndexFilingInput(BaseModel):
    ticker: str = Field(description="Stock ticker, e.g. 'AAPL'.")
    filing_type: FilingType = Field(description="'10-K' or '10-Q'.")
    filing_date: str | None = Field(
        default=None,
        description="ISO filing date of a specific filing as list_sec_filings reports it; omit for the latest.",
    )


def create_filing_tools(store: FilingStore, retriever: Retriever) -> list[BaseTool]:
    """Build the agent's tools over an injected store and retriever.

    Args:
        store: The shared FilingStore (also written to by index_filing).
        retriever: The retriever search_filings runs.

    Returns:
        [search_filings, list_available_filings, list_sec_filings, index_filing].
    """

    def search_filings(
        query: str,
        ticker: str | None = None,
        filing_type: str | None = None,
        fiscal_year: int | None = None,
        fiscal_period: str | None = None,
        item_code: str | None = None,
        part: str | None = None,
        content_type: str | None = None,
    ) -> tuple[str, list[Document]]:
        filters = {
            "ticker": ticker.strip().upper() if ticker else None,
            "filing_type": filing_type,
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
            "item_code": _normalize_item_code(item_code),
            "part": part,
            "content_type": content_type,
        }
        filters = {key: value for key, value in filters.items() if value is not None}

        indexed_tickers = store.distinct_values("ticker")
        if "ticker" in filters and filters["ticker"] not in indexed_tickers:
            return (
                f"No filings are indexed for ticker {filters['ticker']!r}. Indexed tickers: "
                f"{', '.join(indexed_tickers) or 'none'}. Use list_sec_filings and index_filing to add one.",
                [],
            )

        passages = retriever.retrieve(query, filters or None)
        if not passages:
            return (
                f"No passages matched {query!r} with filters {filters}. Check the values with "
                "list_available_filings, or drop a filter.",
                [],
            )
        return _render_passages(passages), passages

    def list_available_filings() -> tuple[str, list[dict]]:
        rows = store.filing_summaries()
        if not rows:
            return "Nothing is indexed yet.", []
        return "Indexed filings:\n" + render_catalog(rows), rows

    def list_sec_filings(ticker: str, filing_type: str, limit: int = 8) -> tuple[str, list[dict]]:
        try:
            listings = sec_loader.list_filings(ticker, filing_type, limit=limit)
        except (ValueError, RuntimeError) as e:
            return f"Could not list SEC filings: {e}", []
        if not listings:
            return f"SEC lists no {filing_type} filings for {ticker.upper()}.", []
        indexed = store.doc_ids()
        rows = [
            {
                "doc_id": listing.doc_id,
                "company_name": listing.company_name,
                "filing_type": listing.filing_type,
                "filing_date": listing.filing_date,
                "period_end_date": listing.report_date or None,
                "indexed": listing.doc_id in indexed,
            }
            for listing in listings
        ]
        lines = [
            f"- {row['filing_type']} filed {row['filing_date']}, period ended {row['period_end_date'] or 'unknown'}"
            f"{' (already indexed)' if row['indexed'] else ''}"
            for row in rows
        ]
        return f"{listings[0].company_name} ({ticker.upper()}) {filing_type} filings on SEC EDGAR, newest first:\n" + "\n".join(lines), rows

    def index_filing(ticker: str, filing_type: str, filing_date: str | None = None) -> tuple[str, dict]:
        try:
            result = run_index_pipeline(ticker, filing_type, filing_date, store=store, progress=emit_progress)
        except (ValueError, RuntimeError) as e:
            return f"Indexing failed: {e}", {"status": "failed", "error": str(e)}
        if result["status"] == "skipped":
            return f"{result['doc_id']} was already indexed; nothing to do.", result
        return f"Indexed {result['doc_id']} ({result['chunk_count']} chunks); it is now searchable.", result

    return [
        StructuredTool.from_function(
            search_filings,
            name="search_filings",
            description=(
                "Search the indexed SEC filings and return the most relevant passages, each labeled with an id "
                "like [NVDA_10-K_2026-02-25#77] and its company, filing, section and sub-section. Use filters "
                "when the question names a company, form, period or section; leave them out otherwise."
            ),
            args_schema=SearchFilingsInput,
            response_format="content_and_artifact",
        ),
        StructuredTool.from_function(
            list_available_filings,
            name="list_available_filings",
            description="List every filing currently indexed (ticker, form, fiscal year and period, filing date).",
            response_format="content_and_artifact",
        ),
        StructuredTool.from_function(
            list_sec_filings,
            name="list_sec_filings",
            description=(
                "List a company's recent 10-K or 10-Q filings on SEC EDGAR (filing date, period end), marking "
                "which are already indexed. Read-only; use it to pick a filing for index_filing."
            ),
            args_schema=ListSecFilingsInput,
            response_format="content_and_artifact",
        ),
        StructuredTool.from_function(
            index_filing,
            name="index_filing",
            description=(
                "Download one 10-K or 10-Q from SEC EDGAR and index it so search_filings can find it. Slow "
                "(minutes) and changes the index, so it needs the user's approval. Only use it when the "
                "question needs a filing that list_available_filings doesn't show."
            ),
            args_schema=IndexFilingInput,
            response_format="content_and_artifact",
            metadata={REQUIRES_APPROVAL: True},
        ),
    ]


def render_catalog(rows: list[dict]) -> str:
    """One line per indexed filing (FilingStore.filing_summaries rows), for tool output and prompts."""
    if not rows:
        return "(nothing is indexed yet)"
    return "\n".join(
        f"- {row['doc_id']}: {row['filing_name'] or row['ticker']}, {row['filing_type']}, "
        f"fiscal {row['fiscal_year']} {row['fiscal_period']} (period ended {row['period_end_date']}), "
        f"filed {row['filing_date']}"
        for row in rows
    )


def _render_passages(passages: list[Document]) -> str:
    """Passages for the LLM: id, then the text (whose header names company/filing/section), truncated."""
    blocks = []
    for document in passages:
        text = document.page_content
        if len(text) > PASSAGE_CHAR_LIMIT:
            text = text[:PASSAGE_CHAR_LIMIT] + " …[truncated]"
        blocks.append(f"[{passage_id(document)}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def _normalize_item_code(item_code: str | None) -> str | None:
    """'Item 1A.' / 'item 7a' / '7' -> '1A' / '7A' / '7'."""
    if not item_code:
        return None
    code = item_code.strip().upper().removeprefix("ITEM").strip().rstrip(".")
    return code or None
