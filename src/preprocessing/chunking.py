"""Markdown -> chunk splitting for parsed SEC filings.

Splits a filing's Markdown (as produced by parse_html.py) into sections on
the H2 heading contract (^##\\s+), then runs RecursiveCharacterTextSplitter
independently per section so a chunk can never span two sections. Each
resulting chunk becomes one Document tagged with the §4.3 metadata schema.
Every chunk's page_content is prefixed with a short contextual header
(company/filing/section) built from filing_meta, applied after splitting so
it never affects the splitter's chunk_size accounting.
"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_FILING_META_KEYS = {"doc_id", "ticker", "filing_type", "filing_date", "filing_name", "source", "url"}
_SECTION_HEADING_PATTERN = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def chunk_markdown(md_path: str | Path, filing_meta: dict) -> list[Document]:
    """Split a parsed filing's Markdown into per-section chunks as Documents.

    Args:
        md_path: Path to the Markdown file produced by parse_html.py.
        filing_meta: Non-derivable filing fields to stamp onto every chunk's
            metadata: doc_id, ticker, filing_type, filing_date, filing_name,
            source, url.

    Returns:
        One Document per chunk, in section then in-section order, each with
        metadata matching the §4.3 schema (section is still the raw H2 text,
        unaffected by the page_content prefix) and page_content prefixed
        with a "Company: ... / Filing: ... / Section: ..." header. chunk_id
        is sequential per doc_id starting at 0, counted across all sections.

    Raises:
        ValueError: filing_meta is missing a required key.
        FileNotFoundError: md_path does not exist.
    """
    missing = _REQUIRED_FILING_META_KEYS - filing_meta.keys()
    if missing:
        raise ValueError(f"filing_meta is missing required keys: {sorted(missing)}")

    path = Path(md_path)
    logger.info("Chunking %s", path)
    text = path.read_text(encoding="utf-8")

    sections = _split_sections(text)
    if not sections:
        logger.warning("No '## ' section headings found in %s; nothing to chunk.", path)
        return []

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    documents: list[Document] = []
    chunk_id = 0

    for section_title, section_body in sections:
        if not section_body:
            continue
        prefix = _build_chunk_prefix(filing_meta, section_title)
        for chunk_text in splitter.split_text(section_body):
            if not chunk_text.strip():
                continue
            page_content = f"{prefix}\n\n{chunk_text}" if prefix else chunk_text
            metadata = {
                "doc_id": filing_meta["doc_id"],
                "ticker": filing_meta["ticker"],
                "filing_type": filing_meta["filing_type"],
                "filing_date": filing_meta["filing_date"],
                "source": filing_meta["source"],
                "filing_name": filing_meta["filing_name"],
                "section": section_title,
                "url": filing_meta["url"],
                "chunk_id": chunk_id,
            }
            documents.append(Document(page_content=page_content, metadata=metadata))
            chunk_id += 1

    logger.info("Produced %d chunks across %d sections for doc_id=%s", len(documents), len(sections), filing_meta["doc_id"])
    return documents


def _build_chunk_prefix(filing_meta: dict, section: str) -> str:
    """Build the short contextual header prepended to every chunk's page_content.

    Uses only fields already present in filing_meta (filing_name, ticker,
    filing_type, filing_date) plus the section's own H2 text — no new
    metadata fields, no SEC calls. A line whose required field(s) are
    missing or empty is omitted entirely, rather than rendering a "None"
    placeholder.

    Args:
        filing_meta: The same dict passed to chunk_markdown.
        section: The section's exact H2 text.

    Returns:
        The header block (one line per available field, "\\n"-joined, no
        trailing blank line). Callers join it to the chunk body with
        "\\n\\n"; empty string if every line's fields were missing.
    """
    lines = []

    filing_name = filing_meta.get("filing_name")
    ticker = filing_meta.get("ticker")
    if filing_name and ticker:
        lines.append(f"Company: {filing_name} ({ticker})")

    filing_type = filing_meta.get("filing_type")
    filing_date = filing_meta.get("filing_date")
    if filing_type and filing_date:
        lines.append(f"Filing: {filing_type} filed {filing_date}")

    if section:
        lines.append(f"Section: {section}")

    return "\n".join(lines)


def _split_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading_text, body_text) pairs on the ^##\\s+ contract.

    Content before the first H2 (the H1 title line) is discarded here; the
    Heading Contract guarantees no real content precedes the first H2, since
    parse_html.py always emits a "## Preamble" section for any pre-heading
    content.
    """
    parts = _SECTION_HEADING_PATTERN.split(markdown_text)
    sections: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((heading, body))
    return sections
