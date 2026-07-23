"""Query-first chat UI over the saved FAISS index, plus an explicit indexing form.

The chat side is query-only: it never downloads, parses, chunks, or builds
the index itself. Ingestion happens ONLY through the sidebar's "Index a new
filing" form, which calls main.index_filing() — the single indexing code
path shared with the `python main.py` CLI. Nothing here infers an indexing
action from a chat question or a retrieval miss.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from main import index_filing
from src.embeddings.embedder import get_embedding_model
from src.generation.rag_chain import ask, create_rag_chain, get_llm
from src.retrieval.retriever import create_retriever
from src.retrieval.vector_store import load_vectorstore
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Financial Report Assistant", page_icon="\U0001F4CA")


# --- cached resources -------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model...")
def _cached_embedding():
    return get_embedding_model()


@st.cache_resource(show_spinner="Loading vector store...")
def _cached_vectorstore(_embedding):
    return load_vectorstore(_embedding)


@st.cache_resource(show_spinner=False)
def _cached_retriever(_vectorstore):
    return create_retriever(_vectorstore)


@st.cache_resource(show_spinner="Connecting to Gemini...")
def _cached_chain(_retriever):
    return create_rag_chain(_retriever, get_llm())


# --- sidebar: filters -------------------------------------------------------


def _distinct_metadata_values(vectorstore, key: str) -> list[str]:
    """Sorted distinct non-empty values for a metadata key across the live index."""
    if vectorstore is None:
        return []
    values = {doc.metadata.get(key) for doc in vectorstore.docstore._dict.values() if doc.metadata.get(key)}
    return sorted(values)


def _render_filters(vectorstore) -> dict | None:
    """Render the ticker/filing-type selectboxes; return the filters dict (None if both are "All")."""
    st.header("Filters")
    tickers = _distinct_metadata_values(vectorstore, "ticker")
    filing_types = _distinct_metadata_values(vectorstore, "filing_type")
    selected_ticker = st.selectbox("Ticker", ["All"] + tickers)
    selected_filing_type = st.selectbox("Filing type", ["All"] + filing_types)

    filters = {}
    if selected_ticker != "All":
        filters["ticker"] = selected_ticker
    if selected_filing_type != "All":
        filters["filing_type"] = selected_filing_type
    return filters or None


# --- sidebar: indexed filings panel -----------------------------------------


def _indexed_filings(vectorstore) -> list[dict]:
    """One row per distinct doc_id (ticker, filing_type, filing_date, chunk_count).

    The vectorstore is the source of truth for what's actually retrievable,
    not manifest.json. Sorted by ticker ascending, then filing_date
    descending within each ticker (via two stable sorts).
    """
    if vectorstore is None:
        return []

    chunk_counts: dict[str, int] = {}
    info: dict[str, dict] = {}
    for doc in vectorstore.docstore._dict.values():
        doc_id = doc.metadata.get("doc_id")
        if doc_id is None:
            continue
        chunk_counts[doc_id] = chunk_counts.get(doc_id, 0) + 1
        info.setdefault(
            doc_id,
            {
                "ticker": doc.metadata.get("ticker"),
                "filing_type": doc.metadata.get("filing_type"),
                "filing_date": doc.metadata.get("filing_date"),
            },
        )

    rows = [{**info[doc_id], "chunk_count": count} for doc_id, count in chunk_counts.items()]
    rows.sort(key=lambda r: r["filing_date"] or "", reverse=True)
    rows.sort(key=lambda r: r["ticker"] or "")
    return rows


def _render_indexed_filings_panel(vectorstore) -> None:
    """Compact table of what's actually indexed, scanned inline (microseconds; not cache_data-able)."""
    filings = _indexed_filings(vectorstore)
    with st.expander(f"Indexed filings ({len(filings)})"):
        if filings:
            st.dataframe(filings, use_container_width=True, hide_index=True)
        else:
            st.caption("No filings indexed yet.")


# --- sidebar: index a new filing --------------------------------------------


def _friendly_index_error(e: Exception) -> str:
    """Classify a few common indexing failures into an actionable hint; generic fallback otherwise."""
    message = str(e)
    lowered = message.lower()
    if "403" in message:
        return f"SEC EDGAR rejected the request (403 Forbidden) — check SEC_USER_AGENT in .env. ({message})"
    if "429" in message:
        return f"SEC EDGAR is rate-limiting requests (429) — wait a moment and try again. ({message})"
    if "found" in lowered:
        return f"Not found — check the ticker symbol, or try the other filing type. ({message})"
    return f"Indexing failed: {message}"


def _validated_ticker(ticker_input: str) -> str | None:
    """Normalize a raw ticker input; shows an error and returns None if invalid."""
    ticker = ticker_input.strip().upper()
    if not ticker or not ticker.isalnum():
        st.error("Enter a valid ticker (letters/digits only).")
        return None
    return ticker


def _run_indexing(ticker: str, filing_type: str) -> dict | None:
    """Call index_filing under a spinner; shows a friendly error and returns None on failure."""
    with st.spinner(f"Indexing {ticker} {filing_type} — this can take a minute (SEC download + parse + embed)..."):
        try:
            return index_filing(ticker, filing_type)
        except Exception as e:
            logger.error("Indexing failed for ticker=%s filing_type=%s", ticker, filing_type, exc_info=True)
            st.error(_friendly_index_error(e))
            return None


def _render_indexing_form() -> None:
    """Explicit indexing entry point. Ingestion happens ONLY when this form is submitted."""
    with st.expander("Index a new filing"):
        st.caption("Writes to the shared FAISS index. Assumes a single local user.")
        with st.form("index_filing_form"):
            ticker_input = st.text_input("Ticker", placeholder="e.g. NVDA")
            filing_type_input = st.selectbox("Filing type", ["10-K", "10-Q"])
            submitted = st.form_submit_button("Index filing")

        if not submitted:
            return

        ticker = _validated_ticker(ticker_input)
        if ticker is None:
            return

        result = _run_indexing(ticker, filing_type_input)
        if result is None:
            return

        st.success(f"doc_id={result['doc_id']} — {result['status']} ({result['chunk_count']} chunks).")
        # Cached resources hold the OLD vectorstore/retriever/chain in memory;
        # without clearing them the new filing would be invisible until restart.
        st.cache_resource.clear()
        st.rerun()


# --- sidebar assembly --------------------------------------------------------


def _render_sidebar(vectorstore) -> dict | None:
    """Render filters, indexed filings, the indexing form, and clear-conversation; return filters."""
    with st.sidebar:
        filters = _render_filters(vectorstore)
        st.divider()
        _render_indexed_filings_panel(vectorstore)
        st.divider()
        _render_indexing_form()
        st.divider()
        if st.button("Clear conversation"):
            st.session_state.messages = []
    return filters


# --- chat --------------------------------------------------------------------


def _strip_header_for_display(page_content: str) -> str:
    """Strip the Company/Filing/Section header for display; it's for embedding quality, not reading."""
    _, separator, body = page_content.partition("\n\n")
    return body if separator else page_content


def _render_citations(citations: list[dict] | None, documents: list | None) -> None:
    """Render each deduplicated citation as an expander, with its matching (header-stripped) chunk text."""
    if not citations:
        return
    by_triple = {}
    if documents:
        for doc in documents:
            triple = (doc.metadata.get("source"), doc.metadata.get("section"), doc.metadata.get("chunk_id"))
            by_triple.setdefault(triple, doc)

    for citation in citations:
        triple = (citation["source"], citation["section"], citation["chunk_id"])
        label = f"{citation['source']} — {citation['section']} — chunk {citation['chunk_id']}"
        with st.expander(label):
            matching_doc = by_triple.get(triple)
            if matching_doc is not None:
                st.markdown(_strip_header_for_display(matching_doc.page_content))
            else:
                st.write(citation)


def _render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_citations(message.get("citations"), message.get("documents"))


def _handle_new_question(chain, filters: dict | None) -> None:
    """Take one new chat_input question and answer it. History is display-only: ask() sees
    only this question, never prior turns — the chain is stateless."""
    question = st.chat_input("Ask a question about the indexed filings...", disabled=chain is None)
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question, "citations": None, "documents": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = ask(chain, question, filters)
            except Exception as e:
                logger.error("ask() failed for question=%r filters=%r: %s", question, filters, e, exc_info=True)
                st.error(f"Something went wrong answering that question: {e}")
                return

        st.markdown(result["answer"])
        _render_citations(result.get("citations"), result.get("documents"))
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "citations": result.get("citations"),
                "documents": result.get("documents"),
            }
        )


def main() -> None:
    """Load cached resources, render the sidebar, then the chat."""
    st.title("Financial Report Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    embedding = _cached_embedding()
    vectorstore = _cached_vectorstore(embedding)

    chain = None
    if vectorstore is not None:
        try:
            retriever = _cached_retriever(vectorstore)
            chain = _cached_chain(retriever)
        except Exception as e:
            logger.error("Failed to initialize the RAG chain: %s", e, exc_info=True)
            st.error(f"Could not initialize the assistant (check GEMINI_API_KEY in .env): {e}")

    filters = _render_sidebar(vectorstore)

    if vectorstore is None:
        st.info('No filings indexed yet. Use "Index a new filing" in the sidebar to get started.')

    _render_chat_history()
    _handle_new_question(chain, filters)


if __name__ == "__main__":
    main()
