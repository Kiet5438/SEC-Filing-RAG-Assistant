"""Chat UI over the shared Qdrant store, answered by the sidekick agent.

Each question runs the agent graph (src/agent/sidekick.py): progress events
stream into a status box, the answer streams in token by token, and an
expander under it shows how it was found, step by step. The chat is one
conversation — one agent thread — so a follow-up ("and Microsoft?") is read
in the context of earlier turns; "Clear conversation" starts a new thread.
When the agent needs a filing that isn't indexed, it pauses and asks here
before indexing it; the run resumes with the user's answer. The sidebar's "Index a
new filing" form remains the direct way to index, through the same
src.ingestion.pipeline.index_filing() code path as the `python main.py` CLI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from langgraph.types import Command

from src.agent.progress import ModelProgress
from src.agent.sidekick import SYNTHESIZER, build_sidekick, new_thread
from src.embeddings.embedder import get_embedding_model, get_sparse_embedding_model
from src.generation.citations import citation_key
from src.generation.llm import get_llm
from src.ingestion.pipeline import index_filing
from src.retrieval.reranker import get_reranker
from src.retrieval.retriever import create_retriever
from src.retrieval.vector_store import open_store
from src.utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(page_title="Financial Report Assistant", page_icon="\U0001F4CA")


# --- cached resources -------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model...")
def _cached_embedding():
    return get_embedding_model()


@st.cache_resource(show_spinner="Loading BM25 model...")
def _cached_sparse_embedding():
    return get_sparse_embedding_model()


@st.cache_resource(show_spinner="Opening vector store...")
def _cached_store(_embedding, _sparse_embedding):
    # One process-wide client: in local mode Qdrant locks the storage folder,
    # so indexing must reuse this store rather than open a second client.
    return open_store(_embedding, _sparse_embedding)


@st.cache_resource(show_spinner="Loading reranker...")
def _cached_reranker():
    # None (re-ranking off) unless RERANKER_MODEL_NAME is set.
    return get_reranker()


@st.cache_resource(show_spinner=False)
def _cached_retriever(_store, _reranker):
    return create_retriever(_store, reranker=_reranker)


@st.cache_resource(show_spinner="Connecting to the LLM...")
def _cached_sidekick(_store, _retriever):
    # Cached with its in-memory checkpointer, so a run paused for approval
    # can be resumed on a later rerun.
    return build_sidekick(_store, _retriever, get_llm("main"), get_llm("lite"))


# --- sidebar: filters -------------------------------------------------------


def _render_filters(store) -> dict | None:
    """Render the ticker/filing-type selectboxes; return the filters dict (None if both are "All")."""
    st.header("Filters")
    tickers = store.distinct_values("ticker")
    filing_types = store.distinct_values("filing_type")
    selected_ticker = st.selectbox("Ticker", ["All"] + tickers)
    selected_filing_type = st.selectbox("Filing type", ["All"] + filing_types)

    filters = {}
    if selected_ticker != "All":
        filters["ticker"] = selected_ticker
    if selected_filing_type != "All":
        filters["filing_type"] = selected_filing_type
    return filters or None


# --- sidebar: indexed filings panel -----------------------------------------


def _render_indexed_filings_panel(store) -> None:
    """Compact table of what's actually indexed, read live from the store on every rerun."""
    filings = [
        {key: row[key] for key in ("ticker", "filing_type", "fiscal_year", "fiscal_period", "filing_date", "chunk_count")}
        for row in store.filing_summaries()
    ]
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


def _run_indexing(store, ticker: str, filing_type: str) -> dict | None:
    """Call index_filing under a spinner; shows a friendly error and returns None on failure."""
    with st.spinner(f"Indexing {ticker} {filing_type} — this can take a minute (SEC download + parse + embed)..."):
        try:
            return index_filing(ticker, filing_type, store=store)
        except Exception as e:
            logger.error("Indexing failed for ticker=%s filing_type=%s", ticker, filing_type, exc_info=True)
            st.error(_friendly_index_error(e))
            return None


def _render_indexing_form(store) -> None:
    """Explicit indexing entry point, independent of the agent."""
    with st.expander("Index a new filing"):
        st.caption("Writes to the shared Qdrant store. Assumes a single local user.")
        with st.form("index_filing_form"):
            ticker_input = st.text_input("Ticker", placeholder="e.g. NVDA")
            filing_type_input = st.selectbox("Filing type", ["10-K", "10-Q"])
            submitted = st.form_submit_button("Index filing")

        if not submitted:
            return

        ticker = _validated_ticker(ticker_input)
        if ticker is None:
            return

        result = _run_indexing(store, ticker, filing_type_input)
        if result is None:
            return

        st.success(f"doc_id={result['doc_id']} — {result['status']} ({result['chunk_count']} chunks).")
        # Qdrant serves new points immediately, so the cached store/retriever/
        # agent stay valid; a rerun just refreshes the sidebar. Never clear
        # the resource cache here: a fresh client would hit the folder lock
        # still held by the cached one.
        st.rerun()


# --- sidebar assembly --------------------------------------------------------


def _render_sidebar(store) -> dict | None:
    """Render filters, indexed filings, the indexing form, and clear-conversation; return filters."""
    with st.sidebar:
        filters = _render_filters(store)
        st.divider()
        _render_indexed_filings_panel(store)
        st.divider()
        _render_indexing_form(store)
        st.divider()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.pending = None
            st.session_state.thread = new_thread()  # the agent forgets the old conversation too
    return filters


# --- chat --------------------------------------------------------------------


def _md(text: str) -> str:
    """Escape "$" for st.markdown, which renders text between two dollar signs as
    LaTeX math — "$193,737 million ... $162,361 million" would come out garbled."""
    return text.replace("$", "\\$")


def _strip_header_for_display(page_content: str) -> str:
    """Strip the Company/Filing/Section/Subsection header for display; it's for retrieval, not reading."""
    _, separator, body = page_content.partition("\n\n")
    return body if separator else page_content


def _render_citations(citations: list[dict] | None, documents: list | None) -> None:
    """Render each cited passage as an expander, labeled [n] to match the answer's markers."""
    if not citations:
        return
    by_key = {}
    if documents:
        for doc in documents:
            by_key.setdefault(citation_key(doc.metadata), doc)

    for citation in citations:
        location = " › ".join(part for part in (citation["section"], citation["subsection"]) if part)
        label = f"[{citation['number']}] {citation['source']} — {location}"
        with st.expander(label):
            matching_doc = by_key.get(citation_key(citation))
            if matching_doc is not None:
                st.markdown(_md(_strip_header_for_display(matching_doc.page_content)))
            else:
                st.write(citation)


_TRACE_LABELS = {"planner": "Planner", "executor": "Executor", "run_tools": "Tools", "synthesizer": "Answer"}


def _render_trace(trace: list[dict] | None) -> None:
    """How an answer was found: each node's event, with the model that served its LLM call."""
    if not trace:
        return
    with st.expander(f"How this was answered ({len(trace)} events)"):
        for number, event in enumerate(trace, start=1):
            model = f" · `{event['model']}`" if event.get("model") else ""
            st.markdown(f"**{number}. {_TRACE_LABELS.get(event['node'], event['node'])}**{model}  \n{_md(event['summary'])}")


def _render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(_md(message["content"]))
            if message["role"] == "assistant":
                if message.get("caption"):
                    st.caption(message["caption"])
                _render_trace(message.get("trace"))
                _render_citations(message.get("citations"), message.get("documents"))


def _run_caption(values: dict) -> str:
    """'1 step(s) · answered by gemini-3.8-flash'."""
    answering = next((event["model"] for event in reversed(values.get("trace", [])) if event["node"] == SYNTHESIZER), None)
    caption = f"{values.get('steps_taken', 0)} step(s)"
    return caption + (f" · answered by {answering}" if answering else "")


def _run_agent(graph, payload, config) -> tuple[str, dict | None]:
    """Stream one run (a new question, or a resume after an approval) into the current chat message.

    Progress events go into a collapsible status box; the synthesizer's
    tokens go into the answer as they arrive.

    Returns:
        ("done", final state values), ("paused", the approval request) or
        ("failed", None).
    """
    status = st.status("Working...", expanded=False)
    answer_box = st.empty()
    streamed = ""
    run_config = {**config, "callbacks": [ModelProgress()]}  # which model each call waits on, and why one was skipped
    try:
        for mode, chunk in graph.stream(payload, run_config, stream_mode=["custom", "messages"]):
            if mode == "custom":
                status.write(chunk["message"])
                status.update(label=chunk["message"])
                continue
            message, metadata = chunk
            if metadata.get("langgraph_node") == SYNTHESIZER and message.text:
                streamed += message.text
                answer_box.markdown(_md(streamed))
        snapshot = graph.get_state(config)
    except Exception as e:
        logger.error("Agent run failed: %s", e, exc_info=True)
        status.update(label="Failed", state="error")
        st.error(f"Something went wrong answering that question: {e}")
        return "failed", None

    if snapshot.interrupts:
        status.update(label="Waiting for your approval", state="complete")
        return "paused", snapshot.interrupts[0].value

    values = snapshot.values
    status.update(label=_run_caption(values), state="complete")
    answer_box.markdown(_md(values["answer"]))  # final text: citation markers normalized
    _render_trace(values.get("trace"))
    _render_citations(values.get("citations"), values.get("evidence"))
    return "done", values


def _settle(outcome: str, result: dict | None, config: dict) -> None:
    """Record a run's outcome: an answer joins the history; a pause waits for the approval UI."""
    if outcome == "paused":
        st.session_state.pending = {"config": config, "request": result}
        st.rerun()
    if outcome == "done":
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "caption": _run_caption(result),
                "trace": result.get("trace"),
                "citations": result.get("citations"),
                "documents": result.get("evidence"),
            }
        )


def _render_pending_approval(graph) -> None:
    """Ask before the agent indexes a filing; resume the paused run with the answer."""
    pending = st.session_state.pending
    if not pending or graph is None:
        return
    request = pending["request"]
    args = request["args"]
    filing = f"{args.get('ticker', '?')} {args.get('filing_type', '')} filed {args.get('filing_date') or '(latest)'}"
    with st.chat_message("assistant"):
        st.markdown(
            f"To answer this I need **{filing}**, which isn't indexed yet. Download and index it from "
            "SEC EDGAR? It takes a few minutes and adds it to the shared index."
        )
        st.caption(f"Plan step: {request['step']}")
        approve_column, decline_column = st.columns(2)
        approved = approve_column.button("Index it", type="primary", key="approve_index")
        declined = decline_column.button("Don't index", key="decline_index")
        if not (approved or declined):
            return
        st.session_state.pending = None
        outcome, result = _run_agent(graph, Command(resume={"approved": approved}), pending["config"])
    _settle(outcome, result, pending["config"])
    st.rerun()  # refresh the sidebar: a filing may have been indexed


def _handle_new_question(graph, filters: dict | None) -> None:
    """Take one new chat_input question and run it as the next turn of the conversation's thread."""
    question = st.chat_input(
        "Ask about the indexed filings, or about a company to fetch from SEC...",
        disabled=graph is None or bool(st.session_state.pending),
    )
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(_md(question))

    config = st.session_state.thread
    with st.chat_message("assistant"):
        outcome, result = _run_agent(graph, {"question": question, "filters": filters or {}}, config)
    _settle(outcome, result, config)


def main() -> None:
    """Load cached resources, render the sidebar, then the chat."""
    st.title("Financial Report Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending" not in st.session_state:
        st.session_state.pending = None
    if "thread" not in st.session_state:
        st.session_state.thread = new_thread()  # one agent thread per conversation

    embedding = _cached_embedding()
    try:
        store = _cached_store(embedding, _cached_sparse_embedding())
    except RuntimeError as e:
        logger.error("Failed to open the vector store: %s", e)
        st.error(str(e))
        st.stop()

    graph = None
    try:
        retriever = _cached_retriever(store, _cached_reranker())
        graph = _cached_sidekick(store, retriever)
    except Exception as e:
        logger.error("Failed to initialize the agent: %s", e, exc_info=True)
        st.error(f"Could not initialize the assistant (check GEMINI_API_KEY in .env): {e}")

    filters = _render_sidebar(store)

    if store.count() == 0:
        st.info(
            "No filings indexed yet. Ask about a company and the assistant will offer to fetch its filing "
            'from SEC EDGAR, or use "Index a new filing" in the sidebar.'
        )

    _render_chat_history()
    _render_pending_approval(graph)
    _handle_new_question(graph, filters)


if __name__ == "__main__":
    main()
