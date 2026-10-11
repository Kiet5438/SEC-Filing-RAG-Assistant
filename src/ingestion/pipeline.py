"""The indexing pipeline: ingest, rebuild, or remove SEC filings in the shared store.

index_filing() is the single indexing code path — the main.py CLI, the
Streamlit app's indexing form and the agent's index_filing tool all call it,
so there is exactly one place that knows how ingestion -> parsing ->
chunking -> embedding -> upsert fits together. No retrieval, no generation
here.
"""

import time
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timezone

from langchain_core.documents import Document

from src.embeddings.embedder import get_embedding_model, get_sparse_embedding_model
from src.ingestion.sec_loader import FilingRef, load_filing
from src.preprocessing.chunking import ChunkedFiling, chunk_markdown
from src.preprocessing.filing_facts import FILING_FACT_KEYS, extract_filing_facts
from src.preprocessing.parse_html import parse_html
from src.retrieval.vector_store import FilingStore, open_store
from src.utils.config import PROCESSED_DIR
from src.utils.helpers import manifest_get, raw_path, read_manifest, remove_manifest_entry, upsert_manifest
from src.utils.logger import get_logger

logger = get_logger(__name__)


def index_filing(
    ticker: str,
    filing_type: str,
    filing_date: str | None = None,
    *,
    force_rebuild: bool = False,
    store: FilingStore | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Index one filing, or rebuild the whole store from processed Markdown.

    This is the one and only indexing code path in the project (see the
    module docstring for its callers).

    Args:
        ticker: Stock ticker, e.g. "NVDA". Ignored when force_rebuild=True.
        filing_type: Exact SEC form string, e.g. "10-K". Ignored when force_rebuild=True.
        filing_date: ISO filing date of a specific filing (as
            sec_loader.list_filings reports it), or None for the latest one.
            Ignored when force_rebuild=True.
        force_rebuild: If True, rebuild the entire store from
            data/processed/*.md (recovering filing_meta from the manifest)
            instead of indexing a single new filing.
        store: An already-open FilingStore to write into. Pass the caller's
            own store when it has one open (the Streamlit app does): in
            local mode a second client on the same folder would fail. If
            None, a store is opened for this call and closed afterwards.
        progress: Optional callback receiving a short message as each stage
            of a single-filing index starts (download, parse, embed), e.g.
            for the agent to stream to the UI.

    Returns:
        {"doc_id": str | None, "status": "indexed" | "skipped" | "rebuilt",
        "chunk_count": int}, chunk_count counting the embedded (child)
        chunks. doc_id is None for "rebuilt", since many filings are
        involved rather than one.

    Raises:
        ValueError: The ticker/form couldn't be resolved to a filing, or
            chunking produced zero chunks.
        RuntimeError: A network or SEC API request failed, or the local
            store is already open in another client.
    """
    owns_store = store is None
    if owns_store:
        store = open_store(get_embedding_model(), get_sparse_embedding_model())
    try:
        if force_rebuild:
            chunk_count = _force_rebuild(store)
            return {"doc_id": None, "status": "rebuilt", "chunk_count": chunk_count}
        return _index_single_filing(ticker, filing_type, filing_date, store, progress or _no_progress)
    finally:
        if owns_store:
            store.close()


def remove_filing(doc_id: str, store: FilingStore | None = None) -> dict:
    """Remove one filing's chunks (children and parents) from the store and its manifest entry.

    The raw HTML and processed Markdown are left on disk (they're caches):
    re-indexing the same filing later reuses them instead of re-downloading,
    and --force-rebuild skips them since they no longer have a manifest entry.

    Args:
        doc_id: The filing to remove, e.g. "NVDA_10-K_2026-02-25".
        store: An already-open FilingStore, or None to open one for this call.

    Returns:
        {"doc_id": str, "status": "removed" | "not_found", "chunk_count": int},
        chunk_count being how many child chunks were deleted.
    """
    owns_store = store is None
    if owns_store:
        store = open_store(get_embedding_model(), get_sparse_embedding_model())
    try:
        deleted = store.delete_filing(doc_id)
        in_manifest = remove_manifest_entry(doc_id)
    finally:
        if owns_store:
            store.close()

    status = "removed" if deleted or in_manifest else "not_found"
    return {"doc_id": doc_id, "status": status, "chunk_count": deleted}


def _no_progress(message: str) -> None:
    pass


def _index_single_filing(
    ticker: str, filing_type: str, filing_date: str | None, store: FilingStore, progress: Callable[[str], None]
) -> dict:
    """Ingest one filing if it isn't already indexed: download -> parse -> chunk -> embed -> upsert."""
    call_start = time.time()
    progress(f"Fetching {ticker.upper()} {filing_type} {filing_date or '(latest)'} from SEC EDGAR")
    ref = load_filing(ticker, filing_type, filing_date)
    # A 1s buffer guards against filesystem timestamp truncation; write_bytes()
    # on the fresh-download path always sets mtime to "now", so a file that
    # predates this call's start was necessarily reused from cache, not written by it.
    downloaded_this_run = ref.raw_path.stat().st_mtime >= call_start - 1.0

    if ref.doc_id in read_manifest() or ref.doc_id in store.doc_ids():
        logger.info("doc_id=%s is already indexed; skipping.", ref.doc_id)
        return {"doc_id": ref.doc_id, "status": "skipped", "chunk_count": 0}

    progress(f"Parsing and chunking {ref.doc_id}")
    facts = extract_filing_facts(ref.raw_path)
    chunked = _parse_and_chunk_filing(ref, facts, downloaded_this_run)

    progress(f"Embedding {len(chunked.children)} chunks of {ref.doc_id} (a few minutes on CPU)")
    store.add_filing(chunked.parents, chunked.children)
    _record_manifest_entry(ref, facts)

    logger.info("Indexed doc_id=%s (%d parents, %d children).", ref.doc_id, len(chunked.parents), len(chunked.children))
    return {"doc_id": ref.doc_id, "status": "indexed", "chunk_count": len(chunked.children)}


def _parse_and_chunk_filing(ref: FilingRef, facts: dict, downloaded_this_run: bool) -> ChunkedFiling:
    """Parse the raw HTML and chunk it; on failure, clean up a freshly-downloaded file and re-raise."""
    try:
        md_path = parse_html(ref.raw_path)
        chunked = chunk_markdown(md_path, filing_meta={**asdict(ref), **facts})
        if not chunked.children:
            raise ValueError(
                f"chunk_markdown produced zero chunks for doc_id={ref.doc_id}; "
                "the parsed Markdown may be empty or malformed."
            )
    except Exception as e:
        if downloaded_this_run:
            ref.raw_path.unlink(missing_ok=True)
            logger.error(
                "Parsing/chunking failed for doc_id=%s; deleted the freshly-downloaded raw HTML "
                "(%s) so the next attempt re-downloads instead of reusing a possibly truncated or "
                "error-page file. Cause: %s",
                ref.doc_id,
                ref.raw_path,
                e,
            )
        else:
            logger.error(
                "Parsing/chunking failed for doc_id=%s using raw HTML reused from a previous run "
                "(%s); left it in place since this run didn't download it. Cause: %s",
                ref.doc_id,
                ref.raw_path,
                e,
            )
        raise
    return chunked


def _record_manifest_entry(ref: FilingRef, facts: dict) -> None:
    """Upsert the manifest entry for a freshly indexed filing, fiscal facts included."""
    upsert_manifest(
        ref.doc_id,
        {
            "ticker": ref.ticker,
            "filing_type": ref.filing_type,
            "filing_date": ref.filing_date,
            "filing_name": ref.filing_name,
            "source": ref.source,
            "url": ref.url,
            "cik": ref.cik,
            "accession": ref.accession,
            **facts,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _force_rebuild(store: FilingStore) -> int:
    """Rebuild the entire store from data/processed/*.md, recovering filing_meta from the manifest.

    Every file is chunked before the store is touched, so a chunking
    failure can't leave it half-wiped. Only then are the collection and the
    parent store emptied and refilled — a clean slate, so chunks from a
    changed chunking scheme never linger next to the new ones.

    Manifest entries written before fiscal facts existed are backfilled
    from the cached raw HTML (still offline), and the manifest is updated.

    Returns:
        Total child-chunk count across every rebuilt file (0 if nothing was rebuilt).
    """
    md_files = sorted(PROCESSED_DIR.glob("*.md"))
    if not md_files:
        logger.warning("No processed Markdown files found in %s; nothing to rebuild.", PROCESSED_DIR)
        return 0

    all_parents: list[Document] = []
    all_children: list[Document] = []
    rebuilt_files = 0
    for md_path in md_files:
        doc_id = md_path.stem
        entry = manifest_get(doc_id)
        if entry is None:
            logger.warning("No manifest entry for doc_id=%s (%s); skipping.", doc_id, md_path)
            continue

        entry = _backfill_filing_facts(doc_id, entry)
        try:
            chunked = chunk_markdown(md_path, filing_meta={"doc_id": doc_id, **entry})
        except ValueError as e:
            logger.error("Failed to chunk %s: %s", md_path, e)
            continue

        all_parents.extend(chunked.parents)
        all_children.extend(chunked.children)
        rebuilt_files += 1
        logger.info("Recovered %d children for doc_id=%s from manifest.", len(chunked.children), doc_id)

    if not all_children:
        logger.warning("No chunks recovered from any file; the store was not rebuilt.")
        return 0

    store.reset()
    store.add_filing(all_parents, all_children)
    logger.info(
        "Rebuilt the store from %d file(s): %d parents, %d children.", rebuilt_files, len(all_parents), len(all_children)
    )
    return len(all_children)


def _backfill_filing_facts(doc_id: str, entry: dict) -> dict:
    """Add fiscal facts to a pre-facts manifest entry from the cached raw HTML, persisting them.

    Leaves the entry unchanged if it already has them, or if the raw HTML
    isn't cached (the facts then stay absent and the chunks carry None).
    """
    if all(key in entry for key in FILING_FACT_KEYS):
        return entry
    html_path = raw_path(doc_id)
    if not html_path.exists():
        logger.warning("No cached raw HTML for doc_id=%s; indexing it without fiscal facts.", doc_id)
        return entry
    entry = {**entry, **extract_filing_facts(html_path)}
    upsert_manifest(doc_id, entry)
    return entry
