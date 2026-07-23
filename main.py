"""Indexing entry point: ingest one SEC filing into the shared FAISS vector store.

Usage:
    python main.py --ticker NVDA --filing 10-K
    python main.py --ticker NVDA --filing 10-K --force-rebuild

index_filing() is the single indexing code path — both this CLI and
app/streamlit_app.py's indexing form call it directly, so there is exactly
one place that knows how ingestion -> parsing -> chunking -> embedding ->
save fits together. No retrieval, no generation here.
"""

import argparse
import time
from dataclasses import asdict
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.embeddings.embedder import get_embedding_model
from src.ingestion.sec_loader import FilingRef, load_filing
from src.preprocessing.chunking import chunk_markdown
from src.preprocessing.parse_html import parse_html
from src.retrieval.vector_store import build_vectorstore, indexed_doc_ids, load_vectorstore, save_vectorstore, update_vectorstore
from src.utils.config import PROCESSED_DIR
from src.utils.helpers import manifest_get, read_manifest, upsert_manifest
from src.utils.logger import get_logger

logger = get_logger(__name__)


def index_filing(ticker: str, filing_type: str, force_rebuild: bool = False) -> dict:
    """Index one filing, or rebuild the whole index from processed Markdown.

    This is the one and only indexing code path in the project: main()
    below and app/streamlit_app.py's indexing form both call this directly
    rather than duplicating any pipeline logic.

    Args:
        ticker: Stock ticker, e.g. "NVDA". Ignored when force_rebuild=True.
        filing_type: Exact SEC form string, e.g. "10-K". Ignored when force_rebuild=True.
        force_rebuild: If True, rebuild the entire index from
            data/processed/*.md (recovering filing_meta from the manifest)
            instead of indexing a single new filing.

    Returns:
        {"doc_id": str | None, "status": "indexed" | "skipped" | "rebuilt",
        "chunk_count": int}. doc_id is None for "rebuilt", since many
        filings are involved rather than one.

    Raises:
        ValueError: The ticker/form couldn't be resolved to a filing, or
            chunking produced zero chunks.
        RuntimeError: A network or SEC API request failed.
    """
    embedding = get_embedding_model()

    if force_rebuild:
        chunk_count = _force_rebuild(embedding)
        return {"doc_id": None, "status": "rebuilt", "chunk_count": chunk_count}

    return _index_single_filing(ticker, filing_type, embedding)


def _index_single_filing(ticker: str, filing_type: str, embedding: HuggingFaceEmbeddings) -> dict:
    """Ingest one filing if it isn't already indexed: download -> parse -> chunk -> embed -> save."""
    call_start = time.time()
    ref = load_filing(ticker, filing_type)
    # A 1s buffer guards against filesystem timestamp truncation; write_bytes()
    # on the fresh-download path always sets mtime to "now", so a file that
    # predates this call's start was necessarily reused from cache, not written by it.
    downloaded_this_run = ref.raw_path.stat().st_mtime >= call_start - 1.0

    vectorstore = load_vectorstore(embedding)
    if ref.doc_id in read_manifest() or ref.doc_id in indexed_doc_ids(vectorstore):
        logger.info("doc_id=%s is already indexed; skipping.", ref.doc_id)
        return {"doc_id": ref.doc_id, "status": "skipped", "chunk_count": 0}

    documents = _parse_and_chunk_filing(ref, downloaded_this_run)

    vectorstore = update_vectorstore(vectorstore, documents, embedding)
    save_vectorstore(vectorstore)
    _record_manifest_entry(ref)

    logger.info("Indexed doc_id=%s (%d chunks).", ref.doc_id, len(documents))
    return {"doc_id": ref.doc_id, "status": "indexed", "chunk_count": len(documents)}


def _parse_and_chunk_filing(ref: FilingRef, downloaded_this_run: bool) -> list[Document]:
    """Parse the raw HTML and chunk it; on failure, clean up a freshly-downloaded file and re-raise."""
    try:
        md_path = parse_html(ref.raw_path)
        documents = chunk_markdown(md_path, filing_meta=asdict(ref))
        if not documents:
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
    return documents


def _record_manifest_entry(ref: FilingRef) -> None:
    """Upsert the manifest entry for a freshly indexed filing."""
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
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _force_rebuild(embedding: HuggingFaceEmbeddings) -> int:
    """Rebuild the entire index from data/processed/*.md, recovering filing_meta from the manifest.

    Builds one fresh index from every recovered chunk and saves it in a
    single call; FAISS has no partial delete/replace, so a full rebuild is
    always all-or-nothing rather than file-by-file.

    Returns:
        Total chunk count across every rebuilt file (0 if nothing was rebuilt).
    """
    md_files = sorted(PROCESSED_DIR.glob("*.md"))
    if not md_files:
        logger.warning("No processed Markdown files found in %s; nothing to rebuild.", PROCESSED_DIR)
        return 0

    all_documents: list[Document] = []
    for md_path in md_files:
        doc_id = md_path.stem
        entry = manifest_get(doc_id)
        if entry is None:
            logger.warning("No manifest entry for doc_id=%s (%s); skipping.", doc_id, md_path)
            continue

        filing_meta = {"doc_id": doc_id, **entry}
        try:
            documents = chunk_markdown(md_path, filing_meta=filing_meta)
        except ValueError as e:
            logger.error("Failed to chunk %s: %s", md_path, e)
            continue

        all_documents.extend(documents)
        logger.info("Recovered %d chunks for doc_id=%s from manifest.", len(documents), doc_id)

    if not all_documents:
        logger.warning("No chunks recovered from any file; vector store was not rebuilt.")
        return 0

    vectorstore = build_vectorstore(all_documents, embedding)
    save_vectorstore(vectorstore)
    logger.info("Rebuilt FAISS index from %d file(s), %d total chunks.", len(md_files), len(all_documents))
    return len(all_documents)


def main() -> None:
    """Thin CLI wrapper: parse args, call index_filing, log the result."""
    parser = argparse.ArgumentParser(description="Index a SEC filing into the shared FAISS vector store.")
    parser.add_argument("--ticker", required=True, help="Stock ticker, e.g. NVDA")
    parser.add_argument("--filing", required=True, help="Exact SEC form string, e.g. 10-K")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild the entire index from data/processed/*.md")
    args = parser.parse_args()

    try:
        result = index_filing(args.ticker, args.filing, force_rebuild=args.force_rebuild)
    except Exception as e:
        logger.error("Indexing failed for ticker=%s filing=%s: %s", args.ticker, args.filing, e)
        return

    if result["status"] == "skipped":
        logger.info("Result: doc_id=%s already indexed; nothing to do.", result["doc_id"])
    elif result["status"] == "rebuilt":
        logger.info("Result: rebuilt the index (%d total chunks).", result["chunk_count"])
    else:
        logger.info("Result: indexed doc_id=%s (%d chunks).", result["doc_id"], result["chunk_count"])


if __name__ == "__main__":
    main()
