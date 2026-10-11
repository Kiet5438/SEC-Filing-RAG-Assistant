"""Indexing CLI: ingest, rebuild, or remove SEC filings in the shared Qdrant store.

Usage:
    python main.py --ticker NVDA --filing 10-K
    python main.py --ticker NVDA --filing 10-K --filing-date 2025-02-26
    python main.py --force-rebuild
    python main.py --remove NVDA_10-K_2026-02-25

A thin wrapper: the pipeline itself lives in src/ingestion/pipeline.py,
shared with the Streamlit app and the agent's tools.
"""

import argparse

from src.ingestion.pipeline import index_filing, remove_filing
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Thin CLI wrapper: parse args, call index_filing / remove_filing, log the result."""
    parser = argparse.ArgumentParser(description="Index, rebuild, or remove SEC filings in the shared Qdrant store.")
    parser.add_argument("--ticker", help="Stock ticker, e.g. NVDA")
    parser.add_argument("--filing", help="Exact SEC form string, e.g. 10-K")
    parser.add_argument("--filing-date", help="ISO filing date of a specific filing (default: the latest)")
    parser.add_argument("--force-rebuild", action="store_true", help="Rebuild the entire store from data/processed/*.md")
    parser.add_argument("--remove", metavar="DOC_ID", help="Remove one filing, e.g. NVDA_10-K_2026-02-25")
    args = parser.parse_args()

    if args.force_rebuild and args.remove:
        parser.error("--force-rebuild and --remove can't be combined.")
    if not (args.force_rebuild or args.remove) and not (args.ticker and args.filing):
        parser.error("--ticker and --filing are required unless --force-rebuild or --remove is given.")

    try:
        if args.remove:
            result = remove_filing(args.remove)
        else:
            result = index_filing(args.ticker, args.filing, args.filing_date, force_rebuild=args.force_rebuild)
    except Exception as e:
        logger.error("Command failed (ticker=%s filing=%s remove=%s): %s", args.ticker, args.filing, args.remove, e)
        return

    if result["status"] == "skipped":
        logger.info("Result: doc_id=%s already indexed; nothing to do.", result["doc_id"])
    elif result["status"] == "rebuilt":
        logger.info("Result: rebuilt the store (%d total chunks).", result["chunk_count"])
    elif result["status"] == "removed":
        logger.info("Result: removed doc_id=%s (%d chunks).", result["doc_id"], result["chunk_count"])
    elif result["status"] == "not_found":
        logger.info("Result: doc_id=%s was not indexed; nothing removed.", result["doc_id"])
    else:
        logger.info("Result: indexed doc_id=%s (%d chunks).", result["doc_id"], result["chunk_count"])


if __name__ == "__main__":
    main()
