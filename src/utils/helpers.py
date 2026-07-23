"""Shared, stateless helper functions used across ingestion, preprocessing,
and retrieval modules: doc_id naming, path resolution, HTTP with retries,
manifest read/write, and HTML table -> GFM Markdown conversion.
"""

import json
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.config import MANIFEST_PATH, PROCESSED_DIR, RAW_DIR, SEC_USER_AGENT
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_doc_id(ticker: str, filing_type: str, filing_date: str) -> str:
    """Build the canonical document id used to name files and tag metadata.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
        filing_type: Original SEC form name, e.g. "10-K".
        filing_date: ISO date string, e.g. "2025-02-21".

    Returns:
        A doc_id of the form "{TICKER}_{filing_type}_{filing_date}".
    """
    return f"{ticker.upper()}_{filing_type}_{filing_date}"


def parse_doc_id(doc_id: str) -> dict[str, str]:
    """Invert build_doc_id, recovering ticker, filing_type, and filing_date.

    Args:
        doc_id: A doc_id previously produced by build_doc_id.

    Returns:
        A dict with keys "ticker", "filing_type", "filing_date".

    Raises:
        ValueError: If doc_id does not have the expected 3-part structure.
    """
    parts = doc_id.split("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Malformed doc_id, expected 3 '_'-separated parts: {doc_id!r}")
    ticker, filing_type, filing_date = parts
    return {"ticker": ticker, "filing_type": filing_type, "filing_date": filing_date}


def raw_path(doc_id: str) -> Path:
    """Return the raw HTML file path for a given doc_id."""
    return RAW_DIR / f"{doc_id}.html"


def processed_path(doc_id: str) -> Path:
    """Return the processed Markdown file path for a given doc_id."""
    return PROCESSED_DIR / f"{doc_id}.md"


def http_get(
    url: str,
    extra_headers: dict[str, str] | None = None,
    timeout: int = 30,
) -> requests.Response:
    """GET a URL with the SEC user agent attached and automatic retries.

    Args:
        url: Target URL.
        extra_headers: Optional additional headers, merged over the defaults.
        timeout: Request timeout in seconds.

    Returns:
        The successful requests.Response object.

    Raises:
        requests.HTTPError: If the final response status is not 2xx.
    """
    headers = {"User-Agent": SEC_USER_AGENT}
    if extra_headers:
        headers.update(extra_headers)

    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    logger.info("GET %s", url)
    response = session.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


def read_manifest() -> dict[str, Any]:
    """Read the ingestion manifest, returning an empty dict if absent."""
    if not MANIFEST_PATH.exists():
        return {}
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def upsert_manifest(doc_id: str, entry: dict[str, Any]) -> None:
    """Insert or update a single doc_id entry in the manifest and persist it.

    Args:
        doc_id: The document id to key the entry under.
        entry: Manifest entry fields (ticker, filing_type, filing_date,
            source, url, cik, accession, indexed_at).
    """
    manifest = read_manifest()
    manifest[doc_id] = entry
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest updated for doc_id=%s", doc_id)


def manifest_get(doc_id: str) -> dict[str, Any] | None:
    """Look up a single manifest entry by doc_id, or None if not present."""
    return read_manifest().get(doc_id)


def _safe_span(value: str | None) -> int:
    """Parse an HTML colspan/rowspan attribute, defaulting invalid values to 1."""
    try:
        parsed = int(value) if value else 1
        return parsed if parsed > 0 else 1
    except ValueError:
        return 1


def html_table_to_markdown(table_tag: Any) -> str:
    """Convert a BeautifulSoup <table> tag into a GitHub-Flavored Markdown table.

    Builds an explicit grid so that colspan/rowspan and missing cells are
    handled tolerantly: a colspan'd cell's text is placed once, in its
    leftmost covered column, with the remaining spanned columns left blank
    (real-world filing HTML frequently uses colspan purely for layout
    spacing, so repeating the text into every spanned column would
    triplicate it); a rowspan'd cell's text is repeated down every covered
    row, since that does represent the same value applying to each row.
    Columns that end up blank across every row (pure spacer columns) are
    dropped entirely.

    Args:
        table_tag: A bs4.Tag for a <table> element.

    Returns:
        A GFM Markdown table as a string. Empty string if the table has no
        rows or no non-blank columns.
    """
    rows = table_tag.find_all("tr")
    if not rows:
        return ""

    grid: list[dict[int, str]] = []
    span_tracker: dict[int, list[Any]] = {}  # col -> [text, rows_remaining]
    known_width = 0

    for row in rows:
        cells = row.find_all(["td", "th"])
        current_row: dict[int, str] = {}
        col = 0
        cell_idx = 0

        while cell_idx < len(cells) or col < known_width:
            tracked = span_tracker.get(col)
            if tracked is not None and tracked[1] > 0:
                current_row[col] = tracked[0]
                tracked[1] -= 1
                if tracked[1] <= 0:
                    del span_tracker[col]
                col += 1
                continue

            if cell_idx < len(cells):
                cell = cells[cell_idx]
                cell_idx += 1
                text = cell.get_text(" ", strip=True).replace("|", "\\|")
                colspan = _safe_span(cell.get("colspan"))
                rowspan = _safe_span(cell.get("rowspan"))
                for i in range(colspan):
                    cell_text = text if i == 0 else ""
                    current_row[col + i] = cell_text
                    if rowspan > 1:
                        span_tracker[col + i] = [cell_text, rowspan - 1]
                col += colspan
                continue

            current_row.setdefault(col, "")
            col += 1

        known_width = max(known_width, col)
        grid.append(current_row)

    kept_cols = [c for c in range(known_width) if any(row.get(c, "").strip() for row in grid)]
    if not kept_cols:
        return ""

    def render_row(row: dict[int, str]) -> str:
        cells = [row.get(c, "") for c in kept_cols]
        return "| " + " | ".join(cells) + " |"

    header, *body = grid
    lines = [render_row(header), "| " + " | ".join(["---"] * len(kept_cols)) + " |"]
    lines.extend(render_row(r) for r in body)
    return "\n".join(lines)
