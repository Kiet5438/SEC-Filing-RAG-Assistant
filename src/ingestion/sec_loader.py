"""SEC EDGAR filing discovery and download.

Resolves a ticker to a CIK via the SEC's company_tickers.json, lists a
company's filings of a given form type via the JSON submissions API (its
"recent" window first, then the older history pages it links to, fetched
only when needed), downloads (and caches) the chosen filing's primary
document HTML, and returns a FilingRef describing it. The submissions API
is always tried first; a lightweight parse of the filing's own Archives
index page is used only as a fallback when the API response omits the
primary document (older filings). The EDGAR full-text search page is never
scraped.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.utils.config import (
    SEC_ARCHIVES_BASE,
    SEC_SUBMISSIONS_PAGE_URL,
    SEC_SUBMISSIONS_URL,
    SEC_TICKERS_URL,
    TICKERS_CACHE_PATH,
)
from src.utils.helpers import build_doc_id, http_get, raw_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FilingRef:
    """Reference to a single downloaded SEC filing."""

    doc_id: str
    ticker: str
    filing_type: str
    filing_date: str
    filing_name: str
    url: str
    source: str
    cik: str | None
    accession: str | None
    raw_path: Path


@dataclass(frozen=True)
class FilingListing:
    """One filing in a company's SEC submissions history (not downloaded)."""

    ticker: str
    company_name: str
    filing_type: str
    filing_date: str
    report_date: str  # period end date ("" if SEC lists none)
    accession: str
    primary_document: str

    @property
    def doc_id(self) -> str:
        return build_doc_id(self.ticker, self.filing_type, self.filing_date)


def list_filings(ticker: str, filing_type: str, limit: int = 10) -> list[FilingListing]:
    """List a company's filings of one form type, newest first, without downloading them.

    Reads the submissions API's "recent" window, then older history pages
    only until limit filings have been found.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
        filing_type: Exact SEC form string, e.g. "10-K" (amendments such as
            "10-K/A" are a different form and are not included).
        limit: Maximum number of filings to return.

    Returns:
        Up to limit FilingListings, newest filing_date first.

    Raises:
        ValueError: The ticker cannot be resolved to a CIK.
        RuntimeError: A network or SEC API request fails.
    """
    ticker = ticker.upper()
    cik = _resolve_cik(ticker)
    submissions = _fetch_submissions(cik)
    listings: list[FilingListing] = []
    for match in _iter_matching_filings(submissions, filing_type):
        listings.append(_to_listing(ticker, submissions, filing_type, match))
        if len(listings) >= limit:
            break
    return listings


def load_filing(ticker: str, filing_type: str, filing_date: str | None = None) -> FilingRef:
    """Discover and download one filing of a given type for a ticker.

    Resolves ticker -> CIK -> submissions API -> the matching filing (the
    latest, or the one filed on filing_date), then downloads (or reuses a
    cached copy of) the primary document HTML.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
        filing_type: Exact SEC form string, e.g. "10-K", "10-Q", "8-K".
        filing_date: ISO date the filing was filed (as list_filings reports
            it), or None for the latest filing of that type.

    Returns:
        A FilingRef describing and pointing to the downloaded filing.

    Raises:
        ValueError: Ticker cannot be resolved to a CIK, or no matching
            filing is found.
        RuntimeError: A network or SEC API request fails.
    """
    ticker = ticker.upper()
    cik = _resolve_cik(ticker)
    submissions = _fetch_submissions(cik)
    match = _find_filing(submissions, filing_type, filing_date)

    cik_int_str = str(int(cik))
    accession_nodash = match["accession"].replace("-", "")
    primary_document = match["primary_document"]
    if not primary_document:
        logger.warning(
            "Submissions API returned no primaryDocument for %s %s (accession=%s); "
            "falling back to parsing the filing's own Archives index page.",
            ticker,
            filing_type,
            match["accession"],
        )
        primary_document = _fallback_primary_document(cik_int_str, accession_nodash)

    url = f"{SEC_ARCHIVES_BASE}/{cik_int_str}/{accession_nodash}/{primary_document}"
    filing_date = match["filing_date"]
    doc_id = build_doc_id(ticker, filing_type, filing_date)
    entity_name = submissions.get("name", ticker)
    filing_name = f"{entity_name} {filing_type} {filing_date[:4]}"
    source = f"{doc_id}.html"

    downloaded_path = _get_or_download_raw(doc_id, url)

    return FilingRef(
        doc_id=doc_id,
        ticker=ticker,
        filing_type=filing_type,
        filing_date=filing_date,
        filing_name=filing_name,
        url=url,
        source=source,
        cik=cik,
        accession=match["accession"],
        raw_path=downloaded_path,
    )


def _get_or_download_raw(doc_id: str, url: str) -> Path:
    """Return the cached raw HTML path for doc_id, downloading it if absent."""
    path = raw_path(doc_id)
    if path.exists():
        logger.info("Raw filing already cached for doc_id=%s (%s); skipping download.", doc_id, path)
        return path

    logger.info("Downloading filing doc_id=%s from %s", doc_id, url)
    try:
        response = http_get(url)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download filing doc_id={doc_id} from {url}: {e}") from e

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def _load_ticker_map(force_refresh: bool = False) -> dict[str, str]:
    """Load the ticker->CIK map, using the on-disk cache unless refreshing.

    Args:
        force_refresh: If True, re-download company_tickers.json even if a
            cached copy exists (used when a ticker isn't found in the cache,
            since it may be stale for recently listed companies).
    """
    if TICKERS_CACHE_PATH.exists() and not force_refresh:
        try:
            with open(TICKERS_CACHE_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return _build_ticker_map(raw)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Ticker cache at %s is unreadable (%s); re-downloading.", TICKERS_CACHE_PATH, e)

    logger.info("Downloading SEC company_tickers.json to %s", TICKERS_CACHE_PATH)
    try:
        response = http_get(SEC_TICKERS_URL)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download SEC company_tickers.json: {e}") from e

    raw = response.json()
    TICKERS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TICKERS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    return _build_ticker_map(raw)


def _build_ticker_map(raw: dict) -> dict[str, str]:
    """Convert the raw company_tickers.json payload into {TICKER: zero-padded CIK}."""
    return {entry["ticker"].upper(): f"{int(entry['cik_str']):010d}" for entry in raw.values()}


def _resolve_cik(ticker: str) -> str:
    """Resolve a ticker to its 10-digit zero-padded CIK, refreshing the cache once if needed."""
    ticker_map = _load_ticker_map()
    cik = ticker_map.get(ticker)
    if cik is None:
        logger.info("Ticker %s not found in cached company_tickers.json; refreshing cache.", ticker)
        ticker_map = _load_ticker_map(force_refresh=True)
        cik = ticker_map.get(ticker)
    if cik is None:
        raise ValueError(
            f"Ticker '{ticker}' was not found in SEC's company_tickers.json. "
            "Verify the symbol is correct and currently registered with the SEC."
        )
    return cik


def _fetch_submissions(cik: str) -> dict:
    """Fetch the JSON submissions history for a given zero-padded CIK."""
    url = SEC_SUBMISSIONS_URL.format(cik=cik)
    try:
        response = http_get(url)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch SEC submissions for CIK {cik}: {e}") from e
    return response.json()


def _find_filing(submissions: dict, filing_type: str, filing_date: str | None) -> dict:
    """The latest filing of filing_type, or the one filed on filing_date.

    Raises:
        ValueError: No such filing in the company's submissions history.
    """
    for match in _iter_matching_filings(submissions, filing_type):
        if filing_date is None or match["filing_date"] == filing_date:
            return match
    entity = submissions.get("name", "this company")
    wanted = f"'{filing_type}' filed on {filing_date}" if filing_date else f"'{filing_type}' filings"
    raise ValueError(
        f"No {wanted} found for {entity} in its SEC submissions history. Confirm the exact SEC form "
        "string (e.g. '10-K', '10-Q') and, for a specific filing, the filing date list_filings reports."
    )


def _iter_matching_filings(submissions: dict, filing_type: str) -> Iterator[dict]:
    """Yield filings exactly matching filing_type, newest first, across all history pages.

    The "recent" window comes inline with the submissions response; older
    history lives in separate pages listed under filings.files, each fetched
    only once the newer ones are exhausted. Every table is columnar (one list
    per field) and sorted newest first.
    """
    for table in _iter_filing_tables(submissions):
        forms = table.get("form", [])
        rows = [
            {
                "accession": table["accessionNumber"][i],
                "filing_date": table["filingDate"][i],
                "report_date": _column_value(table, "reportDate", i),
                "primary_document": _column_value(table, "primaryDocument", i),
            }
            for i, form in enumerate(forms)
            if form == filing_type
        ]
        yield from sorted(rows, key=lambda row: row["filing_date"], reverse=True)


def _iter_filing_tables(submissions: dict) -> Iterator[dict]:
    """The "recent" filings table, then each older history page (fetched lazily)."""
    filings = submissions.get("filings", {})
    yield filings.get("recent", {})
    for page in filings.get("files", []):
        url = SEC_SUBMISSIONS_PAGE_URL.format(name=page["name"])
        try:
            yield http_get(url).json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch SEC submissions history page {url}: {e}") from e


def _column_value(table: dict, column: str, index: int) -> str:
    values = table.get(column, [])
    return values[index] if index < len(values) else ""


def _to_listing(ticker: str, submissions: dict, filing_type: str, match: dict) -> FilingListing:
    return FilingListing(
        ticker=ticker,
        company_name=submissions.get("name", ticker),
        filing_type=filing_type,
        filing_date=match["filing_date"],
        report_date=match["report_date"],
        accession=match["accession"],
        primary_document=match["primary_document"],
    )


def _fallback_primary_document(cik_int_str: str, accession_nodash: str) -> str:
    """Discover the primary document filename from the filing's own Archives index page.

    Used only when the submissions API omits primaryDocument. This parses the
    filing's official EDGAR directory listing, not the EDGAR full-text search UI.

    Raises:
        RuntimeError: The index page can't be fetched or no document is found.
    """
    index_url = f"{SEC_ARCHIVES_BASE}/{cik_int_str}/{accession_nodash}/"
    try:
        response = http_get(index_url)
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to fetch filing index page at {index_url}: {e}") from e

    soup = BeautifulSoup(response.text, "lxml")
    for link in soup.find_all("a"):
        href = link.get("href", "")
        if href.lower().endswith((".htm", ".html")) and "index" not in href.lower():
            return href.rsplit("/", 1)[-1]

    raise RuntimeError(f"Could not discover a primary document from filing index at {index_url}")
