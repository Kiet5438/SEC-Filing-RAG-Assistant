"""SEC EDGAR filing discovery and download.

Resolves a ticker to a CIK via the SEC's company_tickers.json, looks up the
latest filing of a given form type via the JSON submissions API, downloads
(and caches) the primary document HTML, and returns a FilingRef describing
it. The submissions API is always tried first; a lightweight parse of the
filing's own Archives index page is used only as a fallback when the API
response omits the primary document (older filings). The EDGAR full-text
search page is never scraped.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from src.utils.config import SEC_ARCHIVES_BASE, SEC_SUBMISSIONS_URL, SEC_TICKERS_URL, TICKERS_CACHE_PATH
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


def load_filing(ticker: str, filing_type: str) -> FilingRef:
    """Discover and download the latest filing of a given type for a ticker.

    Resolves ticker -> CIK -> submissions API -> latest matching filing,
    then downloads (or reuses a cached copy of) the primary document HTML.

    Args:
        ticker: Stock ticker symbol, e.g. "NVDA".
        filing_type: Exact SEC form string, e.g. "10-K", "10-Q", "8-K".

    Returns:
        A FilingRef describing and pointing to the downloaded filing.

    Raises:
        ValueError: Ticker cannot be resolved to a CIK, or no filing of the
            requested type is found.
        RuntimeError: A network or SEC API request fails.
    """
    ticker = ticker.upper()
    cik = _resolve_cik(ticker)
    submissions = _fetch_submissions(cik)
    match = _select_latest_filing(submissions, filing_type)

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


def _select_latest_filing(submissions: dict, filing_type: str) -> dict:
    """Pick the most recent filing exactly matching filing_type from the recent window.

    Raises:
        ValueError: No filing of that form is present in the recent window.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_documents = recent.get("primaryDocument", [])

    matches = [
        {
            "accession": accessions[i],
            "filing_date": filing_dates[i],
            "primary_document": primary_documents[i] if i < len(primary_documents) else "",
        }
        for i, form in enumerate(forms)
        if form == filing_type
    ]
    if not matches:
        entity = submissions.get("name", "this company")
        raise ValueError(
            f"No '{filing_type}' filings found for {entity} in the SEC submissions 'recent' window. "
            "Confirm the exact SEC form string (e.g. '10-K', '10-Q', '8-K'); very old filings that "
            "have rolled out of the recent window are not currently supported."
        )
    return max(matches, key=lambda m: m["filing_date"])


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
