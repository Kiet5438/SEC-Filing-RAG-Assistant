"""Document-level facts read from a filing's inline-XBRL cover page.

Every 10-K/10-Q primary document since 2019 is inline XBRL, and its cover
page tags a few Document and Entity Information (dei:) facts. Three of them
are worth stamping on every chunk because they can't be derived from the
filing date: the fiscal year (GOOGL's FY2025 10-K is filed in 2026; NVIDIA's
fiscal year runs a year ahead of the calendar), the fiscal period (FY, Q1,
Q2, Q3), and the period end date. Reading them here keeps rebuilds offline:
the raw HTML is already on disk, no SEC API call needed.
"""

import re
import warnings
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup, SoupStrainer, XMLParsedAsHTMLWarning

from src.utils.logger import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_FISCAL_YEAR = "dei:DocumentFiscalYearFocus"
_FISCAL_PERIOD = "dei:DocumentFiscalPeriodFocus"
_PERIOD_END = "dei:DocumentPeriodEndDate"
# Only the dei: facts are parsed, not the whole multi-megabyte document.
_DEI_STRAINER = SoupStrainer(attrs={"name": re.compile(r"^dei:")})
# "January 25, 2026", "June 30,2026", "January 25 , 2026" (nested ix tags), any spaces.
_MONTH_NAME_DATE = re.compile(r"([A-Za-z]+)\s*(\d{1,2})\s*,?\s*(\d{4})")
_FISCAL_PERIODS = {"FY", "Q1", "Q2", "Q3", "Q4"}

FILING_FACT_KEYS = ("fiscal_year", "fiscal_period", "period_end_date")


def extract_filing_facts(raw_html_path: str | Path) -> dict:
    """Read fiscal year, fiscal period, and period end date from a filing's dei: tags.

    Args:
        raw_html_path: Path to the raw inline-XBRL filing HTML.

    Returns:
        {"fiscal_year": int | None, "fiscal_period": "FY"|"Q1"|... | None,
        "period_end_date": "YYYY-MM-DD" | None}. A fact that's missing or
        unparseable is None rather than an error, so a non-XBRL filing still
        indexes, just without these fields.
    """
    path = Path(raw_html_path)
    soup = BeautifulSoup(path.read_bytes(), "lxml", parse_only=_DEI_STRAINER)
    facts = {
        "fiscal_year": _parse_year(_first_text(soup, _FISCAL_YEAR)),
        "fiscal_period": _parse_period(_first_text(soup, _FISCAL_PERIOD)),
        "period_end_date": _parse_date(_first_text(soup, _PERIOD_END)),
    }
    missing = [key for key, value in facts.items() if value is None]
    if missing:
        logger.warning("Could not read %s from the dei: tags of %s", missing, path.name)
    return facts


def _first_text(soup: BeautifulSoup, name: str) -> str | None:
    tag = soup.find(attrs={"name": name})
    return tag.get_text(" ", strip=True) if tag is not None else None


def _parse_year(text: str | None) -> int | None:
    return int(text) if text and text.strip().isdigit() else None


def _parse_period(text: str | None) -> str | None:
    period = (text or "").strip().upper()
    return period if period in _FISCAL_PERIODS else None


def _parse_date(text: str | None) -> str | None:
    match = _MONTH_NAME_DATE.search((text or "").replace("\xa0", " "))
    if match is None:
        return None
    try:
        return datetime.strptime(" ".join(match.groups()), "%B %d %Y").date().isoformat()
    except ValueError:
        return None
