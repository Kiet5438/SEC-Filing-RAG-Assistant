"""Form-type-aware section-detection profiles.

Each SEC form type numbers its Items differently. A 10-K's Item numbers are
globally unique, but a 10-Q reuses Item 1-4 under both Part I and Part II
(Part I Item 1 = Financial Statements, Part II Item 1 = Legal Proceedings),
so section identity there requires Part scoping to stay unambiguous. A
FilingProfile captures those per-form rules — which Item headings to look
for, whether Part scoping applies, and how much following content a heading
needs to count as a real section — so parse_html.py stays form-agnostic:
adding a new form type means adding a profile HERE, not editing the parser.

This module is the single source of truth for section-detection regexes;
config.py no longer defines any.
"""

import re
from dataclasses import dataclass

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FilingProfile:
    """Per-form-type section-detection strategy.

    Attributes:
        filing_type: "10-K", "10-Q", or "GENERIC".
        item_patterns: Compiled regexes matching this form's Item headings.
            Empty for GENERIC (which uses heading-only structural detection).
        uses_parts: Whether Part I/II scoping applies (10-Q only).
        part_pattern: Regex matching a "PART I"/"Part II" heading, or None.
        min_content_chars: Minimum following-body length for a heading
            candidate to be confirmed as a real section.
    """

    filing_type: str
    item_patterns: list[re.Pattern]
    uses_parts: bool
    part_pattern: re.Pattern | None
    min_content_chars: int


def _item_pattern(item: str) -> re.Pattern:
    """Compile a heading regex for one SEC Item label (e.g. "1A", "7", "10").

    Mirrors the historical pattern shape exactly ("^Item <n>.? <space>") so
    10-K detection — and therefore 10-K output — is unchanged by the move to
    profiles.
    """
    return re.compile(rf"^Item\s+{item}\.?\s+", re.IGNORECASE)


# Roman numerals longest-first so alternation doesn't stop early on "II".
_PART_PATTERN = re.compile(r"^Part\s+(IV|III|II|I)\b", re.IGNORECASE)

_TEN_K_ITEMS = [
    "1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8",
    "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16",
]
# Part I: Items 1-4. Part II: Items 1-6 (1A is Risk Factors, standard in 10-Qs).
_TEN_Q_ITEMS = ["1", "1A", "2", "3", "4", "5", "6"]

TEN_K = FilingProfile(
    filing_type="10-K",
    item_patterns=[_item_pattern(i) for i in _TEN_K_ITEMS],
    uses_parts=False,
    part_pattern=None,
    min_content_chars=1,
)

TEN_Q = FilingProfile(
    filing_type="10-Q",
    item_patterns=[_item_pattern(i) for i in _TEN_Q_ITEMS],
    uses_parts=True,
    part_pattern=_PART_PATTERN,
    min_content_chars=1,
)

GENERIC = FilingProfile(
    filing_type="GENERIC",
    item_patterns=[],
    uses_parts=False,
    part_pattern=None,
    # Heading-only detection has no Item-pattern filter, so require more
    # following content to avoid treating every stray heading as a section.
    min_content_chars=200,
)

PROFILES: dict[str, FilingProfile] = {
    "10-K": TEN_K,
    "10-Q": TEN_Q,
}


def get_profile(filing_type: str) -> FilingProfile:
    """Return the FilingProfile for a form type, or GENERIC if unregistered.

    Args:
        filing_type: Exact SEC form string, e.g. "10-K".

    Returns:
        The registered profile, or GENERIC (with a logged warning naming the
        unsupported type) for anything not in PROFILES.
    """
    profile = PROFILES.get(filing_type)
    if profile is None:
        logger.warning(
            "No FilingProfile registered for filing_type=%r; using GENERIC heading-only detection.",
            filing_type,
        )
        return GENERIC
    return profile
