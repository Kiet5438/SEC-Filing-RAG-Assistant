"""HTML -> Markdown parsing for SEC filings.

Cleans a raw filing HTML file, detects section (Item) headings while
preserving original document order, converts financial tables to GFM
Markdown, and writes a single Markdown file. Emits no Document objects and
attaches no metadata. Heading Contract: every real section heading is an
H2, and nothing else is; sub-headings inside a section (short all-bold
blocks) are H3; every block is its own blank-line-separated paragraph; page
furniture (running headers, page numbers, ToC back-links) is dropped.

Section detection is form-type-aware via a FilingProfile (filing_profiles.py):
the profile supplies the Item regexes, whether Part scoping applies, and the
minimum following-content threshold. 10-Q section strings are Part-qualified
("Part II — Item 1. Legal Proceedings") because a 10-Q reuses Item 1-4 under
both Parts; 10-K strings stay exactly as the raw H2 text.
"""

import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from src.preprocessing.filing_profiles import FilingProfile, get_profile
from src.utils.helpers import html_table_to_markdown, parse_doc_id, processed_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# SEC inline-XBRL filings are XHTML with an XML declaration/xbrl namespaces up
# top, which makes bs4 suspect the document is XML. We deliberately parse as
# HTML per spec ("lxml") regardless; the unrecognized ix:/xbrli: tags are
# still walked as generic elements and their text is extracted normally, so
# this warning doesn't reflect an actual parsing problem here.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

_HEADING_TAGS = {"h1", "h2", "h3", "b", "strong", "a"}
_GENERIC_HEADING_TAGS = {"h1", "h2", "h3"}
_LEAF_BLOCK_TAGS = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "b", "strong", "a"}
_CONTAINER_CHECK_TAGS = ["p", "div", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"]
_HIDDEN_STYLE = re.compile(r"display:\s*none", re.IGNORECASE)
_BOLD_STYLE = re.compile(r"font-weight:\s*(bold|[6-9]00)", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")  # includes non-breaking spaces
# Leading labels that filings separate from the following text only with CSS
# spacing, so a rendering-faithful join glues them on: "ITEM 1.BUSINESS",
# "•content", "(1)Includes". A space is restored after them.
_GLUED_LEADING_LABEL = re.compile(r"^(item\s+\d{1,2}[a-c]?\.|[•●◦▪‣]|\((?:\d{1,2}|[a-z])\))(?=[^\s\d])", re.IGNORECASE)

_MIN_CONFIDENT_STRUCTURAL_SECTIONS = 3
_PART_SEPARATOR = " — "  # " — " (em dash), e.g. "Part II — Item 1. Legal Proceedings"

# An all-bold block up to this long is a sub-heading (H3). Long enough to
# cover a risk factor's one-sentence bold headline, so each risk factor
# becomes its own subsection; longer bold blocks stay body text.
_SUBHEADING_MAX_CHARS = 400
# Bold bullet points (e.g. a summary list in MD&A) are list items, not headings.
_BULLET_PREFIX = re.compile(r"^[•●◦▪‣\-–]\s")
# Short blocks repeated this often are page furniture (running headers like
# "NVIDIA Corporation and Subsidiaries" or "(Continued)"), not content. Real
# sub-headings do repeat across segments ("Google Cloud", "Competition"),
# but measured at most 6 times in the indexed filings.
_RUNNING_HEADER_MIN_REPEATS = 8
_RUNNING_HEADER_MAX_CHARS = 120
_NOISE_PATTERNS = [
    re.compile(r"^table of contents$", re.IGNORECASE),
    re.compile(r"^(page\s+)?\d{1,3}\.?$", re.IGNORECASE),  # page numbers ("37", "37.", "Page 37")
    re.compile(r"^part\s+(iv|iii|ii|i)$", re.IGNORECASE),  # bare Part label (divider/running header)
    re.compile(r"^item\s+\d{1,2}[a-c]?\.?$", re.IGNORECASE),  # bare Item label (running header)
    re.compile(r"^[_\-=\s]{5,}$"),  # horizontal-rule lines
]


@dataclass
class _Block:
    """A single piece of extracted document content, in original DOM order."""

    kind: str  # "text" | "table"
    tag_name: str
    text: str
    length: int
    bold: bool = False  # every visible character is bold (tag or CSS font-weight)


def parse_html(raw_html_path: str | Path, filing_type: str | None = None) -> Path:
    """Parse a raw filing HTML file into a section-aware Markdown file.

    Args:
        raw_html_path: Path to the raw HTML file (data/raw/{doc_id}.html).
        filing_type: SEC form string selecting the detection profile. When
            None (the normal path), it is derived from the doc_id filename
            via parse_doc_id, so existing single-argument call sites keep
            working unchanged; the override exists for testing.

    Returns:
        Path to the written Markdown file (data/processed/{doc_id}.md).

    Raises:
        FileNotFoundError: raw_html_path does not exist.
    """
    path = Path(raw_html_path)
    if filing_type is None:
        filing_type = parse_doc_id(path.stem)["filing_type"]
    profile = get_profile(filing_type)

    logger.info("Parsing %s (profile=%s)", path, profile.filing_type)
    raw_bytes = path.read_bytes()
    soup = BeautifulSoup(raw_bytes, "lxml")

    title = _derive_title(path)
    soup = _clean(soup, profile)
    blocks = _extract_blocks(soup)

    confirmed = _detect_sections(blocks, profile)
    markdown = _assemble_markdown(blocks, confirmed, title, profile)

    out_path = processed_path(path.stem)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    logger.info("Wrote %s (%d confirmed sections)", out_path, len(confirmed))
    return out_path


def _derive_title(raw_html_path: Path) -> str:
    """Derive the document H1 title from the doc_id filename stem.

    SEC inline-XBRL filings routinely set <title> to the raw document
    filename (e.g. "nvda-20260125"), so it isn't a reliable source of a
    human-readable title; the doc_id (e.g. "NVDA_10-K_2025-02-21") already
    encodes ticker/form/date and is guaranteed present and consistent.
    """
    return raw_html_path.stem.replace("_", " ")


def _clean(soup: BeautifulSoup, profile: FilingProfile) -> BeautifulSoup:
    """Strip scripts/styles/nav/hidden elements/ToC blocks from the soup, in place."""
    for tag in soup.find_all(["script", "style", "noscript", "nav", "head"]):
        tag.decompose()
    for tag in soup.find_all(style=_HIDDEN_STYLE):
        tag.decompose()
    _strip_toc(soup, profile)
    return soup


def _strip_toc(soup: BeautifulSoup, profile: FilingProfile) -> None:
    """Remove table-of-contents blocks: containers dense with internal-anchor jump links.

    Two independent signals, either sufficient on its own: several anchors
    whose own text matches an Item pattern (e.g. "Item 7. Management's..."
    as one link), or a high raw count of same-page anchors regardless of
    text (e.g. ToC layouts that split the item number and title into
    separate linked cells, so neither one's text alone matches).
    """
    for container in soup.find_all(["table", "div"]):
        if container.decomposed:
            continue
        internal_anchors = [a for a in container.find_all("a", href=True) if a["href"].startswith("#")]
        item_text_matches = sum(1 for a in internal_anchors if _is_item_heading(a.get_text(strip=True), profile))
        if item_text_matches >= 3 or len(internal_anchors) >= 8:
            container.decompose()


def _is_item_heading(text: str, profile: FilingProfile) -> bool:
    """Return True if text starts with an Item heading recognized by the profile."""
    return any(pattern.match(text) for pattern in profile.item_patterns)


def _extract_blocks(soup: BeautifulSoup) -> list[_Block]:
    """Flatten the cleaned soup into an ordered list of text/table content blocks.

    Only "leaf" content elements are emitted (elements with no nested
    block-level descendant), so a wrapping <div>/<p> around several
    paragraphs is skipped in favor of its children, and a heading tag
    nested inside another heading tag (e.g. <h2><strong>...</strong></h2>)
    is recorded only once. Tables are emitted whole, converted to GFM.
    """
    blocks: list[_Block] = []
    consumed: set[int] = set()
    body = soup.body or soup

    for el in body.find_all(True):
        if not isinstance(el, Tag) or id(el) in consumed:
            continue

        if el.name == "table":
            if el.find_parent("table") is not None:
                continue
            markdown_table = html_table_to_markdown(el)
            if markdown_table.strip():
                blocks.append(_Block(kind="table", tag_name="table", text=markdown_table, length=len(markdown_table)))
            for descendant in el.find_all(True):
                consumed.add(id(descendant))
            continue

        if el.find_parent("table") is not None:
            continue

        if el.name not in _LEAF_BLOCK_TAGS:
            continue
        if el.find(_CONTAINER_CHECK_TAGS) is not None:
            continue

        text = _block_text(el)
        if not text:
            continue
        blocks.append(_Block(kind="text", tag_name=el.name, text=text, length=len(text), bold=_is_all_bold(el)))
        for descendant in el.find_all(True):
            consumed.add(id(descendant))

    return blocks


def _block_text(el: Tag) -> str:
    """A leaf block's visible text, joined the way a browser renders it.

    A leaf block holds only inline content, so its text runs are
    concatenated with no separator: filings split single words across
    <span>s for kerning ("RIS" + "K FACTORS", "CORP" + "ORATION"), which a
    space-separated join turns into broken words, and put punctuation in its
    own span ("2025" + ", "), which it turns into "2025 ,". <br> becomes a
    space, whitespace runs (incl. non-breaking spaces) collapse to one
    space, and a space is restored after a CSS-spaced leading label.
    """
    for br in el.find_all("br"):
        br.replace_with(" ")
    text = _WHITESPACE.sub(" ", el.get_text("")).strip()
    return _GLUED_LEADING_LABEL.sub(r"\1 ", text)


def _is_all_bold(el: Tag) -> bool:
    """True if every non-blank text node under el is bold, via <b>/<strong> or CSS font-weight.

    Inline-XBRL filings rarely use <b>: emphasis is a style on a <span>
    (e.g. "font-weight:700" or "font-weight:bold"), so both are checked on
    every ancestor between the text node and el itself.
    """
    strings = [s for s in el.find_all(string=True) if s.strip()]
    if not strings:
        return False
    for string in strings:
        node = string.parent
        while node is not None:
            if node.name in {"b", "strong"} or _BOLD_STYLE.search(node.get("style", "")):
                break
            if node is el:
                return False
            node = node.parent
        else:
            return False
    return True


def _filter_sufficient_content(blocks: list[_Block], candidate_idxs: list[int], min_content_chars: int) -> list[int]:
    """Keep only heading candidates followed by enough content before the next candidate/end."""
    confirmed = []
    for pos, idx in enumerate(candidate_idxs):
        next_idx = candidate_idxs[pos + 1] if pos + 1 < len(candidate_idxs) else len(blocks)
        following_len = sum(b.length for b in blocks[idx + 1 : next_idx])
        if following_len >= min_content_chars:
            confirmed.append(idx)
    return confirmed


def _detect_sections(blocks: list[_Block], profile: FilingProfile) -> list[int]:
    """Detect confirmed section-heading block indices per the profile's strategy."""
    if not profile.item_patterns:
        return _sections_generic(blocks, profile)
    confirmed = _sections_structural(blocks, profile)
    if confirmed is None:
        logger.info("Structural heading detection not confident; falling back to regex.")
        confirmed = _sections_regex(blocks, profile)
    return confirmed


def _item_heading_idxs(blocks: list[_Block], profile: FilingProfile, tag_names: set[str] | None = None) -> list[int]:
    """Indices of text blocks matching an Item pattern, optionally restricted to tag_names."""
    return [
        i
        for i, b in enumerate(blocks)
        if b.kind == "text" and (tag_names is None or b.tag_name in tag_names) and _is_item_heading(b.text, profile)
    ]


def _sections_structural(blocks: list[_Block], profile: FilingProfile) -> list[int] | None:
    """Detect Item headings among heading-styled tags (h1-h3/b/strong/a).

    Returns:
        Confirmed heading block indices if there are enough of them to be
        confident, else None to signal the caller should try regex fallback.
    """
    candidate_idxs = _item_heading_idxs(blocks, profile, _HEADING_TAGS)
    confirmed = _filter_sufficient_content(blocks, candidate_idxs, profile.min_content_chars)
    if len(confirmed) >= _MIN_CONFIDENT_STRUCTURAL_SECTIONS:
        return confirmed
    return None


def _sections_regex(blocks: list[_Block], profile: FilingProfile) -> list[int]:
    """Detect Item headings by scanning every text block's start against the profile's patterns."""
    candidate_idxs = _item_heading_idxs(blocks, profile)
    return _filter_sufficient_content(blocks, candidate_idxs, profile.min_content_chars)


def _sections_generic(blocks: list[_Block], profile: FilingProfile) -> list[int]:
    """Heading-only detection for forms without Item patterns: h1-h3 blocks with enough content."""
    candidate_idxs = [i for i, b in enumerate(blocks) if b.kind == "text" and b.tag_name in _GENERIC_HEADING_TAGS]
    return _filter_sufficient_content(blocks, candidate_idxs, profile.min_content_chars)


def _running_headers(blocks: list[_Block]) -> set[str]:
    """Short blocks repeated often enough across the document to be page furniture.

    Tables count too: some filings lay their running header out as a tiny
    table (e.g. "| Table of Contents | Alphabet Inc. |" on every page).
    """
    counts = Counter(b.text for b in blocks if b.length <= _RUNNING_HEADER_MAX_CHARS)
    return {text for text, count in counts.items() if count >= _RUNNING_HEADER_MIN_REPEATS}


def _is_bold_subheading(block: _Block) -> bool:
    """A short all-bold block that isn't a bullet point."""
    return block.bold and block.length <= _SUBHEADING_MAX_CHARS and not _BULLET_PREFIX.match(block.text)


def _render_body_block(block: _Block, running_headers: set[str]) -> str | None:
    """Render a non-section-heading block as Markdown body content, or None to drop it.

    Sub-headings become H3 — an HTML heading tag, or a short all-bold block
    (how inline-XBRL filings actually style them). Page furniture (running
    headers, page numbers, "Table of Contents" back-links, bare Part/Item
    labels, rule lines) is dropped.
    """
    if block.text in running_headers:
        return None
    if block.kind == "table":
        return block.text
    if any(pattern.match(block.text) for pattern in _NOISE_PATTERNS):
        return None
    if block.tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"} or _is_bold_subheading(block):
        return f"### {block.text}"
    if block.tag_name in {"b", "strong"}:
        return f"**{block.text}**"
    return block.text


def _part_label_by_index(blocks: list[_Block], profile: FilingProfile) -> dict[int, str | None]:
    """Map each block index to the Part label active at that point (None before any Part heading)."""
    labels: dict[int, str | None] = {}
    current: str | None = None
    for i, block in enumerate(blocks):
        if block.kind == "text" and profile.part_pattern is not None:
            match = profile.part_pattern.match(block.text)
            if match:
                current = f"Part {match.group(1).upper()}"
        labels[i] = current
    return labels


def _section_heading_text(item_text: str, idx: int, part_labels: dict[int, str | None] | None) -> str:
    """Qualify an Item heading with its active Part when Part scoping is in effect.

    With no scoping (part_labels is None) or before any Part heading has been
    seen (label is None), the raw Item text is emitted unqualified rather
    than guessing a Part.
    """
    if part_labels is None:
        return item_text
    label = part_labels.get(idx)
    return f"{label}{_PART_SEPARATOR}{item_text}" if label else item_text


def _assemble_markdown(blocks: list[_Block], confirmed_idxs: list[int], title: str, profile: FilingProfile) -> str:
    """Assemble the final Markdown: H1 title, Preamble, then one H2 per confirmed section.

    Every block is its own Markdown paragraph (blank-line separated), so
    chunking can recover block boundaries: a paragraph, a whole GFM table,
    or an H3 sub-heading line.
    """
    parts = [f"# {title}"]
    part_labels = _part_label_by_index(blocks, profile) if profile.uses_parts else None
    running_headers = _running_headers(blocks)

    def render(body_blocks: list[_Block]) -> list[str]:
        rendered = (_render_body_block(b, running_headers) for b in body_blocks)
        return [text for text in rendered if text]

    first_section_start = confirmed_idxs[0] if confirmed_idxs else len(blocks)
    preamble = render(blocks[:first_section_start])
    if preamble:
        parts.append("## Preamble")
        parts.extend(preamble)

    for pos, idx in enumerate(confirmed_idxs):
        next_idx = confirmed_idxs[pos + 1] if pos + 1 < len(confirmed_idxs) else len(blocks)
        section_text = _section_heading_text(blocks[idx].text, idx, part_labels)
        parts.append(f"## {section_text}")
        parts.extend(render(blocks[idx + 1 : next_idx]))

    return "\n\n".join(parts).strip() + "\n"
