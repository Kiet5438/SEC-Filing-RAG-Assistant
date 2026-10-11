"""Markdown -> parent/child chunks for parsed SEC filings.

Splits a filing's Markdown (as produced by parse_html.py) into sections on
the H2 heading contract (^##\\s+), then each section into sub-sections on its
H3 headings. Each sub-section becomes a *parent* — merged into the next one
if tiny, split at block boundaries if huge — which is the unit the LLM
reads. Each parent is cut into small *children*, the unit that is embedded
and searched: prose as overlapping windows, and tables kept whole (or as row
groups with the header repeated), never mixed with prose. Neither kind of
chunk ever spans two sections. Retrieval searches children and hands back
their parents (see retriever.py).

Every chunk's page_content is prefixed with a short contextual header
(company / filing / section / sub-section) built from filing_meta, applied
after splitting so it never affects size accounting.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_MAX_CHARS,
    PARENT_MIN_CHARS,
    TABLE_CAPTION_MAX_CHARS,
    TABLE_CHILD_MAX_CHARS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_FILING_META_KEYS = {"doc_id", "ticker", "filing_type", "filing_date", "filing_name", "source", "url"}
# Optional: read from the filing's dei: tags (filing_facts.py); None when unavailable.
_OPTIONAL_FILING_META_KEYS = ("fiscal_year", "fiscal_period", "period_end_date")
_SECTION_HEADING_PATTERN = re.compile(r"^##[ \t]+(.+?)[ \t]*$", re.MULTILINE)
_SUBHEADING_PREFIX = "### "
_TABLE_PREFIX = "|"
_BLOCK_SEPARATOR = re.compile(r"\n[ \t]*\n")
# "Item 1A. Risk Factors" -> (None, "1A"); "Part II — Item 1. Legal ..." -> ("II", "1")
_SECTION_ID_PATTERN = re.compile(r"^(?:Part\s+(IV|III|II|I)\s*—\s*)?Item\s+(\d{1,2}[A-C]?)\b", re.IGNORECASE)
_SUBSECTION_TITLE_SEPARATOR = " > "


@dataclass
class ChunkedFiling:
    """A filing's parents (what the LLM reads) and children (what gets embedded)."""

    parents: list[Document]
    children: list[Document]


@dataclass
class _Group:
    """A run of blocks that becomes one parent, titled by the sub-heading(s) it starts with."""

    titles: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(block) for block in self.blocks)


def chunk_markdown(md_path: str | Path, filing_meta: dict) -> ChunkedFiling:
    """Split a parsed filing's Markdown into parent and child chunks.

    Args:
        md_path: Path to the Markdown file produced by parse_html.py.
        filing_meta: Filing fields to stamp onto every chunk's metadata:
            required doc_id, ticker, filing_type, filing_date, filing_name,
            source, url; optional fiscal_year, fiscal_period, period_end_date.

    Returns:
        A ChunkedFiling. Every parent and child carries the filing fields
        plus section (the raw H2 text), item_code / part (parsed from it,
        None for Preamble and for 10-K parts), subsection (the H3 title(s),
        None before a section's first H3) and parent_id (sequential per
        doc_id). Children additionally carry chunk_id (sequential per doc_id
        across all sections) and content_type ("text" | "table").

    Raises:
        ValueError: filing_meta is missing a required key.
        FileNotFoundError: md_path does not exist.
    """
    missing = _REQUIRED_FILING_META_KEYS - filing_meta.keys()
    if missing:
        raise ValueError(f"filing_meta is missing required keys: {sorted(missing)}")

    path = Path(md_path)
    logger.info("Chunking %s", path)
    sections = _split_sections(path.read_text(encoding="utf-8"))
    if not sections:
        logger.warning("No '## ' section headings found in %s; nothing to chunk.", path)
        return ChunkedFiling(parents=[], children=[])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator="end",
    )
    parents: list[Document] = []
    children: list[Document] = []

    for section_title, section_body in sections:
        if not section_body:
            continue
        part, item_code = _section_ids(section_title)
        for group in _parent_groups(_split_blocks(section_body)):
            subsection = _SUBSECTION_TITLE_SEPARATOR.join(group.titles) or None
            prefix = _build_chunk_prefix(filing_meta, section_title, subsection)
            metadata = {
                **_filing_fields(filing_meta),
                "section": section_title,
                "item_code": item_code,
                "part": part,
                "subsection": subsection,
                "parent_id": len(parents),
            }
            parents.append(Document(page_content=_with_prefix(prefix, "\n\n".join(group.blocks)), metadata=metadata))
            for content_type, body in _child_bodies(group, splitter):
                child_metadata = {**metadata, "chunk_id": len(children), "content_type": content_type}
                children.append(Document(page_content=_with_prefix(prefix, body), metadata=child_metadata))

    logger.info(
        "Produced %d parents and %d children across %d sections for doc_id=%s",
        len(parents),
        len(children),
        len(sections),
        filing_meta["doc_id"],
    )
    return ChunkedFiling(parents=parents, children=children)


def _build_chunk_prefix(filing_meta: dict, section: str, subsection: str | None = None) -> str:
    """Build the short contextual header prepended to every chunk's page_content.

    Uses only fields already present in filing_meta plus the chunk's own
    section/sub-section titles — no SEC calls. A line whose required
    field(s) are missing or empty is omitted entirely, rather than rendering
    a "None" placeholder; so is the fiscal-period suffix on the Filing line.

    Args:
        filing_meta: The same dict passed to chunk_markdown.
        section: The section's exact H2 text.
        subsection: The chunk's sub-section title(s), if any.

    Returns:
        The header block (one line per available field, "\\n"-joined, no
        trailing blank line). Callers join it to the chunk body with
        "\\n\\n"; empty string if every line's fields were missing.
    """
    lines = []

    filing_name = filing_meta.get("filing_name")
    ticker = filing_meta.get("ticker")
    if filing_name and ticker:
        lines.append(f"Company: {filing_name} ({ticker})")

    filing_type = filing_meta.get("filing_type")
    filing_date = filing_meta.get("filing_date")
    if filing_type and filing_date:
        lines.append(f"Filing: {filing_type} filed {filing_date}{_fiscal_suffix(filing_meta)}")

    if section:
        lines.append(f"Section: {section}")

    if subsection:
        lines.append(f"Subsection: {subsection}")

    return "\n".join(lines)


def _fiscal_suffix(filing_meta: dict) -> str:
    """", fiscal year 2026" for annual reports, ", fiscal 2027 Q1" for quarters, "" if unknown."""
    fiscal_year = filing_meta.get("fiscal_year")
    fiscal_period = filing_meta.get("fiscal_period")
    if not fiscal_year:
        return ""
    if fiscal_period and fiscal_period != "FY":
        return f", fiscal {fiscal_year} {fiscal_period}"
    return f", fiscal year {fiscal_year}"


def _with_prefix(prefix: str, body: str) -> str:
    return f"{prefix}\n\n{body}" if prefix else body


def _filing_fields(filing_meta: dict) -> dict:
    """The per-filing metadata fields, required ones plus the optional fiscal facts (None if absent)."""
    fields = {key: filing_meta[key] for key in sorted(_REQUIRED_FILING_META_KEYS)}
    fields.update({key: filing_meta.get(key) for key in _OPTIONAL_FILING_META_KEYS})
    return fields


def _section_ids(section: str) -> tuple[str | None, str | None]:
    """(part, item_code) parsed from a section title, e.g. ("II", "1A"); None where absent."""
    match = _SECTION_ID_PATTERN.match(section)
    if match is None:
        return None, None
    part = match.group(1).upper() if match.group(1) else None
    return part, match.group(2).upper()


def _split_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading_text, body_text) pairs on the ^##\\s+ contract.

    Content before the first H2 (the H1 title line) is discarded here; the
    Heading Contract guarantees no real content precedes the first H2, since
    parse_html.py always emits a "## Preamble" section for any pre-heading
    content.
    """
    parts = _SECTION_HEADING_PATTERN.split(markdown_text)
    sections: list[tuple[str, str]] = []
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((heading, body))
    return sections


def _split_blocks(section_body: str) -> list[str]:
    """A section body's blocks: paragraphs, whole GFM tables, and H3 lines (blank-line separated)."""
    return [block.strip() for block in _BLOCK_SEPARATOR.split(section_body) if block.strip()]


def _is_subheading(block: str) -> bool:
    return block.startswith(_SUBHEADING_PREFIX)


def _is_table(block: str) -> bool:
    return block.startswith(_TABLE_PREFIX)


def _parent_groups(blocks: list[str]) -> list[_Group]:
    """Group a section's blocks into parents: one per H3 sub-section, size-normalized.

    A sub-section smaller than PARENT_MIN_CHARS (often a bare heading, like a
    risk-factor category right before its first risk factor) merges into the
    next one, and the merged parent is titled by both headings ("A > B"). A
    too-small final sub-section merges back into the previous parent instead.
    Parents over PARENT_MAX_CHARS are then split at block boundaries.
    """
    subsections: list[_Group] = []
    for block in blocks:
        if _is_subheading(block) or not subsections:
            subsections.append(_Group())
        if _is_subheading(block):
            subsections[-1].titles.append(block[len(_SUBHEADING_PREFIX) :].strip())
        subsections[-1].blocks.append(block)

    groups: list[_Group] = []
    pending: _Group | None = None
    for subsection in subsections:
        if pending is None:
            pending = subsection
        else:
            pending.titles.extend(subsection.titles)
            pending.blocks.extend(subsection.blocks)
        if pending.chars >= PARENT_MIN_CHARS:
            groups.append(pending)
            pending = None
    if pending is not None:
        if groups:
            groups[-1].blocks.extend(pending.blocks)
        else:
            groups.append(pending)

    return [window for group in groups for window in _split_oversized(group)]


def _split_oversized(group: _Group) -> list[_Group]:
    """Split a group over PARENT_MAX_CHARS into consecutive block windows sharing its titles.

    A single block over the limit (a very large table) becomes a parent on
    its own rather than being cut mid-block.
    """
    if group.chars <= PARENT_MAX_CHARS:
        return [group]
    windows: list[_Group] = []
    current = _Group(titles=group.titles)
    for block in group.blocks:
        if current.blocks and current.chars + len(block) > PARENT_MAX_CHARS:
            windows.append(current)
            current = _Group(titles=group.titles)
        current.blocks.append(block)
    windows.append(current)
    return windows


def _child_bodies(group: _Group, splitter: RecursiveCharacterTextSplitter) -> list[tuple[str, str]]:
    """Cut one parent's blocks into (content_type, body) children.

    Consecutive prose blocks are joined and split into overlapping windows.
    Each table becomes its own child(ren), carrying as a caption the short
    paragraph right before it (typically "The following table sets
    forth..." or "(In millions)"). Sub-heading lines that title this parent
    are skipped (they're already in the header); any other sub-heading
    merged into it is kept as a plain line of prose.
    """
    bodies: list[tuple[str, str]] = []
    prose: list[str] = []

    def flush_prose() -> None:
        if prose:
            bodies.extend(("text", text) for text in splitter.split_text("\n\n".join(prose)) if text.strip())
            prose.clear()

    for block in group.blocks:
        if _is_subheading(block):
            title = block[len(_SUBHEADING_PREFIX) :].strip()
            if title not in group.titles:
                prose.append(title)
        elif _is_table(block):
            caption = prose.pop() if prose and len(prose[-1]) <= TABLE_CAPTION_MAX_CHARS else None
            flush_prose()
            bodies.extend(("table", body) for body in _split_table(block, caption))
        else:
            prose.append(block)
    flush_prose()
    return bodies


def _split_table(table: str, caption: str | None) -> list[str]:
    """Split a GFM table into pieces of at most ~TABLE_CHILD_MAX_CHARS, each with the header rows.

    Never splits inside a row; a single row longer than the budget becomes
    its own piece.
    """
    lines = table.split("\n")
    header, rows = lines[:2], lines[2:]
    lead = f"{caption}\n\n" if caption else ""
    budget = TABLE_CHILD_MAX_CHARS - len(lead) - sum(len(line) + 1 for line in header)

    pieces: list[list[str]] = []
    current: list[str] = []
    size = 0
    for row in rows:
        if current and size + len(row) + 1 > budget:
            pieces.append(current)
            current, size = [], 0
        current.append(row)
        size += len(row) + 1
    pieces.append(current)
    return [lead + "\n".join(header + piece) for piece in pieces]
