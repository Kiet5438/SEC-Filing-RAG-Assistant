"""Numbered context for the answering LLM, and citations built from the same passages.

Shared by the single-shot RAG chain and the agent's synthesizer: the LLM
sees passages labeled SOURCE [1..n] and cites by those numbers; the
citations are then built in code from the very same passage list, so a
"[2]" in an answer always points at citation number 2 — never at an
identifier the LLM produced.
"""

import re

from langchain_core.documents import Document

# Citation styles models fall into despite the prompt asking for "[n]":
# OpenAI-style 【2†L1-L9】 markers, and "[SOURCE 2]" echoing the context label.
_CITATION_VARIANTS = re.compile(r"【(\d+)(?:†[^】]*)?】|\[SOURCE\s+(\d+)\]", re.IGNORECASE)


def normalize_citation_markers(answer: str) -> str:
    """Rewrite other citation styles to the "[n]" markers the citations are numbered by."""
    return _CITATION_VARIANTS.sub(lambda match: f"[{match.group(1) or match.group(2)}]", answer)


def render_context(documents: list[Document]) -> str:
    """Render passages with explicit numbered source separators.

    Only the passage text goes in: its header already names the company,
    filing, section and sub-section. Raw metadata (ids, chunk numbers) is
    deliberately left out, since the model tends to echo such ids back as
    made-up citations; the SOURCE number is the only identifier it sees.
    """
    if not documents:
        return "No relevant context was retrieved."
    return "\n\n".join(f"SOURCE [{i}]\n{doc.page_content}" for i, doc in enumerate(documents, start=1))


def citation_key(metadata: dict) -> tuple:
    """Identity of a cited passage: (source, section, parent_id)."""
    return (metadata.get("source"), metadata.get("section"), metadata.get("parent_id"))


def build_citations(documents: list[Document]) -> list[dict]:
    """One citation per distinct passage, in order, numbered by its first SOURCE [n] position."""
    seen: set[tuple] = set()
    citations = []
    for number, doc in enumerate(documents, start=1):
        metadata = doc.metadata
        key = citation_key(metadata)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "number": number,
                "source": metadata.get("source"),
                "section": metadata.get("section"),
                "subsection": metadata.get("subsection"),
                "parent_id": metadata.get("parent_id"),
            }
        )
    return citations
