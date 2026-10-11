"""Synthesizer: writes the final answer from the collected passages.

Every run ends here, once the planner is done or the step budget is spent.
The prompt holds only the question,
the passages numbered SOURCE [1..n] (the same rendering as the single-shot
chain) and short notes on what could not be found — not the plan or the
tool chatter — so the model cites by SOURCE number and nothing else.
Citations are then built in code from the same passage list, and the
answer is written in the language the user asked in, detected in code from
their own words (the standalone rewrite of a follow-up may be in English).

When the planner ran no step at all — a greeting, thanks, a question about
the assistant, a request outside the filings — there is nothing to cite,
and the research prompt would answer "nothing was found". Such a turn gets
a conversational prompt instead: reply briefly, say what the assistant can
do, and name the indexed filings.
"""

import re
from collections.abc import Callable
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.memory import current_question
from src.agent.progress import emit_progress
from src.agent.render import describe_call
from src.agent.state import AgentState, StepResult
from src.agent.tools import render_catalog
from src.generation.citations import build_citations, normalize_citation_markers, render_context
from src.generation.llm import ChatModel, answered_by
from src.retrieval.vector_store import FilingStore
from src.utils.config import AGENT_MAX_EVIDENCE

LANGUAGE_NAMES = {"vi": "Vietnamese", "en": "English"}
NOTE_CHARS = 300  # per research note

# Unlike the single-shot chain's all-or-nothing prompt, partial answers are
# asked for: a question that bundles a figure and its drivers should get the
# drivers even when the figure isn't in the passages, with the gap named.
_SYSTEM_PROMPT = """You are a financial analyst answering questions about SEC 10-K and 10-Q filings.
Use ONLY the passages under CONTEXT.
- Answer every part of the question the passages support, quoting figures exactly as written.
- Report a figure only under the label its passage gives it. If the passages give a related but different
  measure (a segment that contains the asked-for business, another period, a subtotal), name it as what it
  is and say the exact figure asked for is not given; never relabel one line item as another.
- If some part is not covered, say plainly which part is missing; do not withhold the parts you can answer.
- If nothing in the context answers the question, say so, and explain why from the RESEARCH NOTES when they
  show it (e.g. the filing is not indexed, the user declined indexing it, a search failed).
- Cite each fact with the number of its SOURCE in square brackets, e.g. [2]. Use no other citation format.
- RESEARCH NOTES, when present, list lookups that returned no passages (listings, indexing, empty or failed
  searches). Use them to explain what is missing, never as a source of facts.
- Write the entire answer in {language}."""

_CONVERSATION_PROMPT = """You are the assistant of a research tool over SEC 10-K and 10-Q filings.
The user's latest message needed no lookup in the filings: a greeting, thanks, a question about what you can do,
or a request outside the filings.
- Reply briefly and naturally. Do not say that nothing was found.
- Say what you can help with: questions about the indexed filings listed below, and fetching other companies'
  10-K/10-Q filings from SEC EDGAR, which the user approves before anything is downloaded.
- If the message asks for something the filings cannot answer (e.g. today's stock price), say so plainly.
- Write the entire reply in {language}."""


def make_synthesizer(
    llm: ChatModel, max_evidence: int = AGENT_MAX_EVIDENCE, store: FilingStore | None = None
) -> Callable[[AgentState], dict]:
    """Build the synthesizer node.

    Args:
        llm: The main chat model.
        max_evidence: The most passages put in the prompt, in the order found.
        store: Read for the catalog of indexed filings named in conversational
            replies (turns with no lookup); without it the reply names none.

    Returns:
        A node returning answer, citations and a trace event.
    """

    def synthesizer(state: AgentState) -> dict:
        language = LANGUAGE_NAMES[question_language(state["question"])]
        if not state.get("step_results"):
            return _converse(state, language)
        passages = state.get("evidence", [])[:max_evidence]
        human = f" CONTEXT \n{render_context(passages)}\n\n"
        notes = research_notes(state.get("step_results", []))
        if notes:
            human += f" RESEARCH NOTES \n{notes}\n\n"
        human += f" QUESTION \n{current_question(state)}\n\n(Answer in {language}.)"
        system = _SYSTEM_PROMPT.format(language=language)

        emit_progress("Writing the answer")
        message = llm.invoke([SystemMessage(system), HumanMessage(human)])
        return {
            "answer": normalize_citation_markers(message.text),
            "citations": build_citations(passages),
            "trace": [{"node": "synthesizer", "summary": f"answered from {len(passages)} passages", "model": answered_by(message)}],
        }

    def _converse(state: AgentState, language: str) -> dict:
        catalog = render_catalog(store.filing_summaries()) if store is not None else "(not available)"
        human = f"Indexed filings:\n{catalog}\n\nMESSAGE\n{state['question']}\n\n(Answer in {language}.)"
        emit_progress("Writing the reply")
        message = llm.invoke([SystemMessage(_CONVERSATION_PROMPT.format(language=language)), HumanMessage(human)])
        return {
            "answer": message.text,
            "citations": [],
            "trace": [{"node": "synthesizer", "summary": "conversational reply (no lookup needed)", "model": answered_by(message)}],
        }

    return synthesizer


# Every letter of the Vietnamese alphabet that carries a diacritic. A few
# (à, é, ó...) also occur in French or Spanish, but users ask only in
# Vietnamese or English, so any of them means Vietnamese.
_VIETNAMESE_LETTERS = re.compile(
    r"[ăâđêôơư"
    r"àáảãạằắẳẵặầấẩẫậ"
    r"èéẻẽẹềếểễệ"
    r"ìíỉĩị"
    r"òóỏõọồốổỗộờớởỡợ"
    r"ùúủũụừứửữự"
    r"ỳýỷỹỵ]"
)


def question_language(text: str) -> Literal["vi", "en"]:
    """'vi' if the text has a Vietnamese diacritic letter, else 'en' — the two languages users ask in."""
    return "vi" if _VIETNAMESE_LETTERS.search(text.lower()) else "en"


def research_notes(step_results: list[StepResult]) -> str:
    """One line per tool call that brought back no passages: what was tried and what came back."""
    lines = []
    for step in step_results:
        if step["note"]:
            lines.append(f"- {step['goal']}: {step['note']}")
        for call in step["tool_calls"]:
            if call["passage_ids"]:
                continue
            content = " ".join(call["content"].split())
            if len(content) > NOTE_CHARS:
                content = content[:NOTE_CHARS] + " …"
            lines.append(f"- {describe_call(call['name'], call['args'])}: {content}")
    return "\n".join(lines)
