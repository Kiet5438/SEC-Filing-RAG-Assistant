"""Planner: plans the research, and reflects on the results after every step.

Every question starts here. A question one search can answer gets a
one-step plan, and the review after that step is what checks the passages
actually answer it — a missing part (a figure, a second period) gets a
step of its own before anything is written.

Planning and reflecting are the same call. Before any step has run, the
planner turns the question into steps; after each step, it reads what the
step found and decides again: done (the evidence answers the question, or
nothing more can be done), or the steps still to run — the old plan, an
adjusted one, or new steps the results revealed (e.g. index a filing that
list_sec_filings turned up). So replanning needs no separate node, and the
plan is always the planner's latest view, not a script fixed up front.

The planner reads each passage a step found whole (each passage once), so
its review sees what the synthesizer will see: cut to its first 1,500
characters, a passage could hide the very table row that answers the
question, and the review would plan a search for something already found.

The planner never answers the question; the synthesizer writes the answer
from the collected passages.
"""

from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.memory import current_question, render_conversation
from src.agent.progress import emit_progress
from src.agent.render import render_steps
from src.agent.state import AgentState
from src.agent.tools import render_catalog
from src.generation.llm import ChatModel, answered_by, without_streaming
from src.retrieval.vector_store import FilingStore
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You plan and review research on SEC 10-K and 10-Q filings. An executor carries out each step with these tools:
- search_filings: search the indexed filings; filters for ticker, form, fiscal year/period, SEC Item, Part, text or table
- list_available_filings: what is indexed
- list_sec_filings: a company's 10-K/10-Q history on SEC EDGAR, with filing dates and period ends
- index_filing: download and index one filing (slow, and the user must approve it)

Rules:
- When the latest question follows earlier turns of the conversation, first resolve what it refers to ("and Microsoft?", "the year before") and write it as standalone_question; plan for that.
- Plan the fewest steps that answer the question. A question one search can answer gets a one-step plan. A question that needs no filings at all (a greeting, a question about you) gets none: set done to true.
- One step is one lookup: one company, one period, one topic. Comparing two companies or two periods takes one search step each.
- Write every step in English, naming the ticker, form, fiscal period and SEC Item from the catalog when they apply; a small model turns each step into a tool call, so leave nothing to infer.
- Ground steps in the catalog of indexed filings. If the question needs a company or period that is not indexed, plan a list_sec_filings step, then an index_filing step naming the filing date that covers the asked period, then the search.
- Figures (revenue by segment or market, income, counts per period) usually sit in tables, which prose passages only summarize. When the question asks for a number, or a number is still missing after a step, plan a step that searches tables only and say so in the step ("search tables only").
- After each step, judge its results. If the passages answer every part of the question, set done to true. If something cannot be done (not on SEC, the user declined indexing), set done to true; the answer will explain.
- If a part is still missing, the next step must change tactics, never repeat the last search with new wording only. First widen it: drop a filter that may have excluded the answer and say so in the step ("no Item filter", "text and tables") — an SEC Item filter is the usual culprit, since segment and market figures often sit in the financial-statement notes rather than in MD&A. Then try tables only, or another section.
- next_steps lists only steps still to run, never completed ones, and stays within the remaining budget.
- Never answer the question yourself."""


class PlannedStep(BaseModel):
    goal: str = Field(
        description=(
            "One English sentence naming one lookup: company, period, form and topic, e.g. "
            "'Search NVDA fiscal 2026 10-K MD&A for data center revenue and its growth drivers'."
        )
    )


class PlannerDecision(BaseModel):
    standalone_question: str = Field(
        default="",
        description=(
            "Only when the latest question follows earlier turns: that question rewritten to stand alone, in "
            "the user's language, e.g. 'and Microsoft?' after a question on NVIDIA's competition -> 'How does "
            "Microsoft describe competition in its latest 10-K?'. Empty otherwise."
        ),
    )
    assessment: str = Field(
        description="One to three sentences: what the completed steps found and what is still missing. Empty before any step."
    )
    done: bool = Field(description="True when the evidence answers the question or nothing more can usefully be done.")
    next_steps: list[PlannedStep] = Field(
        default_factory=list, description="The steps still to run, in order; empty when done."
    )


def make_planner(llm: ChatModel, store: FilingStore, max_steps: int) -> Callable[[AgentState], dict]:
    """Build the planner node.

    Args:
        llm: The main chat model.
        store: Read for the catalog of indexed filings (which index_filing can grow mid-run).
        max_steps: The executor-step budget per question.

    Returns:
        A node returning the remaining plan (empty = go answer) and a trace event.
    """
    structured = without_streaming(llm).with_structured_output(PlannerDecision, include_raw=True)

    def planner(state: AgentState) -> dict:
        steps_taken = state.get("steps_taken", 0)
        budget = max_steps - steps_taken
        emit_progress("Planning" if not steps_taken else f"Reviewing step {steps_taken}")
        result = structured.invoke([SystemMessage(_SYSTEM_PROMPT), HumanMessage(_render_request(state, store, budget))])
        decision = result["parsed"]
        if decision is None:
            logger.warning("Planner output could not be parsed (%s); answering with what was found.", result["parsing_error"])
            decision = PlannerDecision(assessment="(the plan could not be read; answering with what was found)", done=True)

        steps = [] if decision.done else [{"goal": step.goal} for step in decision.next_steps[:budget]]
        summary = decision.assessment or "Initial plan"
        summary += " | done" if not steps else " | next: " + " / ".join(step["goal"] for step in steps)
        update = {"plan": steps}
        if _is_follow_up(state) and decision.standalone_question.strip():
            update["standalone_question"] = decision.standalone_question.strip()
            summary = f"Read as: {update['standalone_question']} | {summary}"
        emit_progress("Enough to answer" if not steps else f"Next: {steps[0]['goal']}")
        update["trace"] = [{"node": "planner", "summary": summary, "model": answered_by(result["raw"])}]
        return update

    return planner


def _is_follow_up(state: AgentState) -> bool:
    """The first call of a turn that has earlier turns before it."""
    return not state.get("steps_taken", 0) and bool(state.get("messages"))


def _render_request(state: AgentState, store: FilingStore, budget: int) -> str:
    if _is_follow_up(state):
        parts = [
            f"Conversation so far:\n{render_conversation(state['messages'])}",
            f"Latest question (it may refer to the conversation): {state['question']}",
        ]
    else:
        parts = [f"Question: {current_question(state)}"]
    if state.get("filters"):
        parts.append(f"The user restricted every search to: {state['filters']}")
    parts.append(f"Indexed filings:\n{render_catalog(store.filing_summaries())}")
    parts.append(
        "Completed steps:\n"
        + render_steps(state.get("step_results", []), state.get("evidence", []), show_passages=True)
    )
    parts.append(f"Remaining budget: at most {budget} more step(s).")
    return "\n\n".join(parts)
