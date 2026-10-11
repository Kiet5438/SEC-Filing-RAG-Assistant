"""The agent's shared state: what flows between the sidekick's nodes.

LangGraph owns the state. Each node reads it and returns only the keys it
changes; keys annotated with a reducer (step_results, evidence, trace) are
merged instead of overwritten, so every step appends to them.

Everything here is plain data — strings, dicts, lists, LangChain
Documents and messages — never models, tools or stores (those are injected into the
nodes when the graph is built). That keeps the state serializable by the
checkpointer without registering any custom class, and keeps the nodes
provider-agnostic: each one renders the state into its own prompt instead
of replaying another model's message history.
"""

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class PlanStep(TypedDict):
    goal: str  # one lookup, e.g. "NVIDIA fiscal 2026 data center revenue (10-K, MD&A)"


class ToolCallRecord(TypedDict):
    id: str
    name: str
    args: dict
    content: str  # the tool's text result; for searches, a one-line summary
    passage_ids: list[str]  # passages a search returned (they are in state["evidence"])
    approved: bool | None  # for tools needing approval: the user's decision; otherwise None


class StepResult(TypedDict):
    goal: str
    tool_calls: list[ToolCallRecord]
    note: str  # e.g. why no tool ran; "" normally


class TraceEvent(TypedDict):
    node: str
    summary: str
    model: str | None  # the model that answered, when the node called one


def passage_id(document: Document) -> str:
    """The stable id of a passage, e.g. 'NVDA_10-K_2026-02-25#77'."""
    return f"{document.metadata.get('doc_id')}#{document.metadata.get('parent_id')}"


def merge_evidence(current: list[Document], new: list[Document]) -> list[Document]:
    """Reducer: append newly found passages, skipping ones already collected."""
    seen = {passage_id(document) for document in current}
    merged = list(current)
    for document in new:
        if passage_id(document) not in seen:
            seen.add(passage_id(document))
            merged.append(document)
    return merged


class AgentState(TypedDict, total=False):
    # --- input (one per turn) ---
    question: str
    filters: dict  # filters the user picked in the UI: enforced on every search

    # --- the conversation, kept across the turns of a thread (see memory.py) ---
    messages: Annotated[list[AnyMessage], add_messages]  # earlier questions and answers
    standalone_question: str  # this turn's question with references to earlier turns resolved

    # --- planner <-> executor loop ---
    plan: list[PlanStep]  # steps still to run; the executor takes the first
    pending_tool_calls: list[dict]  # the executor's chosen calls, run (after approval) by run_tools
    step_results: Annotated[list[StepResult], operator.add]
    evidence: Annotated[list[Document], merge_evidence]
    steps_taken: int

    # --- synthesizer ---
    answer: str
    citations: list[dict]

    trace: Annotated[list[TraceEvent], operator.add]  # one event per node run, for the UI
