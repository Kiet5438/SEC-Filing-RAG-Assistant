"""Conversation memory: what carries over from one turn of a chat to the next.

A conversation is one checkpointer thread, and each run of the graph is one
turn of it:

- start_turn (first node): clears the previous turn's working state — plan,
  steps, evidence, trace, answer — with Overwrite, which bypasses the
  append-only reducers, and keeps the conversation.
- end_turn (last node): appends the turn's question and answer to
  state["messages"].

On its first call of a turn the planner reads the recent conversation and,
for a follow-up ("and Microsoft?"), writes a standalone question, which the
executor and the synthesizer then work from (current_question). So a
follow-up costs no extra LLM call, and retrieval never sees an unresolved
"it" or "that year".
"""

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.types import Overwrite

from src.agent.state import AgentState

MAX_TURNS = 3  # earlier turns the planner reads
ANSWER_CHARS = 1000  # of each earlier answer it reads


def start_turn(state: AgentState) -> dict:
    """Begin a turn: reset the working state, keep state["messages"]."""
    return {
        "standalone_question": state["question"],
        "plan": [],
        "pending_tool_calls": [],
        "steps_taken": 0,
        "step_results": Overwrite([]),
        "evidence": Overwrite([]),
        "trace": Overwrite([]),
        "answer": "",
        "citations": [],
    }


def end_turn(state: AgentState) -> dict:
    """Close a turn: record its question and answer in the conversation."""
    return {"messages": [HumanMessage(state["question"]), AIMessage(state["answer"])]}


def current_question(state: AgentState) -> str:
    """The question this turn is about: the planner's standalone rewrite when there is one."""
    return state.get("standalone_question") or state["question"]


def render_conversation(messages: list[AnyMessage]) -> str:
    """The last MAX_TURNS turns, each answer cut to ANSWER_CHARS, for the planner's prompt."""
    lines = []
    for message in messages[-2 * MAX_TURNS :]:
        text = message.text
        if isinstance(message, AIMessage) and len(text) > ANSWER_CHARS:
            text = text[:ANSWER_CHARS] + " …"
        lines.append(f"{'User' if isinstance(message, HumanMessage) else 'Assistant'}: {text}")
    return "\n".join(lines)
