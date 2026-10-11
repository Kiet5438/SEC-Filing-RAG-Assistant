"""Sidekick: wires the agent's components into one LangGraph workflow.

    START -> start_turn -> planner --done---------------> synthesizer -> end_turn -> END
                             ^   | next step                    ^
                             |   v                              |
                             | executor -> run_tools --budget spent
                             +--------------+

- start_turn / end_turn: one run is one turn of a conversation; a thread
  keeps the conversation across turns (memory.py).

- planner (main model): turns the question into steps — one for a simple
  question — and reviews the results after every step: done, or the steps
  still to run.
- executor (lite model) + run_tools: one step's tool calls; index_filing
  waits for the user's approval via interrupt().
- synthesizer (main model): the answer, in the question's language, and
  its citations.

This module only wires nodes; their logic lives in memory.py, planner.py,
executor.py and synthesizer.py. Interrupts need a checkpointer, so the graph is always
compiled with one (in memory by default).
"""

import uuid

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.executor import make_executor, make_tool_runner
from src.agent.memory import end_turn, start_turn
from src.agent.planner import make_planner
from src.agent.state import AgentState
from src.agent.synthesizer import make_synthesizer
from src.agent.tools import create_filing_tools
from src.generation.llm import ChatModel
from src.retrieval.retriever import Retriever
from src.retrieval.vector_store import FilingStore
from src.utils.config import AGENT_MAX_STEPS, AGENT_MAX_TOOL_CALLS_PER_STEP

START_TURN, PLANNER, EXECUTOR, RUN_TOOLS, SYNTHESIZER, END_TURN = (
    "start_turn",
    "planner",
    "executor",
    "run_tools",
    "synthesizer",
    "end_turn",
)


def build_sidekick(
    store: FilingStore,
    retriever: Retriever,
    main_llm: ChatModel,
    lite_llm: ChatModel,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    max_steps: int = AGENT_MAX_STEPS,
    max_tool_calls: int = AGENT_MAX_TOOL_CALLS_PER_STEP,
) -> CompiledStateGraph:
    """Build and compile the agent graph.

    Args:
        store: The shared FilingStore (catalog reads; index_filing writes).
        retriever: Runs every search.
        main_llm: Planner and synthesizer.
        lite_llm: Executor.
        checkpointer: Where runs are saved, so an interrupted run (waiting
            for approval) can resume; defaults to an in-memory saver.
        max_steps: Executor steps per question before answering regardless.
        max_tool_calls: Tool calls kept per step.

    Returns:
        The compiled graph. Invoke or stream it with {"question": str,
        "filters": dict | None} and a config from new_thread(); reuse the
        config for the next question of the same conversation.
    """
    tools = create_filing_tools(store, retriever)

    builder = StateGraph(AgentState)
    builder.add_node(START_TURN, start_turn)
    builder.add_node(PLANNER, make_planner(main_llm, store, max_steps))
    builder.add_node(EXECUTOR, make_executor(lite_llm, tools, store, max_tool_calls))
    builder.add_node(RUN_TOOLS, make_tool_runner(tools))
    builder.add_node(SYNTHESIZER, make_synthesizer(main_llm, store=store))
    builder.add_node(END_TURN, end_turn)

    builder.add_edge(START, START_TURN)
    builder.add_edge(START_TURN, PLANNER)
    builder.add_conditional_edges(
        PLANNER, lambda state: EXECUTOR if state.get("plan") else SYNTHESIZER, [EXECUTOR, SYNTHESIZER]
    )
    builder.add_edge(EXECUTOR, RUN_TOOLS)
    builder.add_conditional_edges(
        RUN_TOOLS,
        lambda state: SYNTHESIZER if state.get("steps_taken", 0) >= max_steps else PLANNER,
        [SYNTHESIZER, PLANNER],
    )
    builder.add_edge(SYNTHESIZER, END_TURN)
    builder.add_edge(END_TURN, END)
    return builder.compile(checkpointer=checkpointer or InMemorySaver())


def new_thread() -> dict:
    """A run config with a fresh thread id: one per conversation (and reused to resume after an approval)."""
    return {"configurable": {"thread_id": str(uuid.uuid4())}}
