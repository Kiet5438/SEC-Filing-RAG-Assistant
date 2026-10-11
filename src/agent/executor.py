"""Executor: carries out one plan step with the filing tools.

Two nodes:

- executor: the lite model reads the step and chooses the tool call(s)
  for it (tool_choice="any", so it always acts rather than chats). The
  planner writes each step as one concrete lookup — company, form, period,
  section — so turning it into a call is mechanical, and a small model
  does it in about a second.
- run_tools: runs those calls. A tool whose metadata requires approval
  (index_filing) pauses the graph with interrupt() until the user answers.
  This is a separate node because a resumed node re-runs from its start:
  here that replays only the approval questions — answered now, so they
  return at once — and never repeats the LLM call that chose the tools.

Each call is recorded in a StepResult (what ran, with what arguments, what
came back); passages a search returns go to state["evidence"].
"""

from collections.abc import Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.types import interrupt

from src.agent.memory import current_question
from src.agent.progress import emit_progress
from src.agent.render import describe_call, render_steps
from src.agent.state import AgentState, ToolCallRecord, passage_id
from src.agent.tools import REQUIRES_APPROVAL, render_catalog
from src.generation.llm import ChatModel, answered_by, without_streaming
from src.retrieval.vector_store import FilingStore
from src.utils.logger import get_logger

logger = get_logger(__name__)

SEARCH_TOOL = "search_filings"

_SYSTEM_PROMPT = """You carry out one step of a research plan on SEC 10-K and 10-Q filings by calling tools.
Call the tool that accomplishes exactly this step; make several calls only when the step needs independent lookups (at most {max_calls}).
- search_filings: write the query in English, the way a 10-K/10-Q words it. Set filters to exact values from the catalog (ticker, fiscal_year, fiscal_period, filing_type), item_code/part when the step names a section, and content_type='table' when the step says to search tables. Leave out every filter the step says to drop (e.g. "no Item filter" means no item_code or part).
- index_filing: use the exact filing date shown by an earlier list_sec_filings result."""


def make_executor(llm: ChatModel, tools: list[BaseTool], store: FilingStore, max_calls: int) -> Callable[[AgentState], dict]:
    """Build the executor node: choose the tool call(s) for the next plan step.

    Args:
        llm: The lite chat model.
        tools: The filing tools (see src.agent.tools).
        store: Read for the catalog of indexed filings.
        max_calls: The most tool calls kept for one step.

    Returns:
        A node returning pending_tool_calls and a trace event.
    """
    acting = without_streaming(llm).bind_tools(tools, tool_choice="any")
    system = _SYSTEM_PROMPT.format(max_calls=max_calls)

    def executor(state: AgentState) -> dict:
        step = state["plan"][0]
        emit_progress(f"Step {state.get('steps_taken', 0) + 1}: {step['goal']}")
        message = acting.invoke([SystemMessage(system), HumanMessage(_render_request(state, store))])
        calls = [
            {"id": call.get("id") or f"call-{index}", "name": call["name"], "args": call["args"]}
            for index, call in enumerate(message.tool_calls[:max_calls])
        ]
        summary = "; ".join(describe_call(call["name"], call["args"]) for call in calls) or "no tool call"
        return {
            "pending_tool_calls": calls,
            "trace": [{"node": "executor", "summary": summary, "model": answered_by(message)}],
        }

    return executor


def make_tool_runner(tools: list[BaseTool]) -> Callable[[AgentState], dict]:
    """Build the run_tools node: run the executor's calls, asking approval where required.

    Args:
        tools: The filing tools; their metadata says which need approval.

    Returns:
        A node returning the step's result, the passages found, the
        advanced plan and step count, and a trace event.
    """
    by_name = {tool.name: tool for tool in tools}

    def run_tools(state: AgentState) -> dict:
        step = state["plan"][0]
        calls = state.get("pending_tool_calls", [])
        # Every approval is asked before anything runs: on resume this node
        # restarts, and each interrupt() then returns its stored answer in
        # order, so no tool runs twice.
        approvals = {
            call["id"]: _is_approved(
                interrupt({"tool": call["name"], "args": call["args"], "step": step["goal"], "question": state["question"]})
            )
            for call in calls
            if _requires_approval(by_name.get(call["name"]))
        }

        records, evidence = [], []
        for call in calls:
            record, found = _run_call(by_name.get(call["name"]), call, approvals.get(call["id"]), state.get("filters") or {})
            records.append(record)
            evidence.extend(found)

        note = "" if calls else "The executor chose no tool for this step."
        return {
            "step_results": [{"goal": step["goal"], "tool_calls": records, "note": note}],
            "evidence": evidence,
            "plan": state["plan"][1:],
            "pending_tool_calls": [],
            "steps_taken": state.get("steps_taken", 0) + 1,
            "trace": [{"node": "run_tools", "summary": _summarize(records, note), "model": None}],
        }

    return run_tools


def _run_call(tool: BaseTool | None, call: dict, approved: bool | None, user_filters: dict) -> tuple[ToolCallRecord, list]:
    """Run one tool call; failures become the record's content so the planner can react."""
    args = dict(call["args"])
    if call["name"] == SEARCH_TOOL:
        args.update(user_filters)  # the user's UI filters bind every search
    record: ToolCallRecord = {
        "id": call["id"],
        "name": call["name"],
        "args": args,
        "content": "",
        "passage_ids": [],
        "approved": approved,
    }
    if tool is None:
        record["content"] = f"Unknown tool {call['name']!r}."
        return record, []
    if approved is False:
        record["content"] = "Not run: the user declined."
        emit_progress(f"Skipped {call['name']}: declined")
        return record, []

    emit_progress(f"Running {describe_call(call['name'], args)}")
    try:
        message = tool.invoke({"type": "tool_call", "id": call["id"], "name": call["name"], "args": args})
    except Exception as e:  # invalid arguments from the model, or an unexpected tool failure
        logger.warning("Tool %s failed: %s", call["name"], e)
        record["content"] = f"Tool error: {e}"
        return record, []

    if call["name"] == SEARCH_TOOL:
        passages = message.artifact or []
        record["passage_ids"] = [passage_id(document) for document in passages]
        record["content"] = f"{len(passages)} passages" if passages else message.content
        return record, passages
    record["content"] = message.content
    return record, []


def _requires_approval(tool: BaseTool | None) -> bool:
    return tool is not None and bool((tool.metadata or {}).get(REQUIRES_APPROVAL))


def _is_approved(answer) -> bool:
    """Accept {"approved": bool} (what the UI sends) or a bare bool."""
    if isinstance(answer, dict):
        return bool(answer.get("approved"))
    return bool(answer)


def _summarize(records: list[ToolCallRecord], note: str) -> str:
    lines = [f"{describe_call(record['name'], record['args'])} -> {record['content'].splitlines()[0] if record['content'] else ''}" for record in records]
    return "; ".join(lines) or note


def _render_request(state: AgentState, store: FilingStore) -> str:
    parts = [f"Question being researched: {current_question(state)}", f"Step to carry out now: {state['plan'][0]['goal']}"]
    if state.get("filters"):
        parts.append(f"Applied automatically to every search: {state['filters']}")
    parts.append(f"Indexed filings:\n{render_catalog(store.filing_summaries())}")
    parts.append("Earlier steps:\n" + render_steps(state.get("step_results", []), state.get("evidence", [])))
    return "\n\n".join(parts)
