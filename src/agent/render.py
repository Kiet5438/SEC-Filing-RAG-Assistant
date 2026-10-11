"""Render parts of the agent state as prompt text.

Every node builds its prompt from the state rather than from a running
message history, so these views are shared: the planner reads completed
steps with their passages, the executor a compact version, the synthesizer
only the notes about what could not be found.
"""

from langchain_core.documents import Document

from src.agent.state import StepResult, passage_id


def describe_call(name: str, args: dict) -> str:
    """'search_filings(query='revenue', ticker='NVDA')'."""
    return f"{name}({', '.join(f'{key}={value!r}' for key, value in args.items())})"


def render_steps(step_results: list[StepResult], evidence: list[Document], show_passages: bool = False) -> str:
    """Completed steps, each with its tool calls and results.

    Args:
        step_results: state["step_results"].
        evidence: state["evidence"], to look passages up by id.
        show_passages: Also show the passages each search returned, whole,
            the first time each appears (a later step that finds one again
            just names it); otherwise only the search's one-line result.
    """
    if not step_results:
        return "(none yet)"
    by_id = {passage_id(document): document for document in evidence}
    shown: set[str] = set()
    blocks = []
    for number, step in enumerate(step_results, start=1):
        lines = [f"Step {number}: {step['goal']}"]
        if step["note"]:
            lines.append(f"  Note: {step['note']}")
        for call in step["tool_calls"]:
            lines.append(f"  {describe_call(call['name'], call['args'])} -> {call['content']}")
            if not show_passages:
                continue
            for pid in call["passage_ids"]:
                document = by_id.get(pid)
                if document is None:
                    continue
                if pid in shown:
                    lines.append(f"    [{pid}] (shown above)")
                    continue
                shown.add(pid)
                lines.append(_indent(f"[{pid}]\n{document.page_content}", "    "))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())
