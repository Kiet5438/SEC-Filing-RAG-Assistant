"""Progress events from inside a graph run, streamed to the UI as they happen.

Nodes and tools call emit_progress(); a caller streaming the graph with
stream_mode="custom" receives each event as {"message": str, **fields}.
Outside a graph run (CLI, tests, a tool invoked directly) it does nothing,
so the same code works everywhere.

ModelProgress does the same for the model chains: passed as a run callback,
it reports which model each LLM call is waiting on and why one was passed
over, so a slow call — every flash model out of daily quota, the last one
backing off on a per-minute limit — shows up in the UI instead of looking
like a hang.
"""

from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langgraph.config import get_stream_writer


def emit_progress(message: str, **fields) -> None:
    """Send one progress event to the run's custom stream, if there is one."""
    try:
        writer = get_stream_writer()
    except (RuntimeError, KeyError):  # not inside a graph run
        return
    writer({"message": message, **fields})


class ModelProgress(BaseCallbackHandler):
    """Run callback: emit_progress for each model attempt and each model passed over."""

    def __init__(self) -> None:
        self._models: dict[UUID, str] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id: UUID, **kwargs) -> None:
        name = (kwargs.get("metadata") or {}).get("ls_model_name") or (kwargs.get("invocation_params") or {}).get("model")
        model = str(name or "model").removeprefix("models/")
        self._models[run_id] = model
        emit_progress(f"Asking {model}")

    def on_llm_end(self, response, *, run_id: UUID, **kwargs) -> None:
        self._models.pop(run_id, None)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs) -> None:
        model = self._models.pop(run_id, "model")
        emit_progress(f"{model} unavailable ({describe_model_error(error)})")


def describe_model_error(error: BaseException) -> str:
    """A short reason for a failed model call, from the error Gemini returned."""
    text = str(error)
    if "PerDay" in text:
        return "daily quota used up"
    if "PerMinute" in text or "429" in text:
        return "rate limited"
    if "503" in text or "UNAVAILABLE" in text:
        return "overloaded"
    if "timed out" in text.lower() or "Timeout" in type(error).__name__:
        return "timed out"
    return type(error).__name__
