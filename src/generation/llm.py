"""Chat-model factory: the one place that builds the LLMs.

Each role is a chain of Gemini API models (GEMINI_MODELS for "main" —
planner, synthesizer — and GEMINI_LITE_MODELS for "lite" — executor). The
first model serves; a call it can't serve — rate limit or used-up daily
quota, server error, timeout, lost connection — is retried on the next one,
through LangChain's with_fallbacks. The free tier's quota is per model, so
every model in the chain adds its own daily allowance.

The chain's wrapper forwards bind_tools / with_structured_output to every
model in it, so consumers never know a fallback exists; the model that
actually answered is in each AIMessage's response_metadata["model_name"]
(see answered_by). Consumers (the RAG chain, the agent) are written against
the chat-model interface and never import the provider package.
"""

from typing import Literal

import httpx
from langchain_core.exceptions import ModelAPIError, ModelRateLimitError
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableWithFallbacks
from langchain_google_genai import ChatGoogleGenerativeAI

from src.utils.config import (
    GEMINI_API_KEY,
    GEMINI_LITE_MODELS,
    GEMINI_MODELS,
    GEMINI_THINKING_LEVEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

Role = Literal["main", "lite"]
# What get_llm returns: a chat model, or a chain of them (which exposes the
# same invoke / bind_tools / with_structured_output surface).
ChatModel = BaseChatModel | RunnableWithFallbacks

# Errors that mean "this model can't serve the call right now", so the next
# one in the chain should. langchain-google-genai classifies 429s and 5xx
# but lets timeouts and connection failures through as raw httpx errors.
# Anything else (a malformed request, a bad key, a retired model id) is a
# bug to surface, not to mask.
FALLBACK_ERRORS: tuple[type[Exception], ...] = (ModelRateLimitError, ModelAPIError, httpx.TransportError)


def get_llm(role: Role = "main") -> ChatModel:
    """Construct the model chain for one role.

    Args:
        role: "main" or "lite".

    Returns:
        A chat model (a chain when the role lists several), to be injected
        into its consumers rather than constructed by them.

    Raises:
        ValueError: GEMINI_API_KEY or the role's model list is not configured.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set; add it to .env.")
    names = GEMINI_MODELS if role == "main" else GEMINI_LITE_MODELS
    if not names:
        raise ValueError(f"{'GEMINI_MODELS' if role == 'main' else 'GEMINI_LITE_MODELS'} is empty; list at least one model in .env.")
    logger.info("Gemini %s chain: %s", role, " -> ".join(names))
    # Every model but the last fails fast: retrying a rate limit (e.g. a
    # used-up daily quota) only delays the next model taking over. The last
    # one keeps the SDK's retries, which ride out a transient 429 or 500.
    models = [_gemini(name, role, fail_fast=index < len(names) - 1) for index, name in enumerate(names)]
    if len(models) == 1:
        return models[0]
    return models[0].with_fallbacks(models[1:], exceptions_to_handle=FALLBACK_ERRORS)


def without_streaming(llm: ChatModel) -> ChatModel:
    """The same model (and its fallbacks) with token streaming off.

    For calls whose output is parsed, never shown (planner, executor):
    when a graph is streamed with stream_mode="messages", every model call
    would otherwise stream internally, and only the answer is worth
    streaming. Anything that isn't a chat model (a test stand-in) is
    returned as is.
    """
    if isinstance(llm, RunnableWithFallbacks):
        return llm.__class__(
            **{
                **llm.model_dump(),
                "runnable": without_streaming(llm.runnable),
                "fallbacks": [without_streaming(fallback) for fallback in llm.fallbacks],
            }
        )
    if isinstance(llm, BaseChatModel):
        return llm.model_copy(update={"disable_streaming": True})
    return llm


def answered_by(message: BaseMessage | None) -> str | None:
    """The model that produced a response — a later one in the chain when earlier ones failed."""
    return message.response_metadata.get("model_name") if message is not None else None


def _gemini(model: str, role: Role, fail_fast: bool) -> ChatGoogleGenerativeAI:
    options: dict = {"timeout": LLM_TIMEOUT_SECONDS}
    if model.startswith("gemini-3"):
        # Gemini 3 runs at its default temperature: Google's guidance for the
        # family is to leave it there, and 3.5-flash-lite / 3.6-flash ignore
        # it outright. Its thinking depth is tunable; other families take
        # no thinking level (Gemma rejects the request).
        if role == "main" and GEMINI_THINKING_LEVEL:
            options["thinking_level"] = GEMINI_THINKING_LEVEL
    else:
        options["temperature"] = LLM_TEMPERATURE
    if fail_fast:
        options["max_retries"] = 1  # 1 = a single attempt; 0 would mean the SDK default (5 retries)
    return ChatGoogleGenerativeAI(model=model, google_api_key=GEMINI_API_KEY, **options)
