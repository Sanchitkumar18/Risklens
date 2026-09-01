"""LLM provider abstraction.

The assistant must run **without an API key**. ``get_chat_model`` returns a real
OpenAI-compatible chat model when ``LLM_PROVIDER=openai`` and a key is present, and a
deterministic :class:`MockChatModel` otherwise. The LangGraph assistant (Phase 13)
additionally renders explanations deterministically in mock mode, so grounding never
depends on a live model.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger("risklens.llm")


class MockChatModel(BaseChatModel):
    """A deterministic, offline chat model used when no LLM API key is configured.

    It echoes the last human message with a marker so behavior is fully reproducible
    (used in tests and as a safe default). The assistant's grounded explanations in
    mock mode are produced by a template renderer, not by this model.
    """

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        last_human = next(
            (m.content for m in reversed(messages) if m.type == "human"), ""
        )
        text = f"[mock-llm] {last_human}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": "mock"}


def llm_enabled(settings: Settings | None = None) -> bool:
    """True when a real OpenAI-compatible provider is configured with a key."""
    settings = settings or get_settings()
    return settings.llm_provider == "openai" and bool(settings.openai_api_key)


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Return the configured chat model (real when enabled, else the mock)."""
    settings = settings or get_settings()
    if not llm_enabled(settings):
        logger.info("using MockChatModel (no LLM provider configured)")
        return MockChatModel()

    # Imported lazily so the dependency isn't needed in mock mode.
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": settings.model_name,
        "temperature": settings.llm_temperature,
        "api_key": settings.openai_api_key,
    }
    if settings.openai_base_url:
        kwargs["base_url"] = settings.openai_base_url
    logger.info("using ChatOpenAI", extra={"model": settings.model_name})
    return ChatOpenAI(**kwargs)
