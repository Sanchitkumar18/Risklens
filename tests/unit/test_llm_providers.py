"""Unit tests for the LLM provider abstraction."""

from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from app.core.config import Settings
from app.llm.providers import MockChatModel, get_chat_model, llm_enabled


@pytest.mark.unit
def test_mock_used_when_no_key():
    settings = Settings(llm_provider="mock", openai_api_key=None)
    assert llm_enabled(settings) is False
    assert isinstance(get_chat_model(settings), MockChatModel)


@pytest.mark.unit
def test_llm_enabled_requires_key():
    assert llm_enabled(Settings(llm_provider="openai", openai_api_key=None)) is False
    assert llm_enabled(Settings(llm_provider="openai", openai_api_key="sk-x")) is True


@pytest.mark.unit
def test_mock_is_deterministic():
    model = MockChatModel()
    a = model.invoke([HumanMessage(content="hello")])
    b = model.invoke([HumanMessage(content="hello")])
    assert a.content == b.content
    assert "hello" in a.content
