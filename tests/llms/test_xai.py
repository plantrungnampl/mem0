from unittest.mock import Mock, patch

import pytest

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.xai import XAILLM


class XAIConfig(BaseLlmConfig):
    """Test config with xai_base_url attribute to match what XAILLM expects."""

    def __init__(self, xai_base_url=None, **kwargs):
        super().__init__(**kwargs)
        self.xai_base_url = xai_base_url


@pytest.fixture
def mock_openai_client():
    with patch("mem0.llms.xai.OpenAI") as mock_openai:
        mock_client = Mock()
        mock_openai.return_value = mock_client
        yield mock_client


class TestXAILLMInit:
    def test_default_model(self, mock_openai_client):
        config = XAIConfig(api_key="test-key")
        llm = XAILLM(config)
        assert llm.config.model == "grok-2-latest"

    def test_custom_model(self, mock_openai_client):
        config = XAIConfig(model="grok-beta", api_key="test-key")
        llm = XAILLM(config)
        assert llm.config.model == "grok-beta"

    def test_api_key_from_config(self):
        with patch("mem0.llms.xai.OpenAI") as mock_openai:
            config = XAIConfig(api_key="config-key")
            XAILLM(config)
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["api_key"] == "config-key"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "env-key")
        with patch("mem0.llms.xai.OpenAI") as mock_openai:
            config = XAIConfig()
            XAILLM(config)
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["api_key"] == "env-key"

    def test_default_base_url(self, monkeypatch):
        monkeypatch.delenv("XAI_API_BASE", raising=False)
        with patch("mem0.llms.xai.OpenAI") as mock_openai:
            config = XAIConfig(api_key="key")
            XAILLM(config)
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["base_url"] == "https://api.x.ai/v1"

    def test_base_url_from_config(self, monkeypatch):
        monkeypatch.delenv("XAI_API_BASE", raising=False)
        with patch("mem0.llms.xai.OpenAI") as mock_openai:
            config = XAIConfig(api_key="key", xai_base_url="https://custom.xai.api/v1")
            XAILLM(config)
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["base_url"] == "https://custom.xai.api/v1"

    def test_base_url_from_env(self, monkeypatch):
        monkeypatch.setenv("XAI_API_BASE", "https://env.xai.api/v1")
        with patch("mem0.llms.xai.OpenAI") as mock_openai:
            config = XAIConfig(api_key="key")
            XAILLM(config)
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["base_url"] == "https://env.xai.api/v1"


class TestXAILLMGenerateResponse:
    def test_generate_response_basic(self, mock_openai_client):
        config = XAIConfig(model="grok-2-latest", temperature=0.7, max_tokens=100, top_p=0.9, api_key="key")
        llm = XAILLM(config)

        messages = [{"role": "user", "content": "Hello"}]
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Hi there!"))]
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = llm.generate_response(messages)

        mock_openai_client.chat.completions.create.assert_called_once_with(
            model="grok-2-latest",
            messages=messages,
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
        )
        assert result == "Hi there!"

    def test_generate_response_with_response_format(self, mock_openai_client):
        config = XAIConfig(model="grok-2-latest", api_key="key")
        llm = XAILLM(config)

        messages = [{"role": "user", "content": "Give me JSON"}]
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content='{"key": "value"}'))]
        mock_openai_client.chat.completions.create.return_value = mock_response

        result = llm.generate_response(messages, response_format={"type": "json_object"})

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert result == '{"key": "value"}'

    def test_generate_response_without_response_format(self, mock_openai_client):
        config = XAIConfig(model="grok-2-latest", api_key="key")
        llm = XAILLM(config)

        messages = [{"role": "user", "content": "Hello"}]
        mock_response = Mock()
        mock_response.choices = [Mock(message=Mock(content="Response"))]
        mock_openai_client.chat.completions.create.return_value = mock_response

        llm.generate_response(messages)

        call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
        assert "response_format" not in call_kwargs
