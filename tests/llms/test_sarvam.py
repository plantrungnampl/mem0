from unittest.mock import Mock, patch

import pytest
import requests

from mem0.configs.llms.base import BaseLlmConfig
from mem0.llms.sarvam import SarvamLLM


@pytest.fixture
def mock_requests_post():
    with patch("mem0.llms.sarvam.requests.post") as mock_post:
        yield mock_post


class TestSarvamLLMInit:
    def test_default_model(self, mock_requests_post):
        config = BaseLlmConfig(api_key="test-key")
        llm = SarvamLLM(config)
        assert llm.config.model == "sarvam-m"

    def test_custom_model(self, mock_requests_post):
        config = BaseLlmConfig(model="sarvam-2b", api_key="test-key")
        llm = SarvamLLM(config)
        assert llm.config.model == "sarvam-2b"

    def test_api_key_from_config(self, mock_requests_post):
        config = BaseLlmConfig(api_key="config-key")
        llm = SarvamLLM(config)
        assert llm.api_key == "config-key"

    def test_api_key_from_env(self, mock_requests_post, monkeypatch):
        monkeypatch.setenv("SARVAM_API_KEY", "env-key")
        config = BaseLlmConfig()
        llm = SarvamLLM(config)
        assert llm.api_key == "env-key"

    def test_raises_without_api_key(self, mock_requests_post, monkeypatch):
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)
        config = BaseLlmConfig()
        with pytest.raises(ValueError, match="Sarvam API key is required"):
            SarvamLLM(config)

    def test_default_base_url(self, mock_requests_post, monkeypatch):
        monkeypatch.delenv("SARVAM_API_BASE", raising=False)
        config = BaseLlmConfig(api_key="key")
        llm = SarvamLLM(config)
        assert llm.base_url == "https://api.sarvam.ai/v1"

    def test_base_url_from_env(self, mock_requests_post, monkeypatch):
        monkeypatch.setenv("SARVAM_API_BASE", "https://custom.sarvam.ai/v1")
        config = BaseLlmConfig(api_key="key")
        llm = SarvamLLM(config)
        assert llm.base_url == "https://custom.sarvam.ai/v1"


class TestSarvamLLMGenerateResponse:
    def test_generate_response_basic(self, mock_requests_post):
        config = BaseLlmConfig(model="sarvam-m", api_key="test-key", temperature=0.7, max_tokens=100, top_p=0.9)
        llm = SarvamLLM(config)

        messages = [{"role": "user", "content": "Hello"}]
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Namaste!"}}]}
        mock_response.raise_for_status = Mock()
        mock_requests_post.return_value = mock_response

        result = llm.generate_response(messages)

        mock_requests_post.assert_called_once()
        call_args = mock_requests_post.call_args
        assert call_args.kwargs["json"]["messages"] == messages
        assert call_args.kwargs["json"]["model"] == "sarvam-m"
        assert call_args.kwargs["json"]["temperature"] == 0.7
        assert call_args.kwargs["json"]["max_tokens"] == 100
        assert call_args.kwargs["json"]["top_p"] == 0.9
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer test-key"
        assert result == "Namaste!"

    def test_generate_response_with_dict_model(self, mock_requests_post):
        config = BaseLlmConfig(api_key="test-key")
        llm = SarvamLLM(config)
        llm.config.model = {"name": "sarvam-m", "reasoning_effort": "high", "seed": 42}

        messages = [{"role": "user", "content": "Test"}]
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
        mock_response.raise_for_status = Mock()
        mock_requests_post.return_value = mock_response

        result = llm.generate_response(messages)

        call_args = mock_requests_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["model"] == "sarvam-m"
        assert payload["reasoning_effort"] == "high"
        assert payload["seed"] == 42
        assert result == "Response"

    def test_generate_response_no_choices(self, mock_requests_post):
        config = BaseLlmConfig(model="sarvam-m", api_key="test-key")
        llm = SarvamLLM(config)

        messages = [{"role": "user", "content": "Hello"}]
        mock_response = Mock()
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status = Mock()
        mock_requests_post.return_value = mock_response

        with pytest.raises(ValueError, match="No response choices found"):
            llm.generate_response(messages)

    def test_generate_response_request_exception(self, mock_requests_post):
        config = BaseLlmConfig(model="sarvam-m", api_key="test-key")
        llm = SarvamLLM(config)

        messages = [{"role": "user", "content": "Hello"}]
        mock_requests_post.side_effect = requests.exceptions.RequestException("Connection failed")

        with pytest.raises(RuntimeError, match="Sarvam API request failed"):
            llm.generate_response(messages)

    def test_generate_response_url_construction(self, mock_requests_post):
        config = BaseLlmConfig(model="sarvam-m", api_key="test-key")
        llm = SarvamLLM(config)
        llm.base_url = "https://custom.api.com/v1"

        messages = [{"role": "user", "content": "Hello"}]
        mock_response = Mock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_response.raise_for_status = Mock()
        mock_requests_post.return_value = mock_response

        llm.generate_response(messages)

        call_args = mock_requests_post.call_args
        assert call_args[0][0] == "https://custom.api.com/v1/chat/completions"
