import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig


@pytest.fixture
def mock_together():
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_module.Together.return_value = mock_client
    with patch.dict(sys.modules, {"together": mock_module}):
        # Force reimport
        if "mem0.embeddings.together" in sys.modules:
            del sys.modules["mem0.embeddings.together"]
        from mem0.embeddings.together import TogetherEmbedding

        yield mock_module, mock_client, TogetherEmbedding


class TestTogetherEmbeddingInit:
    def test_default_model(self, mock_together):
        _, _, TogetherEmbedding = mock_together
        config = BaseEmbedderConfig()
        embedder = TogetherEmbedding(config)
        assert embedder.config.model == "togethercomputer/m2-bert-80M-8k-retrieval"

    def test_custom_model(self, mock_together):
        _, _, TogetherEmbedding = mock_together
        config = BaseEmbedderConfig(model="custom-embed-model")
        embedder = TogetherEmbedding(config)
        assert embedder.config.model == "custom-embed-model"

    def test_default_embedding_dims(self, mock_together):
        _, _, TogetherEmbedding = mock_together
        config = BaseEmbedderConfig()
        embedder = TogetherEmbedding(config)
        assert embedder.config.embedding_dims == 768

    def test_custom_embedding_dims(self, mock_together):
        _, _, TogetherEmbedding = mock_together
        config = BaseEmbedderConfig(embedding_dims=1024)
        embedder = TogetherEmbedding(config)
        assert embedder.config.embedding_dims == 1024

    def test_api_key_from_config(self, mock_together):
        mock_module, _, TogetherEmbedding = mock_together
        config = BaseEmbedderConfig(api_key="config-key")
        TogetherEmbedding(config)
        mock_module.Together.assert_called_with(api_key="config-key")

    def test_api_key_from_env(self, mock_together, monkeypatch):
        mock_module, _, TogetherEmbedding = mock_together
        monkeypatch.setenv("TOGETHER_API_KEY", "env-key")
        config = BaseEmbedderConfig()
        TogetherEmbedding(config)
        mock_module.Together.assert_called_with(api_key="env-key")


class TestTogetherEmbeddingEmbed:
    def test_embed_text(self, mock_together):
        _, mock_client, TogetherEmbedding = mock_together
        config = BaseEmbedderConfig(api_key="key")
        embedder = TogetherEmbedding(config)

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        result = embedder.embed("Hello world")

        mock_client.embeddings.create.assert_called_once_with(
            model="togethercomputer/m2-bert-80M-8k-retrieval",
            input="Hello world",
        )
        assert result == [0.1, 0.2, 0.3]

    def test_embed_with_custom_model(self, mock_together):
        _, mock_client, TogetherEmbedding = mock_together
        config = BaseEmbedderConfig(model="my-model", api_key="key")
        embedder = TogetherEmbedding(config)

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.4, 0.5])]
        mock_client.embeddings.create.return_value = mock_response

        result = embedder.embed("Test")

        mock_client.embeddings.create.assert_called_once_with(
            model="my-model",
            input="Test",
        )
        assert result == [0.4, 0.5]

    def test_embed_with_memory_action(self, mock_together):
        _, mock_client, TogetherEmbedding = mock_together
        config = BaseEmbedderConfig(api_key="key")
        embedder = TogetherEmbedding(config)

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.7, 0.8, 0.9])]
        mock_client.embeddings.create.return_value = mock_response

        result = embedder.embed("Text", memory_action="search")
        assert result == [0.7, 0.8, 0.9]
