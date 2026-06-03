import sys
from unittest.mock import MagicMock, patch

import pytest

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.openai import OpenAIConfig
from mem0.configs.rerankers.cohere import CohereRerankerConfig
from mem0.utils.factory import (
    EmbedderFactory,
    LlmFactory,
    RerankerFactory,
    VectorStoreFactory,
    load_class,
)


class TestLoadClass:
    def test_load_existing_class(self):
        cls = load_class("mem0.configs.llms.base.BaseLlmConfig")
        assert cls is BaseLlmConfig

    def test_load_nonexistent_module(self):
        with pytest.raises(ModuleNotFoundError):
            load_class("nonexistent.module.ClassName")

    def test_load_nonexistent_class(self):
        with pytest.raises(AttributeError):
            load_class("mem0.configs.llms.base.NonExistentClass")


class TestLlmFactory:
    def test_get_supported_providers(self):
        providers = LlmFactory.get_supported_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "xai" in providers
        assert "sarvam" in providers
        assert "ollama" in providers

    def test_create_with_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported Llm provider"):
            LlmFactory.create("nonexistent_provider")

    @patch("mem0.llms.openai.OpenAI")
    def test_create_openai_with_none_config(self, mock_openai):
        mock_openai.return_value = MagicMock()
        llm = LlmFactory.create("openai", model="gpt-4", api_key="test-key")
        assert llm.config.model == "gpt-4"

    @patch("mem0.llms.openai.OpenAI")
    def test_create_openai_with_dict_config(self, mock_openai):
        mock_openai.return_value = MagicMock()
        config_dict = {"model": "gpt-4", "api_key": "test-key", "temperature": 0.5}
        llm = LlmFactory.create("openai", config=config_dict)
        assert llm.config.model == "gpt-4"
        assert llm.config.temperature == 0.5

    @patch("mem0.llms.openai.OpenAI")
    def test_create_openai_with_base_config(self, mock_openai):
        mock_openai.return_value = MagicMock()
        base_config = BaseLlmConfig(model="gpt-4", api_key="key", temperature=0.8)
        llm = LlmFactory.create("openai", config=base_config)
        assert llm.config.model == "gpt-4"

    @patch("mem0.llms.openai.OpenAI")
    def test_create_openai_with_provider_config(self, mock_openai):
        mock_openai.return_value = MagicMock()
        config = OpenAIConfig(model="gpt-4", api_key="key")
        llm = LlmFactory.create("openai", config=config)
        assert llm.config.model == "gpt-4"

    def test_register_provider(self):
        LlmFactory.register_provider("test_provider", "mem0.llms.openai.OpenAILLM")
        assert "test_provider" in LlmFactory.provider_to_class
        # Cleanup
        del LlmFactory.provider_to_class["test_provider"]

    def test_register_provider_with_config_class(self):
        LlmFactory.register_provider("test_custom", "mem0.llms.openai.OpenAILLM", OpenAIConfig)
        assert LlmFactory.provider_to_class["test_custom"] == ("mem0.llms.openai.OpenAILLM", OpenAIConfig)
        # Cleanup
        del LlmFactory.provider_to_class["test_custom"]


class TestEmbedderFactory:
    def test_create_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported Embedder provider"):
            EmbedderFactory.create("nonexistent", {}, None)

    @patch("mem0.embeddings.openai.OpenAI")
    def test_create_openai_embedder(self, mock_openai):
        mock_openai.return_value = MagicMock()
        config = {"model": "text-embedding-3-small", "api_key": "test-key"}
        embedder = EmbedderFactory.create("openai", config, None)
        assert embedder.config.model == "text-embedding-3-small"

    def test_create_mock_embeddings_for_upstash(self):
        from mem0.embeddings.mock import MockEmbeddings

        vector_config = MagicMock()
        vector_config.enable_embeddings = True
        result = EmbedderFactory.create("upstash_vector", {}, vector_config)
        assert isinstance(result, MockEmbeddings)

    def test_supported_providers_list(self):
        expected_providers = ["openai", "ollama", "huggingface", "azure_openai", "gemini", "together"]
        for provider in expected_providers:
            assert provider in EmbedderFactory.provider_to_class


class TestVectorStoreFactory:
    def test_create_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported VectorStore provider"):
            VectorStoreFactory.create("nonexistent", {})

    def test_supported_providers_list(self):
        expected_providers = ["qdrant", "chroma", "pgvector", "milvus", "pinecone", "redis", "faiss"]
        for provider in expected_providers:
            assert provider in VectorStoreFactory.provider_to_class

    @patch("mem0.vector_stores.qdrant.Qdrant.__init__", return_value=None)
    def test_create_with_dict_config(self, mock_init):
        config = {"collection_name": "test", "embedding_model_dims": 128}
        VectorStoreFactory.create("qdrant", config)
        mock_init.assert_called_once_with(**config)

    @patch("mem0.vector_stores.qdrant.Qdrant.__init__", return_value=None)
    def test_create_with_pydantic_config(self, mock_init):
        mock_config = MagicMock()
        mock_config.model_dump.return_value = {"collection_name": "test", "embedding_model_dims": 128}
        VectorStoreFactory.create("qdrant", mock_config)
        mock_init.assert_called_once_with(collection_name="test", embedding_model_dims=128)


class TestRerankerFactory:
    def test_create_unsupported_provider(self):
        with pytest.raises(ValueError, match="Unsupported reranker provider"):
            RerankerFactory.create("nonexistent")

    def test_supported_providers(self):
        expected = ["cohere", "sentence_transformer", "zero_entropy", "llm_reranker", "huggingface"]
        for provider in expected:
            assert provider in RerankerFactory.provider_to_class

    def test_create_cohere_with_none_config(self):
        mock_cohere_module = MagicMock()
        with patch.dict(sys.modules, {"cohere": mock_cohere_module}):
            sys.modules.pop("mem0.reranker.cohere_reranker", None)
            with patch("mem0.reranker.cohere_reranker.COHERE_AVAILABLE", True, create=True):
                with patch("mem0.reranker.cohere_reranker.cohere", mock_cohere_module, create=True):
                    mock_cohere_module.Client.return_value = MagicMock()
                    reranker = RerankerFactory.create("cohere", api_key="key", model="rerank-english-v3.0")
                    assert reranker.api_key == "key"

    def test_create_cohere_with_dict_config(self):
        mock_cohere_module = MagicMock()
        with patch.dict(sys.modules, {"cohere": mock_cohere_module}):
            sys.modules.pop("mem0.reranker.cohere_reranker", None)
            with patch("mem0.reranker.cohere_reranker.COHERE_AVAILABLE", True, create=True):
                with patch("mem0.reranker.cohere_reranker.cohere", mock_cohere_module, create=True):
                    mock_cohere_module.Client.return_value = MagicMock()
                    config = {"api_key": "dict-key", "model": "rerank-english-v3.0"}
                    reranker = RerankerFactory.create("cohere", config=config)
                    assert reranker.api_key == "dict-key"

    def test_create_cohere_with_config_object(self):
        mock_cohere_module = MagicMock()
        with patch.dict(sys.modules, {"cohere": mock_cohere_module}):
            sys.modules.pop("mem0.reranker.cohere_reranker", None)
            with patch("mem0.reranker.cohere_reranker.COHERE_AVAILABLE", True, create=True):
                with patch("mem0.reranker.cohere_reranker.cohere", mock_cohere_module, create=True):
                    mock_cohere_module.Client.return_value = MagicMock()
                    config = CohereRerankerConfig(api_key="obj-key", model="rerank-english-v3.0")
                    reranker = RerankerFactory.create("cohere", config=config)
                    assert reranker.api_key == "obj-key"

    def test_create_with_invalid_config_type(self):
        with pytest.raises(ValueError, match="Config must be"):
            RerankerFactory.create("cohere", config="invalid")
