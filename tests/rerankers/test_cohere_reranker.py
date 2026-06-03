import sys
from unittest.mock import MagicMock, patch

import pytest

from mem0.configs.rerankers.cohere import CohereRerankerConfig


@pytest.fixture(autouse=True)
def mock_cohere():
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"cohere": mock_module}):
        sys.modules.pop("mem0.reranker.cohere_reranker", None)
        with patch("mem0.reranker.cohere_reranker.COHERE_AVAILABLE", True, create=True):
            with patch("mem0.reranker.cohere_reranker.cohere", mock_module, create=True):
                yield mock_module


@pytest.fixture
def cohere_config():
    return CohereRerankerConfig(
        api_key="test-api-key",
        model="rerank-english-v3.0",
        top_k=3,
        return_documents=False,
        max_chunks_per_doc=10,
    )


@pytest.fixture
def sample_documents():
    return [
        {"memory": "Python is a programming language", "id": "1"},
        {"memory": "JavaScript is used for web development", "id": "2"},
        {"memory": "Machine learning uses Python extensively", "id": "3"},
    ]


class TestCohereRerankerInit:
    def test_init_with_api_key(self, mock_cohere, cohere_config):
        from mem0.reranker.cohere_reranker import CohereReranker

        reranker = CohereReranker(cohere_config)
        assert reranker.api_key == "test-api-key"
        assert reranker.model == "rerank-english-v3.0"
        mock_cohere.Client.assert_called_once_with("test-api-key")

    def test_init_with_env_variable(self, mock_cohere, monkeypatch):
        from mem0.reranker.cohere_reranker import CohereReranker

        monkeypatch.setenv("COHERE_API_KEY", "env-api-key")
        config = CohereRerankerConfig(model="rerank-english-v3.0")
        reranker = CohereReranker(config)
        assert reranker.api_key == "env-api-key"

    def test_init_raises_without_api_key(self, mock_cohere, monkeypatch):
        from mem0.reranker.cohere_reranker import CohereReranker

        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        config = CohereRerankerConfig(model="rerank-english-v3.0")
        with pytest.raises(ValueError, match="Cohere API key is required"):
            CohereReranker(config)

    def test_init_raises_when_cohere_not_available(self, cohere_config):
        with patch("mem0.reranker.cohere_reranker.COHERE_AVAILABLE", False):
            from mem0.reranker.cohere_reranker import CohereReranker

            with pytest.raises(ImportError, match="cohere package is required"):
                CohereReranker(cohere_config)


class TestCohereRerankerRerank:
    def test_rerank_empty_documents(self, mock_cohere, cohere_config):
        from mem0.reranker.cohere_reranker import CohereReranker

        reranker = CohereReranker(cohere_config)
        result = reranker.rerank("test query", [])
        assert result == []

    def test_rerank_with_memory_field(self, mock_cohere, cohere_config, sample_documents):
        from mem0.reranker.cohere_reranker import CohereReranker

        reranker = CohereReranker(cohere_config)

        mock_result_1 = MagicMock()
        mock_result_1.index = 2
        mock_result_1.relevance_score = 0.95
        mock_result_2 = MagicMock()
        mock_result_2.index = 0
        mock_result_2.relevance_score = 0.80

        mock_response = MagicMock()
        mock_response.results = [mock_result_1, mock_result_2]
        reranker.client.rerank.return_value = mock_response

        result = reranker.rerank("Python programming", sample_documents, top_k=2)

        reranker.client.rerank.assert_called_once_with(
            model="rerank-english-v3.0",
            query="Python programming",
            documents=[
                "Python is a programming language",
                "JavaScript is used for web development",
                "Machine learning uses Python extensively",
            ],
            top_n=2,
            return_documents=False,
            max_chunks_per_doc=10,
        )
        assert len(result) == 2
        assert result[0]["rerank_score"] == 0.95
        assert result[0]["id"] == "3"
        assert result[1]["rerank_score"] == 0.80
        assert result[1]["id"] == "1"

    def test_rerank_with_text_field(self, mock_cohere, cohere_config):
        from mem0.reranker.cohere_reranker import CohereReranker

        reranker = CohereReranker(cohere_config)
        docs = [{"text": "doc one"}, {"text": "doc two"}]

        mock_result = MagicMock()
        mock_result.index = 0
        mock_result.relevance_score = 0.9
        mock_response = MagicMock()
        mock_response.results = [mock_result]
        reranker.client.rerank.return_value = mock_response

        result = reranker.rerank("query", docs, top_k=1)
        assert result[0]["rerank_score"] == 0.9

    def test_rerank_with_content_field(self, mock_cohere, cohere_config):
        from mem0.reranker.cohere_reranker import CohereReranker

        reranker = CohereReranker(cohere_config)
        docs = [{"content": "content doc"}]

        mock_result = MagicMock()
        mock_result.index = 0
        mock_result.relevance_score = 0.85
        mock_response = MagicMock()
        mock_response.results = [mock_result]
        reranker.client.rerank.return_value = mock_response

        result = reranker.rerank("query", docs, top_k=1)
        assert result[0]["rerank_score"] == 0.85

    def test_rerank_fallback_on_exception(self, mock_cohere, cohere_config, sample_documents):
        from mem0.reranker.cohere_reranker import CohereReranker

        reranker = CohereReranker(cohere_config)
        reranker.client.rerank.side_effect = Exception("API error")

        result = reranker.rerank("query", sample_documents, top_k=2)
        assert len(result) == 2
        for doc in result:
            assert doc["rerank_score"] == 0.0

    def test_rerank_fallback_without_top_k(self, mock_cohere, monkeypatch):
        from mem0.reranker.cohere_reranker import CohereReranker

        config = CohereRerankerConfig(api_key="key", model="rerank-english-v3.0")
        reranker = CohereReranker(config)
        reranker.client.rerank.side_effect = Exception("API error")

        docs = [{"memory": "doc1"}, {"memory": "doc2"}]
        result = reranker.rerank("query", docs)
        assert len(result) == 2
