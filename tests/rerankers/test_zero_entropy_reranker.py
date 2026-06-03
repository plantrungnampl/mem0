import sys
from unittest.mock import MagicMock, patch

import pytest

from mem0.configs.rerankers.zero_entropy import ZeroEntropyRerankerConfig


@pytest.fixture(autouse=True)
def mock_zeroentropy():
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"zeroentropy": mock_module}):
        sys.modules.pop("mem0.reranker.zero_entropy_reranker", None)
        with patch("mem0.reranker.zero_entropy_reranker.ZERO_ENTROPY_AVAILABLE", True, create=True):
            with patch("mem0.reranker.zero_entropy_reranker.ZeroEntropy", mock_module.ZeroEntropy, create=True):
                yield mock_module


@pytest.fixture
def ze_config():
    return ZeroEntropyRerankerConfig(
        api_key="test-ze-key",
        model="zerank-1",
        top_k=2,
    )


@pytest.fixture
def sample_documents():
    return [
        {"memory": "Transformers revolutionized NLP", "id": "1"},
        {"memory": "SQL is used for database queries", "id": "2"},
        {"memory": "BERT is a transformer model", "id": "3"},
    ]


class TestZeroEntropyRerankerInit:
    def test_init_with_api_key(self, mock_zeroentropy, ze_config):
        from mem0.reranker.zero_entropy_reranker import ZeroEntropyReranker

        reranker = ZeroEntropyReranker(ze_config)
        assert reranker.api_key == "test-ze-key"
        assert reranker.model == "zerank-1"
        mock_zeroentropy.ZeroEntropy.assert_called_once_with(api_key="test-ze-key")

    def test_init_with_env_variable(self, mock_zeroentropy, monkeypatch):
        from mem0.reranker.zero_entropy_reranker import ZeroEntropyReranker

        monkeypatch.setenv("ZERO_ENTROPY_API_KEY", "env-ze-key")
        config = ZeroEntropyRerankerConfig(model="zerank-1")
        reranker = ZeroEntropyReranker(config)
        assert reranker.api_key == "env-ze-key"

    def test_init_raises_without_api_key(self, mock_zeroentropy, monkeypatch):
        from mem0.reranker.zero_entropy_reranker import ZeroEntropyReranker

        monkeypatch.delenv("ZERO_ENTROPY_API_KEY", raising=False)
        config = ZeroEntropyRerankerConfig(model="zerank-1")
        with pytest.raises(ValueError, match="Zero Entropy API key is required"):
            ZeroEntropyReranker(config)

    def test_init_raises_when_not_available(self, ze_config):
        with patch("mem0.reranker.zero_entropy_reranker.ZERO_ENTROPY_AVAILABLE", False):
            from mem0.reranker.zero_entropy_reranker import ZeroEntropyReranker

            with pytest.raises(ImportError, match="zeroentropy package is required"):
                ZeroEntropyReranker(ze_config)

    def test_init_default_model(self, mock_zeroentropy):
        from mem0.reranker.zero_entropy_reranker import ZeroEntropyReranker

        config = ZeroEntropyRerankerConfig(api_key="key", model="zerank-1")
        reranker = ZeroEntropyReranker(config)
        assert reranker.model == "zerank-1"


class TestZeroEntropyRerankerRerank:
    def test_rerank_empty_documents(self, mock_zeroentropy, ze_config):
        from mem0.reranker.zero_entropy_reranker import ZeroEntropyReranker

        reranker = ZeroEntropyReranker(ze_config)
        result = reranker.rerank("test query", [])
        assert result == []

    def test_rerank_returns_sorted_results(self, mock_zeroentropy, ze_config, sample_documents):
        from mem0.reranker.zero_entropy_reranker import ZeroEntropyReranker

        reranker = ZeroEntropyReranker(ze_config)

        mock_result_1 = MagicMock()
        mock_result_1.index = 0
        mock_result_1.relevance_score = 0.9
        mock_result_2 = MagicMock()
        mock_result_2.index = 1
        mock_result_2.relevance_score = 0.3
        mock_result_3 = MagicMock()
        mock_result_3.index = 2
        mock_result_3.relevance_score = 0.85

        mock_response = MagicMock()
        mock_response.results = [mock_result_1, mock_result_2, mock_result_3]
        reranker.client.models.rerank.return_value = mock_response

        result = reranker.rerank("transformers NLP", sample_documents)

        reranker.client.models.rerank.assert_called_once_with(
            model="zerank-1",
            query="transformers NLP",
            documents=[
                "Transformers revolutionized NLP",
                "SQL is used for database queries",
                "BERT is a transformer model",
            ],
        )
        # top_k=2, sorted descending by score
        assert len(result) == 2
        assert result[0]["rerank_score"] == 0.9
        assert result[0]["id"] == "1"
        assert result[1]["rerank_score"] == 0.85
        assert result[1]["id"] == "3"

    def test_rerank_with_text_field(self, mock_zeroentropy, ze_config):
        from mem0.reranker.zero_entropy_reranker import ZeroEntropyReranker

        reranker = ZeroEntropyReranker(ze_config)
        docs = [{"text": "text doc one"}, {"text": "text doc two"}]

        mock_result = MagicMock()
        mock_result.index = 1
        mock_result.relevance_score = 0.7
        mock_response = MagicMock()
        mock_response.results = [mock_result]
        reranker.client.models.rerank.return_value = mock_response

        reranker.rerank("query", docs)
        call_args = reranker.client.models.rerank.call_args
        assert call_args.kwargs["documents"] == ["text doc one", "text doc two"]

    def test_rerank_with_top_k_override(self, mock_zeroentropy, ze_config, sample_documents):
        from mem0.reranker.zero_entropy_reranker import ZeroEntropyReranker

        reranker = ZeroEntropyReranker(ze_config)

        mock_results = []
        for i, score in enumerate([0.9, 0.5, 0.3]):
            r = MagicMock()
            r.index = i
            r.relevance_score = score
            mock_results.append(r)

        mock_response = MagicMock()
        mock_response.results = mock_results
        reranker.client.models.rerank.return_value = mock_response

        result = reranker.rerank("query", sample_documents, top_k=1)
        assert len(result) == 1
        assert result[0]["rerank_score"] == 0.9

    def test_rerank_fallback_on_exception(self, mock_zeroentropy, ze_config, sample_documents):
        from mem0.reranker.zero_entropy_reranker import ZeroEntropyReranker

        reranker = ZeroEntropyReranker(ze_config)
        reranker.client.models.rerank.side_effect = Exception("API error")

        result = reranker.rerank("query", sample_documents, top_k=2)
        assert len(result) == 2
        for doc in result:
            assert doc["rerank_score"] == 0.0

    def test_rerank_fallback_without_top_k(self, mock_zeroentropy):
        from mem0.reranker.zero_entropy_reranker import ZeroEntropyReranker

        config = ZeroEntropyRerankerConfig(api_key="key", model="zerank-1")
        reranker = ZeroEntropyReranker(config)
        reranker.client.models.rerank.side_effect = Exception("API error")

        docs = [{"memory": "doc1"}, {"memory": "doc2"}]
        result = reranker.rerank("query", docs)
        assert len(result) == 2
