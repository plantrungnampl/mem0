import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mem0.configs.rerankers.base import BaseRerankerConfig
from mem0.configs.rerankers.sentence_transformer import SentenceTransformerRerankerConfig


@pytest.fixture(autouse=True)
def mock_sentence_transformers():
    mock_cross_encoder = MagicMock()
    with patch.dict(sys.modules, {"sentence_transformers": MagicMock()}):
        sys.modules.pop("mem0.reranker.sentence_transformer_reranker", None)
        with patch("mem0.reranker.sentence_transformer_reranker.SENTENCE_TRANSFORMERS_AVAILABLE", True, create=True):
            with patch("mem0.reranker.sentence_transformer_reranker.CrossEncoder", mock_cross_encoder, create=True):
                yield mock_cross_encoder


@pytest.fixture
def st_config():
    return SentenceTransformerRerankerConfig(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
        batch_size=16,
        show_progress_bar=False,
        top_k=2,
    )


@pytest.fixture
def sample_documents():
    return [
        {"memory": "Neural networks are great for NLP", "id": "1"},
        {"memory": "Databases store structured data", "id": "2"},
        {"memory": "Deep learning transforms NLP tasks", "id": "3"},
    ]


class TestSentenceTransformerRerankerInit:
    def test_init_with_config(self, mock_sentence_transformers, st_config):
        from mem0.reranker.sentence_transformer_reranker import SentenceTransformerReranker

        reranker = SentenceTransformerReranker(st_config)
        mock_sentence_transformers.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        assert reranker.config == st_config

    def test_init_with_base_config_converts(self, mock_sentence_transformers):
        from mem0.reranker.sentence_transformer_reranker import SentenceTransformerReranker

        base_config = BaseRerankerConfig(provider="sentence_transformer", model="my-model", top_k=5)
        reranker = SentenceTransformerReranker(base_config)
        assert isinstance(reranker.config, SentenceTransformerRerankerConfig)
        assert reranker.config.model == "my-model"
        assert reranker.config.top_k == 5

    def test_init_with_dict_config(self, mock_sentence_transformers):
        from mem0.reranker.sentence_transformer_reranker import SentenceTransformerReranker

        config_dict = {"model": "custom-model", "device": "cpu", "batch_size": 8}
        reranker = SentenceTransformerReranker(config_dict)
        assert reranker.config.model == "custom-model"
        assert reranker.config.batch_size == 8

    def test_init_raises_when_not_available(self, st_config):
        with patch("mem0.reranker.sentence_transformer_reranker.SENTENCE_TRANSFORMERS_AVAILABLE", False):
            from mem0.reranker.sentence_transformer_reranker import SentenceTransformerReranker

            with pytest.raises(ImportError, match="sentence-transformers package is required"):
                SentenceTransformerReranker(st_config)


class TestSentenceTransformerRerankerRerank:
    def test_rerank_empty_documents(self, mock_sentence_transformers, st_config):
        from mem0.reranker.sentence_transformer_reranker import SentenceTransformerReranker

        reranker = SentenceTransformerReranker(st_config)
        result = reranker.rerank("test query", [])
        assert result == []

    def test_rerank_returns_sorted_results(self, mock_sentence_transformers, st_config, sample_documents):
        from mem0.reranker.sentence_transformer_reranker import SentenceTransformerReranker

        reranker = SentenceTransformerReranker(st_config)
        mock_model_instance = mock_sentence_transformers.return_value
        mock_model_instance.predict.return_value = np.array([0.9, 0.1, 0.85])

        result = reranker.rerank("NLP deep learning", sample_documents)

        mock_model_instance.predict.assert_called_once()
        call_args = mock_model_instance.predict.call_args
        assert call_args.kwargs["batch_size"] == 16
        assert call_args.kwargs["show_progress_bar"] is False

        assert len(result) == 2  # top_k=2
        assert result[0]["id"] == "1"  # highest score
        assert result[0]["rerank_score"] == 0.9
        assert result[1]["id"] == "3"
        assert result[1]["rerank_score"] == 0.85

    def test_rerank_with_text_field(self, mock_sentence_transformers, st_config):
        from mem0.reranker.sentence_transformer_reranker import SentenceTransformerReranker

        reranker = SentenceTransformerReranker(st_config)
        docs = [{"text": "first doc"}, {"text": "second doc"}]
        mock_model_instance = mock_sentence_transformers.return_value
        mock_model_instance.predict.return_value = np.array([0.7, 0.3])

        reranker.rerank("query", docs)
        call_args = mock_model_instance.predict.call_args
        pairs = call_args[0][0]
        assert pairs == [["query", "first doc"], ["query", "second doc"]]

    def test_rerank_with_content_field(self, mock_sentence_transformers, st_config):
        from mem0.reranker.sentence_transformer_reranker import SentenceTransformerReranker

        reranker = SentenceTransformerReranker(st_config)
        docs = [{"content": "content doc"}]
        mock_model_instance = mock_sentence_transformers.return_value
        mock_model_instance.predict.return_value = np.array([0.5])

        reranker.rerank("query", docs)
        call_args = mock_model_instance.predict.call_args
        pairs = call_args[0][0]
        assert pairs == [["query", "content doc"]]

    def test_rerank_with_top_k_override(self, mock_sentence_transformers, st_config, sample_documents):
        from mem0.reranker.sentence_transformer_reranker import SentenceTransformerReranker

        reranker = SentenceTransformerReranker(st_config)
        mock_model_instance = mock_sentence_transformers.return_value
        mock_model_instance.predict.return_value = np.array([0.9, 0.5, 0.3])

        result = reranker.rerank("query", sample_documents, top_k=1)
        assert len(result) == 1

    def test_rerank_fallback_on_exception(self, mock_sentence_transformers, st_config, sample_documents):
        from mem0.reranker.sentence_transformer_reranker import SentenceTransformerReranker

        reranker = SentenceTransformerReranker(st_config)
        mock_model_instance = mock_sentence_transformers.return_value
        mock_model_instance.predict.side_effect = Exception("Model error")

        result = reranker.rerank("query", sample_documents)
        assert len(result) == 2  # top_k=2
        for doc in result:
            assert doc["rerank_score"] == 0.0
