import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mem0.configs.rerankers.base import BaseRerankerConfig
from mem0.configs.rerankers.huggingface import HuggingFaceRerankerConfig


@pytest.fixture(autouse=True)
def mock_transformers():
    mock_tokenizer_cls = MagicMock()
    mock_model_cls = MagicMock()
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.no_grad.return_value.__enter__ = MagicMock()
    mock_torch.no_grad.return_value.__exit__ = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "transformers": MagicMock(),
            "torch": mock_torch,
        },
    ):
        # Remove cached module so it reimports with our mocks
        sys.modules.pop("mem0.reranker.huggingface_reranker", None)
        with patch("mem0.reranker.huggingface_reranker.TRANSFORMERS_AVAILABLE", True, create=True):
            with patch("mem0.reranker.huggingface_reranker.AutoTokenizer", mock_tokenizer_cls, create=True):
                with patch(
                    "mem0.reranker.huggingface_reranker.AutoModelForSequenceClassification",
                    mock_model_cls,
                    create=True,
                ):
                    with patch("mem0.reranker.huggingface_reranker.torch", mock_torch, create=True):
                        yield {
                            "tokenizer_cls": mock_tokenizer_cls,
                            "model_cls": mock_model_cls,
                            "torch": mock_torch,
                        }


@pytest.fixture
def hf_config():
    return HuggingFaceRerankerConfig(
        model="BAAI/bge-reranker-base",
        device="cpu",
        batch_size=32,
        max_length=512,
        normalize=True,
        top_k=2,
    )


@pytest.fixture
def sample_documents():
    return [
        {"memory": "Python is great for data science", "id": "1"},
        {"memory": "Java is used for enterprise apps", "id": "2"},
        {"memory": "Python machine learning libraries are popular", "id": "3"},
    ]


class TestHuggingFaceRerankerInit:
    def test_init_with_hf_config(self, mock_transformers, hf_config):
        from mem0.reranker.huggingface_reranker import HuggingFaceReranker

        reranker = HuggingFaceReranker(hf_config)
        assert reranker.config == hf_config
        assert reranker.device == "cpu"
        mock_transformers["tokenizer_cls"].from_pretrained.assert_called_once_with("BAAI/bge-reranker-base")
        mock_transformers["model_cls"].from_pretrained.assert_called_once_with("BAAI/bge-reranker-base")

    def test_init_with_base_config_converts(self, mock_transformers):
        from mem0.reranker.huggingface_reranker import HuggingFaceReranker

        base_config = BaseRerankerConfig(provider="huggingface", model="test-model", top_k=5)
        reranker = HuggingFaceReranker(base_config)
        assert isinstance(reranker.config, HuggingFaceRerankerConfig)
        assert reranker.config.model == "test-model"
        assert reranker.config.top_k == 5

    def test_init_with_dict_config(self, mock_transformers):
        from mem0.reranker.huggingface_reranker import HuggingFaceReranker

        config_dict = {"model": "some-model", "device": "cpu", "batch_size": 16}
        reranker = HuggingFaceReranker(config_dict)
        assert reranker.config.model == "some-model"
        assert reranker.config.batch_size == 16

    def test_init_auto_detects_device(self, mock_transformers):
        from mem0.reranker.huggingface_reranker import HuggingFaceReranker

        config = HuggingFaceRerankerConfig(model="test-model", device=None)
        mock_transformers["torch"].cuda.is_available.return_value = False
        reranker = HuggingFaceReranker(config)
        assert reranker.device == "cpu"

    def test_init_raises_when_transformers_not_available(self, hf_config):
        with patch("mem0.reranker.huggingface_reranker.TRANSFORMERS_AVAILABLE", False):
            from mem0.reranker.huggingface_reranker import HuggingFaceReranker

            with pytest.raises(ImportError, match="transformers package is required"):
                HuggingFaceReranker(hf_config)


class TestHuggingFaceRerankerRerank:
    def test_rerank_empty_documents(self, mock_transformers, hf_config):
        from mem0.reranker.huggingface_reranker import HuggingFaceReranker

        reranker = HuggingFaceReranker(hf_config)
        result = reranker.rerank("test query", [])
        assert result == []

    def test_rerank_returns_sorted_documents(self, mock_transformers, hf_config, sample_documents):
        from mem0.reranker.huggingface_reranker import HuggingFaceReranker

        reranker = HuggingFaceReranker(hf_config)

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        reranker.tokenizer.return_value = mock_inputs

        mock_outputs = MagicMock()
        mock_logits = MagicMock()
        mock_squeezed = MagicMock()
        mock_squeezed.cpu.return_value.numpy.return_value = np.array([0.3, 0.1, 0.9])
        mock_logits.squeeze.return_value = mock_squeezed
        mock_outputs.logits = mock_logits
        reranker.model.return_value = mock_outputs

        result = reranker.rerank("Python data science", sample_documents)

        assert len(result) == 2  # top_k=2
        assert result[0]["rerank_score"] >= result[1]["rerank_score"]

    def test_rerank_with_text_field(self, mock_transformers, hf_config):
        from mem0.reranker.huggingface_reranker import HuggingFaceReranker

        reranker = HuggingFaceReranker(hf_config)
        docs = [{"text": "doc one"}, {"text": "doc two"}]

        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        reranker.tokenizer.return_value = mock_inputs

        mock_outputs = MagicMock()
        mock_logits = MagicMock()
        mock_squeezed = MagicMock()
        mock_squeezed.cpu.return_value.numpy.return_value = np.array([0.8, 0.2])
        mock_logits.squeeze.return_value = mock_squeezed
        mock_outputs.logits = mock_logits
        reranker.model.return_value = mock_outputs

        result = reranker.rerank("query", docs)
        assert len(result) == 2

    def test_rerank_fallback_on_exception(self, mock_transformers, hf_config, sample_documents):
        from mem0.reranker.huggingface_reranker import HuggingFaceReranker

        reranker = HuggingFaceReranker(hf_config)
        reranker.tokenizer.side_effect = Exception("Tokenization failed")

        result = reranker.rerank("query", sample_documents)
        assert len(result) == 2  # top_k=2
        for doc in result:
            assert doc["rerank_score"] == 0.0
