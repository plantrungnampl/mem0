import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig


@pytest.fixture(autouse=True)
def mock_boto3():
    mock_module = MagicMock()
    mock_client = MagicMock()
    mock_module.client.return_value = mock_client
    with patch.dict(sys.modules, {"boto3": mock_module}):
        sys.modules.pop("mem0.embeddings.aws_bedrock", None)
        with patch("mem0.embeddings.aws_bedrock.boto3", mock_module, create=True):
            yield mock_module, mock_client


class TestAWSBedrockEmbeddingInit:
    def test_default_model(self, mock_boto3):
        from mem0.embeddings.aws_bedrock import AWSBedrockEmbedding

        config = BaseEmbedderConfig()
        embedder = AWSBedrockEmbedding(config)
        assert embedder.config.model == "amazon.titan-embed-text-v1"

    def test_custom_model(self, mock_boto3):
        from mem0.embeddings.aws_bedrock import AWSBedrockEmbedding

        config = BaseEmbedderConfig(model="cohere.embed-english-v3")
        embedder = AWSBedrockEmbedding(config)
        assert embedder.config.model == "cohere.embed-english-v3"

    def test_uses_config_aws_credentials(self, mock_boto3):
        mock_module, _ = mock_boto3
        from mem0.embeddings.aws_bedrock import AWSBedrockEmbedding

        config = BaseEmbedderConfig(
            aws_region="us-east-1",
            aws_access_key_id="config-access-key",
            aws_secret_access_key="config-secret-key",
        )
        AWSBedrockEmbedding(config)

        mock_module.client.assert_called_with(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="config-access-key",
            aws_secret_access_key="config-secret-key",
            aws_session_token=None,
        )

    def test_default_region(self, mock_boto3, monkeypatch):
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)

        mock_module, _ = mock_boto3
        from mem0.embeddings.aws_bedrock import AWSBedrockEmbedding

        config = BaseEmbedderConfig()
        AWSBedrockEmbedding(config)

        call_kwargs = mock_module.client.call_args.kwargs
        assert call_kwargs["region_name"] == "us-west-2"


class TestAWSBedrockEmbeddingEmbed:
    def test_embed_amazon_provider(self, mock_boto3):
        _, mock_client = mock_boto3
        from mem0.embeddings.aws_bedrock import AWSBedrockEmbedding

        config = BaseEmbedderConfig(model="amazon.titan-embed-text-v1")
        embedder = AWSBedrockEmbedding(config)

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()
        mock_client.invoke_model.return_value = {"body": mock_body}

        result = embedder.embed("Hello world")

        call_kwargs = mock_client.invoke_model.call_args.kwargs
        body = json.loads(call_kwargs["body"])
        assert body == {"inputText": "Hello world"}
        assert call_kwargs["modelId"] == "amazon.titan-embed-text-v1"
        assert result == [0.1, 0.2, 0.3]

    def test_embed_cohere_provider(self, mock_boto3):
        _, mock_client = mock_boto3
        from mem0.embeddings.aws_bedrock import AWSBedrockEmbedding

        config = BaseEmbedderConfig(model="cohere.embed-english-v3")
        embedder = AWSBedrockEmbedding(config)

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"embeddings": [[0.4, 0.5, 0.6]]}).encode()
        mock_client.invoke_model.return_value = {"body": mock_body}

        result = embedder.embed("Test text")

        call_kwargs = mock_client.invoke_model.call_args.kwargs
        body = json.loads(call_kwargs["body"])
        assert body == {"input_type": "search_document", "texts": ["Test text"]}
        assert result == [0.4, 0.5, 0.6]

    def test_embed_raises_on_error(self, mock_boto3):
        _, mock_client = mock_boto3
        from mem0.embeddings.aws_bedrock import AWSBedrockEmbedding

        config = BaseEmbedderConfig(model="amazon.titan-embed-text-v1")
        embedder = AWSBedrockEmbedding(config)

        mock_client.invoke_model.side_effect = Exception("Service unavailable")

        with pytest.raises(ValueError, match="Error getting embedding from AWS Bedrock"):
            embedder.embed("Hello")

    def test_normalize_vector(self, mock_boto3):
        from mem0.embeddings.aws_bedrock import AWSBedrockEmbedding

        config = BaseEmbedderConfig()
        embedder = AWSBedrockEmbedding(config)

        result = embedder._normalize_vector([3.0, 4.0])
        # 3/5 = 0.6, 4/5 = 0.8
        assert abs(result[0] - 0.6) < 1e-6
        assert abs(result[1] - 0.8) < 1e-6
