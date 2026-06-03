import inspect
import json
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from mem0.configs.llms.base import BaseLlmConfig
from mem0.memory.utils import extract_json


class LLMBase(ABC):
    """
    Base class for all LLM providers.
    Handles common functionality and delegates provider-specific logic to subclasses.
    """

    def __init__(self, config: Optional[Union[BaseLlmConfig, Dict]] = None):
        """Initialize a base LLM class

        :param config: LLM configuration option class or dict, defaults to None
        :type config: Optional[Union[BaseLlmConfig, Dict]], optional
        """
        if config is None:
            self.config = BaseLlmConfig()
        elif isinstance(config, dict):
            # Handle dict-based configuration (backward compatibility)
            self.config = BaseLlmConfig(**config)
        else:
            self.config = config

        # Validate configuration
        self._validate_config()

    @classmethod
    def _convert_config(cls, config, provider_config_class):
        """Convert config to a provider-specific config class.

        Handles None, dict, and BaseLlmConfig inputs. Used by providers that
        have their own config subclass to avoid repeating the same
        ``if/elif/elif`` block in every ``__init__``.

        Args:
            config: None, dict, BaseLlmConfig, or an already-correct config instance.
            provider_config_class: The target config class (e.g. ``OpenAIConfig``).

        Returns:
            An instance of *provider_config_class*.
        """
        if config is None:
            return provider_config_class()
        if isinstance(config, dict):
            return provider_config_class(**config)
        if isinstance(config, BaseLlmConfig) and not isinstance(config, provider_config_class):
            base_kwargs = {
                "model": config.model,
                "temperature": config.temperature,
                "api_key": config.api_key,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "enable_vision": config.enable_vision,
                "vision_details": config.vision_details,
                "reasoning_effort": getattr(config, "reasoning_effort", None),
                "http_client_proxies": config.http_client,
            }
            # Only pass kwargs accepted by the target config class
            sig = inspect.signature(provider_config_class.__init__)
            accepted = set(sig.parameters.keys()) - {"self"}
            return provider_config_class(**{k: v for k, v in base_kwargs.items() if k in accepted})
        return config

    def _validate_config(self):
        """
        Validate the configuration.
        Override in subclasses to add provider-specific validation.
        """
        if not hasattr(self.config, "model"):
            raise ValueError("Configuration must have a 'model' attribute")

        if not hasattr(self.config, "api_key") and not hasattr(self.config, "api_key"):
            # Check if API key is available via environment variable
            # This will be handled by individual providers
            pass

    def _is_reasoning_model(self, model: str) -> bool:
        """
        Check if the model is a reasoning model or GPT-5 series that doesn't support certain parameters.

        Args:
            model: The model name to check

        Returns:
            bool: True if the model is a reasoning model or GPT-5 series
        """
        reasoning_models = {
            "o1",
            "o1-preview",
            "o3-mini",
            "o3",
            "gpt-5",
            "gpt-5o",
            "gpt-5o-mini",
            "gpt-5o-micro",
        }

        model_lower = model.lower()
        # Strip provider prefixes (e.g. "openai/o3-mini" -> "o3-mini")
        base_model = model_lower.rsplit("/", 1)[-1]

        if base_model in reasoning_models:
            return True

        # Match o1/o3 family with prefixes (o1-2024-12-17, o3-2025-04-16)
        # but NOT gpt-5.x variants (gpt-5.4-mini supports temperature)
        if any(base_model.startswith(prefix) for prefix in ["o1-", "o1.", "o3-", "o3."]):
            return True

        return False

    def _get_supported_params(self, **kwargs) -> Dict:
        """
        Get parameters that are supported by the current model.
        Filters out unsupported parameters for reasoning models and GPT-5 series.

        Args:
            **kwargs: Additional parameters to include

        Returns:
            Dict: Filtered parameters dictionary
        """
        model = getattr(self.config, "model", "")

        if self._is_reasoning_model(model):
            supported_params = {}

            if "messages" in kwargs:
                supported_params["messages"] = kwargs["messages"]
            if "response_format" in kwargs:
                supported_params["response_format"] = kwargs["response_format"]
            if "tools" in kwargs:
                supported_params["tools"] = kwargs["tools"]
            if "tool_choice" in kwargs:
                supported_params["tool_choice"] = kwargs["tool_choice"]

            # Add reasoning_effort if configured
            reasoning_effort = getattr(self.config, "reasoning_effort", None)
            if reasoning_effort:
                supported_params["reasoning_effort"] = reasoning_effort

            return supported_params
        else:
            # For regular models, include all common parameters
            return self._get_common_params(**kwargs)

    def _parse_response(self, response, tools):
        """Parse an OpenAI-compatible chat completion response.

        Returns the assistant message content when *tools* is falsy, or a dict
        with ``content`` and ``tool_calls`` keys when tools were requested.

        Providers whose API returns a different shape (e.g. Gemini, Ollama)
        should override this method.
        """
        if tools:
            processed_response = {
                "content": response.choices[0].message.content,
                "tool_calls": [],
            }

            if response.choices[0].message.tool_calls:
                for tool_call in response.choices[0].message.tool_calls:
                    processed_response["tool_calls"].append(
                        {
                            "name": tool_call.function.name,
                            "arguments": json.loads(extract_json(tool_call.function.arguments)),
                        }
                    )

            return processed_response
        else:
            return response.choices[0].message.content

    @abstractmethod
    def generate_response(
        self, messages: List[Dict[str, str]], tools: Optional[List[Dict]] = None, tool_choice: str = "auto", **kwargs
    ):
        """
        Generate a response based on the given messages.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional provider-specific parameters.

        Returns:
            str or dict: The generated response.
        """
        pass

    def _get_common_params(self, **kwargs) -> Dict:
        """
        Get common parameters that most providers use.

        Returns:
            Dict: Common parameters dictionary.
        """
        params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }

        # Add provider-specific parameters from kwargs
        params.update(kwargs)

        return params
