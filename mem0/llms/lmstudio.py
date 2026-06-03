from typing import Dict, List, Optional, Union

from openai import OpenAI

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.lmstudio import LMStudioConfig
from mem0.llms.base import LLMBase


class LMStudioLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, LMStudioConfig, Dict]] = None):
        config = self._convert_config(config, LMStudioConfig)
        super().__init__(config)

        self.config.model = (
            self.config.model
            or "lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/Meta-Llama-3.1-70B-Instruct-IQ2_M.gguf"
        )
        self.config.api_key = self.config.api_key or "lm-studio"

        self.client = OpenAI(base_url=self.config.lmstudio_base_url, api_key=self.config.api_key)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using LM Studio.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional LM Studio-specific parameters.

        Returns:
            str: The generated response.
        """
        params = self._get_supported_params(messages=messages, **kwargs)
        params.update(
            {
                "model": self.config.model,
                "messages": messages,
            }
        )

        if self.config.lmstudio_response_format:
            params["response_format"] = self.config.lmstudio_response_format
        elif response_format:
            params["response_format"] = response_format
        else:
            params["response_format"] = {"type": "json_object"}

        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, tools)
