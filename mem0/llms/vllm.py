import os
from typing import Dict, List, Optional, Union

from openai import OpenAI

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.vllm import VllmConfig
from mem0.llms.base import LLMBase


class VllmLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, VllmConfig, Dict]] = None):
        config = self._convert_config(config, VllmConfig)
        super().__init__(config)

        if not self.config.model:
            self.config.model = "Qwen/Qwen2.5-32B-Instruct"

        self.config.api_key = self.config.api_key or os.getenv("VLLM_API_KEY") or "vllm-api-key"
        base_url = self.config.vllm_base_url or os.getenv("VLLM_BASE_URL")
        self.client = OpenAI(api_key=self.config.api_key, base_url=base_url)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using vLLM.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional vLLM-specific parameters.

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

        if response_format:
            params["response_format"] = response_format
        if tools:
            params["tools"] = tools
            params["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**params)
        return self._parse_response(response, tools)
