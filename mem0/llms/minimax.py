import os
from typing import Dict, List, Optional, Union

from openai import OpenAI

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.minimax import MinimaxConfig
from mem0.llms.base import LLMBase


class MiniMaxLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, MinimaxConfig, Dict]] = None):
        config = self._convert_config(config, MinimaxConfig)
        super().__init__(config)

        if not self.config.model:
            self.config.model = "MiniMax-M2.7"

        api_key = self.config.api_key or os.getenv("MINIMAX_API_KEY")
        base_url = self.config.minimax_base_url or os.getenv("MINIMAX_API_BASE") or "https://api.minimax.io/v1"
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using MiniMax.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to None.
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional MiniMax-specific parameters.

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
