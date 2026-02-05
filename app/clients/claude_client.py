"""Claude API 客户端"""

import json
from typing import AsyncGenerator, Optional, List, Dict, Any

from app.utils.logger import logger
from app.utils.format_converter import (
    openai_tools_to_anthropic,
    openai_tool_choice_to_anthropic,
    convert_messages_for_anthropic,
)

from .base_client import BaseClient


class ClaudeClient(BaseClient):
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.anthropic.com/v1/messages",
        provider: str = "anthropic",
        proxy: str = None,
    ):
        """初始化 Claude 客户端

        Args:
            api_key: Claude API密钥
            api_url: Claude API地址
            provider: API提供商，支持 anthropic、openrouter、oneapi
            proxy: 代理服务器地址
        """
        super().__init__(api_key, api_url, proxy=proxy)
        self.provider = provider

    async def stream_chat(
        self,
        messages: list,
        model_arg: tuple[float, float, float, float],
        model: str,
        stream: bool = True,
        system_prompt: str = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """流式或非流式对话

        Args:
            messages: 消息列表
            model_arg: 模型参数元组[temperature, top_p, presence_penalty, frequency_penalty]
            model: 模型名称
            stream: 是否使用流式输出
            system_prompt: 系统提示
            tools: OpenAI格式的工具定义列表
            tool_choice: 工具选择策略

        Yields:
            tuple[str, Any]: (内容类型, 内容)
                "answer" -> str: 文本内容
                "tool_calls_delta" -> list: OpenAI格式的tool_calls delta
                "finish" -> str: 完成原因 ("stop" or "tool_calls")
        """
        if self.provider == "openrouter":
            yield_gen = self._stream_openrouter(messages, model_arg, model, stream, system_prompt, tools, tool_choice)
        elif self.provider == "oneapi":
            yield_gen = self._stream_oneapi(messages, model_arg, model, stream, system_prompt, tools, tool_choice)
        elif self.provider == "anthropic":
            yield_gen = self._stream_anthropic(messages, model_arg, model, stream, system_prompt, tools, tool_choice)
        else:
            raise ValueError(f"不支持的Claude Provider: {self.provider}")

        async for item in yield_gen:
            yield item

    async def _stream_openrouter(self, messages, model_arg, model, stream, system_prompt, tools, tool_choice):
        model = "anthropic/claude-3.5-sonnet"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ErlichLiu/DeepClaude",
            "X-Title": "DeepClaude",
        }
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": 1 if model_arg[0] < 0 or model_arg[0] > 1 else model_arg[0],
            "presence_penalty": model_arg[2],
            "frequency_penalty": model_arg[3],
        }
        if model_arg[1] is not None:
            data["top_p"] = model_arg[1]
        if tools:
            data["tools"] = tools
        if tool_choice is not None:
            data["tool_choice"] = tool_choice

        logger.debug(f"OpenRouter request: model={model}, tools={len(tools) if tools else 0}")

        async for item in self._parse_openai_format(headers, data, stream):
            yield item

    async def _stream_oneapi(self, messages, model_arg, model, stream, system_prompt, tools, tool_choice):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "temperature": 1 if model_arg[0] < 0 or model_arg[0] > 1 else model_arg[0],
            "presence_penalty": model_arg[2],
            "frequency_penalty": model_arg[3],
        }
        if model_arg[1] is not None:
            data["top_p"] = model_arg[1]
        if tools:
            data["tools"] = tools
        if tool_choice is not None:
            data["tool_choice"] = tool_choice

        logger.debug(f"OneAPI request: model={model}, tools={len(tools) if tools else 0}")

        async for item in self._parse_openai_format(headers, data, stream):
            yield item

    async def _stream_anthropic(self, messages, model_arg, model, stream, system_prompt, tools, tool_choice):
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "accept": "text/event-stream" if stream else "application/json",
        }

        # Convert messages to Anthropic format (handles tool_calls, tool results, etc.)
        extracted_system, converted_messages = convert_messages_for_anthropic(messages)

        # Merge system prompts
        full_system = ""
        if extracted_system:
            full_system = extracted_system
        if system_prompt:
            full_system = f"{full_system}\n{system_prompt}".strip() if full_system else system_prompt

        data = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": 8192,
            "stream": stream,
            "temperature": 1 if model_arg[0] < 0 or model_arg[0] > 1 else model_arg[0],
        }
        if model_arg[1] is not None:
            data["top_p"] = model_arg[1]
        if full_system:
            data["system"] = full_system

        # Add tools in Anthropic format
        if tools:
            data["tools"] = openai_tools_to_anthropic(tools)
            anthro_tc = openai_tool_choice_to_anthropic(tool_choice)
            if anthro_tc:
                data["tool_choice"] = anthro_tc

        logger.debug(f"Anthropic request: model={model}, tools={len(tools) if tools else 0}")

        if stream:
            async for item in self._parse_anthropic_stream(headers, data):
                yield item
        else:
            async for item in self._parse_anthropic_non_stream(headers, data):
                yield item

    async def _parse_openai_format(self, headers, data, stream):
        """Parse OpenAI-compatible format (used by OpenRouter and OneAPI)."""
        if stream:
            async for chunk in self._make_request(headers, data):
                chunk_str = chunk.decode("utf-8")
                if not chunk_str.strip():
                    continue
                for line in chunk_str.split("\n"):
                    if line.startswith("data: "):
                        json_str = line[6:]
                        if json_str.strip() == "[DONE]":
                            return
                        try:
                            resp = json.loads(json_str)
                            choice = resp.get("choices", [{}])[0]
                            delta = choice.get("delta", {})

                            # Text content
                            content = delta.get("content", "")
                            if content:
                                yield "answer", content

                            # Tool calls
                            tool_calls = delta.get("tool_calls")
                            if tool_calls:
                                yield "tool_calls_delta", tool_calls

                            # Finish reason
                            finish = choice.get("finish_reason")
                            if finish:
                                yield "finish", finish
                        except json.JSONDecodeError:
                            continue
        else:
            async for chunk in self._make_request(headers, data):
                try:
                    response = json.loads(chunk.decode("utf-8"))
                    choice = response.get("choices", [{}])[0]
                    message = choice.get("message", {})

                    content = message.get("content", "")
                    if content:
                        yield "answer", content

                    tool_calls = message.get("tool_calls")
                    if tool_calls:
                        yield "tool_calls_complete", tool_calls

                    finish = choice.get("finish_reason", "stop")
                    yield "finish", finish
                except json.JSONDecodeError:
                    continue

    async def _parse_anthropic_stream(self, headers, data):
        """Parse Anthropic streaming format and convert to standardized yields."""
        tool_call_index = -1

        async for chunk in self._make_request(headers, data):
            chunk_str = chunk.decode("utf-8")
            if not chunk_str.strip():
                continue

            for line in chunk_str.split("\n"):
                if not line.startswith("data: "):
                    continue
                json_str = line[6:]
                if json_str.strip() == "[DONE]":
                    return

                try:
                    event = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type", "")

                if event_type == "content_block_start":
                    block = event.get("content_block", {})
                    if block.get("type") == "tool_use":
                        tool_call_index += 1
                        tool_id = block.get("id", "")
                        tool_name = block.get("name", "")
                        yield "tool_calls_delta", [{
                            "index": tool_call_index,
                            "id": tool_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": ""},
                        }]
                        logger.debug(f"Anthropic tool_use start: {tool_name} (id={tool_id})")

                elif event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    delta_type = delta.get("type", "")

                    if delta_type == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            yield "answer", text

                    elif delta_type == "input_json_delta":
                        partial = delta.get("partial_json", "")
                        if partial:
                            yield "tool_calls_delta", [{
                                "index": tool_call_index,
                                "function": {"arguments": partial},
                            }]

                elif event_type == "message_delta":
                    stop_reason = event.get("delta", {}).get("stop_reason", "")
                    if stop_reason == "tool_use":
                        yield "finish", "tool_calls"
                    elif stop_reason == "end_turn":
                        yield "finish", "stop"
                    elif stop_reason:
                        yield "finish", stop_reason

    async def _parse_anthropic_non_stream(self, headers, data):
        """Parse Anthropic non-streaming response."""
        async for chunk in self._make_request(headers, data):
            try:
                response = json.loads(chunk.decode("utf-8"))
            except json.JSONDecodeError:
                continue

            tool_calls = []
            tc_index = 0

            for block in response.get("content", []):
                block_type = block.get("type", "")
                if block_type == "text":
                    text = block.get("text", "")
                    if text:
                        yield "answer", text
                elif block_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                        "index": tc_index,
                    })
                    tc_index += 1

            if tool_calls:
                yield "tool_calls_complete", tool_calls

            stop_reason = response.get("stop_reason", "end_turn")
            if stop_reason == "tool_use":
                yield "finish", "tool_calls"
            else:
                yield "finish", "stop"
