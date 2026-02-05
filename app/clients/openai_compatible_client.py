"""OpenAI 兼容格式的客户端类,用于处理符合 OpenAI API 格式的服务"""

import json
from typing import AsyncGenerator, Optional, Union, Dict, Any, List

import aiohttp
from aiohttp.client_exceptions import ClientError

from app.clients.base_client import BaseClient
from app.utils.logger import logger


class OpenAICompatibleClient(BaseClient):
    """OpenAI 兼容格式的客户端类

    用于处理符合 OpenAI API 格式的服务,如 Gemini 等
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        timeout: Optional[aiohttp.ClientTimeout] = None,
        proxy: str = None,
    ):
        """初始化 OpenAI 兼容客户端

        Args:
            api_key: API密钥
            api_url: API地址
            timeout: 请求超时设置,None则使用默认值
            proxy: 代理服务器地址
        """
        super().__init__(api_key, api_url, timeout, proxy=proxy)

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _prepare_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """处理消息格式"""
        return messages

    async def chat(
        self, messages: List[Dict[str, str]], model: str
    ) -> Dict[str, Any]:
        """非流式对话

        Args:
            messages: 消息列表
            model: 模型名称

        Returns:
            Dict[str, Any]: OpenAI 格式的完整响应
        """
        headers = self._get_headers()
        processed_messages = self._prepare_messages(messages)

        data = {
            "model": model,
            "messages": processed_messages,
            "stream": False,
        }

        try:
            response_chunks = []
            async for chunk in self._make_request(headers, data):
                response_chunks.append(chunk)

            response_text = b"".join(response_chunks).decode("utf-8")
            return json.loads(response_text)

        except Exception as e:
            error_msg = f"Chat请求失败: {str(e)}"
            logger.error(error_msg)
            raise ClientError(error_msg)

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> AsyncGenerator[tuple[str, Any], None]:
        """流式对话

        Args:
            messages: 消息列表
            model: 模型名称
            tools: OpenAI格式的工具定义列表
            tool_choice: 工具选择策略

        Yields:
            tuple[str, Any]: (content_type, content)
                "assistant"/role -> str: 文本内容
                "tool_calls_delta" -> list: tool_calls delta
                "finish" -> str: 完成原因
                Or legacy format: "assistant" -> {"finish_reason": "stop"}
        """
        headers = self._get_headers()
        processed_messages = self._prepare_messages(messages)

        data = {
            "model": model,
            "messages": processed_messages,
            "stream": True,
        }
        if tools:
            data["tools"] = tools
            logger.debug(f"OpenAI compatible client: forwarding {len(tools)} tools")
        if tool_choice is not None:
            data["tool_choice"] = tool_choice

        buffer = ""
        try:
            async for chunk in self._make_request(headers, data):
                buffer += chunk.decode("utf-8")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()

                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        json_str = line[6:].strip()
                        try:
                            response = json.loads(json_str)
                            logger.debug(f"收到响应数据: {json.dumps(response, ensure_ascii=False)[:200]}")

                            if "choices" in response and len(response["choices"]) > 0:
                                choice = response["choices"][0]
                                delta = choice.get("delta", {})

                                # Text content
                                if "content" in delta and delta["content"]:
                                    yield "assistant", delta["content"]

                                # Tool calls
                                tool_calls = delta.get("tool_calls")
                                if tool_calls:
                                    yield "tool_calls_delta", tool_calls

                                # Finish reason
                                finish_reason = choice.get("finish_reason")
                                if finish_reason == "stop":
                                    yield "assistant", {"finish_reason": "stop"}
                                elif finish_reason == "tool_calls":
                                    yield "finish", "tool_calls"
                                elif finish_reason:
                                    yield "finish", finish_reason

                                if "delta" not in choice or (not delta.get("content") and not delta.get("tool_calls")):
                                    if not finish_reason:
                                        logger.debug(f"收到不包含内容的响应: {json.dumps(choice, ensure_ascii=False)[:200]}")
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON解析错误: {str(e)}, 原始数据: {json_str[:200]}")
                            continue

        except Exception as e:
            error_msg = f"Stream chat请求失败: {str(e)}"
            logger.error(error_msg)
            raise ClientError(error_msg)
