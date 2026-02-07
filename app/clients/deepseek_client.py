"""DeepSeek API 客户端"""

import os
import json
from typing import AsyncGenerator, List, Dict, Any

from app.utils.logger import logger

from .base_client import BaseClient


def _sanitize_messages_for_deepseek(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sanitize messages for DeepSeek-reasoner API compatibility.

    DeepSeek-reasoner requires:
    1. All assistant messages MUST have a `reasoning_content` field (can be None)
    2. Content must be a string, not structured blocks (list of dicts)
    3. tool_calls should be in OpenAI format as a top-level field
    4. tool role messages are supported natively

    Roo Code sends structured content like:
        [{"type": "reasoning", "text": "..."}, {"type": "text", "text": "..."}, {"type": "tool_use", ...}]
    This must be converted to DeepSeek's expected flat format.

    Per DeepSeek docs: when starting a new user question, previous reasoning_content
    should be cleared (set to None) to save bandwidth.
    """
    sanitized = []
    for msg in messages:
        role = msg.get("role", "")
        new_msg = {}

        if role == "assistant":
            new_msg["role"] = "assistant"
            raw_content = msg.get("content")
            reasoning = msg.get("reasoning_content")

            if isinstance(raw_content, list):
                # Structured content from Roo Code - extract parts
                text_parts = []
                tool_calls = []
                for block in raw_content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type", "")
                    if btype == "reasoning":
                        # Extract reasoning from content blocks
                        if reasoning is None:
                            reasoning = block.get("text", "")
                    elif btype == "text":
                        text = block.get("text", "")
                        if text:
                            text_parts.append(text)
                    elif btype == "tool_use":
                        # Convert Roo's tool_use to OpenAI tool_calls format
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })

                new_msg["content"] = "\n".join(text_parts) if text_parts else ""
                if tool_calls:
                    new_msg["tool_calls"] = tool_calls
                # Also preserve existing tool_calls from the original message
                if not tool_calls and msg.get("tool_calls"):
                    new_msg["tool_calls"] = msg["tool_calls"]
            else:
                # Content is already a string or None
                new_msg["content"] = raw_content if raw_content else ""
                if msg.get("tool_calls"):
                    new_msg["tool_calls"] = msg["tool_calls"]

            # CRITICAL: DeepSeek-reasoner requires reasoning_content on ALL assistant messages
            # Set to None for previous turns (saves bandwidth per docs)
            new_msg["reasoning_content"] = None

        elif role == "tool":
            # Tool result messages - DeepSeek supports these natively
            new_msg["role"] = "tool"
            new_msg["content"] = msg.get("content", "")
            if msg.get("tool_call_id"):
                new_msg["tool_call_id"] = msg["tool_call_id"]

        elif role == "user":
            new_msg["role"] = "user"
            raw_content = msg.get("content")

            if isinstance(raw_content, list):
                # Structured content - flatten to string
                text_parts = []
                for block in raw_content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text":
                            text_parts.append(block.get("text", ""))
                        elif btype == "tool_result":
                            # Convert inline tool_result to text context
                            result_content = block.get("content", "")
                            tool_use_id = block.get("tool_use_id", "")
                            text_parts.append(f"[Tool result for {tool_use_id}]: {result_content}")
                    elif isinstance(block, str):
                        text_parts.append(block)
                new_msg["content"] = "\n".join(text_parts) if text_parts else ""
            else:
                new_msg["content"] = raw_content if raw_content else ""

        elif role == "system":
            new_msg["role"] = "system"
            raw_content = msg.get("content")
            if isinstance(raw_content, list):
                text_parts = [
                    b.get("text", "") for b in raw_content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                new_msg["content"] = "\n".join(text_parts)
            else:
                new_msg["content"] = raw_content if raw_content else ""
        else:
            # Unknown role - pass through
            new_msg = dict(msg)

        sanitized.append(new_msg)

    logger.debug(f"Sanitized {len(sanitized)} messages for DeepSeek (ensured reasoning_content on assistant msgs)")
    return sanitized


class DeepSeekClient(BaseClient):
    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.siliconflow.cn/v1/chat/completions",
        proxy: str = None,
        system_config: dict = None,
    ):
        """初始化 DeepSeek 客户端

        Args:
            api_key: DeepSeek API密钥
            api_url: DeepSeek API地址
            proxy: 代理服务器地址
            system_config: 系统配置，包含 save_deepseek_tokens 等设置
        """
        super().__init__(api_key, api_url, proxy=proxy)
        self.system_config = system_config or {}

    def _process_think_tag_content(self, content: str) -> tuple[bool, str]:
        """处理包含 think 标签的内容

        Args:
            content: 需要处理的内容字符串

        Returns:
            tuple[bool, str]:
                bool: 是否检测到完整的 think 标签对
                str: 处理后的内容
        """
        has_start = "<think>" in content
        has_end = "</think>" in content

        if has_start and has_end:
            return True, content
        elif has_start:
            return False, content
        elif not has_start and not has_end:
            return False, content
        else:
            return True, content

    async def stream_chat(
        self,
        messages: list,
        model: str = "deepseek-ai/DeepSeek-R1",
        is_origin_reasoning: bool = True,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """流式对话

        Args:
            messages: 消息列表
            model: 模型名称

        Yields:
            tuple[str, str]: (内容类型, 内容)
                内容类型: "reasoning" 或 "content"
                内容: 实际的文本内容
        """
        # Sanitize messages for DeepSeek-reasoner compatibility
        # Ensures all assistant messages have reasoning_content field
        sanitized_messages = _sanitize_messages_for_deepseek(messages)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        data = {
            "model": model,
            "messages": sanitized_messages,
            "stream": True
        }

        # 检查系统配置中的 save_deepseek_tokens 设置
        save_deepseek_tokens = self.system_config.get("save_deepseek_tokens", False)
        max_tokens_limit = self.system_config.get("save_deepseek_tokens_max_tokens", 5)

        logger.info(f"DeepSeek 客户端配置 - save_deepseek_tokens: {save_deepseek_tokens}, max_tokens_limit: {max_tokens_limit}")

        # 如果模型包含 deepseek-v3.1-terminus，启用 thinking 模式
        if "terminus" in model.lower():
            data["enable_thinking"] = True

        # 只在支持原生推理且开启了节省 tokens 功能时才添加 max_tokens 参数
        if is_origin_reasoning and save_deepseek_tokens:
            data["max_tokens"] = max_tokens_limit
            logger.info(f"已开启节省 DeepSeek tokens 功能，设置 max_tokens 为: {max_tokens_limit}")

        logger.debug(f"开始流式对话：{data}")

        accumulated_content = ""
        is_collecting_think = False

        async for chunk in self._make_request(headers, data):
            chunk_str = chunk.decode("utf-8")

            try:
                lines = chunk_str.splitlines()
                for line in lines:
                    if line.startswith("data: "):
                        json_str = line[len("data: ") :]
                        if json_str == "[DONE]":
                            return

                        data = json.loads(json_str)
                        if (
                            data
                            and data.get("choices")
                            and data["choices"][0].get("delta")
                        ):
                            delta = data["choices"][0]["delta"]

                            if is_origin_reasoning:
                                # 处理 reasoning_content
                                if delta.get("reasoning_content"):
                                    content = delta["reasoning_content"]
                                    logger.debug(f"提取推理内容：{content}")
                                    yield "reasoning", content

                                if delta.get("reasoning_content") is None and delta.get(
                                    "content"
                                ):
                                    content = delta["content"]
                                    logger.info(
                                        f"提取内容信息，推理阶段结束: {content}"
                                    )
                                    yield "content", content
                            else:
                                # 处理其他模型的输出
                                if delta.get("content"):
                                    content = delta["content"]
                                    if content == "":  # 只跳过完全空的字符串
                                        continue
                                    logger.debug(f"非原生推理内容：{content}")
                                    accumulated_content += content

                                    # 检查累积的内容是否包含完整的 think 标签对
                                    is_complete, processed_content = (
                                        self._process_think_tag_content(
                                            accumulated_content
                                        )
                                    )

                                    if "<think>" in content and not is_collecting_think:
                                        # 开始收集推理内容
                                        logger.debug(f"开始收集推理内容：{content}")
                                        is_collecting_think = True
                                        yield "reasoning", content
                                    elif is_collecting_think:
                                        if "</think>" in content:
                                            # 推理内容结束
                                            logger.debug(f"推理内容结束：{content}")
                                            is_collecting_think = False
                                            yield "reasoning", content
                                            # 输出空的 content 来触发 Claude 处理
                                            yield "content", ""
                                            # 重置累积内容
                                            accumulated_content = ""
                                        else:
                                            # 继续收集推理内容
                                            yield "reasoning", content
                                    else:
                                        # 普通内容
                                        yield "content", content

            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析错误: {e}")
            except Exception as e:
                logger.error(f"处理 chunk 时发生错误: {e}")
