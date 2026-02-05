"""DeepClaude 服务，用于协调 DeepSeek 和 Claude API 的调用"""

import asyncio
import json
import time
from typing import AsyncGenerator, Optional, List, Dict, Any

import tiktoken

from app.clients import ClaudeClient, DeepSeekClient
from app.utils.logger import logger
from app.utils.xml_tool_filter import StreamXMLFilter


class DeepClaude:
    """处理 DeepSeek 和 Claude API 的流式输出衔接"""

    def __init__(
        self,
        deepseek_api_key: str,
        claude_api_key: str,
        deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions",
        claude_api_url: str = "https://api.anthropic.com/v1/messages",
        claude_provider: str = "anthropic",
        is_origin_reasoning: bool = True,
        reasoner_proxy: str = None,
        target_proxy: str = None,
        system_config: dict = None
    ):
        """初始化 API 客户端"""
        self.system_config = system_config or {}
        self.deepseek_client = DeepSeekClient(
            deepseek_api_key,
            deepseek_api_url,
            proxy=reasoner_proxy,
            system_config=self.system_config
        )
        self.claude_client = ClaudeClient(
            claude_api_key, claude_api_url, claude_provider, proxy=target_proxy
        )
        self.is_origin_reasoning = is_origin_reasoning

    async def chat_completions_with_stream(
        self,
        messages: list,
        model_arg: tuple[float, float, float, float],
        deepseek_model: str = "deepseek-reasoner",
        claude_model: str = "claude-3-5-sonnet-20241022",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> AsyncGenerator[bytes, None]:
        """处理完整的流式输出过程

        Args:
            messages: 初始消息列表
            model_arg: 模型参数
            deepseek_model: DeepSeek 模型名称
            claude_model: Claude 模型名称
            tools: OpenAI格式的工具定义列表
            tool_choice: 工具选择策略
        """
        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())
        output_queue = asyncio.Queue()
        claude_queue = asyncio.Queue()
        reasoning_content = []

        async def process_deepseek():
            logger.info(f"开始处理 DeepSeek 流，使用模型：{deepseek_model}")
            try:
                async for content_type, content in self.deepseek_client.stream_chat(
                    messages, deepseek_model, self.is_origin_reasoning
                ):
                    if content_type == "reasoning":
                        reasoning_content.append(content)
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": deepseek_model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "reasoning_content": content,
                                    "content": "",
                                },
                            }],
                        }
                        await output_queue.put(
                            f"data: {json.dumps(response)}\n\n".encode("utf-8")
                        )
                    elif content_type == "content":
                        logger.info(
                            f"DeepSeek 推理完成，收集到的推理内容长度：{len(''.join(reasoning_content))}"
                        )
                        await claude_queue.put("".join(reasoning_content))
                        break
            except Exception as e:
                logger.error(f"处理 DeepSeek 流时发生错误: {e}")
                error_response = _build_error_response(chat_id, created_time, deepseek_model, e)
                await output_queue.put(f"data: {json.dumps(error_response)}\n\n".encode("utf-8"))
                await output_queue.put(b"data: [DONE]\n\n")
                await output_queue.put(None)
                return
            logger.info("DeepSeek 任务处理完成，标记结束")
            await output_queue.put(None)

        async def process_claude():
            try:
                logger.info("等待获取 DeepSeek 的推理内容...")
                reasoning = await claude_queue.get()
                logger.debug(
                    f"获取到推理内容，内容长度：{len(reasoning) if reasoning else 0}"
                )
                if not reasoning:
                    logger.warning("未能获取到有效的推理内容，将使用默认提示继续")
                    reasoning = "获取推理内容失败"

                # 构造 Claude 的输入消息
                claude_messages = messages.copy()
                combined_content = f"""
                ******The above is user information*****
The following is the reasoning process of another model:****\n{reasoning}\n\n ****
Based on this reasoning, combined with your knowledge, when the current reasoning conflicts with your knowledge, you are more confident that you can adopt your own knowledge, which is completely acceptable. Please provide the user with a complete answer directly. You do not need to repeat the request or make your own reasoning. Please be sure to reply completely:"""

                # 提取 system message 并同时过滤掉 system messages
                system_content = ""
                non_system_messages = []
                for message in claude_messages:
                    if message.get("role", "") == "system":
                        system_content += message.get("content", "") + "\n"
                    else:
                        non_system_messages.append(message)

                claude_messages = non_system_messages

                if not claude_messages:
                    raise ValueError("消息列表为空，无法处理 Claude 请求")

                last_message = claude_messages[-1]
                if last_message.get("role", "") != "user":
                    raise ValueError("最后一个消息的角色不是用户，无法处理请求")

                # 修改最后一个消息的内容
                original_content = last_message["content"]
                fixed_content = f"Here's my original input:\n{original_content}\n\n{combined_content}"
                last_message["content"] = fixed_content

                logger.info(
                    f"开始处理 Claude 流，使用模型: {claude_model}, 提供商: {self.claude_client.provider}, tools={len(tools) if tools else 0}"
                )

                system_content = system_content.strip() if system_content else None
                if system_content:
                    logger.debug(f"使用系统提示: {system_content[:100]}...")

                xml_filter = StreamXMLFilter()

                async for content_type, content in self.claude_client.stream_chat(
                    messages=claude_messages,
                    model_arg=model_arg,
                    model=claude_model,
                    system_prompt=system_content,
                    tools=tools,
                    tool_choice=tool_choice,
                ):
                    if content_type == "answer":
                        # Filter out XML tool call markup from Claude's output
                        filtered = xml_filter.process(content)
                        if not filtered:
                            continue
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": claude_model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": filtered},
                            }],
                        }
                        await output_queue.put(
                            f"data: {json.dumps(response)}\n\n".encode("utf-8")
                        )

                    elif content_type == "tool_calls_delta":
                        # Flush any buffered text from XML filter before tool calls
                        flushed = xml_filter.flush()
                        if flushed:
                            flush_resp = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": claude_model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": flushed},
                                }],
                            }
                            await output_queue.put(
                                f"data: {json.dumps(flush_resp)}\n\n".encode("utf-8")
                            )

                        # Forward tool_calls delta in OpenAI format
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": claude_model,
                            "choices": [{
                                "index": 0,
                                "delta": {"tool_calls": content},
                            }],
                        }
                        await output_queue.put(
                            f"data: {json.dumps(response)}\n\n".encode("utf-8")
                        )

                    elif content_type == "finish":
                        # Flush remaining buffered text
                        flushed = xml_filter.flush()
                        if flushed:
                            flush_resp = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": claude_model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": flushed},
                                }],
                            }
                            await output_queue.put(
                                f"data: {json.dumps(flush_resp)}\n\n".encode("utf-8")
                            )

                        # Send finish reason
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": claude_model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": content,
                            }],
                        }
                        await output_queue.put(
                            f"data: {json.dumps(response)}\n\n".encode("utf-8")
                        )

            except Exception as e:
                logger.error(f"处理 Claude 流时发生错误: {e}")
                error_response = _build_error_response(chat_id, created_time, claude_model, e)
                await output_queue.put(f"data: {json.dumps(error_response)}\n\n".encode("utf-8"))
                await output_queue.put(b"data: [DONE]\n\n")
                await output_queue.put(None)
                return
            logger.info("Claude 任务处理完成，标记结束")
            await output_queue.put(None)

        # 创建并发任务
        asyncio.create_task(process_deepseek())
        asyncio.create_task(process_claude())

        # 等待两个任务完成
        finished_tasks = 0
        while finished_tasks < 2:
            item = await output_queue.get()
            if item is None:
                finished_tasks += 1
            else:
                yield item

        yield b"data: [DONE]\n\n"

    async def chat_completions_without_stream(
        self,
        messages: list,
        model_arg: tuple[float, float, float, float],
        deepseek_model: str = "deepseek-reasoner",
        claude_model: str = "claude-3-5-sonnet-20241022",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> dict:
        """处理非流式输出过程"""
        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())
        reasoning_content = []

        # 1. 获取 DeepSeek 的推理内容（仍然使用流式）
        try:
            async for content_type, content in self.deepseek_client.stream_chat(
                messages, deepseek_model, self.is_origin_reasoning
            ):
                if content_type == "reasoning":
                    reasoning_content.append(content)
                elif content_type == "content":
                    break
        except Exception as e:
            logger.error(f"获取 DeepSeek 推理内容时发生错误: {e}")
            reasoning_content = ["获取推理内容失败"]

        # 2. 构造 Claude 的输入消息
        reasoning = "".join(reasoning_content)
        claude_messages = messages.copy()

        combined_content = f"""
            ******The above is user information*****
The following is the reasoning process of another model:****\n{reasoning}\n\n ****
Based on this reasoning, combined with your knowledge, when the current reasoning conflicts with your knowledge, you are more confident that you can adopt your own knowledge, which is completely acceptable. Please provide the user with a complete answer directly. You do not need to repeat the request or make your own reasoning. Please be sure to reply completely:"""

        # 提取 system message 并同时从原始 messages 中过滤掉 system messages
        system_content = ""
        non_system_messages = []
        for message in claude_messages:
            if message.get("role", "") == "system":
                system_content += message.get("content", "") + "\n"
            else:
                non_system_messages.append(message)

        claude_messages = non_system_messages

        last_message = claude_messages[-1]
        if last_message.get("role", "") == "user":
            original_content = last_message["content"]
            fixed_content = (
                f"Here's my original input:\n{original_content}\n\n{combined_content}"
            )
            last_message["content"] = fixed_content

        # 拼接所有 content 为一个字符串，计算 token
        token_content = "\n".join(
            [message.get("content", "") for message in claude_messages
             if isinstance(message.get("content", ""), str)]
        )
        encoding = tiktoken.encoding_for_model("gpt-4o")
        input_tokens = encoding.encode(token_content)
        logger.debug(f"输入 Tokens: {len(input_tokens)}")

        # 3. 获取 Claude 的响应
        try:
            answer = ""
            tool_calls_list = []
            finish_reason = "stop"
            output_tokens = []

            system_content = system_content.strip() if system_content else None
            if system_content:
                logger.debug(f"使用系统提示: {system_content[:100]}...")

            xml_filter = StreamXMLFilter()
            async for content_type, content in self.claude_client.stream_chat(
                messages=claude_messages,
                model_arg=model_arg,
                model=claude_model,
                stream=False,
                system_prompt=system_content,
                tools=tools,
                tool_choice=tool_choice,
            ):
                if content_type == "answer":
                    filtered = xml_filter.process(content)
                    answer += filtered
                    output_tokens = encoding.encode(answer)
                elif content_type == "tool_calls_complete":
                    tool_calls_list = content
                elif content_type == "finish":
                    finish_reason = content
                logger.debug(f"输出 Tokens: {len(output_tokens)}")

            # Flush remaining XML filter buffer
            flushed = xml_filter.flush()
            if flushed:
                answer += flushed

            # 4. 构造 OpenAI 格式的响应
            response_message = {
                "role": "assistant",
                "content": answer if answer else None,
                "reasoning_content": reasoning,
            }
            if tool_calls_list:
                response_message["tool_calls"] = tool_calls_list

            return {
                "id": chat_id,
                "object": "chat.completion",
                "created": created_time,
                "model": claude_model,
                "choices": [{
                    "index": 0,
                    "message": response_message,
                    "finish_reason": finish_reason,
                }],
                "usage": {
                    "prompt_tokens": len(input_tokens),
                    "completion_tokens": len(output_tokens),
                    "total_tokens": len(input_tokens) + len(output_tokens),
                },
            }
        except Exception as e:
            logger.error(f"获取 Claude 响应时发生错误: {e}")
            raise e


def _build_error_response(chat_id: str, created_time: int, model: str, error: Exception) -> dict:
    """Build a standardized error response."""
    error_message = str(error)
    error_info = {
        "message": error_message,
        "type": "api_error",
        "code": "invalid_request_error",
    }
    if "Input length" in error_message:
        error_info["message"] = "输入的上下文内容过长，超过了模型的最大处理长度限制。请减少输入内容或分段处理。"
    elif "InvalidParameter" in error_message:
        error_info["message"] = "请求参数无效，请检查输入内容。"
    elif "BadRequest" in error_message:
        error_info["message"] = "请求格式错误，请检查输入内容。"

    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "error": error_info,
    }
