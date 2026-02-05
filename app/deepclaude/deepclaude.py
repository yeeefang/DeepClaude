"""DeepClaude 服務，用於協調 DeepSeek 和 Claude API 的調用"""

import asyncio
import copy
import json
import time
from typing import AsyncGenerator, Optional, List, Dict, Any

import tiktoken

from app.clients import ClaudeClient, DeepSeekClient
from app.utils.logger import logger
from app.utils.xml_tool_filter import StreamXMLFilter


def _condense_reasoning(reasoning: str, max_tokens: int) -> str:
    """Condense reasoning to fit within max_tokens (approximate).

    Keeps the end of reasoning since conclusions are typically there.
    Rough estimate: 1 token ~= 3 chars for mixed Chinese/English.
    """
    if max_tokens <= 0 or not reasoning:
        return reasoning
    max_chars = max_tokens * 3
    if len(reasoning) <= max_chars:
        return reasoning
    return "...\n" + reasoning[-max_chars:]


def _build_bridge_content(original_content: str, reasoning: str, system_config: dict) -> str:
    """Build optimized bridge prompt for target model.

    Minimizes token usage while preserving context.
    """
    max_reasoning_tokens = system_config.get("max_reasoning_tokens", 0)
    condensed = _condense_reasoning(reasoning, max_reasoning_tokens)

    return (
        f"{original_content}\n\n"
        f"<reasoning_context>\n{condensed}\n</reasoning_context>\n\n"
        "Based on the above reasoning context, provide a complete and helpful answer. "
        "You may use available tools if needed to gather information or perform actions. "
        "Override the reasoning with your own knowledge when it conflicts or seems incorrect."
    )


class DeepClaude:
    """處理 DeepSeek 和 Claude API 的流式輸出銜接"""

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
        system_config: dict = None,
        is_same_model: bool = False
    ):
        """初始化 API 客戶端"""
        self.system_config = system_config or {}
        self.is_same_model = is_same_model
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
        stream_options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[bytes, None]:
        """處理完整的流式輸出過程

        Args:
            messages: 初始訊息列表
            model_arg: 模型參數
            deepseek_model: DeepSeek 模型名稱
            claude_model: Claude 模型名稱
            tools: OpenAI 格式的工具定義列表
            tool_choice: 工具選擇策略
            stream_options: 串流選項 (e.g. {"include_usage": true})
        """
        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())
        output_queue = asyncio.Queue()
        claude_queue = asyncio.Queue()
        reasoning_content = []

        include_usage = bool(stream_options and stream_options.get("include_usage"))
        completion_token_count = 0
        encoding = None
        if include_usage:
            try:
                encoding = tiktoken.encoding_for_model("gpt-4o")
            except Exception:
                encoding = tiktoken.get_encoding("cl100k_base")

        async def process_deepseek():
            logger.info(f"開始處理 DeepSeek 流，使用模型：{deepseek_model}")
            try:
                # 若為單一模型模式，跳過推理流程
                if self.is_same_model:
                    logger.info("單一模型模式：跳過推理階段")
                    await claude_queue.put("")
                    return

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
                            f"DeepSeek 推理完成，收集到的推理內容長度：{len(''.join(reasoning_content))}"
                        )
                        await claude_queue.put("".join(reasoning_content))
                        break
            except Exception as e:
                logger.error(f"處理 DeepSeek 流時發生錯誤: {e}")
                error_response = _build_error_response(chat_id, created_time, deepseek_model, e)
                await output_queue.put(f"data: {json.dumps(error_response)}\n\n".encode("utf-8"))
                await output_queue.put(b"data: [DONE]\n\n")
                # FIX: Always notify Claude task so it doesn't deadlock
                await claude_queue.put("")
                await output_queue.put(None)
                return

            # FIX: Ensure Claude queue is always populated even if no content signal
            if claude_queue.empty():
                await claude_queue.put("".join(reasoning_content))

            logger.info("DeepSeek 任務處理完成，標記結束")
            await output_queue.put(None)

        async def process_claude():
            nonlocal completion_token_count
            try:
                logger.info("等待取得 DeepSeek 的推理內容...")
                try:
                    reasoning = await asyncio.wait_for(claude_queue.get(), timeout=300)
                except asyncio.TimeoutError:
                    logger.error("等待 DeepSeek 推理內容逾時 (300s)")
                    reasoning = ""

                if not reasoning:
                    logger.warning("未能取得有效的推理內容，將使用預設提示繼續")
                    reasoning = "（無推理內容）"

                # FIX: Deep copy to avoid mutating original messages
                claude_messages = copy.deepcopy(messages)

                # Extract system messages
                system_content = ""
                non_system_messages = []
                for message in claude_messages:
                    if message.get("role", "") == "system":
                        system_content += message.get("content", "") + "\n"
                    else:
                        non_system_messages.append(message)

                claude_messages = non_system_messages

                if not claude_messages:
                    raise ValueError("訊息列表為空，無法處理 Claude 請求")

                last_message = claude_messages[-1]
                if last_message.get("role", "") != "user":
                    raise ValueError("最後一則訊息的角色不是使用者，無法處理請求")

                # Optimized bridge prompt (minimal token usage)
                # Skip bridge if same model or no reasoning
                if not self.is_same_model and reasoning and reasoning != "（無推理內容）":
                    original_content = last_message["content"]
                    last_message["content"] = _build_bridge_content(
                        original_content, reasoning, self.system_config
                    )

                logger.info(
                    f"開始處理 Claude 流，使用模型: {claude_model}, 提供商: {self.claude_client.provider}, tools={len(tools) if tools else 0}"
                )

                system_content = system_content.strip() if system_content else None
                if system_content:
                    logger.debug(f"使用系統提示: {system_content[:100]}...")

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
                        filtered = xml_filter.process(content)
                        if not filtered:
                            continue
                        if encoding:
                            completion_token_count += len(encoding.encode(filtered))
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
                        flushed = xml_filter.flush()
                        if flushed:
                            if encoding:
                                completion_token_count += len(encoding.encode(flushed))
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
                        flushed = xml_filter.flush()
                        if flushed:
                            if encoding:
                                completion_token_count += len(encoding.encode(flushed))
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
                logger.error(f"處理 Claude 流時發生錯誤: {e}")
                error_response = _build_error_response(chat_id, created_time, claude_model, e)
                await output_queue.put(f"data: {json.dumps(error_response)}\n\n".encode("utf-8"))
                await output_queue.put(b"data: [DONE]\n\n")
                await output_queue.put(None)
                return
            logger.info("Claude 任務處理完成，標記結束")
            await output_queue.put(None)

        # 建立並行任務
        asyncio.create_task(process_deepseek())
        asyncio.create_task(process_claude())

        # 等待兩個任務完成
        finished_tasks = 0
        while finished_tasks < 2:
            item = await output_queue.get()
            if item is None:
                finished_tasks += 1
            else:
                yield item

        # Streaming usage: send final usage chunk before [DONE] (OpenAI spec)
        if include_usage and encoding:
            # Estimate prompt tokens from messages
            prompt_text = "\n".join(
                msg.get("content", "") for msg in messages
                if isinstance(msg.get("content", ""), str)
            )
            prompt_token_count = len(encoding.encode(prompt_text))
            usage_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": claude_model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_token_count,
                    "completion_tokens": completion_token_count,
                    "total_tokens": prompt_token_count + completion_token_count,
                },
            }
            yield f"data: {json.dumps(usage_chunk)}\n\n".encode("utf-8")

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
        """處理非流式輸出過程"""
        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())
        reasoning_content = []

        # 1. 取得 DeepSeek 的推理內容（仍然使用流式）
        try:
            async for content_type, content in self.deepseek_client.stream_chat(
                messages, deepseek_model, self.is_origin_reasoning
            ):
                if content_type == "reasoning":
                    reasoning_content.append(content)
                elif content_type == "content":
                    break
        except Exception as e:
            logger.error(f"取得 DeepSeek 推理內容時發生錯誤: {e}")
            reasoning_content = ["（取得推理內容失敗）"]

        # 2. 構造 Claude 的輸入訊息 (FIX: deep copy)
        reasoning = "".join(reasoning_content)
        claude_messages = copy.deepcopy(messages)

        # Extract system messages
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
            last_message["content"] = _build_bridge_content(
                original_content, reasoning, self.system_config
            )

        # Count input tokens
        token_content = "\n".join(
            [message.get("content", "") for message in claude_messages
             if isinstance(message.get("content", ""), str)]
        )
        try:
            encoding = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        input_tokens = encoding.encode(token_content)
        logger.debug(f"輸入 Tokens: {len(input_tokens)}")

        # 3. 取得 Claude 的回應
        try:
            answer = ""
            tool_calls_list = []
            finish_reason = "stop"
            output_tokens = []

            system_content = system_content.strip() if system_content else None
            if system_content:
                logger.debug(f"使用系統提示: {system_content[:100]}...")

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
                logger.debug(f"輸出 Tokens: {len(output_tokens)}")

            # Flush remaining XML filter buffer
            flushed = xml_filter.flush()
            if flushed:
                answer += flushed

            # 4. 構造 OpenAI 格式的回應
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
            logger.error(f"取得 Claude 回應時發生錯誤: {e}")
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
        error_info["message"] = "輸入的上下文內容過長，超過了模型的最大處理長度限制。請減少輸入內容或分段處理。"
    elif "InvalidParameter" in error_message:
        error_info["message"] = "請求參數無效，請檢查輸入內容。"
    elif "BadRequest" in error_message:
        error_info["message"] = "請求格式錯誤，請檢查輸入內容。"

    return {
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model,
        "error": error_info,
    }
