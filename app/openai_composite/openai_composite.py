"""OpenAI 兼容的組合模型服務，用於協調 DeepSeek 和其他 OpenAI 兼容模型的調用"""

import asyncio
import copy
import json
import time
from typing import AsyncGenerator, Dict, Any, List, Optional

import tiktoken

from app.clients import DeepSeekClient
from app.clients.openai_compatible_client import OpenAICompatibleClient
from app.utils.logger import logger
from app.utils.xml_tool_filter import StreamXMLFilter
from app.deepclaude.deepclaude import _build_bridge_content


class OpenAICompatibleComposite:
    """處理 DeepSeek 和其他 OpenAI 兼容模型的流式輸出銜接"""

    def __init__(
        self,
        deepseek_api_key: str,
        openai_api_key: str,
        deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions",
        openai_api_url: str = "",
        is_origin_reasoning: bool = True,
        reasoner_proxy: str = None,
        target_proxy: str = None,
        system_config: dict = None
    ):
        """初始化 API 客戶端"""
        self.system_config = system_config or {}
        self.deepseek_client = DeepSeekClient(
            deepseek_api_key,
            deepseek_api_url,
            proxy=reasoner_proxy,
            system_config=self.system_config
        )
        self.openai_client = OpenAICompatibleClient(openai_api_key, openai_api_url, proxy=target_proxy)
        self.is_origin_reasoning = is_origin_reasoning

    async def chat_completions_with_stream(
        self,
        messages: List[Dict[str, str]],
        model_arg: tuple[float, float, float, float],
        deepseek_model: str = "deepseek-reasoner",
        target_model: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        stream_options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[bytes, None]:
        """處理完整的流式輸出過程"""
        chat_id = f"chatcmpl-{hex(int(time.time() * 1000))[2:]}"
        created_time = int(time.time())
        output_queue = asyncio.Queue()
        reasoning_queue = asyncio.Queue()
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
                        await reasoning_queue.put("".join(reasoning_content))
                        break
            except Exception as e:
                logger.error(f"處理 DeepSeek 流時發生錯誤: {e}")
                error_response = _build_error_response(chat_id, created_time, deepseek_model, e)
                await output_queue.put(f"data: {json.dumps(error_response)}\n\n".encode("utf-8"))
                await output_queue.put(b"data: [DONE]\n\n")
                # FIX: Always notify target task to prevent deadlock
                await reasoning_queue.put("")
                await output_queue.put(None)
                return

            # FIX: Ensure reasoning_queue is always populated
            if reasoning_queue.empty():
                await reasoning_queue.put("".join(reasoning_content))

            logger.info("DeepSeek 任務處理完成，標記結束")
            await output_queue.put(None)

        async def process_openai():
            nonlocal completion_token_count
            try:
                logger.info("等待取得 DeepSeek 的推理內容...")
                try:
                    reasoning = await asyncio.wait_for(reasoning_queue.get(), timeout=300)
                except asyncio.TimeoutError:
                    logger.error("等待 DeepSeek 推理內容逾時 (300s)")
                    reasoning = ""

                if not reasoning:
                    logger.warning("未能取得有效的推理內容，將使用預設提示繼續")
                    reasoning = "（無推理內容）"

                # FIX: Deep copy to avoid mutating original messages
                openai_messages = copy.deepcopy(messages)

                if not openai_messages:
                    raise ValueError("訊息列表為空，無法處理請求")

                last_message = openai_messages[-1]
                if last_message.get("role", "") != "user":
                    raise ValueError("最後一則訊息的角色不是使用者，無法處理請求")

                # Optimized bridge prompt
                original_content = last_message["content"]
                last_message["content"] = _build_bridge_content(
                    original_content, reasoning, self.system_config
                )

                logger.info(f"開始處理 OpenAI 兼容流，使用模型: {target_model}, tools={len(tools) if tools else 0}")

                xml_filter = StreamXMLFilter()

                async for role, content in self.openai_client.stream_chat(
                    messages=openai_messages,
                    model=target_model,
                    tools=tools,
                    tool_choice=tool_choice,
                ):
                    if isinstance(content, dict) and content.get("finish_reason") == "stop":
                        flushed = xml_filter.flush()
                        if flushed:
                            if encoding:
                                completion_token_count += len(encoding.encode(flushed))
                            flush_resp = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": target_model,
                                "choices": [{"index": 0, "delta": {"role": "assistant", "content": flushed}}],
                            }
                            await output_queue.put(f"data: {json.dumps(flush_resp)}\n\n".encode("utf-8"))
                        end_response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": target_model,
                            "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                        }
                        await output_queue.put(f"data: {json.dumps(end_response)}\n\n".encode("utf-8"))
                        break

                    if role == "tool_calls_delta":
                        flushed = xml_filter.flush()
                        if flushed:
                            if encoding:
                                completion_token_count += len(encoding.encode(flushed))
                            flush_resp = {
                                "id": chat_id, "object": "chat.completion.chunk", "created": created_time,
                                "model": target_model,
                                "choices": [{"index": 0, "delta": {"role": "assistant", "content": flushed}}],
                            }
                            await output_queue.put(f"data: {json.dumps(flush_resp)}\n\n".encode("utf-8"))
                        response = {
                            "id": chat_id, "object": "chat.completion.chunk", "created": created_time,
                            "model": target_model,
                            "choices": [{"index": 0, "delta": {"tool_calls": content}}],
                        }
                        await output_queue.put(f"data: {json.dumps(response)}\n\n".encode("utf-8"))
                        continue

                    if role == "finish":
                        flushed = xml_filter.flush()
                        if flushed:
                            if encoding:
                                completion_token_count += len(encoding.encode(flushed))
                            flush_resp = {
                                "id": chat_id, "object": "chat.completion.chunk", "created": created_time,
                                "model": target_model,
                                "choices": [{"index": 0, "delta": {"role": "assistant", "content": flushed}}],
                            }
                            await output_queue.put(f"data: {json.dumps(flush_resp)}\n\n".encode("utf-8"))
                        response = {
                            "id": chat_id, "object": "chat.completion.chunk", "created": created_time,
                            "model": target_model,
                            "choices": [{"index": 0, "delta": {}, "finish_reason": content}],
                        }
                        await output_queue.put(f"data: {json.dumps(response)}\n\n".encode("utf-8"))
                        break

                    filtered_content = xml_filter.process(content)
                    if not filtered_content:
                        continue
                    if encoding:
                        completion_token_count += len(encoding.encode(filtered_content))
                    response = {
                        "id": chat_id, "object": "chat.completion.chunk", "created": created_time,
                        "model": target_model,
                        "choices": [{"index": 0, "delta": {"role": role, "content": filtered_content}}],
                    }
                    await output_queue.put(f"data: {json.dumps(response)}\n\n".encode("utf-8"))

            except Exception as e:
                logger.error(f"處理 OpenAI 兼容流時發生錯誤: {e}")
                error_response = _build_error_response(chat_id, created_time, target_model, e)
                await output_queue.put(f"data: {json.dumps(error_response)}\n\n".encode("utf-8"))
                await output_queue.put(b"data: [DONE]\n\n")
                await output_queue.put(None)
                return
            logger.info("OpenAI 兼容任務處理完成，標記結束")
            await output_queue.put(None)

        asyncio.create_task(process_deepseek())
        asyncio.create_task(process_openai())

        finished_tasks = 0
        while finished_tasks < 2:
            item = await output_queue.get()
            if item is None:
                finished_tasks += 1
                continue
            yield item

        # Streaming usage chunk (OpenAI spec: stream_options.include_usage)
        if include_usage and encoding:
            prompt_text = "\n".join(
                msg.get("content", "") for msg in messages
                if isinstance(msg.get("content", ""), str)
            )
            prompt_token_count = len(encoding.encode(prompt_text))
            usage_chunk = {
                "id": chat_id, "object": "chat.completion.chunk", "created": created_time,
                "model": target_model, "choices": [],
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
        messages: List[Dict[str, str]],
        model_arg: tuple[float, float, float, float],
        deepseek_model: str = "deepseek-reasoner",
        target_model: str = "",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """處理非流式輸出請求"""
        full_response = {
            "id": f"chatcmpl-{hex(int(time.time() * 1000))[2:]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": target_model,
            "choices": [],
            "usage": {},
        }

        content_parts = []
        reasoning_parts = []
        tool_calls_list = []
        finish_reason = "stop"

        try:
            # Pass include_usage so we get token counts from streaming
            async for chunk in self.chat_completions_with_stream(
                messages, model_arg, deepseek_model, target_model, tools, tool_choice,
                stream_options={"include_usage": True},
            ):
                if chunk != b"data: [DONE]\n\n":
                    try:
                        response_data = json.loads(chunk.decode("utf-8")[6:])
                        if "choices" in response_data and len(response_data["choices"]) > 0:
                            choice = response_data["choices"][0]
                            delta = choice.get("delta", {})
                            if "content" in delta and delta["content"]:
                                content_parts.append(delta["content"])
                            if "reasoning_content" in delta and delta["reasoning_content"]:
                                reasoning_parts.append(delta["reasoning_content"])
                            if "tool_calls" in delta:
                                for tc_delta in delta["tool_calls"]:
                                    idx = tc_delta.get("index", 0)
                                    while len(tool_calls_list) <= idx:
                                        tool_calls_list.append({
                                            "id": "", "type": "function",
                                            "function": {"name": "", "arguments": ""},
                                        })
                                    if "id" in tc_delta:
                                        tool_calls_list[idx]["id"] = tc_delta["id"]
                                    if "function" in tc_delta:
                                        if "name" in tc_delta["function"]:
                                            tool_calls_list[idx]["function"]["name"] = tc_delta["function"]["name"]
                                        if "arguments" in tc_delta["function"]:
                                            tool_calls_list[idx]["function"]["arguments"] += tc_delta["function"]["arguments"]
                            fr = choice.get("finish_reason")
                            if fr:
                                finish_reason = fr
                        # Capture usage from the streaming usage chunk
                        if "usage" in response_data and response_data["usage"]:
                            full_response["usage"] = response_data["usage"]
                    except json.JSONDecodeError:
                        continue

            response_message = {
                "role": "assistant",
                "content": "".join(content_parts) if content_parts else None,
                "reasoning_content": "".join(reasoning_parts),
            }
            if tool_calls_list:
                response_message["tool_calls"] = tool_calls_list

            full_response["choices"] = [{
                "index": 0,
                "message": response_message,
                "finish_reason": finish_reason,
            }]

            # Fallback: estimate tokens if streaming didn't provide usage
            if not full_response.get("usage"):
                try:
                    enc = tiktoken.encoding_for_model("gpt-4o")
                except Exception:
                    enc = tiktoken.get_encoding("cl100k_base")
                prompt_text = "\n".join(
                    msg.get("content", "") for msg in messages
                    if isinstance(msg.get("content", ""), str)
                )
                pt = len(enc.encode(prompt_text))
                ct = len(enc.encode("".join(content_parts))) if content_parts else 0
                full_response["usage"] = {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}

            return full_response
        except Exception as e:
            logger.error(f"處理非流式請求時發生錯誤: {e}")
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
        "id": chat_id, "object": "chat.completion.chunk", "created": created_time,
        "model": model, "error": error_info,
    }
