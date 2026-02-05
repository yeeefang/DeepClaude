"""Gemini 組合模型服務 - 支援 Gemini 原生 API 格式"""

import asyncio
import copy
import json
import time
from typing import AsyncGenerator, Dict, Any, List, Optional

import tiktoken

from app.clients import DeepSeekClient, GeminiClient
from app.utils.logger import logger
from app.utils.xml_tool_filter import StreamXMLFilter
from app.deepclaude.deepclaude import _build_bridge_content, _build_error_response


class GeminiComposite:
    """處理 DeepSeek 和 Gemini 原生 API 的流式輸出銜接"""

    def __init__(
        self,
        deepseek_api_key: str,
        gemini_api_key: str,
        deepseek_api_url: str = "https://api.deepseek.com/v1/chat/completions",
        gemini_api_url: str = "https://generativelanguage.googleapis.com",
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
        self.gemini_client = GeminiClient(gemini_api_key, gemini_api_url, proxy=target_proxy)
        self.is_origin_reasoning = is_origin_reasoning

    async def chat_completions_with_stream(
        self,
        messages: List[Dict[str, str]],
        model_arg: tuple[float, float, float, float],
        deepseek_model: str = "deepseek-reasoner",
        target_model: str = "gemini-2.5-flash-preview-04-17",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        stream_options: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[bytes, None]:
        """處理完整的流式輸出過程

        Args:
            messages: 訊息列表
            model_arg: (temperature, top_p, presence_penalty, frequency_penalty)
            deepseek_model: DeepSeek 模型名稱
            target_model: Gemini 模型名稱
            tools: 工具定義 (Gemini 不支援)
            tool_choice: 工具選擇 (Gemini 不支援)
            stream_options: 串流選項
        """
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
                # 若為單一模型模式，跳過推理流程
                if self.is_same_model:
                    logger.info("單一模型模式：跳過推理階段")
                    await reasoning_queue.put("")
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
                        await reasoning_queue.put("".join(reasoning_content))
                        break
            except Exception as e:
                logger.error(f"處理 DeepSeek 流時發生錯誤: {e}")
                error_response = _build_error_response(chat_id, created_time, deepseek_model, e)
                await output_queue.put(f"data: {json.dumps(error_response)}\n\n".encode("utf-8"))
                await output_queue.put(b"data: [DONE]\n\n")
                await reasoning_queue.put("")
                await output_queue.put(None)
                return

            if reasoning_queue.empty():
                await reasoning_queue.put("".join(reasoning_content))

            logger.info("DeepSeek 任務處理完成，標記結束")
            await output_queue.put(None)

        async def process_gemini():
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

                gemini_messages = copy.deepcopy(messages)

                if not gemini_messages:
                    raise ValueError("訊息列表為空，無法處理請求")

                last_message = gemini_messages[-1]
                if last_message.get("role", "") != "user":
                    raise ValueError("最後一則訊息的角色不是使用者，無法處理請求")

                # Skip bridge if same model or no reasoning
                if not self.is_same_model and reasoning and reasoning != "（無推理內容）":
                    original_content = last_message["content"]
                    last_message["content"] = _build_bridge_content(
                        original_content, reasoning, self.system_config
                    )

                logger.info(f"開始處理 Gemini 流，使用模型: {target_model}")

                xml_filter = StreamXMLFilter()

                async for content_type, content in self.gemini_client.stream_chat(
                    messages=gemini_messages,
                    model_arg=model_arg,
                    model=target_model,
                    tools=tools,
                    tool_choice=tool_choice,
                ):
                    if content_type == "assistant":
                        filtered = xml_filter.process(content)
                        if not filtered:
                            continue
                        if encoding:
                            completion_token_count += len(encoding.encode(filtered))
                        response = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": target_model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": filtered},
                            }],
                        }
                        await output_queue.put(
                            f"data: {json.dumps(response)}\n\n".encode("utf-8")
                        )
                    elif content_type == "finish":
                        finish_chunk = {
                            "id": chat_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": target_model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": content,
                            }],
                        }
                        await output_queue.put(
                            f"data: {json.dumps(finish_chunk)}\n\n".encode("utf-8")
                        )

                        if include_usage:
                            usage_chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": target_model,
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": 0,
                                    "completion_tokens": completion_token_count,
                                    "total_tokens": completion_token_count,
                                }
                            }
                            await output_queue.put(
                                f"data: {json.dumps(usage_chunk)}\n\n".encode("utf-8")
                            )

                        await output_queue.put(b"data: [DONE]\n\n")
                        break

            except Exception as e:
                logger.error(f"處理 Gemini 流時發生錯誤: {e}")
                error_response = _build_error_response(chat_id, created_time, target_model, e)
                await output_queue.put(f"data: {json.dumps(error_response)}\n\n".encode("utf-8"))
                await output_queue.put(b"data: [DONE]\n\n")
            finally:
                await output_queue.put(None)

        deepseek_task = asyncio.create_task(process_deepseek())
        gemini_task = asyncio.create_task(process_gemini())

        try:
            while True:
                chunk = await output_queue.get()
                if chunk is None:
                    finished_count = sum([deepseek_task.done(), gemini_task.done()])
                    if finished_count >= 2:
                        break
                    continue
                yield chunk
        finally:
            for task in [deepseek_task, gemini_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    async def chat_completions_without_stream(
        self,
        messages: List[Dict[str, str]],
        model_arg: tuple[float, float, float, float],
        deepseek_model: str = "deepseek-reasoner",
        target_model: str = "gemini-2.5-flash-preview-04-17",
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """非串流對話 - 透過串流實作後累積結果"""
        full_content = ""
        full_reasoning = ""
        finish_reason = "stop"

        async for chunk in self.chat_completions_with_stream(
            messages, model_arg, deepseek_model, target_model, tools, tool_choice,
            stream_options={"include_usage": True}
        ):
            chunk_str = chunk.decode("utf-8")
            if chunk_str.startswith("data: "):
                data_str = chunk_str[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    if "choices" in data and data["choices"]:
                        delta = data["choices"][0].get("delta", {})
                        if "reasoning_content" in delta:
                            full_reasoning += delta["reasoning_content"]
                        if "content" in delta:
                            full_content += delta["content"]
                        if "finish_reason" in data["choices"][0] and data["choices"][0]["finish_reason"]:
                            finish_reason = data["choices"][0]["finish_reason"]
                except json.JSONDecodeError:
                    pass

        return {
            "id": f"chatcmpl-{hex(int(time.time() * 1000))[2:]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": target_model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_content,
                    "reasoning_content": full_reasoning,
                },
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(full_content),
                "total_tokens": len(full_content),
            }
        }
