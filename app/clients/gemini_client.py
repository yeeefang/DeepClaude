"""Gemini 原生 API 客戶端 - 支援 Google Gemini 原生格式"""

import json
from typing import AsyncGenerator, Optional, Dict, Any, List

import aiohttp
from aiohttp.client_exceptions import ClientError

from app.clients.base_client import BaseClient
from app.utils.logger import logger


class GeminiClient(BaseClient):
    """Gemini 原生 API 客戶端

    支援 Google Gemini API 原生格式：
    - Endpoint: /v1beta/models/{model}:generateContent
    - Auth: x-goog-api-key header
    - Format: contents array with parts
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        timeout: Optional[aiohttp.ClientTimeout] = None,
        proxy: str = None,
    ):
        """初始化 Gemini 客戶端

        Args:
            api_key: Google API Key
            api_url: API 基礎 URL (e.g., https://generativelanguage.googleapis.com)
            timeout: 請求超時設置
            proxy: 代理服務器地址
        """
        super().__init__(api_key, api_url, timeout, proxy=proxy)

    def _get_headers(self) -> Dict[str, str]:
        """獲取請求頭 - Gemini 使用 x-goog-api-key"""
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }

    def _convert_openai_to_gemini(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """轉換 OpenAI 格式訊息到 Gemini 格式

        OpenAI: [{"role": "user", "content": "text"}]
        Gemini: [{"parts": [{"text": "text"}]}]
        """
        gemini_contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Gemini 只支援 user/model role
            if role == "system":
                # System messages 合併到第一個 user message
                continue

            gemini_role = "model" if role == "assistant" else "user"
            gemini_contents.append({
                "role": gemini_role,
                "parts": [{"text": content}]
            })

        return gemini_contents

    def _convert_gemini_to_openai(self, gemini_text: str, role: str = "assistant") -> tuple[str, str]:
        """轉換 Gemini 回應到 OpenAI 格式

        Returns:
            tuple: (content_type, content)
                - content_type: "assistant" for text
                - content: actual text content
        """
        return ("assistant", gemini_text)

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model_arg: tuple[float, float, float, float],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> AsyncGenerator[tuple[str, str], None]:
        """串流對話

        Args:
            messages: OpenAI 格式訊息列表
            model_arg: (temperature, top_p, presence_penalty, frequency_penalty)
            model: 模型 ID (e.g., "gemini-2.5-flash-preview-04-17")
            tools: 工具定義 (Gemini 不支援)
            tool_choice: 工具選擇 (Gemini 不支援)

        Yields:
            tuple[str, str]: (content_type, content)
        """
        temperature, top_p, _, _ = model_arg

        # 轉換訊息格式
        gemini_contents = self._convert_openai_to_gemini(messages)

        # 構建 Gemini 請求體
        request_body = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
            }
        }

        # Gemini endpoint: /v1beta/models/{model}:streamGenerateContent
        endpoint = f"{self.api_url}/v1beta/models/{model}:streamGenerateContent"

        logger.info(f"Gemini API 串流請求: {endpoint}")

        try:
            async with self.session.post(
                endpoint,
                headers=self._get_headers(),
                json=request_body,
                proxy=self.proxy,
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Gemini API 錯誤 {response.status}: {error_text}")
                    raise ClientError(f"Gemini API 錯誤: 狀態碼 {response.status}, {error_text}")

                # 處理 Gemini 串流回應 (Server-Sent Events)
                buffer = b""
                async for chunk in response.content.iter_chunked(1024):
                    buffer += chunk

                    # 處理完整的 JSON 行
                    while b'\n' in buffer:
                        line, buffer = buffer.split(b'\n', 1)
                        line = line.strip()

                        if not line:
                            continue

                        try:
                            # Gemini 返回 JSON 格式
                            data = json.loads(line)

                            # 提取文本內容
                            if "candidates" in data:
                                for candidate in data["candidates"]:
                                    if "content" in candidate:
                                        for part in candidate["content"].get("parts", []):
                                            if "text" in part:
                                                yield ("assistant", part["text"])

                            # 檢查完成
                            if "candidates" in data:
                                for candidate in data["candidates"]:
                                    finish_reason = candidate.get("finishReason")
                                    if finish_reason:
                                        # 轉換 finish_reason
                                        openai_reason = "stop" if finish_reason == "STOP" else "length"
                                        yield ("finish", openai_reason)
                                        return

                        except json.JSONDecodeError:
                            logger.warning(f"無法解析 Gemini 回應: {line[:100]}")
                            continue

                # 處理剩餘 buffer
                if buffer:
                    try:
                        data = json.loads(buffer)
                        if "candidates" in data:
                            for candidate in data["candidates"]:
                                if "content" in candidate:
                                    for part in candidate["content"].get("parts", []):
                                        if "text" in part:
                                            yield ("assistant", part["text"])
                    except json.JSONDecodeError:
                        pass

                yield ("finish", "stop")

        except ClientError as e:
            logger.error(f"Gemini 串流請求失敗: {e}")
            raise
        except Exception as e:
            logger.error(f"Gemini 串流處理異常: {e}")
            raise
