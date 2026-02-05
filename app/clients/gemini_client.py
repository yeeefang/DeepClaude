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
        Gemini: {"contents": [{"parts": [{"text": "text"}]}]} (無 role 字段)
        """
        gemini_contents = []
        system_message = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # 保存 system message 稍後合併
            if role == "system":
                system_message = content
                continue

            # Gemini API 不接受 role 字段在 contents 中
            # 直接構建 parts 格式
            gemini_contents.append({
                "parts": [{"text": content}]
            })

        # 如果有 system message，插入到開頭
        if system_message and gemini_contents:
            gemini_contents[0]["parts"][0]["text"] = f"{system_message}\n\n{gemini_contents[0]['parts'][0]['text']}"

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

        # 處理代理地址格式
        proxy_url = None
        if self.proxy:
            if not self.proxy.startswith(('http://', 'https://', 'socks://', 'socks5://')):
                proxy_url = f"http://{self.proxy}"
            else:
                proxy_url = self.proxy
            logger.info(f"使用代理: {proxy_url}")

        try:
            connector = aiohttp.TCPConnector(limit=100, force_close=True)
            async with aiohttp.ClientSession(connector=connector, timeout=self.timeout) as session:
                async with session.post(
                    endpoint,
                    headers=self._get_headers(),
                    json=request_body,
                    proxy=proxy_url,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Gemini API 錯誤 {response.status}: {error_text}")
                        raise ClientError(f"Gemini API 錯誤: 狀態碼 {response.status}, {error_text}")

                    # 處理 Gemini 串流回應 - 累積完整響應
                    buffer = b""
                    async for chunk in response.content.iter_chunked(1024):
                        buffer += chunk

                    # 解析完整的 JSON 響應
                    try:
                        response_text = buffer.decode('utf-8')

                        # Gemini 可能返回 JSON 數組 [{}] 或單個對象 {}
                        if response_text.startswith('['):
                            # JSON 數組格式
                            data_list = json.loads(response_text)
                            if data_list:
                                data = data_list[0]
                            else:
                                data = {}
                        else:
                            # 單個 JSON 對象
                            data = json.loads(response_text)

                        # 提取文本內容
                        if "candidates" in data:
                            for candidate in data["candidates"]:
                                if "content" in candidate and "parts" in candidate["content"]:
                                    for part in candidate["content"]["parts"]:
                                        if "text" in part:
                                            yield ("assistant", part["text"])

                        # 檢查完成狀態
                        finish_reason = "stop"
                        if "candidates" in data and data["candidates"]:
                            candidate_finish = data["candidates"][0].get("finishReason")
                            if candidate_finish:
                                if candidate_finish == "STOP":
                                    finish_reason = "stop"
                                elif candidate_finish in ["MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"]:
                                    finish_reason = "length"
                                elif candidate_finish == "MALFORMED_FUNCTION_CALL":
                                    logger.warning(f"Gemini 函數調用格式錯誤: {data['candidates'][0].get('finishMessage', '')}")
                                    finish_reason = "stop"

                        yield ("finish", finish_reason)

                    except json.JSONDecodeError as e:
                        logger.error(f"無法解析 Gemini 完整回應: {e}")
                        logger.debug(f"回應內容: {buffer[:500]}")
                        yield ("finish", "stop")

        except ClientError as e:
            logger.error(f"Gemini 串流請求失敗: {e}")
            raise
        except Exception as e:
            logger.error(f"Gemini 串流處理異常: {e}")
            raise
