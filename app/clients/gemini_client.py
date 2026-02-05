"""Gemini 客戶端 - 使用 Google 官方 SDK"""

import asyncio
from typing import AsyncGenerator, Optional, Dict, Any, List

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool, ContentDict

from app.clients.base_client import BaseClient
from app.utils.logger import logger


class GeminiClient(BaseClient):
    """Gemini 客戶端 - 使用 Google 官方 SDK

    參考: https://ai.google.dev/api?hl=zh-tw
    """

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://generativelanguage.googleapis.com",
        timeout: Optional[Any] = None,
        proxy: str = None,
    ):
        """初始化 Gemini 客戶端

        Args:
            api_key: Google API Key
            api_url: API 基礎 URL (未使用，SDK 自動處理)
            timeout: 請求超時設置 (未使用，SDK 自動處理)
            proxy: 代理服務器地址 (未使用，需要通過環境變量設置)
        """
        super().__init__(api_key, api_url, timeout, proxy=proxy)

        # 配置 SDK
        genai.configure(api_key=api_key)

        if proxy:
            logger.warning(
                f"Google SDK 不直接支援 proxy 參數。"
                f"請通過環境變量設置: HTTP_PROXY={proxy} HTTPS_PROXY={proxy}"
            )

    def _convert_openai_to_gemini(
        self,
        messages: List[Dict[str, str]]
    ) -> tuple[Optional[str], List[ContentDict]]:
        """轉換 OpenAI 格式訊息到 Gemini SDK 格式

        Args:
            messages: OpenAI 格式訊息列表

        Returns:
            tuple: (system_instruction, contents)
                - system_instruction: 系統指令 (可選)
                - contents: Gemini 格式的對話內容
        """
        system_instruction = None
        contents: List[ContentDict] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # 系統訊息作為 system_instruction
                system_instruction = content
            elif role == "user":
                contents.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                contents.append({"role": "model", "parts": [content]})

        return system_instruction, contents

    def _convert_openai_tools_to_gemini(
        self,
        openai_tools: List[Dict[str, Any]]
    ) -> List[Tool]:
        """轉換 OpenAI 工具格式到 Gemini SDK 格式

        Args:
            openai_tools: OpenAI 格式的工具列表
                例如: [{"type": "function", "function": {"name": "...", "parameters": {...}}}]

        Returns:
            Gemini Tool 列表

        參考: https://ai.google.dev/api/python/google/generativeai/types/Tool
        """
        if not openai_tools:
            return []

        function_declarations = []

        for tool in openai_tools:
            if tool.get("type") != "function":
                logger.warning(f"跳過非函數類型工具: {tool.get('type')}")
                continue

            func_spec = tool.get("function", {})
            name = func_spec.get("name")
            description = func_spec.get("description", "")
            parameters = func_spec.get("parameters", {})

            if not name:
                logger.warning("跳過沒有名稱的工具")
                continue

            try:
                # 創建 Gemini FunctionDeclaration
                # Gemini 使用 JSON Schema 格式的參數定義
                func_decl = FunctionDeclaration(
                    name=name,
                    description=description,
                    parameters=parameters  # 直接使用 OpenAI 的 JSON Schema
                )
                function_declarations.append(func_decl)
                logger.debug(f"轉換工具: {name}")

            except Exception as e:
                logger.error(f"轉換工具 {name} 失敗: {e}")
                continue

        if not function_declarations:
            return []

        # 將所有函數聲明包裝成一個 Tool
        return [Tool(function_declarations=function_declarations)]

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
            model: 模型 ID (e.g., "gemini-3-flash-preview")
            tools: 工具定義 (OpenAI 格式，將被轉換為 Gemini 格式)
            tool_choice: 工具選擇 (Gemini 不支援，將被忽略)

        Yields:
            tuple[str, str]: (content_type, content)
                - content_type: "assistant", "tool_call", 或 "finish"
                - content: 文本內容、工具調用 JSON 或完成原因
        """
        temperature, top_p, _, _ = model_arg

        # 轉換訊息格式
        system_instruction, contents = self._convert_openai_to_gemini(messages)

        # 轉換工具格式
        gemini_tools = None
        if tools:
            try:
                gemini_tools = self._convert_openai_tools_to_gemini(tools)
                if gemini_tools:
                    logger.info(f"已轉換 {len(tools)} 個工具到 Gemini 格式")
                else:
                    logger.warning("工具轉換結果為空")
            except Exception as e:
                logger.error(f"工具轉換失敗: {e}，將不使用工具")
                gemini_tools = None

        if tool_choice:
            logger.warning("Gemini 不支援 tool_choice 參數，將被忽略")

        try:
            # 創建生成配置
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
            )

            # 創建模型實例
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                system_instruction=system_instruction,
                tools=gemini_tools,  # 傳遞工具定義
            )

            logger.info(f"使用 Gemini SDK 串流請求: {model}")

            # 在異步環境中運行同步的 generate_content
            loop = asyncio.get_event_loop()

            # 生成內容 (streaming)
            response = await loop.run_in_executor(
                None,
                lambda: model_instance.generate_content(
                    contents,
                    stream=True,
                )
            )

            # 處理串流回應
            has_content = False
            for chunk in response:
                # 處理文本內容
                if chunk.text:
                    has_content = True
                    yield ("assistant", chunk.text)

                # 處理函數調用
                # Gemini SDK 會在 chunk.parts 中包含 function_call
                if hasattr(chunk, 'parts'):
                    for part in chunk.parts:
                        if hasattr(part, 'function_call'):
                            func_call = part.function_call
                            # 轉換為 OpenAI 格式的工具調用
                            tool_call_json = {
                                "id": f"call_{func_call.name}",
                                "type": "function",
                                "function": {
                                    "name": func_call.name,
                                    "arguments": str(func_call.args)
                                }
                            }
                            logger.info(f"Gemini 調用工具: {func_call.name}")
                            yield ("tool_call", str(tool_call_json))

            # 檢查完成原因
            finish_reason = "stop"

            # 檢查安全性過濾
            try:
                if hasattr(response, 'prompt_feedback'):
                    feedback = response.prompt_feedback
                    if feedback and hasattr(feedback, 'block_reason'):
                        if feedback.block_reason:
                            logger.warning(f"Gemini 回應被阻止: {feedback.block_reason}")
                            finish_reason = "content_filter"
                            if not has_content:
                                yield ("assistant", "⚠️ 回應因安全過濾而被阻止")
            except Exception as e:
                logger.debug(f"無法檢查完成原因: {e}")

            yield ("finish", finish_reason)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Gemini SDK 請求失敗: {error_msg}")

            # 返回錯誤訊息而不是拋出異常
            yield ("assistant", f"⚠️ Gemini API 錯誤: {error_msg[:200]}")
            yield ("finish", "stop")
