"""Format converter between OpenAI and Anthropic API formats.

Handles conversion of tool definitions, tool_choice, messages with tool_calls,
and tool result messages between the two API formats.
"""

import json
from typing import List, Dict, Any, Optional

from app.utils.logger import logger


def openai_tools_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI tools format to Anthropic tools format.

    OpenAI: [{"type": "function", "function": {"name": "x", "description": "y", "parameters": {...}}}]
    Anthropic: [{"name": "x", "description": "y", "input_schema": {...}}]
    """
    anthropic_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool["function"]
            anthropic_tools.append({
                "name": func["name"],
                "description": func.get("description", ""),
                "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
            })
    return anthropic_tools


def openai_tool_choice_to_anthropic(tool_choice: Any) -> Optional[Dict[str, Any]]:
    """Convert OpenAI tool_choice to Anthropic tool_choice.

    OpenAI: "auto" | "required" | "none" | {"type": "function", "function": {"name": "x"}}
    Anthropic: {"type": "auto"} | {"type": "any"} | {"type": "tool", "name": "x"}
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "required":
            return {"type": "any"}
        elif tool_choice == "none":
            return None
    elif isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            return {"type": "tool", "name": tool_choice["function"]["name"]}
    return {"type": "auto"}


def convert_messages_for_anthropic(messages: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    """Convert OpenAI-format messages for Anthropic API.

    Handles:
    - System messages -> extracted as system prompt
    - Assistant messages with tool_calls -> content blocks with tool_use
    - Tool role messages -> user messages with tool_result content
    - Structured content (list format) -> proper Anthropic text blocks
    - Consecutive same-role messages -> merged

    Returns:
        tuple[str, List[Dict]]: (system_prompt, converted_messages)
    """
    system_parts = []
    converted = []

    for msg in messages:
        role = msg.get("role", "")

        if role == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                system_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_parts.append(block.get("text", ""))
            continue

        if role == "assistant":
            content_blocks = _convert_assistant_content(msg)
            _append_message(converted, "assistant", content_blocks)

        elif role == "tool":
            tool_result = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": msg.get("content", ""),
            }
            _append_message(converted, "user", [tool_result])

        elif role == "user":
            content_blocks = _convert_user_content(msg)
            _append_message(converted, "user", content_blocks)

    system_prompt = "\n".join(system_parts).strip()
    return system_prompt, converted


def _convert_assistant_content(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert an assistant message's content to Anthropic content blocks."""
    blocks = []
    raw_content = msg.get("content")

    if isinstance(raw_content, str) and raw_content:
        blocks.append({"type": "text", "text": raw_content})
    elif isinstance(raw_content, list):
        for block in raw_content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    text = block.get("text", "")
                    if text:
                        blocks.append({"type": "text", "text": text})
                # Skip reasoning, thinking, and other Roo-specific block types

    # Convert tool_calls to tool_use blocks
    for tc in msg.get("tool_calls", []):
        if tc.get("type") == "function":
            args = tc["function"].get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            blocks.append({
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": tc["function"]["name"],
                "input": args,
            })

    if not blocks:
        blocks.append({"type": "text", "text": " "})

    return blocks


def _convert_user_content(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a user message's content to Anthropic content blocks."""
    raw_content = msg.get("content")
    if isinstance(raw_content, str):
        return [{"type": "text", "text": raw_content}]
    elif isinstance(raw_content, list):
        blocks = []
        for block in raw_content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    blocks.append({"type": "text", "text": block.get("text", "")})
                elif block.get("type") == "image_url":
                    # Pass through image blocks
                    blocks.append(block)
                else:
                    # Try to preserve other block types
                    blocks.append(block)
            elif isinstance(block, str):
                blocks.append({"type": "text", "text": block})
        return blocks if blocks else [{"type": "text", "text": " "}]
    return [{"type": "text", "text": str(raw_content) if raw_content else " "}]


def _append_message(messages: List[Dict], role: str, content_blocks: List[Dict]):
    """Append content to messages, merging with previous message if same role.

    Anthropic requires strict alternation between user and assistant roles.
    """
    if messages and messages[-1].get("role") == role:
        # Merge with previous message of same role
        prev_content = messages[-1].get("content", [])
        if isinstance(prev_content, str):
            prev_content = [{"type": "text", "text": prev_content}]
        prev_content.extend(content_blocks)
        messages[-1]["content"] = prev_content
    else:
        messages.append({"role": role, "content": content_blocks})
