"""XML tool call markup filter for Roo Code / Cline compatibility.

Strips XML tool call tags (e.g. <read_file>...</read_file>) from message
content to prevent errors with clients that require native tool calling.
"""

import re
from typing import List, Dict

from app.utils.logger import logger

# Known XML tool tags used by Roo Code / Cline
XML_TOOL_TAGS = [
    'read_file', 'write_to_file', 'execute_command', 'search_files',
    'list_files', 'list_code_definition_names', 'browser_action',
    'use_mcp_tool', 'access_mcp_resource', 'ask_followup_question',
    'attempt_completion', 'switch_mode', 'new_task', 'update_todo_list',
    'insert_content', 'apply_diff', 'search_and_replace', 'replace_in_file',
]

# Pre-compiled regex patterns
_TAGS_ALTERNATION = '|'.join(re.escape(t) for t in XML_TOOL_TAGS)
_XML_TOOL_BLOCK_RE = re.compile(
    rf'<({_TAGS_ALTERNATION})\b[^>]*>[\s\S]*?</\1>',
    re.DOTALL,
)
_XML_TOOL_SELF_CLOSE_RE = re.compile(
    rf'<({_TAGS_ALTERNATION})\b[^>]*/>',
    re.DOTALL,
)
_MULTI_NEWLINE_RE = re.compile(r'\n{3,}')

# Set of tag names for fast lookup
_TOOL_TAG_SET = set(XML_TOOL_TAGS)


def strip_xml_tool_tags(text: str) -> str:
    """Strip XML tool call markup from text.

    Removes <tool_name>...</tool_name> blocks for all known tool tags.
    Returns original text if stripping would produce empty output.
    """
    if not text or '<' not in text:
        return text

    cleaned = _XML_TOOL_BLOCK_RE.sub('', text)
    cleaned = _XML_TOOL_SELF_CLOSE_RE.sub('', cleaned)
    cleaned = _MULTI_NEWLINE_RE.sub('\n\n', cleaned)
    cleaned = cleaned.strip()

    return cleaned if cleaned else text


def clean_message_content(content) -> str | list:
    """Clean XML tool markup from a single message's content field.

    Handles both string content and structured content (list of blocks).
    """
    if isinstance(content, str):
        return strip_xml_tool_tags(content)
    elif isinstance(content, list):
        cleaned_blocks = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                new_block = block.copy()
                new_block["text"] = strip_xml_tool_tags(block.get("text", ""))
                cleaned_blocks.append(new_block)
            else:
                cleaned_blocks.append(block)
        return cleaned_blocks
    return content


def clean_messages(messages: List[Dict]) -> List[Dict]:
    """Clean XML tool markup from all messages in a conversation.

    Returns a new list with cleaned message content.
    """
    cleaned = []
    for msg in messages:
        new_msg = msg.copy()
        if "content" in new_msg:
            new_msg["content"] = clean_message_content(new_msg["content"])
        cleaned.append(new_msg)
    return cleaned


class StreamXMLFilter:
    """Stateful filter for streaming content that strips XML tool tags.

    Handles XML tags that may span across multiple streaming chunks
    by buffering content when a potential tool tag opening is detected.
    """

    def __init__(self):
        self._buffer = ""
        self._in_tool_tag = False
        self._current_tag = ""
        self._max_tag_len = max(len(t) for t in XML_TOOL_TAGS)

    def process(self, chunk: str) -> str:
        """Process a streaming chunk. Returns filtered text to output."""
        if not chunk:
            return chunk

        self._buffer += chunk
        output = []

        while self._buffer:
            if self._in_tool_tag:
                # Looking for closing tag </current_tag>
                close_tag = f"</{self._current_tag}>"
                close_idx = self._buffer.find(close_tag)
                if close_idx >= 0:
                    # Found closing tag — skip everything including it
                    self._buffer = self._buffer[close_idx + len(close_tag):]
                    self._in_tool_tag = False
                    self._current_tag = ""
                else:
                    # Still inside tool tag, consume entire buffer
                    self._buffer = ""
                    break
            else:
                lt_idx = self._buffer.find('<')
                if lt_idx < 0:
                    # No '<' at all — output everything
                    output.append(self._buffer)
                    self._buffer = ""
                else:
                    # Output text before '<'
                    if lt_idx > 0:
                        output.append(self._buffer[:lt_idx])
                        self._buffer = self._buffer[lt_idx:]

                    # Check if what follows '<' is a known tool tag
                    matched_tag = self._match_tool_tag()
                    if matched_tag is True:
                        # Matched and entered tool tag mode — loop continues
                        continue
                    elif matched_tag is False:
                        # Definitely not a tool tag — output the '<'
                        output.append(self._buffer[0])
                        self._buffer = self._buffer[1:]
                    else:
                        # None = not enough data to decide, wait for more
                        break

        return "".join(output)

    def flush(self) -> str:
        """Flush any remaining buffered content."""
        if self._in_tool_tag:
            self._buffer = ""
            self._in_tool_tag = False
            return ""
        remaining = self._buffer
        self._buffer = ""
        return remaining

    def _match_tool_tag(self) -> bool | None:
        """Check if buffer starts with an XML tool tag.

        Returns:
            True  — matched a tool tag, entered suppression mode
            False — definitely not a tool tag
            None  — buffer too short to decide, need more data
        """
        # Buffer starts with '<', check what follows
        rest = self._buffer[1:]

        if not rest:
            return None  # Need more data

        # Quick reject: if first char after '<' is not a letter, not a tag
        if not rest[0].isalpha():
            return False

        # Try to match against known tags
        for tag in XML_TOOL_TAGS:
            if rest.startswith(tag):
                # Check the char after tag name
                after_tag_idx = len(tag)
                if after_tag_idx >= len(rest):
                    return None  # Need more data to confirm
                next_char = rest[after_tag_idx]
                if next_char in ('>', ' ', '\n', '\r', '\t', '/'):
                    # Confirmed tool tag opening
                    self._in_tool_tag = True
                    self._current_tag = tag
                    # Skip past the opening tag
                    gt_idx = self._buffer.find('>')
                    if gt_idx >= 0:
                        self._buffer = self._buffer[gt_idx + 1:]
                    else:
                        self._buffer = ""
                    logger.debug(f"XML filter: stripping <{tag}> block")
                    return True

        # Check if buffer might contain a partial tag name
        # e.g., buffer is "<read_" and we're waiting for "file"
        for tag in XML_TOOL_TAGS:
            if tag.startswith(rest.split('>')[0].split(' ')[0].split('\n')[0]):
                if len(rest) < len(tag) + 1:
                    return None  # Might be a partial match, need more data

        return False
