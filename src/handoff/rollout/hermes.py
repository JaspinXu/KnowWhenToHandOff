"""Parse Hermes-style tool calls from generated text (Qwen chat format).

The evaluation driver receives tool calls parsed by vLLM's ``hermes`` parser
(``Hermes2ProToolParser.extract_tool_calls`` in vLLM 0.12) and turned into tau2
``ToolCall`` objects; the training driver generates raw tokens and must parse
them itself. This module reproduces that whole chain so both drivers treat the
same generation the same way:

- no ``<tool_call>`` tag: the whole text is a message to the user;
- every ``<tool_call>{json}</tool_call>`` block (a missing closing tag is
  accepted) holds a JSON object with a string ``name`` and an ``arguments``
  key: the text before the first tag is the content and each block becomes a
  tool call;
- a block is not valid JSON, or lacks ``name`` or ``arguments``: vLLM gives up
  and the whole text goes to the user (``error="invalid_tool_json"``); the
  markup left in the message is counted by the format check
  (``unparsed_tool_call``) in both drivers;
- every block parses but some ``arguments`` value is not a JSON object: vLLM
  forwards it, tau2 cannot build a tool call from it and the turn is lost
  (``error="invalid_arguments"``). Both drivers then end the episode with
  ``agent_error`` (see :mod:`handoff.rollout.tau2_convert`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_CALL_START = "<tool_call>"
_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)


@dataclass(frozen=True)
class ParsedToolCall:
    """One tool call decoded from a generation."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ParsedGeneration:
    """Content and tool calls decoded from one assistant generation."""

    content: str | None
    tool_calls: tuple[ParsedToolCall, ...] = field(default_factory=tuple)
    error: str | None = None


def _decode_block(raw: str) -> dict[str, Any] | None:
    """The block's JSON object if vLLM would accept it, else ``None``."""
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        return None
    if "arguments" not in payload:
        return None
    return payload


def parse_hermes(text: str) -> ParsedGeneration:
    """Split generated text into user-facing content and tool calls."""
    if TOOL_CALL_START not in text:
        return ParsedGeneration(content=text)
    payloads = [
        _decode_block(closed or unclosed)
        for closed, unclosed in _TOOL_CALL.findall(text)
    ]
    if not payloads or any(payload is None for payload in payloads):
        return ParsedGeneration(content=text, error="invalid_tool_json")
    content = text[: text.find(TOOL_CALL_START)] or None
    if any(not isinstance(payload["arguments"], dict) for payload in payloads):
        return ParsedGeneration(content=content, error="invalid_arguments")
    calls = tuple(ParsedToolCall(p["name"], dict(p["arguments"])) for p in payloads)
    return ParsedGeneration(content=content, tool_calls=calls)
