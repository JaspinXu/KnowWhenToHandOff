"""Validate chat-completion bytes without executing tools or computing rewards."""

import json
import math
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any


class CompletionError(ValueError):
    """A bounded diagnostic code; never embeds untrusted model response content."""

    def __init__(self, code: str) -> None:
        """Create a diagnostic message that does not embed response text."""
        self.code = code
        super().__init__(code)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("nonfinite JSON number")
    return parsed


def _reject_constant(value: str) -> None:
    raise ValueError("nonstandard JSON constant")


def _validate_unicode(value: object) -> None:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            # UTF-8 wire validation alone misses JSON-escaped lone surrogates.
            item.encode("utf-8")
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)


def _json_object(raw: str, code: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        _validate_unicode(value)
        return value
    except (ValueError, RecursionError):
        raise CompletionError(code) from None


@dataclass(frozen=True)
class ToolCall:
    """A parsed function call with the original JSON argument string intact."""

    id: str = field(repr=False)
    name: str
    arguments: str = field(repr=False)

    def parsed_arguments(self) -> dict[str, Any]:
        """Decode independent arguments; tool schema validation is separate."""
        return _json_object(self.arguments, "INVALID_ARGUMENTS")


@dataclass(frozen=True)
class Completion:
    """One validated assistant turn; absent usage remains explicitly unknown."""

    model: str = field(repr=False)
    content: str | None = field(repr=False)
    tool_calls: tuple[ToolCall, ...]
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    def to_message(self) -> dict[str, Any]:
        """Return a new chat message, preserving call IDs and argument strings."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        return message


def _tool_calls(
    value: object, allowed_tools: Collection[str], limit: int
) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise CompletionError("INVALID_TOOL_CALL")
    if len(value) > limit:
        raise CompletionError("TOO_MANY_TOOL_CALLS")
    calls: list[ToolCall] = []
    ids: set[str] = set()
    for raw in value:
        if not isinstance(raw, dict) or raw.get("type") != "function":
            raise CompletionError("INVALID_TOOL_CALL")
        call_id, function = raw.get("id"), raw.get("function")
        if (
            not isinstance(call_id, str)
            or not call_id.strip()
            or call_id in ids
            or not isinstance(function, dict)
        ):
            raise CompletionError("INVALID_TOOL_CALL")
        name, arguments = function.get("name"), function.get("arguments")
        if not isinstance(name, str) or name not in allowed_tools:
            raise CompletionError("UNKNOWN_TOOL")
        if not isinstance(arguments, str):
            raise CompletionError("INVALID_ARGUMENTS")
        _json_object(arguments, "INVALID_ARGUMENTS")
        calls.append(ToolCall(call_id, name, arguments))
        ids.add(call_id)
    return tuple(calls)


def _usage(value: object) -> tuple[int | None, int | None, int | None]:
    if value is None:
        return None, None, None
    if not isinstance(value, dict):
        raise CompletionError("INVALID_USAGE")
    prompt, completion, total = (
        value.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    )
    if (
        any(
            type(count) is not int or count < 0 for count in (prompt, completion, total)
        )
        or prompt + completion != total
    ):
        raise CompletionError("INVALID_USAGE")
    return prompt, completion, total


def parse_completion(
    raw: bytes, *, allowed_tools: Collection[str], max_tool_calls: int
) -> Completion:
    """Parse one nonstreaming completion; reject unusable or ambiguous tool calls.

    Transport code must bound response bytes before calling this pure parser.
    Known tool names do not prove valid arguments or authorized actions: the
    environment still owns argument-schema checks and deterministic policy rules.
    Returned text is not a substitute for sampled token IDs in an RL rollout.
    """
    if (
        isinstance(allowed_tools, str | bytes)
        or not isinstance(allowed_tools, Collection)
        or any(not isinstance(name, str) or not name.strip() for name in allowed_tools)
        or type(max_tool_calls) is not int
        or max_tool_calls < 1
    ):
        raise ValueError("invalid tool parser configuration")
    if not isinstance(raw, bytes):
        raise ValueError("completion input must be UTF-8 bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeError:
        raise CompletionError("INVALID_JSON") from None
    payload = _json_object(text, "INVALID_JSON")
    model, choices = payload.get("model"), payload.get("choices")
    if (
        not isinstance(model, str)
        or not model.strip()
        or not isinstance(choices, list)
        or len(choices) != 1
        or not isinstance(choices[0], dict)
    ):
        raise CompletionError("INVALID_COMPLETION")
    choice = choices[0]
    if type(choice.get("index")) is not int or choice["index"] != 0:
        raise CompletionError("INVALID_COMPLETION")
    reason = choice.get("finish_reason")
    if reason == "length":
        raise CompletionError("TRUNCATED_COMPLETION")
    if reason == "content_filter":
        raise CompletionError("FILTERED_COMPLETION")
    if reason not in ("stop", "tool_calls"):
        raise CompletionError("INVALID_COMPLETION")
    message = choice.get("message")
    if (
        not isinstance(message, dict)
        or message.get("role") != "assistant"
        or message.get("function_call") is not None
    ):
        raise CompletionError("INVALID_MESSAGE")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise CompletionError("INVALID_MESSAGE")
    calls = _tool_calls(message.get("tool_calls"), allowed_tools, max_tool_calls)
    if reason == "tool_calls" and not calls:
        raise CompletionError("INVALID_TOOL_CALL")
    if not calls and (content is None or not content.strip()):
        raise CompletionError("EMPTY_ASSISTANT")
    usage = _usage(payload.get("usage"))
    return Completion(model, content, calls, reason, *usage)
