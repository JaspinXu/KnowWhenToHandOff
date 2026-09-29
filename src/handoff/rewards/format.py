"""Detect malformed agent actions in a normalized trajectory.

``N_format`` in the README reward counts these errors. They are separate from
business-logic tool errors (for example "reservation not found"), which are a
normal part of solving a task and are not penalized.

Error kinds:

- ``unparsed_tool_call``: tool-call markup or JSON left in a message to the user
  because the serving parser could not turn it into a tool call.
- ``unknown_tool``: a call to a tool that the domain does not provide.
- ``invalid_arguments``: arguments the tool signature rejected.
- ``empty_message``: an assistant turn with neither text nor tool calls.
- ``protocol_error``: tau2 stopped the episode with ``agent_error``.
- ``truncated_generation``: a generation hit the per-turn token limit; both
  drivers report it through the stop sentinel (see
  :mod:`handoff.rollout.tau2_convert`).

Tool-call markup that the Hermes parser could not decode stays in the message
text in both drivers (vLLM falls back to plain text, and so does the training
parser), so it is counted once, as ``unparsed_tool_call``.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from handoff.rollout.schema import FormatError, assistant_messages

FORMAT_ERROR_KINDS = (
    "unparsed_tool_call",
    "unknown_tool",
    "invalid_arguments",
    "empty_message",
    "protocol_error",
    "truncated_generation",
)

_TOOL_MARKUP = re.compile(r"</?tool_call>|<function=|\"arguments\"\s*:", re.IGNORECASE)
_UNKNOWN_TOOL = re.compile(r"Tool '[^']*' not found", re.IGNORECASE)
_BAD_ARGUMENTS = re.compile(
    r"unexpected keyword argument|required positional argument|"
    r"missing \d+ required|takes \d+ positional|validation error|"
    r"Field required|Input should be",
    re.IGNORECASE,
)


def find_format_errors(
    messages: Sequence[Mapping[str, Any]],
    *,
    known_tools: Collection[str] | None = None,
    termination_reason: str | None = None,
    count_protocol_error: bool = True,
) -> list[FormatError]:
    """Return format errors in turn order.

    ``known_tools`` enables unknown-tool detection at call time; without it,
    unknown tools are detected from the environment's error message instead.
    ``count_protocol_error`` is disabled by drivers that already reported the
    malformed turn that ended the episode (to avoid counting it twice).
    """
    errors: list[FormatError] = []
    flagged_unknown_ids: set[str] = set()
    for index, message in enumerate(assistant_messages(messages)):
        turn = int(message.get("turn", index))
        content = message.get("content")
        calls = message.get("tool_calls") or []
        text = content if isinstance(content, str) else ""
        if not calls and not text.strip():
            errors.append(FormatError("empty_message", turn))
        if text and _TOOL_MARKUP.search(text):
            errors.append(FormatError("unparsed_tool_call", turn, text[:80]))
        if known_tools is not None:
            for call in calls:
                if call.get("name") not in known_tools:
                    errors.append(
                        FormatError("unknown_tool", turn, str(call.get("name")))
                    )
                    flagged_unknown_ids.add(str(call.get("id", "")))
    for index, message in enumerate(messages):
        if message.get("role") != "tool" or message.get("requestor") == "user":
            continue
        if not message.get("error", False):
            continue
        turn = int(message.get("turn", index))
        content = str(message.get("content") or "")
        if _UNKNOWN_TOOL.search(content):
            if str(message.get("tool_call_id", "")) not in flagged_unknown_ids:
                errors.append(FormatError("unknown_tool", turn, content[:80]))
        elif _BAD_ARGUMENTS.search(content):
            errors.append(FormatError("invalid_arguments", turn, content[:80]))
    if termination_reason == "agent_error" and count_protocol_error:
        last_turn = int(messages[-1].get("turn", len(messages) - 1)) if messages else 0
        errors.append(FormatError("protocol_error", last_turn))
    errors.sort(key=lambda e: (e.turn, FORMAT_ERROR_KINDS.index(e.kind)))
    return errors
