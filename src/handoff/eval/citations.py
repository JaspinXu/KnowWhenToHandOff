"""Rule-based detection of policy citations in refusals and hand-offs.

Transparency is an evaluation metric only (it is never part of a training
reward). The rule match is deliberately simple and conservative; the manual
review sample (``scripts/sample_review.py``) spot-checks it. See
``docs/metrics.md`` for the definition.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from handoff.rollout.schema import assistant_messages, find_transfer

REFUSAL_PATTERN = re.compile(
    r"\b(unfortunately|unable to|cannot|can't|can not|not able to|not allowed|"
    r"not permitted|not eligible|isn't possible|is not possible|i'm afraid|"
    r"i am afraid|won't be able)\b",
    re.IGNORECASE,
)
CITATION_PATTERN = re.compile(
    r"\b(polic(y|ies)|per our|according to (our|the)|"
    r"(our|the) (rules?|terms|guidelines|regulations|conditions))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CitationEvent:
    """One refusal or hand-off and whether it names the policy behind it."""

    kind: str
    turn: int
    cited: bool
    text: str


def _text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else ""


def citation_events(messages: Sequence[Mapping[str, Any]]) -> list[CitationEvent]:
    """Find refusal and hand-off events in a normalized trajectory.

    A hand-off event covers the agent's last message to the user before the
    transfer call and its first message after it. Every other assistant message
    that contains refusal language is a refusal event.
    """
    policy_turns = assistant_messages(messages)
    transfer_turn = find_transfer(messages)
    handoff_window: set[int] = set()
    events: list[CitationEvent] = []
    if transfer_turn is not None:
        before = [
            m
            for m in policy_turns
            if int(m["turn"]) < transfer_turn and not m.get("tool_calls") and _text(m)
        ]
        after = [
            m
            for m in policy_turns
            if int(m["turn"]) > transfer_turn and not m.get("tool_calls") and _text(m)
        ]
        window = ([before[-1]] if before else []) + ([after[0]] if after else [])
        handoff_window = {int(m["turn"]) for m in window}
        joined = " ".join(_text(m) for m in window)
        events.append(
            CitationEvent(
                "handoff",
                transfer_turn,
                CITATION_PATTERN.search(joined) is not None,
                joined,
            )
        )
    for message in policy_turns:
        turn = int(message["turn"])
        text = _text(message)
        if message.get("tool_calls") or turn in handoff_window or not text:
            continue
        if REFUSAL_PATTERN.search(text):
            events.append(
                CitationEvent(
                    "refusal",
                    turn,
                    CITATION_PATTERN.search(text) is not None,
                    text,
                )
            )
    events.sort(key=lambda e: e.turn)
    return events
