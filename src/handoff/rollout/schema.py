"""Shared episode record used by evaluation, rewards, metrics and training.

The same :class:`Trajectory` is produced by the evaluation driver (tau2 runner)
and by the training driver (verl agent loop), so every reward term and metric is
computed from one representation. Messages are normalized plain dictionaries so
that rules, rewards and metrics stay pure and testable without tau2 installed.

Normalized message keys:

- ``role``: ``"assistant"``, ``"user"`` or ``"tool"``.
- ``content``: text or ``None``.
- ``tool_calls``: list of ``{"id", "name", "arguments"}`` dictionaries, or ``None``.
- ``requestor``: for tool results, ``"assistant"`` or ``"user"``.
- ``tool_call_id`` and ``error``: for tool results.
- ``turn``: position in the trajectory, starting at 0.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
TRANSFER_TOOL = "transfer_to_human_agents"
EPISODE_KINDS = ("upstream", "explicit_request", "out_of_scope", "hard_negative")
DEFAULT_GREETING = "Hi! How can I help you today?"


@dataclass(frozen=True)
class Violation:
    """One deterministic policy-rule breach found in a trajectory."""

    rule: str
    turn: int
    detail: str = ""


@dataclass(frozen=True)
class FormatError:
    """One malformed action (unparsed tool call, unknown tool, bad arguments)."""

    kind: str
    turn: int
    detail: str = ""


@dataclass
class Trajectory:
    """One simulated conversation plus the labels and checks needed downstream.

    ``task_reward`` always comes from the tau2 evaluator. ``violations`` and
    ``format_errors`` come from :mod:`handoff.rewards.rules` and
    :mod:`handoff.rewards.format`. Reward composition happens elsewhere, so the
    record stays valid for any lambda setting.
    """

    run_id: str
    task_id: str
    source_task_id: str
    domain: str
    split: str
    kind: str
    persona: str
    trial: int
    seed: int | None
    policy_ckpt: str
    user_sim: str
    messages: list[dict[str, Any]]
    termination_reason: str
    task_reward: float
    reward_basis: list[str] | None
    db_match: bool | None
    transferred: bool
    transfer_turn: int | None
    should_transfer: bool
    violations: list[Violation] = field(default_factory=list)
    format_errors: list[FormatError] = field(default_factory=list)
    n_turns: int = 0
    n_tool_calls: int = 0
    n_tokens: int | None = None
    duration_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    @property
    def success(self) -> bool:
        """Return whether the tau2 evaluator gave full task reward."""
        return self.task_reward >= 1.0

    @property
    def episode_id(self) -> str:
        """Return a stable identifier of the (task, persona, trial) episode."""
        return f"{self.domain}/{self.task_id}/{self.persona}/{self.trial}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible primitives."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Trajectory:
        """Rebuild a trajectory from :meth:`to_dict` output, rejecting unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown trajectory fields: {sorted(unknown)}")
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported trajectory schema_version {version}")
        values = dict(data)
        values["violations"] = [Violation(**v) for v in data.get("violations", [])]
        values["format_errors"] = [
            FormatError(**e) for e in data.get("format_errors", [])
        ]
        return cls(**values)


def write_trajectories(path: Path, trajectories: Iterable[Trajectory]) -> int:
    """Write trajectories as JSON lines; return the number written."""
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for trajectory in trajectories:
            stream.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def iter_trajectories(path: Path) -> Iterator[Trajectory]:
    """Stream trajectories from a JSON-lines file."""
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield Trajectory.from_dict(json.loads(line))
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: {error}") from None


def read_trajectories(paths: Sequence[Path]) -> list[Trajectory]:
    """Read and concatenate trajectories from several JSON-lines files."""
    result: list[Trajectory] = []
    for path in paths:
        result.extend(iter_trajectories(path))
    return result


def assistant_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return assistant messages generated by the policy (the greeting excluded).

    tau2 starts every conversation with a fixed agent greeting that the policy
    did not produce. It is the first message and has no preceding user turn.
    """
    result: list[dict[str, Any]] = []
    seen_user = False
    for message in messages:
        role = message.get("role")
        if role == "user":
            seen_user = True
        elif role == "assistant":
            if not seen_user and message.get("content") == DEFAULT_GREETING:
                continue
            result.append(dict(message))
    return result


def tool_calls_with_results(
    messages: Sequence[Mapping[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any] | None, int]]:
    """Pair every assistant tool call with its tool result and turn index.

    Returns ``(call, result_or_None, turn_of_call)`` tuples in order. Results are
    matched by ``tool_call_id``; unmatched calls get ``None``.
    """
    results_by_id: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.get("role") == "tool" and message.get("requestor") != "user":
            call_id = message.get("tool_call_id")
            if isinstance(call_id, str):
                results_by_id.setdefault(call_id, dict(message))
    pairs: list[tuple[dict[str, Any], dict[str, Any] | None, int]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        turn = int(message.get("turn", index))
        for call in message.get("tool_calls") or []:
            result = results_by_id.get(call.get("id", ""))
            pairs.append((dict(call), result, turn))
    return pairs


def find_transfer(messages: Sequence[Mapping[str, Any]]) -> int | None:
    """Return the turn of the first transfer_to_human_agents call.

    The call is the agent's hand-off decision whether or not the tool returned an
    error, as in tau2's own action check (which ignores tool results), so the
    task reward and the hand-off label never disagree about whether it happened.
    """
    for call, _result, turn in tool_calls_with_results(messages):
        if call.get("name") == TRANSFER_TOOL:
            return turn
    return None
