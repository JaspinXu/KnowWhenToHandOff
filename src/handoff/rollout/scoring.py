"""Turn a finished episode into a :class:`Trajectory` with rules and labels applied.

Both drivers call :func:`build_trajectory`, which is the single place where
violations, format errors, the transfer decision and cost counters are derived.
This is what makes the eval/train parity test meaningful.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from handoff.rewards.format import FORMAT_ERROR_KINDS, find_format_errors
from handoff.rewards.rules import GoldAction, find_violations, rules_for_domain
from handoff.rollout.schema import (
    EPISODE_KINDS,
    FormatError,
    Trajectory,
    assistant_messages,
    find_transfer,
)


@dataclass(frozen=True)
class EpisodeContext:
    """Everything about an episode that is known before it runs."""

    run_id: str
    domain: str
    task_id: str
    source_task_id: str
    split: str
    kind: str
    persona: str
    should_transfer: bool
    trial: int
    seed: int | None
    policy_ckpt: str
    user_sim: str
    gold_actions: tuple[GoldAction, ...] = ()
    write_tools: frozenset[str] | None = None
    known_tools: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.kind not in EPISODE_KINDS:
            raise ValueError(f"unknown episode kind {self.kind!r}")
        if self.split not in ("train", "dev", "test"):
            raise ValueError(f"unknown split {self.split!r}")


def with_turns(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return copies of messages where every message carries a ``turn`` index."""
    result: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        copy = dict(message)
        if copy.get("turn") is None:
            copy["turn"] = index
        result.append(copy)
    return result


def build_trajectory(
    ctx: EpisodeContext,
    *,
    messages: Sequence[Mapping[str, Any]],
    termination_reason: str,
    task_reward: float,
    reward_basis: list[str] | None = None,
    db_match: bool | None = None,
    n_tokens: int | None = None,
    duration_s: float | None = None,
    driver_format_errors: Sequence[FormatError] = (),
    extra: Mapping[str, Any] | None = None,
    check_off_reference: bool = True,
) -> Trajectory:
    """Apply rules, format checks and labels to a finished episode."""
    normalized = with_turns(messages)
    rules = rules_for_domain(ctx.domain, ctx.write_tools)
    violations = find_violations(
        normalized, rules, ctx.gold_actions, check_off_reference=check_off_reference
    )
    format_errors = find_format_errors(
        normalized,
        known_tools=ctx.known_tools,
        termination_reason=termination_reason,
        # A driver-reported error already explains why the policy stopped.
        count_protocol_error=not driver_format_errors,
    )
    format_errors.extend(driver_format_errors)
    format_errors.sort(key=lambda e: (e.turn, FORMAT_ERROR_KINDS.index(e.kind)))
    transfer_turn = find_transfer(normalized)
    policy_turns = assistant_messages(normalized)
    return Trajectory(
        run_id=ctx.run_id,
        task_id=ctx.task_id,
        source_task_id=ctx.source_task_id,
        domain=ctx.domain,
        split=ctx.split,
        kind=ctx.kind,
        persona=ctx.persona,
        trial=ctx.trial,
        seed=ctx.seed,
        policy_ckpt=ctx.policy_ckpt,
        user_sim=ctx.user_sim,
        messages=normalized,
        termination_reason=termination_reason,
        task_reward=float(task_reward),
        reward_basis=list(reward_basis) if reward_basis is not None else None,
        db_match=db_match,
        transferred=transfer_turn is not None,
        transfer_turn=transfer_turn,
        should_transfer=ctx.should_transfer,
        violations=violations,
        format_errors=format_errors,
        n_turns=len(policy_turns),
        n_tool_calls=sum(len(m.get("tool_calls") or []) for m in policy_turns),
        n_tokens=n_tokens,
        duration_s=duration_s,
        extra=dict(extra or {}),
    )
