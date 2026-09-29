"""Adapters between tau2 objects and the project's normalized trajectory.

tau2 is imported lazily so that the pure modules stay importable in the CPU-only
development environment. The functions here accept tau2 ``Task``,
``SimulationRun`` and message objects (duck-typed where possible).

Both drivers end an episode early in the same way. When the policy's turn
cannot be used (a generation cut at the token limit, an empty message, a
context overflow), the agent returns a *stop sentinel*: an assistant message
whose ``raw_data`` records why the policy stopped and which format errors the
turn had. tau2 ends the episode on it (the text contains tau2's stop token);
:func:`finalize_simulation` then restores the real termination reason and
re-runs the deterministic evaluator, so the task reward is 0 as for any other
premature termination. The sentinel itself never appears in a
:class:`Trajectory`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from handoff.rewards.rules import GoldAction
from handoff.rollout.schema import FormatError, Trajectory
from handoff.rollout.scoring import EpisodeContext, build_trajectory

NL_ASSERTION = "NL_ASSERTION"
SENTINEL_KEY = "handoff_stop"
STOP_TEXT = "###STOP###"
STOP_REASONS = ("context_window_exceeded", "agent_error")


def stop_marker(reason: str, format_errors: Sequence[str] = ()) -> dict[str, Any]:
    """``raw_data`` payload of a stop sentinel."""
    if reason not in STOP_REASONS:
        raise ValueError(f"unknown stop reason {reason!r}")
    return {SENTINEL_KEY: {"reason": reason, "format_errors": list(format_errors)}}


def stop_message(
    reason: str,
    format_errors: Sequence[str] = (),
    *,
    usage: dict[str, Any] | None = None,
) -> Any:
    """A tau2 assistant message that ends the episode for ``reason``."""
    from tau2.data_model.message import AssistantMessage

    return AssistantMessage(
        role="assistant",
        content=STOP_TEXT,
        cost=0.0,
        usage=usage,
        raw_data=stop_marker(reason, format_errors),
    )


def sentinel_of(message: Any) -> dict[str, Any] | None:
    """The stop record of a sentinel message, or ``None`` for other messages."""
    raw = getattr(message, "raw_data", None)
    if isinstance(raw, dict) and isinstance(raw.get(SENTINEL_KEY), dict):
        return raw[SENTINEL_KEY]
    return None


def finalize_simulation(simulation: Any, task: Any, domain: str) -> list[FormatError]:
    """Apply a stop sentinel (if any) and make sure the episode is evaluated.

    Returns the format errors recorded by the sentinel. The evaluator runs when
    the simulation has no reward yet (training driver) or when a sentinel
    changed the termination reason (both drivers).
    """
    from tau2.data_model.simulation import TerminationReason

    errors: list[FormatError] = []
    stopped = False
    for index, message in enumerate(simulation.messages or []):
        record = sentinel_of(message)
        if record is None:
            continue
        turn = getattr(message, "turn_idx", None)
        turn = index if turn is None else int(turn)
        errors.extend(FormatError(kind, turn) for kind in record["format_errors"])
        simulation.termination_reason = TerminationReason(record["reason"])
        stopped = True
    if stopped or getattr(simulation, "reward_info", None) is None:
        simulation.reward_info = evaluate_deterministic(simulation, task, domain)
    return errors


def normalize_tau2_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """Convert tau2 message objects into normalized dictionaries (sans sentinels)."""
    result: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if sentinel_of(message) is not None:
            continue
        role = getattr(message, "role", None)
        turn = getattr(message, "turn_idx", None)
        turn = index if turn is None else int(turn)
        if role in ("assistant", "user"):
            calls = getattr(message, "tool_calls", None)
            result.append(
                {
                    "role": role,
                    "content": getattr(message, "content", None),
                    "tool_calls": (
                        None
                        if not calls
                        else [
                            {
                                "id": call.id,
                                "name": call.name,
                                "arguments": dict(call.arguments),
                            }
                            for call in calls
                        ]
                    ),
                    "turn": turn,
                }
            )
        elif role == "tool":
            result.append(
                {
                    "role": "tool",
                    "content": getattr(message, "content", None),
                    "tool_call_id": getattr(message, "id", None),
                    "requestor": getattr(message, "requestor", "assistant"),
                    "error": bool(getattr(message, "error", False)),
                    "turn": turn,
                }
            )
    return result


def gold_actions_from_task(task: Any) -> tuple[GoldAction, ...]:
    """Return the assistant actions of a tau2 task's reference solution."""
    criteria = getattr(task, "evaluation_criteria", None)
    actions = getattr(criteria, "actions", None) or []
    return tuple(
        GoldAction(
            name=action.name,
            arguments=dict(action.arguments),
            compare_args=(
                None if action.compare_args is None else tuple(action.compare_args)
            ),
        )
        for action in actions
        if getattr(action, "requestor", "assistant") == "assistant"
    )


def reward_basis_names(task: Any) -> list[str]:
    """Return the task's reward basis as plain strings."""
    criteria = getattr(task, "evaluation_criteria", None)
    basis = getattr(criteria, "reward_basis", None) or []
    return [getattr(item, "value", str(item)) for item in basis]


def assert_deterministic_reward_basis(task: Any) -> None:
    """Refuse tasks whose official reward would need an LLM judge.

    AGENTS hard rule 3: training rewards must be deterministic. Airline and retail
    tasks at the pinned revision do not use NL assertions; this guard makes any
    change to that fail loudly before a rollout starts.
    """
    if NL_ASSERTION in reward_basis_names(task):
        raise ValueError(
            f"task {getattr(task, 'id', '?')} requires NL_ASSERTION; "
            "LLM-judged rewards are not allowed"
        )


def evaluate_deterministic(simulation: Any, task: Any, domain: str) -> Any:
    """Run tau2's evaluator in ``ALL`` mode after the NL-assertion guard."""
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

    assert_deterministic_reward_basis(task)
    return evaluate_simulation(
        simulation=simulation,
        task=task,
        evaluation_type=EvaluationType.ALL,
        solo_mode=False,
        domain=domain,
    )


def agent_completion_tokens(messages: Sequence[Any]) -> int | None:
    """Sum completion tokens reported for assistant messages, if any were."""
    total = 0
    seen = False
    for message in messages:
        if getattr(message, "role", None) != "assistant":
            continue
        usage = getattr(message, "usage", None) or {}
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if isinstance(tokens, int):
            total += tokens
            seen = True
    return total if seen else None


def domain_tool_sets(environment: Any) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(write_tools, all_agent_tools)`` from a tau2 environment."""
    from tau2.environment.toolkit import ToolType, get_tool_types

    types = get_tool_types(environment.tools)
    writes = frozenset(name for name, kind in types.items() if kind == ToolType.WRITE)
    return writes, frozenset(types)


def trajectory_from_simulation(
    ctx: EpisodeContext,
    simulation: Any,
    *,
    extra_format_errors: Sequence[FormatError] = (),
    extra: dict[str, Any] | None = None,
) -> Trajectory:
    """Build a :class:`Trajectory` from an evaluated tau2 ``SimulationRun``."""
    reward_info = getattr(simulation, "reward_info", None)
    if reward_info is None:
        raise ValueError("simulation has no reward_info; evaluate it first")
    messages = list(simulation.messages or [])
    db_check = getattr(reward_info, "db_check", None)
    basis = getattr(reward_info, "reward_basis", None)
    termination = getattr(simulation.termination_reason, "value", None) or str(
        simulation.termination_reason
    )
    info = dict(extra or {})
    info.setdefault("simulation_id", getattr(simulation, "id", None))
    return build_trajectory(
        ctx,
        messages=normalize_tau2_messages(messages),
        termination_reason=termination,
        task_reward=float(reward_info.reward),
        reward_basis=(
            None if basis is None else [getattr(b, "value", str(b)) for b in basis]
        ),
        db_match=None if db_check is None else bool(db_check.db_match),
        n_tokens=agent_completion_tokens(messages),
        duration_s=getattr(simulation, "duration", None),
        driver_format_errors=list(extra_format_errors),
        extra=info,
    )
