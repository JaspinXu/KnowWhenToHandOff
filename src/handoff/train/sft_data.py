"""Teacher-trajectory selection and tokenization for the B2 warm start (T2.1).

Only episodes that are (1) on train-split tasks according to the frozen
registry, (2) successful, (3) violation-free, (4) free of format errors and
(5) made the correct transfer decision are kept. The split check uses the
registry, never the split recorded in the trajectory, so a mislabelled run
cannot leak dev or test tasks into training data.

Kept conversations are rendered with the *student's* chat template; the mask
marks assistant turns (content plus end-of-turn token) as targets. Records use
the ``prompt_ids / response_ids / response_mask`` format of
:mod:`handoff.train.sft_batch`, so SFT and RL share one token representation.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from handoff.envs.registry import TaskRegistry
from handoff.rollout.chat_delta import ChatDeltaTokenizer
from handoff.rollout.schema import Trajectory, assistant_messages

PROMPT_MESSAGES = 3  # system prompt, tau2 greeting, first user message


@dataclass(frozen=True)
class TeacherFilter:
    """Which teacher episodes are eligible as imitation targets."""

    require_success: bool = True
    require_no_violations: bool = True
    require_no_format_errors: bool = True
    require_correct_transfer: bool = True
    max_per_task_persona: int | None = None


def rejection_reason(
    trajectory: Trajectory, registry: TaskRegistry, rules: TeacherFilter
) -> str | None:
    """Return why a teacher episode is rejected, or ``None`` to keep it."""
    try:
        ref = registry.get(trajectory.domain, trajectory.task_id)
    except KeyError:
        return "unknown_task"
    if ref.split != "train":
        return f"split_{ref.split}"
    if ref.source_task_id != trajectory.source_task_id:
        return "lineage_mismatch"
    if not assistant_messages(trajectory.messages):
        return "no_policy_turn"
    if rules.require_success and not trajectory.success:
        return "task_failed"
    if rules.require_no_violations and trajectory.violations:
        return "policy_violation"
    if rules.require_no_format_errors and trajectory.format_errors:
        return "format_error"
    if rules.require_correct_transfer and (
        trajectory.transferred != ref.should_transfer
    ):
        return "wrong_transfer_decision"
    return None


def select_teacher_episodes(
    trajectories: Iterable[Trajectory], registry: TaskRegistry, rules: TeacherFilter
) -> tuple[list[Trajectory], Counter[str]]:
    """Filter teacher episodes; return kept ones and rejection counts."""
    kept: list[Trajectory] = []
    reasons: Counter[str] = Counter()
    per_group: Counter[tuple[str, str, str]] = Counter()
    for trajectory in trajectories:
        reason = rejection_reason(trajectory, registry, rules)
        if reason is None and rules.max_per_task_persona is not None:
            group = (trajectory.domain, trajectory.task_id, trajectory.persona)
            if per_group[group] >= rules.max_per_task_persona:
                reason = "group_cap"
            else:
                per_group[group] += 1
        if reason is None:
            kept.append(trajectory)
        else:
            reasons[reason] += 1
    return kept, reasons


def chat_messages(trajectory: Trajectory, system_prompt: str) -> list[dict[str, Any]]:
    """Convert a normalized trajectory into template-ready chat messages."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for message in trajectory.messages:
        role = message.get("role")
        if role == "user":
            messages.append({"role": "user", "content": message.get("content") or ""})
        elif role == "assistant":
            converted: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content") or "",
            }
            calls = message.get("tool_calls") or []
            if calls:
                converted["tool_calls"] = [
                    {
                        "id": call.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": dict(call.get("arguments") or {}),
                        },
                    }
                    for call in calls
                ]
            messages.append(converted)
        elif role == "tool" and message.get("requestor", "assistant") == "assistant":
            messages.append(
                {
                    "role": "tool",
                    "content": message.get("content") or "",
                    "tool_call_id": message.get("tool_call_id", ""),
                }
            )
    return messages


def tokenize_episode(
    trajectory: Trajectory,
    delta: ChatDeltaTokenizer,
    system_prompt: str,
) -> dict[str, Any]:
    """Render one kept episode into an SFT token record (split is always train)."""
    messages = chat_messages(trajectory, system_prompt)
    if len(messages) <= PROMPT_MESSAGES:
        raise ValueError("episode has no assistant turn after the first user message")
    masked = delta.mask_conversation(messages, prompt_messages=PROMPT_MESSAGES)
    return {
        "split": "train",
        "prompt_ids": masked.prompt_ids,
        "response_ids": masked.response_ids,
        "response_mask": masked.response_mask,
    }


def write_sft_records(
    kept: Sequence[Trajectory],
    *,
    delta_for: Callable[[str], ChatDeltaTokenizer],
    system_prompt_for: Callable[[str], str],
    records_path: Path,
    meta_path: Path,
    max_length: int | None = None,
) -> Counter[str]:
    """Write token records and a parallel metadata file; return skip counts."""
    skipped: Counter[str] = Counter()
    records_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        records_path.open("w", encoding="utf-8", newline="\n") as records,
        meta_path.open("w", encoding="utf-8", newline="\n") as meta,
    ):
        for trajectory in kept:
            record = tokenize_episode(
                trajectory,
                delta_for(trajectory.domain),
                system_prompt_for(trajectory.domain),
            )
            length = len(record["prompt_ids"]) + len(record["response_ids"])
            if max_length is not None and length > max_length:
                skipped["too_long"] += 1
                continue
            records.write(json.dumps(record, separators=(",", ":")) + "\n")
            meta.write(
                json.dumps(
                    {
                        "run_id": trajectory.run_id,
                        "domain": trajectory.domain,
                        "task_id": trajectory.task_id,
                        "source_task_id": trajectory.source_task_id,
                        "kind": trajectory.kind,
                        "persona": trajectory.persona,
                        "trial": trajectory.trial,
                        "n_tokens": length,
                        "n_target_tokens": sum(record["response_mask"]),
                    }
                )
                + "\n"
            )
    return skipped


def dataset_statistics(
    kept: Sequence[Trajectory],
    rejected: Mapping[str, int],
    skipped: Mapping[str, int],
    meta_path: Path,
) -> dict[str, Any]:
    """Summary written next to the dataset (T2.1 definition of done)."""
    rows = [json.loads(line) for line in meta_path.read_text().splitlines() if line]
    by: dict[str, Counter[str]] = {
        key: Counter(str(row[key]) for row in rows)
        for key in ("domain", "kind", "persona")
    }
    tokens = [row["n_tokens"] for row in rows]
    targets = [row["n_target_tokens"] for row in rows]
    return {
        "n_candidates": len(kept) + sum(rejected.values()),
        "n_kept_episodes": len(kept),
        "n_records": len(rows),
        "rejected": dict(rejected),
        "skipped": dict(skipped),
        "by_domain": dict(by["domain"]),
        "by_kind": dict(by["kind"]),
        "by_persona": dict(by["persona"]),
        "n_unique_tasks": len({(r["domain"], r["task_id"]) for r in rows}),
        "tokens_total": sum(tokens),
        "target_tokens_total": sum(targets),
        "tokens_max": max(tokens) if tokens else 0,
        "tokens_mean": (sum(tokens) / len(tokens)) if tokens else 0.0,
        "splits": ["train"],
    }
