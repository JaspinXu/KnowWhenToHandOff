"""Human-review packets and error analysis (T1.3, T1.4, T2.3, T3.5, T3.6).

Everything here produces Markdown for a person to read and tick: sampled
dialogues with the automatic annotations (violations, format errors, hand-offs,
citation events) and a short rubric. Sampling is seeded and stratified so the
packets are reproducible.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from handoff.eval.citations import citation_events
from handoff.rollout.schema import Trajectory

DIALOGUE_RUBRIC = (
    "Policy followed (no unlisted violation)",
    "Hand-off decision correct for this user",
    "Refusals and hand-offs explain the policy",
    "Simulated user stayed within the scenario",
    "Simulated user followed the communication style",
)
TASK_RUBRIC = (
    "Instructions are coherent and solvable as labelled",
    "Label (should-transfer) is correct under the tool docstring and policy",
    "Evaluation criteria match the intended behaviour",
    "No task facts or identifiers were altered",
)


def stratified_sample(
    items: Sequence[Any],
    key: Callable[[Any], Any],
    n: int,
    *,
    seed: int,
) -> list[Any]:
    """Round-robin sample of up to ``n`` items across strata, seeded."""
    groups: dict[Any, list[Any]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    rng = random.Random(seed)
    for members in groups.values():
        rng.shuffle(members)
    ordered_keys = sorted(groups, key=str)
    picked: list[Any] = []
    while len(picked) < n and any(groups.values()):
        for group_key in ordered_keys:
            if groups[group_key] and len(picked) < n:
                picked.append(groups[group_key].pop())
    return picked


def is_failure(trajectory: Trajectory) -> bool:
    """Task failed, a rule was broken, or the hand-off decision was wrong."""
    return (
        not trajectory.success
        or bool(trajectory.violations)
        or trajectory.transferred != trajectory.should_transfer
    )


def _content(message: Mapping[str, Any]) -> str:
    text = message.get("content") or ""
    calls = message.get("tool_calls") or []
    parts = [str(text).strip()] if str(text).strip() else []
    for call in calls:
        arguments = json.dumps(call.get("arguments"), ensure_ascii=False)
        parts.append(f"`{call.get('name')}({arguments})`")
    return "<br>".join(parts) or "(empty)"


def dialogue_markdown(trajectory: Trajectory, *, max_tool_chars: int = 400) -> str:
    """A reviewable transcript with automatic annotations and the rubric."""
    lines = [
        f"### {trajectory.episode_id}",
        "",
        f"- Kind: {trajectory.kind} · split: {trajectory.split} · "
        f"should transfer: {trajectory.should_transfer} · transferred: "
        f"{trajectory.transferred}",
        f"- Task reward: {trajectory.task_reward} · termination: "
        f"{trajectory.termination_reason} · turns: {trajectory.n_turns}",
        f"- Violations: {[v.rule for v in trajectory.violations] or 'none'}",
        f"- Format errors: {[e.kind for e in trajectory.format_errors] or 'none'}",
    ]
    events = citation_events(trajectory.messages)
    if events:
        lines.append(
            "- Citation events: "
            + ", ".join(
                f"{e.kind}@{e.turn}:{'cited' if e.cited else 'not cited'}"
                for e in events
            )
        )
    lines += ["", "| # | Role | Message |", "| --- | --- | --- |"]
    for message in trajectory.messages:
        role = message.get("role")
        text = _content(message)
        if role == "tool":
            text = f"`{str(message.get('content') or '')[:max_tool_chars]}`"
        text = text.replace("|", "\\|").replace("\n", "<br>")
        lines.append(f"| {message.get('turn')} | {role} | {text} |")
    lines += ["", "Review:"]
    lines += [f"- [ ] {item}" for item in DIALOGUE_RUBRIC]
    lines += ["- Notes:", ""]
    return "\n".join(lines)


def task_markdown(task: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    """A reviewable hand-off task with its generation metadata and rubric."""
    scenario = task.get("user_scenario", {})
    criteria = task.get("evaluation_criteria", {})
    lines = [
        f"### {meta.get('domain')}/{task.get('id')}",
        "",
        f"- Kind: {meta.get('kind')} · template: {meta.get('template_id')} · split: "
        f"{meta.get('split')} · source: {meta.get('source_task_id')} · should "
        f"transfer: {meta.get('should_transfer')}",
        "",
        "```json",
        json.dumps(scenario.get("instructions"), indent=2, ensure_ascii=False),
        "```",
        "",
        f"Reward basis: {criteria.get('reward_basis')} · actions: "
        f"{[a.get('name') for a in criteria.get('actions') or []]}",
        "",
        "Review:",
        *[f"- [ ] {item}" for item in TASK_RUBRIC],
        "- Notes:",
        "",
    ]
    return "\n".join(lines)


def per_task_changes(
    candidate: Sequence[Trajectory], baseline: Sequence[Trajectory]
) -> dict[str, list[str]]:
    """Tasks whose success rate went up or down between two runs."""

    def rates(trajectories: Sequence[Trajectory]) -> dict[tuple[str, str], float]:
        grouped: dict[tuple[str, str], list[bool]] = defaultdict(list)
        for t in trajectories:
            grouped[(t.domain, t.task_id)].append(t.success)
        return {k: sum(v) / len(v) for k, v in grouped.items()}

    a, b = rates(candidate), rates(baseline)
    improved = sorted(
        f"{d}/{t}" for (d, t) in a if (d, t) in b and a[(d, t)] > b[(d, t)]
    )
    regressed = sorted(
        f"{d}/{t}" for (d, t) in a if (d, t) in b and a[(d, t)] < b[(d, t)]
    )
    return {"improved": improved, "regressed": regressed}
