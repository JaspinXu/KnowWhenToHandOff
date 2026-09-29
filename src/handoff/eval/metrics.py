"""Evaluation metrics with task-level bootstrap confidence intervals.

Definitions are frozen in ``docs/metrics.md`` (ADR-005). All functions are pure:
they take :class:`~handoff.rollout.schema.Trajectory` records and return numbers.
Undefined values (for example precision without any transfer) are ``None``,
never silently replaced by zero.

Bootstrap unit: one task, ``(domain, task_id)``, with all of its trials and
personas, resampled with replacement (1,000 resamples, percentile interval).
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import comb
from typing import Any

from handoff.eval.citations import citation_events
from handoff.rewards.rules import RULE_IDS
from handoff.rollout.schema import Trajectory

Units = Sequence[Sequence[Trajectory]]
Statistic = Callable[[Units], float | None]


@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap percentile interval.

    ``n_resamples`` counts the resamples on which the statistic was defined
    (for example precision needs a transfer), i.e. those behind the interval.
    """

    point: float | None
    low: float | None
    high: float | None
    n_units: int
    n_resamples: int

    def as_dict(self) -> dict[str, float | int | None]:
        """Flatten for JSON output."""
        return {
            "point": self.point,
            "low": self.low,
            "high": self.high,
            "n_units": self.n_units,
            "n_resamples": self.n_resamples,
        }


def format_interval(interval: Mapping[str, Any] | None, digits: int = 3) -> str:
    """``0.512 [0.44, 0.58]`` or ``n/a``."""
    if not interval or interval.get("point") is None:
        return "n/a"
    point = f"{interval['point']:.{digits}f}"
    if interval.get("low") is None:
        return point
    return f"{point} [{interval['low']:.{digits}f}, {interval['high']:.{digits}f}]"


def _flatten(units: Units) -> list[Trajectory]:
    return [episode for unit in units for episode in unit]


def group_by(
    trajectories: Iterable[Trajectory], key: Callable[[Trajectory], Hashable]
) -> dict[Hashable, list[Trajectory]]:
    """Group trajectories by a key while keeping insertion order."""
    groups: dict[Hashable, list[Trajectory]] = defaultdict(list)
    for trajectory in trajectories:
        groups[key(trajectory)].append(trajectory)
    return dict(groups)


def task_key(trajectory: Trajectory) -> tuple[str, str]:
    """Bootstrap unit key: one task in one domain."""
    return (trajectory.domain, trajectory.task_id)


def task_units(trajectories: Iterable[Trajectory]) -> list[list[Trajectory]]:
    """Split trajectories into bootstrap units (one list per task)."""
    return list(group_by(trajectories, task_key).values())


def pass_hat_k_from_counts(counts: Sequence[tuple[int, int]], k: int) -> float:
    """Average ``C(c, k) / C(n, k)`` over groups given ``(successes, trials)``."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if not counts:
        raise ValueError("no groups to average")
    values = []
    for successes, trials in counts:
        if not 0 <= successes <= trials:
            raise ValueError("successes must be between 0 and trials")
        if trials < k:
            raise ValueError(f"pass^{k} needs at least {k} trials per group")
        values.append(comb(successes, k) / comb(trials, k))
    return sum(values) / len(values)


def _pass_groups(units: Units) -> list[tuple[int, int]]:
    counts: list[tuple[int, int]] = []
    for unit in units:
        for episodes in group_by(unit, lambda t: t.persona).values():
            counts.append((sum(t.success for t in episodes), len(episodes)))
    return counts


def success_statistic(units: Units) -> float | None:
    """pass^1: mean per-(task, persona) success fraction."""
    counts = _pass_groups(units)
    if not counts:
        return None
    return sum(s / n for s, n in counts) / len(counts)


def pass_k_statistic(k: int) -> Statistic:
    """Return a statistic computing pass^k, or ``None`` if trials are too few."""

    def statistic(units: Units) -> float | None:
        counts = _pass_groups(units)
        if not counts or any(n < k for _, n in counts):
            return None
        return pass_hat_k_from_counts(counts, k)

    return statistic


def _rate(predicate: Callable[[Trajectory], bool]) -> Statistic:
    def statistic(units: Units) -> float | None:
        episodes = _flatten(units)
        if not episodes:
            return None
        return sum(predicate(t) for t in episodes) / len(episodes)

    return statistic


def _conditional_rate(
    predicate: Callable[[Trajectory], bool], condition: Callable[[Trajectory], bool]
) -> Statistic:
    def statistic(units: Units) -> float | None:
        episodes = [t for t in _flatten(units) if condition(t)]
        if not episodes:
            return None
        return sum(predicate(t) for t in episodes) / len(episodes)

    return statistic


violation_rate = _rate(lambda t: bool(t.violations))
violation_rate_rules_only = _rate(
    lambda t: any(v.rule != "off_reference_write" for v in t.violations)
)
format_error_rate = _rate(lambda t: bool(t.format_errors))
transfer_rate = _rate(lambda t: t.transferred)
over_escalation_rate = _conditional_rate(
    lambda t: t.transferred, lambda t: not t.should_transfer
)
under_escalation_rate = _conditional_rate(
    lambda t: not t.transferred, lambda t: t.should_transfer
)


def rule_rate(rule: str) -> Statistic:
    """Share of episodes with at least one violation of ``rule``."""
    if rule not in RULE_IDS:
        raise ValueError(f"unknown rule {rule!r}")
    return _rate(lambda t: any(v.rule == rule for v in t.violations))


@dataclass(frozen=True)
class Confusion:
    """Hand-off decisions against should-transfer labels."""

    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float | None:
        """TP / (TP + FP); ``None`` when the agent never transferred."""
        denominator = self.tp + self.fp
        return None if denominator == 0 else self.tp / denominator

    @property
    def recall(self) -> float | None:
        """TP / (TP + FN); ``None`` when no episode should transfer."""
        denominator = self.tp + self.fn
        return None if denominator == 0 else self.tp / denominator

    @property
    def f1(self) -> float | None:
        """2TP / (2TP + FP + FN); ``None`` only without positives and transfers.

        Equal to the harmonic mean of precision and recall when both exist, and
        0 for an agent that never transfers although some episode should.
        """
        denominator = 2 * self.tp + self.fp + self.fn
        return None if denominator == 0 else 2 * self.tp / denominator


def handoff_confusion(trajectories: Iterable[Trajectory]) -> Confusion:
    """Count hand-off outcomes over episodes."""
    tp = fp = fn = tn = 0
    for t in trajectories:
        if t.transferred and t.should_transfer:
            tp += 1
        elif t.transferred:
            fp += 1
        elif t.should_transfer:
            fn += 1
        else:
            tn += 1
    return Confusion(tp, fp, fn, tn)


def _confusion_statistic(attribute: str) -> Statistic:
    def statistic(units: Units) -> float | None:
        return getattr(handoff_confusion(_flatten(units)), attribute)

    return statistic


def citation_statistic(units: Units) -> float | None:
    """Share of refusal and hand-off events that name the policy (rule match)."""
    events = [e for t in _flatten(units) for e in citation_events(t.messages)]
    if not events:
        return None
    return sum(e.cited for e in events) / len(events)


def _mean(value: Callable[[Trajectory], float | None]) -> Statistic:
    def statistic(units: Units) -> float | None:
        values = [v for t in _flatten(units) if (v := value(t)) is not None]
        return None if not values else sum(values) / len(values)

    return statistic


mean_turns = _mean(lambda t: float(t.n_turns))
mean_tool_calls = _mean(lambda t: float(t.n_tool_calls))
mean_tokens = _mean(lambda t: None if t.n_tokens is None else float(t.n_tokens))
mean_messages = _mean(lambda t: float(len(t.messages)))
user_stop_failure_rate = _rate(
    lambda t: (
        t.termination_reason == "user_stop" and not t.success and not t.transferred
    )
)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("no values")
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap[U](
    units: Sequence[U],
    statistic: Callable[[Sequence[U]], float | None],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap over units (tasks, or pairs of tasks) with a fixed seed."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    point = statistic(units)
    if not units or point is None:
        return Interval(point, None, None, len(units), 0)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        resampled = [units[rng.randrange(len(units))] for _ in units]
        value = statistic(resampled)
        if value is not None:
            samples.append(value)
    if not samples:
        return Interval(point, None, None, len(units), 0)
    samples.sort()
    alpha = (1 - confidence) / 2
    return Interval(
        point,
        _percentile(samples, alpha),
        _percentile(samples, 1 - alpha),
        len(units),
        len(samples),
    )


def metric_statistics(k: int = 4) -> dict[str, Statistic]:
    """All headline metrics keyed by their report names."""
    statistics: dict[str, Statistic] = {
        "success": success_statistic,
        f"pass^{k}": pass_k_statistic(k),
        "violation_rate": violation_rate,
        "violation_rate_rules_only": violation_rate_rules_only,
        "format_error_rate": format_error_rate,
        "transfer_rate": transfer_rate,
        "handoff_precision": _confusion_statistic("precision"),
        "handoff_recall": _confusion_statistic("recall"),
        "handoff_f1": _confusion_statistic("f1"),
        "over_escalation_rate": over_escalation_rate,
        "under_escalation_rate": under_escalation_rate,
        "citation_rate": citation_statistic,
        "user_stop_failure_rate": user_stop_failure_rate,
        "mean_turns": mean_turns,
        "mean_tool_calls": mean_tool_calls,
        "mean_tokens": mean_tokens,
    }
    for rule in RULE_IDS:
        statistics[f"rule/{rule}"] = rule_rate(rule)
    return statistics


def summarize(
    trajectories: Sequence[Trajectory],
    *,
    k: int = 4,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Interval]:
    """Compute every headline metric with a task-level bootstrap CI."""
    units = task_units(trajectories)
    return {
        name: bootstrap(
            units, stat, n_resamples=n_resamples, confidence=confidence, seed=seed
        )
        for name, stat in metric_statistics(k).items()
    }


def style_success(units: Units) -> dict[str, float]:
    """Success (pass^1) per communication style."""
    by_style = group_by(_flatten(units), lambda t: t.persona)
    result: dict[str, float] = {}
    for style, episodes in by_style.items():
        value = success_statistic(task_units(episodes))
        if value is not None:
            result[str(style)] = value
    return result


def style_gap_statistic(units: Units) -> float | None:
    """Max minus min success across styles (needs at least two styles)."""
    values = style_success(units)
    if len(values) < 2:
        return None
    return max(values.values()) - min(values.values())


def worst_style_statistic(units: Units) -> float | None:
    """Success of the worst-served style."""
    values = style_success(units)
    return None if not values else min(values.values())


def style_breakdown(
    trajectories: Sequence[Trajectory],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Interval]:
    """Per-style success plus the style gap and worst-style success, with CIs."""
    units = task_units(trajectories)

    def interval(selected: Units, statistic: Statistic) -> Interval:
        return bootstrap(
            selected,
            statistic,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )

    result: dict[str, Interval] = {}
    for style in sorted({t.persona for t in trajectories}):
        style_units = [
            [t for t in unit if t.persona == style]
            for unit in units
            if any(t.persona == style for t in unit)
        ]
        result[f"success/{style}"] = interval(style_units, success_statistic)
    result["style_gap"] = interval(units, style_gap_statistic)
    result["worst_style_success"] = interval(units, worst_style_statistic)
    return result


def paired_difference(
    candidate: Sequence[Trajectory],
    baseline: Sequence[Trajectory],
    statistic: Statistic = success_statistic,
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap CI of ``statistic(candidate) - statistic(baseline)``.

    Only tasks present in both runs are used and both runs are resampled with
    the same task indices (a paired comparison, README protocol).
    """
    a = group_by(candidate, task_key)
    b = group_by(baseline, task_key)
    shared = [key for key in a if key in b]
    pairs = [(a[key], b[key]) for key in shared]

    def difference(selected: Sequence[tuple[list[Trajectory], list[Trajectory]]]):
        left = statistic([p[0] for p in selected])
        right = statistic([p[1] for p in selected])
        return None if left is None or right is None else left - right

    return bootstrap(
        pairs,
        difference,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )


def hacking_indicators(trajectories: Sequence[Trajectory]) -> dict[str, float | None]:
    """Reward-hacking watch-list values (AGENTS section 7), without CIs."""
    units = task_units(trajectories)
    return {
        "transfer_rate_on_negatives": over_escalation_rate(units),
        "mean_messages": mean_messages(units),
        "mean_turns": mean_turns(units),
        "format_error_rate": format_error_rate(units),
        "user_stop_failure_rate": user_stop_failure_rate(units),
    }


def drifted_indicators(
    candidate: dict[str, float | None],
    baseline: dict[str, float | None],
    *,
    factor: float = 2.0,
    floor: float = 0.02,
) -> list[str]:
    """Return watch-list keys whose value exceeds ``factor`` times the baseline.

    ``floor`` avoids flagging tiny absolute changes from a near-zero baseline:
    the threshold is ``max(factor * baseline, floor)``.
    """
    flagged = []
    for key, value in candidate.items():
        reference = baseline.get(key)
        if value is None or reference is None:
            continue
        if value > max(factor * reference, floor):
            flagged.append(key)
    return flagged
