"""Metrics: pass^k against brute force, bootstrap CIs, style gap, F1 edge cases."""

import itertools
import random
from math import comb

import pytest
from fakes import trajectory

from handoff.eval.metrics import (
    Confusion,
    bootstrap,
    drifted_indicators,
    hacking_indicators,
    handoff_confusion,
    metric_statistics,
    paired_difference,
    pass_hat_k_from_counts,
    style_breakdown,
    success_statistic,
    summarize,
    task_units,
)
from handoff.rollout.schema import Violation


def brute_force_pass_k(outcomes: list[bool], k: int) -> float:
    subsets = list(itertools.combinations(range(len(outcomes)), k))
    return sum(all(outcomes[i] for i in s) for s in subsets) / len(subsets)


@pytest.mark.parametrize("seed", range(5))
def test_pass_k_matches_brute_force(seed):
    rng = random.Random(seed)
    groups = [[rng.random() < 0.6 for _ in range(rng.randint(4, 8))] for _ in range(6)]
    for k in (1, 2, 4):
        expected = sum(brute_force_pass_k(g, k) for g in groups) / len(groups)
        counts = [(sum(g), len(g)) for g in groups]
        assert pass_hat_k_from_counts(counts, k) == pytest.approx(expected)


def test_pass_k_input_validation():
    with pytest.raises(ValueError):
        pass_hat_k_from_counts([(1, 2)], 4)
    with pytest.raises(ValueError):
        pass_hat_k_from_counts([(3, 2)], 1)
    with pytest.raises(ValueError):
        pass_hat_k_from_counts([], 1)
    assert pass_hat_k_from_counts([(4, 4), (0, 4)], 4) == 0.5
    assert pass_hat_k_from_counts([(2, 4)], 2) == comb(2, 2) / comb(4, 2)


def episodes(task_id, outcomes, **kw):
    return [
        trajectory(task_id=task_id, trial=i, task_reward=float(o), **kw)
        for i, o in enumerate(outcomes)
    ]


def test_summary_success_pass_k_and_ci_shape():
    data = (
        episodes("1", [1, 1, 1, 1])
        + episodes("2", [1, 0, 1, 0])
        + episodes("3", [0, 0, 0, 0])
    )
    summary = summarize(data, k=4, n_resamples=200, seed=1)
    assert summary["success"].point == pytest.approx(0.5)
    assert summary["pass^4"].point == pytest.approx(1 / 3)
    interval = summary["success"]
    assert interval.low <= interval.point <= interval.high
    assert interval.n_units == 3 and interval.n_resamples == 200
    again = summarize(data, k=4, n_resamples=200, seed=1)["success"]
    assert (again.low, again.high) == (interval.low, interval.high)


def test_pass_k_is_none_when_trials_are_too_few():
    data = episodes("1", [1, 1])
    assert summarize(data, k=4, n_resamples=10)["pass^4"].point is None


def test_bootstrap_of_empty_or_undefined():
    interval = bootstrap([], success_statistic, n_resamples=10)
    assert interval.point is None and interval.low is None
    assert interval.n_resamples == 0
    with pytest.raises(ValueError):
        bootstrap([[trajectory()]], success_statistic, n_resamples=0)


def test_bootstrap_counts_only_resamples_where_the_statistic_exists():
    """Precision needs a transfer: resamples without one are not in the CI."""
    units = [
        [trajectory(task_id="a", transferred=True, should_transfer=True)],
        *[[trajectory(task_id=str(i))] for i in range(9)],
    ]
    precision = metric_statistics()["handoff_precision"]
    interval = bootstrap(units, precision, n_resamples=200, seed=1)
    assert interval.point == 1.0
    assert 0 < interval.n_resamples < 200
    assert bootstrap(units, success_statistic, n_resamples=50).n_resamples == 50


def test_handoff_f1_edge_cases():
    no_positive = Confusion(tp=0, fp=0, fn=0, tn=5)
    assert no_positive.precision is None
    assert no_positive.recall is None
    assert no_positive.f1 is None
    zero = Confusion(tp=0, fp=2, fn=3, tn=0)
    assert zero.precision == 0.0 and zero.recall == 0.0 and zero.f1 == 0.0
    # An agent that never transfers although it should: precision is undefined,
    # but F1 = 2TP / (2TP + FP + FN) is 0, not missing.
    never = Confusion(tp=0, fp=0, fn=10, tn=50)
    assert never.precision is None and never.recall == 0.0 and never.f1 == 0.0
    assert Confusion(tp=3, fp=1, fn=2, tn=4).f1 == pytest.approx(6 / 9)
    data = [
        trajectory(task_id="a", transferred=True, should_transfer=True),
        trajectory(task_id="b", transferred=True, should_transfer=False),
        trajectory(task_id="c", transferred=False, should_transfer=True),
        trajectory(task_id="d", transferred=False, should_transfer=False),
    ]
    confusion = handoff_confusion(data)
    assert (confusion.tp, confusion.fp, confusion.fn, confusion.tn) == (1, 1, 1, 1)
    assert confusion.f1 == pytest.approx(0.5)
    summary = summarize(data, n_resamples=20)
    assert summary["over_escalation_rate"].point == 0.5
    assert summary["under_escalation_rate"].point == 0.5
    assert summary["transfer_rate"].point == 0.5


def test_violation_rates_split_rules_and_reference():
    data = [
        trajectory(task_id="a", violations=[Violation("off_reference_write", 3)]),
        trajectory(
            task_id="b", violations=[Violation("write_without_confirmation", 3)]
        ),
        trajectory(task_id="c"),
        trajectory(task_id="d"),
    ]
    summary = summarize(data, n_resamples=20)
    assert summary["violation_rate"].point == 0.5
    assert summary["violation_rate_rules_only"].point == 0.25
    assert summary["rule/off_reference_write"].point == 0.25


def test_style_gap_and_worst_style():
    data = []
    for task in ("1", "2"):
        data += episodes(task, [1, 1], persona="neutral")
        data += episodes(task, [1, 0], persona="terse")
        data += episodes(task, [0, 0], persona="impatient")
    breakdown = style_breakdown(data, n_resamples=50)
    assert breakdown["success/neutral"].point == 1.0
    assert breakdown["success/terse"].point == 0.5
    assert breakdown["style_gap"].point == 1.0
    assert breakdown["worst_style_success"].point == 0.0


def test_paired_difference_uses_shared_tasks():
    a = episodes("1", [1, 1]) + episodes("2", [1, 1]) + episodes("9", [1, 1])
    b = episodes("1", [0, 0]) + episodes("2", [1, 1])
    interval = paired_difference(a, b, n_resamples=100, seed=3)
    assert interval.n_units == 2
    assert interval.point == pytest.approx(0.5)
    assert interval.low <= interval.point <= interval.high


def test_hacking_indicators_and_drift():
    data = [
        trajectory(task_id="a", transferred=True, should_transfer=False),
        trajectory(task_id="b", termination_reason="user_stop", task_reward=0.0),
    ]
    indicators = hacking_indicators(data)
    assert indicators["transfer_rate_on_negatives"] == 0.5
    assert indicators["user_stop_failure_rate"] == 0.5
    flagged = drifted_indicators(
        {"x": 0.5, "y": 0.03, "z": None}, {"x": 0.2, "y": 0.001, "z": 0.1}
    )
    assert flagged == ["x", "y"]
    assert drifted_indicators({"y": 0.015}, {"y": 0.001}) == []


def test_task_units_group_trials_and_personas():
    data = episodes("1", [1, 0], persona="neutral") + episodes(
        "1", [1], persona="terse"
    )
    units = task_units(data)
    assert len(units) == 1 and len(units[0]) == 3
    assert success_statistic(units) == pytest.approx((0.5 + 1.0) / 2)
