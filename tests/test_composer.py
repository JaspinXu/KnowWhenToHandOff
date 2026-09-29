"""Reward composer: every term in isolation, λ = 0 reduces to the task reward."""

import pytest
from fakes import trajectory

from handoff.rewards.composer import RewardComposer, RewardConfig
from handoff.rollout.schema import FormatError, Violation


def compose(config: RewardConfig, **overrides):
    return RewardComposer(config)(trajectory(**overrides))


def test_all_lambdas_zero_is_task_reward():
    config = RewardConfig()
    for reward in (0.0, 1.0):
        breakdown = compose(
            config,
            task_reward=reward,
            transferred=True,
            should_transfer=False,
            violations=[Violation("off_reference_write", 3)],
            format_errors=[FormatError("unknown_tool", 2)],
        )
        assert breakdown.total == reward


def test_violation_term_uses_weights():
    config = RewardConfig(
        lambda_violation=0.5,
        violation_weights={
            "off_reference_write": 0.0,
            "write_without_confirmation": 2.0,
        },
    )
    breakdown = compose(
        config,
        violations=[
            Violation("off_reference_write", 3),
            Violation("write_without_confirmation", 3),
        ],
    )
    assert breakdown.n_violations == 2.0
    assert breakdown.violation_penalty == 1.0
    assert breakdown.total == 0.0


@pytest.mark.parametrize(
    ("transferred", "should", "fp", "fn"),
    [
        (True, False, True, False),
        (False, True, False, True),
        (True, True, False, False),
        (False, False, False, False),
    ],
)
def test_handoff_terms(transferred, should, fp, fn):
    config = RewardConfig(lambda_fp=0.5, lambda_fn=1.0)
    breakdown = compose(
        config, task_reward=0.0, transferred=transferred, should_transfer=should
    )
    assert (breakdown.false_positive, breakdown.false_negative) == (fp, fn)
    assert breakdown.total == -(0.5 * fp + 1.0 * fn)


def test_format_term_counts_errors():
    config = RewardConfig(lambda_format=0.1)
    breakdown = compose(
        config,
        format_errors=[FormatError("unknown_tool", 1), FormatError("empty_message", 2)],
    )
    assert breakdown.n_format_errors == 2
    assert breakdown.total == pytest.approx(0.8)


def test_breakdown_metrics_are_flat_floats():
    metrics = compose(RewardConfig(lambda_fp=1.0), transferred=True).as_metrics()
    assert metrics["reward/false_positive"] == 1.0
    assert metrics["reward/total"] == 0.0
    assert all(isinstance(v, float) for v in metrics.values())


def test_config_validation():
    with pytest.raises(ValueError):
        RewardConfig(lambda_fp=-1.0)
    with pytest.raises(ValueError):
        RewardConfig.from_mapping({"lambda_x": 1.0})
    with pytest.raises(ValueError):
        RewardConfig.from_mapping({"violation_weights": {"made_up": 1.0}})
    with pytest.raises(ValueError):
        RewardConfig.from_mapping({"lambda_fp": float("nan")})
    config = RewardConfig.from_mapping({"lambda_violation": 0.25, "lambda_fn": 1})
    assert config.violation_weights["off_reference_write"] == 1.0
    assert config.to_dict()["lambda_fn"] == 1.0
