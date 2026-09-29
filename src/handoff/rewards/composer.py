"""Compose the README reward from a trajectory, logging every term separately.

.. math::

    R = R_task - λ_v N_viol - λ_FP 1[transfer ∧ ¬y] - λ_FN 1[¬transfer ∧ y]
        - λ_f N_format

All inputs are deterministic: the tau2 evaluator's task reward, rule-based
violations, the should-transfer label and format checks. No LLM judge is used.
With every lambda at zero (run R1) the total equals the task reward.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from handoff.rewards.rules import RULE_IDS
from handoff.rollout.schema import Trajectory

_CONFIG_KEYS = {
    "lambda_violation",
    "lambda_fp",
    "lambda_fn",
    "lambda_format",
    "violation_weights",
}


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


@dataclass(frozen=True)
class RewardConfig:
    """Lambda weights of the composite reward; all zero reproduces run R1."""

    lambda_violation: float = 0.0
    lambda_fp: float = 0.0
    lambda_fn: float = 0.0
    lambda_format: float = 0.0
    violation_weights: Mapping[str, float] = field(
        default_factory=lambda: dict.fromkeys(RULE_IDS, 1.0)
    )

    def __post_init__(self) -> None:
        for name in ("lambda_violation", "lambda_fp", "lambda_fn", "lambda_format"):
            if _finite(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must be non-negative")
        unknown = set(self.violation_weights) - set(RULE_IDS)
        if unknown:
            raise ValueError(f"unknown violation rules: {sorted(unknown)}")
        for rule, weight in self.violation_weights.items():
            if _finite(weight, f"violation_weights.{rule}") < 0:
                raise ValueError("violation weights must be non-negative")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> RewardConfig:
        """Validate a resolved Hydra ``reward`` mapping."""
        unknown = set(config) - _CONFIG_KEYS
        if unknown:
            raise ValueError(f"unknown reward config keys: {sorted(unknown)}")
        weights = dict.fromkeys(RULE_IDS, 1.0)
        weights.update(dict(config.get("violation_weights") or {}))
        return cls(
            lambda_violation=_finite(config.get("lambda_violation", 0.0), "lambda_v"),
            lambda_fp=_finite(config.get("lambda_fp", 0.0), "lambda_fp"),
            lambda_fn=_finite(config.get("lambda_fn", 0.0), "lambda_fn"),
            lambda_format=_finite(config.get("lambda_format", 0.0), "lambda_f"),
            violation_weights=weights,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for run records."""
        data = asdict(self)
        data["violation_weights"] = dict(self.violation_weights)
        return data


@dataclass(frozen=True)
class RewardBreakdown:
    """Every reward term of one episode; ``total`` is what the trainer sees."""

    task: float
    n_violations: float
    violation_penalty: float
    false_positive: bool
    false_negative: bool
    fp_penalty: float
    fn_penalty: float
    n_format_errors: int
    format_penalty: float
    total: float

    def as_metrics(self, prefix: str = "reward/") -> dict[str, float]:
        """Return a flat float mapping for trackers (one key per term)."""
        return {f"{prefix}{key}": float(value) for key, value in asdict(self).items()}


class RewardComposer:
    """Pure function object turning a :class:`Trajectory` into a breakdown."""

    def __init__(self, config: RewardConfig) -> None:
        """Store a validated configuration."""
        if not isinstance(config, RewardConfig):
            raise ValueError("config must be a RewardConfig")
        self.config = config

    def __call__(self, trajectory: Trajectory) -> RewardBreakdown:
        """Compute every term for one trajectory."""
        cfg = self.config
        task = float(trajectory.task_reward)
        n_violations = sum(
            cfg.violation_weights.get(violation.rule, 0.0)
            for violation in trajectory.violations
        )
        false_positive = trajectory.transferred and not trajectory.should_transfer
        false_negative = (not trajectory.transferred) and trajectory.should_transfer
        n_format = len(trajectory.format_errors)
        violation_penalty = cfg.lambda_violation * n_violations
        fp_penalty = cfg.lambda_fp * float(false_positive)
        fn_penalty = cfg.lambda_fn * float(false_negative)
        format_penalty = cfg.lambda_format * n_format
        total = task - violation_penalty - fp_penalty - fn_penalty - format_penalty
        return RewardBreakdown(
            task=task,
            n_violations=float(n_violations),
            violation_penalty=violation_penalty,
            false_positive=false_positive,
            false_negative=false_negative,
            fp_penalty=fp_penalty,
            fn_penalty=fn_penalty,
            n_format_errors=n_format,
            format_penalty=format_penalty,
            total=total,
        )
