"""Training monitors: reward-hacking watch-list and learning curves (T3.2–T3.5).

The agent loop writes one record per training episode. These functions compute
the watch-list from the most recent records and compare it with the B2
evaluation (see :func:`handoff.eval.metrics.hacking_indicators`); an indicator
is flagged when it exceeds ``max(2 x baseline, 0.02)``.

Learning curves come from verl's own per-step dumps (``trainer.rollout_data_dir``
and ``trainer.validation_data_dir``, one ``<step>.jsonl`` per step), which carry
every reward term the agent loop reports. verl rewrites a step's file when a
resumed run repeats that step, so each step is counted once.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Per-step means shown in the learning curves (keys of the agent loop's
# ``reward_extra_info``; ``total`` is the reward the trainer optimizes).
CURVE_KEYS = (
    "total",
    "task",
    "violation_penalty",
    "fp_penalty",
    "fn_penalty",
    "format_penalty",
    "success",
    "transferred",
    "n_turns",
    "infra_failure",
)


def read_rollout_records(directory: Path) -> list[dict[str, Any]]:
    """Read every per-process rollout record file in ``directory``, oldest first.

    Records are ordered by their ``time`` stamp, not by file: after a resumed
    job, new worker processes write new files whose names may sort anywhere.
    """
    records: list[dict[str, Any]] = []
    if not directory.exists():
        return records
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except ValueError:
                    continue  # a line being written concurrently
    records.sort(key=lambda record: float(record.get("time", 0.0)))
    return records


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def rollout_indicators(
    records: Iterable[Mapping[str, Any]], *, last: int | None = 512
) -> dict[str, float | None]:
    """Watch-list values over the most recent non-validation episodes.

    ``infra_failure_rate`` is the share of recent episodes lost to
    infrastructure (they carry no gradient); the other indicators are computed
    over the episodes that completed.
    """
    recent = [r for r in records if not r.get("validate")]
    if last is not None:
        recent = recent[-last:]
    train = [r for r in recent if not r.get("infra_failure")]
    negatives = [r for r in train if not r.get("should_transfer")]
    return {
        "transfer_rate_on_negatives": _mean(
            [float(r.get("transferred", 0.0)) for r in negatives]
        ),
        "mean_messages": _mean([float(r.get("n_messages", 0.0)) for r in train]),
        "mean_turns": _mean([float(r.get("n_turns", 0.0)) for r in train]),
        "format_error_rate": _mean(
            [float(r.get("n_format_errors", 0.0) > 0) for r in train]
        ),
        "user_stop_failure_rate": _mean(
            [float(r.get("user_stop_failure", 0.0)) for r in train]
        ),
        "success": _mean([float(r.get("success", 0.0)) for r in train]),
        "infra_failure_rate": _mean(
            [float(r.get("infra_failure", 0.0)) for r in recent]
        ),
        "n_episodes": float(len(train)),
    }


def split_self_baseline(
    records: Sequence[Mapping[str, Any]], n_baseline: int
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Split training episodes into the run's first ``n_baseline`` and the rest.

    The first training steps are sampled from the initial policy with the same
    temperature and task mix as the rest of the run, so they are the fair
    reference for drift. A B2 evaluation is greedy on dev and under-counts rare
    actions such as transfers (R1, 7 Oct: 0.22 at step 1 vs 0.04 in B2 dev).
    Validation episodes are dropped from both parts.
    """
    train = [r for r in records if not r.get("validate")]
    return train[:n_baseline], train[n_baseline:]


def reference_max(own: float | None, b2: float | None) -> float | None:
    """Larger of the run's own start and B2 (either may be missing).

    A few hundred start-up episodes can show zero of a rare event (format
    errors: 0 of 128 in R1) although B2 has it, so drift is flagged only above
    both references.
    """
    values = [v for v in (own, b2) if v is not None]
    return max(values) if values else None


def step_curve(directory: Path) -> list[dict[str, float]]:
    """Per-step means of :data:`CURVE_KEYS` from a verl dump directory.

    Each ``<step>.jsonl`` file becomes one row with ``step``, ``n_episodes`` and
    the mean of every key present in it; rows are in step order and a missing
    directory gives no rows.
    """
    rows: list[dict[str, float]] = []
    if not directory.is_dir():
        return rows
    files = [path for path in directory.glob("*.jsonl") if path.stem.isdigit()]
    for path in sorted(files, key=lambda p: int(p.stem)):
        values: dict[str, list[float]] = {key: [] for key in CURVE_KEYS}
        n_episodes = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            n_episodes += 1
            for key in CURVE_KEYS:
                value = record.get(key)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    values[key].append(float(value))
        row = {"step": int(path.stem), "n_episodes": n_episodes}
        row.update({key: sum(v) / len(v) for key, v in values.items() if v})
        rows.append(row)
    return rows
