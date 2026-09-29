"""Reward-hacking watch-list over training rollouts (task T3.5, AGENTS §7).

The agent loop writes one record per training episode. These functions compute
the watch-list from the most recent records and compare it with the B2
evaluation (see :func:`handoff.eval.metrics.hacking_indicators`), flagging any
indicator that drifts more than twice its baseline value.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


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
