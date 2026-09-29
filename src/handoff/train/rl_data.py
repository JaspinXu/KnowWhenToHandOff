"""Rows of the verl RL dataset (tasks T3.3–T3.4).

Each row is one (task, communication style) pair. The fully serialized,
style-injected tau2 task travels in ``extra_info.task_json`` so the agent loop
needs no data files; ``data_source`` groups validation metrics by domain and
task kind. The ``prompt`` column is a placeholder required by verl's dataset:
the agent loop builds the real prompt from the tau2 system prompt and the
simulated user's first message.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from handoff.envs.personas import Persona
from handoff.envs.registry import TaskRef
from handoff.envs.tau2_adapter import styled_task

AGENT_NAME = "tau2_agent"


def stable_seed(*parts: str) -> int:
    """A reproducible 31-bit seed from strings (used for greedy validation)."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def rl_rows(
    refs: Sequence[TaskRef],
    tasks: Mapping[tuple[str, str], Mapping[str, Any]],
    personas: Mapping[str, Persona],
    persona_ids: Sequence[str],
    *,
    allowed_splits: Sequence[str],
) -> list[dict[str, Any]]:
    """Build dataset rows, refusing any task outside ``allowed_splits``."""
    rows: list[dict[str, Any]] = []
    for ref in refs:
        if ref.split not in allowed_splits:
            raise ValueError(
                f"{ref.domain}/{ref.task_id} is {ref.split}; RL data may only use "
                f"{list(allowed_splits)}"
            )
        for persona_id in persona_ids:
            styled = styled_task(tasks[ref.key], personas[persona_id])
            rows.append(
                {
                    "data_source": f"tau2/{ref.domain}/{ref.kind}",
                    "agent_name": AGENT_NAME,
                    "prompt": [
                        {
                            "role": "user",
                            "content": f"tau2 {ref.domain} task {ref.task_id} "
                            f"({persona_id})",
                        }
                    ],
                    "ability": "customer_service",
                    "reward_model": {"style": "rule", "ground_truth": ""},
                    "extra_info": {
                        "index": len(rows),
                        "domain": ref.domain,
                        "task_id": ref.task_id,
                        "source_task_id": ref.source_task_id,
                        "split": ref.split,
                        "kind": ref.kind,
                        "persona": persona_id,
                        "should_transfer": ref.should_transfer,
                        "eval_seed": stable_seed(ref.domain, ref.task_id, persona_id),
                        "task_json": json.dumps(styled, ensure_ascii=False),
                    },
                }
            )
    return rows


def rows_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts by domain, kind, split and persona for the data manifest."""
    summary: dict[str, dict[str, int]] = {}
    for key in ("domain", "kind", "split", "persona"):
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row["extra_info"][key])
            counts[value] = counts.get(value, 0) + 1
        summary[key] = dict(sorted(counts.items()))
    return {"n_rows": len(rows), **summary}
