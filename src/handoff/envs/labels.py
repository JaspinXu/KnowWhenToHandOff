"""Should-transfer labels (task T1.2).

The label ``y`` says whether calling ``transfer_to_human_agents`` is the correct
decision in an episode. The tau2 tool docstring defines when to transfer: only if
the user explicitly asks for a human agent, or if the policy and the available
tools cannot solve the issue. ``docs/labels.md`` explains every rule.

- Upstream tasks: ``y = 1`` iff the reference solution calls the transfer tool
  (rule ``gold_transfer_action``); otherwise ``y = 0``.
- Derived hand-off tasks carry the label of the kind they were generated as:
  ``explicit_request`` and ``out_of_scope`` are positives, ``hard_negative``
  (difficult user, solvable issue) is a negative.
- Audited corrections live in an overrides file, each with a written reason.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from handoff.rollout.schema import TRANSFER_TOOL

LABELS_SCHEMA_VERSION = 1
KIND_LABELS = {
    "explicit_request": (True, "explicit_request"),
    "out_of_scope": (True, "out_of_scope"),
    "hard_negative": (False, "hard_negative"),
}


def canonical_json(value: Any) -> str:
    """Stable JSON used for hashing manifests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    """Hex SHA-256 of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def upstream_label(task: Mapping[str, Any]) -> tuple[bool, str]:
    """Label an upstream task from its reference actions."""
    criteria = task.get("evaluation_criteria") or {}
    actions = criteria.get("actions") or []
    has_transfer = any(action.get("name") == TRANSFER_TOOL for action in actions)
    return has_transfer, "gold_transfer_action" if has_transfer else "no_transfer"


def kind_label(kind: str) -> tuple[bool, str]:
    """Label a derived task from its generation kind."""
    if kind not in KIND_LABELS:
        raise ValueError(f"unknown derived task kind {kind!r}")
    return KIND_LABELS[kind]


def build_label_file(
    tasks_by_domain: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    tau2_commit: str,
    overrides: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Label every upstream task; apply audited overrides with their reasons.

    ``overrides`` maps domain -> task_id -> {"should_transfer": bool, "reason": str}.
    """
    overrides = overrides or {}
    domains: dict[str, Any] = {}
    for domain in sorted(tasks_by_domain):
        labels: dict[str, Any] = {}
        for task in tasks_by_domain[domain]:
            task_id = str(task["id"])
            value, rule = upstream_label(task)
            labels[task_id] = {"should_transfer": value, "rule": rule}
        for task_id, override in (overrides.get(domain) or {}).items():
            if task_id not in labels:
                raise ValueError(f"override for unknown task {domain}/{task_id}")
            reason = override.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("every label override needs a written reason")
            if not isinstance(override.get("should_transfer"), bool):
                raise ValueError("override should_transfer must be a boolean")
            labels[task_id] = {
                "should_transfer": override["should_transfer"],
                "rule": "audited_override",
                "reason": reason,
            }
        positives = sum(entry["should_transfer"] for entry in labels.values())
        domains[domain] = {
            "labels": dict(sorted(labels.items())),
            "n_tasks": len(labels),
            "n_positive": positives,
        }
    body = {"tau2_commit": tau2_commit, "domains": domains}
    return {
        "schema_version": LABELS_SCHEMA_VERSION,
        **body,
        "labels_sha256": sha256_text(canonical_json(body)),
    }


def validate_label_file(data: Mapping[str, Any]) -> None:
    """Check schema version and the content hash of a label file."""
    if data.get("schema_version") != LABELS_SCHEMA_VERSION:
        raise ValueError("unsupported label file schema_version")
    body = {"tau2_commit": data.get("tau2_commit"), "domains": data.get("domains")}
    if sha256_text(canonical_json(body)) != data.get("labels_sha256"):
        raise ValueError("label file hash mismatch; regenerate or restore it")


def labels_by_task(data: Mapping[str, Any]) -> dict[tuple[str, str], bool]:
    """Flatten a validated label file into ``(domain, task_id) -> label``."""
    validate_label_file(data)
    result: dict[tuple[str, str], bool] = {}
    for domain, entry in data["domains"].items():
        for task_id, label in entry["labels"].items():
            result[(domain, task_id)] = bool(label["should_transfer"])
    return result
