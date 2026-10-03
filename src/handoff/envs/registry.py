"""Frozen task registry and split manifests (task T1.1).

The split manifest fixes, per domain, which upstream tasks are train, dev and
test. Train and test come from tau2's official ``split_tasks.json`` at the pinned
commit; dev is carved from train once, stratified by the should-transfer label,
with a recorded seed. The hand-off manifest (T1.3) lists derived tasks, each of
which inherits the split of its source task (lineage-aware separation).

The registry only reads manifests; it never touches tau2 or the network. Use
:mod:`handoff.envs.tau2_adapter` to turn a :class:`TaskRef` into a tau2 ``Task``.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from handoff.envs.labels import canonical_json, sha256_text
from handoff.rollout.schema import EPISODE_KINDS

SPLITS = ("train", "dev", "test")
SPLIT_SCHEMA_VERSION = 1
HANDOFF_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TaskRef:
    """One task of the frozen protocol, with lineage and label."""

    domain: str
    task_id: str
    split: str
    source_task_id: str
    kind: str = "upstream"
    should_transfer: bool = False
    template_id: str | None = None

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}")
        if self.kind not in EPISODE_KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.kind == "upstream" and self.source_task_id != self.task_id:
            raise ValueError("upstream tasks are their own source")

    @property
    def key(self) -> tuple[str, str]:
        """Return ``(domain, task_id)``."""
        return (self.domain, self.task_id)


def carve_dev(
    train_ids: Sequence[str],
    *,
    fraction: float,
    seed: int,
    labels: Mapping[str, bool] | None = None,
) -> tuple[list[str], list[str]]:
    """Split official train IDs into (train, dev), stratified by label.

    Deterministic for a given seed: IDs are sorted, then shuffled per label group
    with ``random.Random(f"{seed}:{label}")`` and the first ``round(fraction * n)``
    of each group go to dev.
    """
    if not 0 <= fraction < 1:
        raise ValueError("dev fraction must be in [0, 1)")
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("duplicate train task IDs")
    labels = labels or {}
    groups: dict[bool, list[str]] = {True: [], False: []}
    for task_id in sorted(train_ids, key=_natural_key):
        groups[bool(labels.get(task_id, False))].append(task_id)
    dev: list[str] = []
    for label, ids in groups.items():
        rng = random.Random(f"{seed}:{label}")
        shuffled = list(ids)
        rng.shuffle(shuffled)
        dev.extend(shuffled[: round(fraction * len(ids))])
    dev_set = set(dev)
    train = [task_id for task_id in train_ids if task_id not in dev_set]
    return sorted(train, key=_natural_key), sorted(dev, key=_natural_key)


def _natural_key(task_id: str) -> tuple[int, str]:
    return (int(task_id), "") if task_id.isdigit() else (1 << 30, task_id)


def build_split_manifest(
    domains: Mapping[str, Mapping[str, Any]],
    *,
    dev_fraction: float,
    seed: int,
    tau2_commit: str,
    labels_sha256: str,
) -> dict[str, Any]:
    """Create the frozen split manifest.

    ``domains`` maps a domain to ``{"official_splits": {...}, "labels": {...},
    "file_sha256": {"tasks": ..., "split_tasks": ...}, "excluded": [...]}`` where
    official splits contain at least ``train`` and ``test``. ``excluded`` lists
    upstream tasks left out of every split (tasks whose reward needs an LLM
    judge); they are recorded in the manifest.
    """
    body_domains: dict[str, Any] = {}
    for domain in sorted(domains):
        source = domains[domain]
        official = source["official_splits"]
        excluded = sorted(
            {str(i) for i in source.get("excluded", [])}, key=_natural_key
        )
        train_ids = [str(i) for i in official["train"] if str(i) not in excluded]
        test_ids = [str(i) for i in official["test"] if str(i) not in excluded]
        overlap = set(train_ids) & set(test_ids)
        if overlap:
            raise ValueError(
                f"{domain}: official train/test overlap: {sorted(overlap)}"
            )
        labels = {str(k): bool(v) for k, v in source.get("labels", {}).items()}
        train, dev = carve_dev(
            train_ids, fraction=dev_fraction, seed=seed, labels=labels
        )
        splits = {
            "train": train,
            "dev": dev,
            "test": sorted(test_ids, key=_natural_key),
        }
        body_domains[domain] = {
            "splits": splits,
            "counts": {name: len(ids) for name, ids in splits.items()},
            "official_counts": {name: len(ids) for name, ids in official.items()},
            "excluded": excluded,
            "positives": {
                name: sum(labels.get(i, False) for i in ids)
                for name, ids in splits.items()
            },
            "file_sha256": dict(source.get("file_sha256", {})),
        }
    body = {
        "tau2_commit": tau2_commit,
        "labels_sha256": labels_sha256,
        "dev_fraction": dev_fraction,
        "seed": seed,
        "domains": body_domains,
    }
    manifest = {"schema_version": SPLIT_SCHEMA_VERSION, **body}
    manifest["manifest_sha256"] = sha256_text(canonical_json(body))
    validate_split_manifest(manifest)
    return manifest


def _split_body(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("tau2_commit", "labels_sha256", "dev_fraction", "seed", "domains")
    return {key: manifest.get(key) for key in keys}


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    """Check schema, hash, and that train, dev and test are disjoint per domain."""
    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError("unsupported split manifest schema_version")
    if sha256_text(canonical_json(_split_body(manifest))) != manifest.get(
        "manifest_sha256"
    ):
        raise ValueError("split manifest hash mismatch")
    for domain, entry in manifest["domains"].items():
        seen: dict[str, str] = {}
        for split in SPLITS:
            for task_id in entry["splits"][split]:
                if task_id in seen:
                    raise ValueError(
                        f"{domain}/{task_id} is in both {seen[task_id]} and {split}"
                    )
                seen[task_id] = split


def build_handoff_manifest(
    generated: Mapping[str, Sequence[TaskRef]],
    *,
    generator_version: str,
    split_manifest_sha256: str,
    tasks_file_sha256: Mapping[str, str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the hand-off manifest listing derived tasks and their lineage."""
    domains: dict[str, Any] = {}
    for domain in sorted(generated):
        refs = sorted(generated[domain], key=lambda r: r.task_id)
        domains[domain] = {
            "tasks": [
                {
                    "task_id": ref.task_id,
                    "source_task_id": ref.source_task_id,
                    "split": ref.split,
                    "kind": ref.kind,
                    "template_id": ref.template_id,
                    "should_transfer": ref.should_transfer,
                }
                for ref in refs
            ],
            "tasks_file_sha256": tasks_file_sha256[domain],
            "counts": {
                split: {
                    kind: sum(1 for r in refs if r.split == split and r.kind == kind)
                    for kind in EPISODE_KINDS[1:]
                }
                for split in SPLITS
            },
        }
    body = {
        "generator_version": generator_version,
        "split_manifest_sha256": split_manifest_sha256,
        "config": dict(config),
        "domains": domains,
    }
    manifest = {"schema_version": HANDOFF_SCHEMA_VERSION, **body}
    manifest["manifest_sha256"] = sha256_text(canonical_json(body))
    return manifest


def _handoff_body(manifest: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("generator_version", "split_manifest_sha256", "config", "domains")
    return {key: manifest.get(key) for key in keys}


def validate_handoff_manifest(
    manifest: Mapping[str, Any], split_manifest: Mapping[str, Any]
) -> None:
    """Check the hash and that every variant inherits its source task's split."""
    if manifest.get("schema_version") != HANDOFF_SCHEMA_VERSION:
        raise ValueError("unsupported hand-off manifest schema_version")
    if sha256_text(canonical_json(_handoff_body(manifest))) != manifest.get(
        "manifest_sha256"
    ):
        raise ValueError("hand-off manifest hash mismatch")
    if manifest["split_manifest_sha256"] != split_manifest["manifest_sha256"]:
        raise ValueError("hand-off manifest was built from a different split manifest")
    for domain, entry in manifest["domains"].items():
        splits = split_manifest["domains"][domain]["splits"]
        split_of = {tid: name for name, ids in splits.items() for tid in ids}
        for task in entry["tasks"]:
            source_split = split_of.get(task["source_task_id"])
            if source_split is None:
                raise ValueError(f"{domain}/{task['task_id']}: unknown source task")
            if source_split != task["split"]:
                raise ValueError(
                    f"{domain}/{task['task_id']}: split {task['split']} differs from "
                    f"source split {source_split}"
                )
            if task["task_id"] in split_of:
                raise ValueError(
                    f"{domain}/{task['task_id']} collides with upstream ID"
                )


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write pretty JSON with a trailing newline (LF endings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, sort_keys=False)
        stream.write("\n")


class TaskRegistry:
    """Read-only view over the split, label and hand-off manifests."""

    def __init__(
        self,
        split_manifest: Mapping[str, Any],
        labels: Mapping[tuple[str, str], bool],
        handoff_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        """Validate manifests and index every task."""
        validate_split_manifest(split_manifest)
        if handoff_manifest is not None:
            validate_handoff_manifest(handoff_manifest, split_manifest)
        self.split_manifest = split_manifest
        self.handoff_manifest = handoff_manifest
        self._refs: dict[tuple[str, str], TaskRef] = {}
        for domain, entry in split_manifest["domains"].items():
            for split, ids in entry["splits"].items():
                for task_id in ids:
                    if (domain, task_id) not in labels:
                        raise ValueError(f"no label for {domain}/{task_id}")
                    self._add(
                        TaskRef(
                            domain=domain,
                            task_id=task_id,
                            split=split,
                            source_task_id=task_id,
                            should_transfer=labels[(domain, task_id)],
                        )
                    )
        if handoff_manifest is not None:
            for domain, entry in handoff_manifest["domains"].items():
                for task in entry["tasks"]:
                    self._add(
                        TaskRef(
                            domain=domain,
                            task_id=task["task_id"],
                            split=task["split"],
                            source_task_id=task["source_task_id"],
                            kind=task["kind"],
                            should_transfer=bool(task["should_transfer"]),
                            template_id=task.get("template_id"),
                        )
                    )

    def _add(self, ref: TaskRef) -> None:
        if ref.key in self._refs:
            raise ValueError(f"duplicate task {ref.domain}/{ref.task_id}")
        self._refs[ref.key] = ref

    @classmethod
    def from_files(
        cls,
        split_path: Path,
        labels_path: Path,
        handoff_path: Path | None = None,
    ) -> TaskRegistry:
        """Load manifests from disk."""
        from handoff.envs.labels import labels_by_task

        split_manifest = read_json(split_path)
        label_file = read_json(labels_path)
        if split_manifest["labels_sha256"] != label_file.get("labels_sha256"):
            raise ValueError("split manifest was built from a different label file")
        handoff = read_json(handoff_path) if handoff_path is not None else None
        return cls(split_manifest, labels_by_task(label_file), handoff)

    @property
    def domains(self) -> list[str]:
        """Domains present in the split manifest."""
        return sorted(self.split_manifest["domains"])

    def get(self, domain: str, task_id: str) -> TaskRef:
        """Return one task reference."""
        try:
            return self._refs[(domain, task_id)]
        except KeyError:
            raise KeyError(f"unknown task {domain}/{task_id}") from None

    def load(
        self,
        domain: str,
        split: str,
        include_handoff_set: bool = False,
        *,
        kinds: Iterable[str] | None = None,
    ) -> list[TaskRef]:
        """Return tasks of one domain and split (README contract).

        ``include_handoff_set`` adds the derived tasks of the same split.
        ``kinds`` filters by episode kind (for example only ``out_of_scope``).
        """
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}")
        allowed = set(kinds) if kinds is not None else None
        refs = [
            ref
            for ref in self._refs.values()
            if ref.domain == domain
            and ref.split == split
            and (include_handoff_set or ref.kind == "upstream")
            and (allowed is None or ref.kind in allowed)
        ]
        return sorted(refs, key=lambda r: (_natural_key(r.source_task_id), r.task_id))

    def all_refs(self) -> list[TaskRef]:
        """Every registered task."""
        return list(self._refs.values())

    def assert_lineage_disjoint(self) -> None:
        """Assert that no source task contributes to more than one split."""
        split_of_source: dict[tuple[str, str], str] = {}
        for ref in self._refs.values():
            key = (ref.domain, ref.source_task_id)
            previous = split_of_source.setdefault(key, ref.split)
            if previous != ref.split:
                raise AssertionError(
                    f"source {ref.domain}/{ref.source_task_id} appears in "
                    f"{previous} and {ref.split}"
                )
