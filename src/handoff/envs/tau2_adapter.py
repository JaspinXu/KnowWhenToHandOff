"""Bridge between the frozen manifests and tau2 ``Task`` objects.

This is the only module in ``handoff.envs`` that imports tau2, and it does so
lazily. It loads upstream tasks and split files from the pinned tau2 data
directory, reads the generated hand-off tasks, injects communication styles, and
records file hashes and the tau2 commit for provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from handoff.envs.personas import Persona, PersonaInjector
from handoff.envs.registry import TaskRef, TaskRegistry

PERSONA_SEPARATOR = "::"


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tau2_data_dir() -> Path:
    """Return the tau2 data directory that the installed package uses."""
    from tau2.utils.utils import DATA_DIR

    return Path(DATA_DIR)


def checkout_commit(source: Path) -> str | None:
    """HEAD of the git checkout whose top level is ``source`` (else ``None``).

    Raises if tracked files of the checkout are modified: upstream tau2 is never
    edited (AGENTS hard rule 2).
    """

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(source), *args],
            text=True,
            timeout=30,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        top = Path(git("rev-parse", "--show-toplevel")).resolve()
    except (OSError, subprocess.SubprocessError):
        return None
    if top != source.resolve():
        return None
    if git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(
            f"the tau2 checkout {source} has modified tracked files; restore it "
            "(upstream tau2 is never edited)"
        )
    return git("rev-parse", "HEAD")


def tau2_commit() -> str:
    """Return the commit of the tau2 source tree that is actually imported.

    tau2 is installed editable from a git checkout (the cluster setup job and the
    test environments do this), so the commit is read from that checkout.
    ``TAU2_COMMIT`` must agree with it when both exist; it is the only source for
    an install without a checkout.
    """
    import tau2

    source = Path(tau2.__file__).resolve().parents[2]
    observed = checkout_commit(source)
    expected = os.environ.get("TAU2_COMMIT", "").strip() or None
    if observed and expected and observed != expected:
        raise RuntimeError(
            f"tau2 at {source} is at commit {observed} but TAU2_COMMIT is "
            f"{expected}; rerun the setup job"
        )
    commit = observed or expected
    if commit is None:
        raise RuntimeError(
            "cannot determine the tau2 commit: install tau2 editable from a git "
            "checkout (slurm/submit.sh setup) or set TAU2_COMMIT"
        )
    return commit


# Files of a tau2 domain that determine its tasks, splits, policy and database.
DOMAIN_FILES = {
    "tasks": "tasks.json",
    "split_tasks": "split_tasks.json",
    "policy": "policy.md",
    "db": "db.json",
}


@dataclass(frozen=True)
class DomainSource:
    """Upstream tasks and official splits of one domain at the pinned commit."""

    domain: str
    tasks: list[dict[str, Any]]
    official_splits: dict[str, list[str]]
    file_sha256: dict[str, str]


def domain_file_hashes(domain: str) -> dict[str, str]:
    """SHA-256 of each domain file that exists (see :data:`DOMAIN_FILES`)."""
    domain_dir = tau2_data_dir() / "tau2" / "domains" / domain
    return {
        key: file_sha256(domain_dir / name)
        for key, name in DOMAIN_FILES.items()
        if (domain_dir / name).exists()
    }


def load_domain_source(domain: str) -> DomainSource:
    """Load serialized upstream tasks, official splits and file hashes."""
    from tau2.registry import registry

    tasks = registry.get_tasks_loader(domain)(task_split_name=None)
    splits_loader = registry.get_task_splits_loader(domain)
    if splits_loader is None:
        raise ValueError(f"tau2 domain {domain!r} has no official splits")
    splits = {name: [str(i) for i in ids] for name, ids in splits_loader().items()}
    return DomainSource(
        domain=domain,
        tasks=[task_to_dict(task) for task in tasks],
        official_splits=splits,
        file_sha256=domain_file_hashes(domain),
    )


def task_to_dict(task: Any) -> dict[str, Any]:
    """Serialize a tau2 ``Task`` to JSON-compatible primitives."""
    return json.loads(task.model_dump_json())


def task_from_dict(data: Mapping[str, Any]) -> Any:
    """Validate a serialized task with tau2's own schema."""
    from tau2.data_model.tasks import Task

    return Task.model_validate(dict(data))


def read_task_file(path: Path) -> list[dict[str, Any]]:
    """Read a JSON list of serialized tasks."""
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of tasks")
    return data


def write_task_file(path: Path, tasks: Sequence[Mapping[str, Any]]) -> str:
    """Write tasks as pretty JSON (LF) and return the file's SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(list(tasks), stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return file_sha256(path)


class TaskStore:
    """Resolve :class:`TaskRef` objects to serialized tau2 tasks."""

    def __init__(
        self,
        upstream: Mapping[str, Sequence[Mapping[str, Any]]],
        derived: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        """Index upstream and derived tasks by ``(domain, task_id)``."""
        self._tasks: dict[tuple[str, str], dict[str, Any]] = {}
        for collection in (upstream, derived or {}):
            for domain, tasks in collection.items():
                for task in tasks:
                    key = (domain, str(task["id"]))
                    if key in self._tasks:
                        raise ValueError(f"duplicate task {key}")
                    self._tasks[key] = dict(task)

    @classmethod
    def from_tau2(
        cls,
        domains: Sequence[str],
        handoff_dir: Path | None = None,
        handoff_manifest: Mapping[str, Any] | None = None,
        *,
        split_manifest: Mapping[str, Any] | None = None,
    ) -> TaskStore:
        """Load upstream tasks from tau2 and derived tasks from ``handoff_dir``.

        With a split manifest, tau2's task, split, policy and database files must
        have the hashes recorded in it; with a hand-off manifest, each derived
        task file must match the hash recorded there.
        """
        sources = {domain: load_domain_source(domain) for domain in domains}
        if split_manifest is not None:
            for domain in domains:
                expected = split_manifest["domains"][domain]["file_sha256"]
                changed = sorted(
                    key
                    for key, digest in expected.items()
                    if sources[domain].file_sha256.get(key) != digest
                )
                if changed:
                    raise ValueError(
                        f"tau2 {domain} data differ from the split manifest "
                        f"({', '.join(changed)}); check TAU2_DATA_DIR and the "
                        "tau2 checkout"
                    )
        upstream = {domain: source.tasks for domain, source in sources.items()}
        derived: dict[str, list[dict[str, Any]]] = {}
        if handoff_dir is not None:
            for domain in domains:
                path = handoff_dir / f"{domain}_tasks.json"
                if not path.exists():
                    continue
                if handoff_manifest is not None:
                    expected = handoff_manifest["domains"][domain]["tasks_file_sha256"]
                    if file_sha256(path) != expected:
                        raise ValueError(f"{path} does not match the hand-off manifest")
                derived[domain] = read_task_file(path)
        return cls(upstream, derived)

    def get(self, ref: TaskRef) -> dict[str, Any]:
        """Return a deep-copyable serialized task for a reference."""
        try:
            return self._tasks[ref.key]
        except KeyError:
            raise KeyError(f"task {ref.domain}/{ref.task_id} not found") from None


def load_frozen_tasks(
    paths: Mapping[str, Any], domains: Sequence[str]
) -> tuple[TaskRegistry, TaskStore]:
    """The frozen registry and its tau2 tasks, checked against the manifests.

    ``paths`` is the resolved ``paths`` config group. The hand-off manifest is
    optional (before the task sets exist, only upstream tasks are available).
    Raises ``ValueError`` when the installed tau2 commit or data differ from the
    ones the manifests were built from.
    """
    handoff_path = Path(paths["handoff_manifest"])
    registry = TaskRegistry.from_files(
        Path(paths["splits_manifest"]),
        Path(paths["labels_file"]),
        handoff_path if handoff_path.exists() else None,
    )
    store = TaskStore.from_tau2(
        domains,
        Path(paths["handoff_tasks_dir"]) if registry.handoff_manifest else None,
        registry.handoff_manifest,
        split_manifest=registry.split_manifest,
    )
    commit = tau2_commit()
    if registry.split_manifest["tau2_commit"] != commit:
        raise ValueError(
            f"installed tau2 {commit} differs from the manifest's "
            f"{registry.split_manifest['tau2_commit']}"
        )
    return registry, store


def episode_task_id(task_id: str, persona: str) -> str:
    """Task ID used inside a tau2 run that mixes personas."""
    return f"{task_id}{PERSONA_SEPARATOR}{persona}"


def split_episode_task_id(episode_id: str) -> tuple[str, str]:
    """Inverse of :func:`episode_task_id`."""
    task_id, separator, persona = episode_id.rpartition(PERSONA_SEPARATOR)
    if not separator:
        raise ValueError(f"not an episode task id: {episode_id!r}")
    return task_id, persona


def styled_task(task: Mapping[str, Any], persona: Persona) -> dict[str, Any]:
    """Inject a style card and suffix the ID with the persona name."""
    styled = PersonaInjector().apply(dict(task), persona)
    styled["id"] = episode_task_id(str(task["id"]), persona.id)
    return styled


def validate_task_set(
    tasks: Sequence[Mapping[str, Any]], personas: Mapping[str, Persona]
) -> list[str]:
    """Problems that would otherwise stop an evaluation or training run later.

    Every task must accept every communication style, validate with tau2's
    schema once styled, and have a deterministic reward basis.
    """
    from handoff.rollout.tau2_convert import assert_deterministic_reward_basis

    problems: list[str] = []
    for task in tasks:
        for persona in personas.values():
            try:
                tau2_task = task_from_dict(styled_task(task, persona))
                assert_deterministic_reward_basis(tau2_task)
            except ValueError as error:
                message = str(error).splitlines()[0]
                problems.append(
                    f"task {task.get('id')} with style {persona.id}: {message}"
                )
                break
    return problems
