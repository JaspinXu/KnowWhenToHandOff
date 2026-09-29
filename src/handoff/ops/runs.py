"""Run bookkeeping used by every CLI: IDs, git state, hardware, model identity.

Complements :mod:`handoff.ops.provenance` (immutable run records) with the
helpers needed to fill those records from a real cluster job.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from handoff.ops.provenance import RUN_ID_PATTERN

FINGERPRINT_CACHE = ".handoff_fingerprint.json"
_READ_BLOCK = 1 << 24


def make_run_id(
    experiment: str,
    *,
    seed: int,
    tags: tuple[str, ...] = (),
    day: date | None = None,
    runs_dir: Path | None = None,
) -> str:
    """Build a run ID following the README naming, e.g. ``R2-s1-20261024``.

    With ``runs_dir``, an ID already taken there (an earlier run of the same
    experiment on the same day) gets a ``try2``, ``try3``, ... tag before the
    seed, so a same-day relaunch never collides with the earlier run's records.
    """
    day = day or datetime.now(UTC).date()
    stamp = day.strftime("%Y%m%d")
    run_id = "-".join([experiment, *tags, f"s{seed}", stamp])
    attempt = 1
    while runs_dir is not None and (runs_dir / run_id).exists():
        attempt += 1
        run_id = "-".join([experiment, *tags, f"try{attempt}", f"s{seed}", stamp])
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(f"invalid run id {run_id!r}")
    return run_id


# Paths whose changes cannot affect a run's numbers: the experiment log, notes
# and generated report assets are edited between runs and committed later.
RUN_IRRELEVANT_PATHS = ("docs", "reports")


def git_state(root: Path) -> tuple[str, bool]:
    """Return ``(commit_sha, is_dirty)`` of the checkout at ``root``.

    ``is_dirty`` covers code, configs and data (tracked or untracked), not the
    paths in :data:`RUN_IRRELEVANT_PATHS`.
    """
    sha = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, timeout=20
    ).strip()
    excludes = [f":(exclude){path}" for path in RUN_IRRELEVANT_PATHS]
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--", ".", *excludes],
        text=True,
        timeout=20,
    )
    return sha, bool(status.strip())


def hardware_info() -> dict[str, Any]:
    """Best-effort description of the machine and Slurm allocation."""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=30,
        )
        info["gpus"] = [line.strip() for line in output.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        info["gpus"] = []
    return info


def _checkpoint_files(path: Path) -> list[Path]:
    """Every top-level file of a checkpoint (hidden files such as caches excluded)."""
    return sorted(
        file
        for file in path.iterdir()
        if file.is_file() and not file.name.startswith(".")
    )


def checkpoint_fingerprint(path: Path) -> str:
    """SHA-256 over the names and full contents of a local checkpoint's files.

    Two checkpoints of the same model have identical file names and sizes, so
    every weight byte is hashed. The result is cached in
    ``.handoff_fingerprint.json`` next to the files, keyed by each file's name,
    size and modification time, so only the first call per checkpoint reads the
    weights (tens of seconds for a 4B model). A read-only directory just skips
    the cache.
    """
    if not path.is_dir():
        raise ValueError(f"checkpoint directory not found: {path}")
    files = _checkpoint_files(path)
    if not files:
        raise ValueError(f"checkpoint directory is empty: {path}")
    signature = [
        [file.name, file.stat().st_size, file.stat().st_mtime_ns] for file in files
    ]
    cache = path / FINGERPRINT_CACHE
    try:
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached["version"] == 1 and cached["files"] == signature:
            return str(cached["fingerprint"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    digest = hashlib.sha256()
    for file in files:
        digest.update(f"{file.name}\0{file.stat().st_size}\0".encode())
        with file.open("rb") as stream:
            for block in iter(lambda stream=stream: stream.read(_READ_BLOCK), b""):
                digest.update(block)
    fingerprint = "sha256:" + digest.hexdigest()
    record = {"version": 1, "files": signature, "fingerprint": fingerprint}
    temporary = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(record) + "\n", encoding="utf-8")
        os.replace(temporary, cache)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink()
    return fingerprint


def read_models_lock(path: Path) -> dict[str, Any]:
    """Read the models lock written by ``scripts/download_models.py``."""
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def model_identity(
    model: Mapping[str, Any], models_lock: Mapping[str, Any]
) -> tuple[str, str]:
    """Return ``(local_path, immutable_revision)`` for a model config.

    A ``local_path`` in the model config (a trained checkpoint) wins and is
    fingerprinted; otherwise the downloaded snapshot recorded in the lock is used.
    """
    local = model.get("local_path")
    if local:
        path = Path(str(local))
        return str(path), checkpoint_fingerprint(path)
    name = str(model.get("name"))
    entry = models_lock.get(name)
    if entry is None:
        raise ValueError(
            f"model {name!r} not in the models lock; run scripts/download_models.py"
        )
    return str(entry["local_path"]), str(entry["revision"])


@dataclass(frozen=True)
class LedgerScope:
    """What a test-split evaluation covers, for the one-evaluation rule.

    ``tasks`` are the ``domain/task_id`` strings of the test tasks it runs and
    ``user_sim`` names the simulator (``served_name@revision``): a checkpoint may
    be evaluated once with each simulator (ADR-005).
    """

    policy_revision: str
    user_sim: str
    tasks: tuple[str, ...]
    personas: tuple[str, ...]


def _overlaps(record: Mapping[str, Any], scope: LedgerScope) -> bool:
    """Whether an earlier ledger record covers part of ``scope``.

    A record without task or persona lists overlaps everything (conservative).
    """
    if record.get("policy_revision") != scope.policy_revision:
        return False
    if record.get("user_sim", scope.user_sim) != scope.user_sim:
        return False
    tasks, personas = record.get("tasks"), record.get("personas")
    return (tasks is None or bool(set(tasks) & set(scope.tasks))) and (
        personas is None or bool(set(personas) & set(scope.personas))
    )


def _ledger_records(ledger: Path) -> list[dict[str, Any]]:
    if not ledger.exists():
        return []
    return [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ledger_conflicts(
    ledger: Path, scope: LedgerScope, *, run_id: str
) -> list[dict[str, Any]]:
    """Earlier test evaluations of the same checkpoint that overlap this one.

    AGENTS hard rule 1 allows one test evaluation per frozen checkpoint: a run
    conflicts with every earlier run of the same checkpoint and simulator that
    shares at least one test task *and* one style, whatever its protocol is
    called. Records of ``run_id`` itself (a resumed run) never conflict.
    """
    return [
        record
        for record in _ledger_records(ledger)
        if record.get("run_id") != run_id and _overlaps(record, scope)
    ]


def record_in_ledger(
    ledger: Path,
    scope: LedgerScope,
    *,
    run_id: str,
    eval_name: str,
    forced_reason: str | None = None,
) -> bool:
    """Append a test evaluation to the ledger once; return whether it was new.

    Call it after the run directory exists, so a run refused by provenance
    checks never uses up the checkpoint's test evaluation. ``forced_reason``
    records why an overlapping evaluation was allowed anyway.
    """
    if any(record.get("run_id") == run_id for record in _ledger_records(ledger)):
        return False
    entry = {
        **asdict(scope),
        "eval_name": eval_name,
        "run_id": run_id,
        "forced_reason": forced_reason,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry) + "\n")
    return True


_LOG_HEADER = "| Date | Run ID | Purpose | Headline metrics (95% CI) | Decision |"


def append_experiment_log(
    path: Path, *, run_id: str, purpose: str, headline: str, decision: str
) -> None:
    """Append one row to ``docs/experiment-log.md`` (AGENTS section 7)."""
    cells = [
        datetime.now(UTC).date().isoformat(),
        run_id,
        purpose,
        headline,
        decision,
    ]
    for cell in cells:
        if "|" in cell or "\n" in cell:
            raise ValueError("experiment-log cells must not contain '|' or newlines")
    if not path.exists() or _LOG_HEADER not in path.read_text(encoding="utf-8"):
        raise ValueError(f"{path} is missing the experiment-log table header")
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write("| " + " | ".join(cells) + " |\n")


def safe_name(text: str) -> str:
    """Filesystem-safe version of a label."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "unnamed"
