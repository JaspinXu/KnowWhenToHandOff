"""Package the study's records for a public dataset release (T4.4).

The release has three parts, all small text files:

- ``tasks/``: the hand-off task set and the split, label and hand-off manifests
  from the repository (``data/``);
- ``runs/<run id>/``: every completed real run whose ID starts with one of
  :data:`RUN_PREFIXES` (evaluations, teacher rollouts, SFT and RL training runs),
  with its trajectories, metrics, summary, provenance and resolved config, and
  for RL runs the per-episode training rollouts;
- ``sft/<dataset>/``: the SFT dataset's manifest and per-record metadata (which
  teacher episodes were kept and why; token arrays are left out).

Model weights and resumable checkpoints are never copied. Text files are
scrubbed of absolute scratch, repository and home paths before they are written.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

RUN_PREFIXES = ("B0-", "B1-", "B2-", "R1-", "R2-", "TEACH-")
RUN_FILES = (
    "trajectories.jsonl",
    "metrics.json",
    "summary.md",
    "provenance.json",
    "config.json",
    "completion.json",
    "infra_failures.jsonl",
)
RUN_DIRS = ("rollouts",)
SFT_FILES = ("manifest.json", "records_meta.jsonl")
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml", ".csv"}


@dataclass(frozen=True)
class Item:
    """One file of the release: where it comes from and where it goes."""

    source: Path
    target: str  # path relative to the release root, POSIX separators

    @property
    def size(self) -> int:
        return self.source.stat().st_size


def scrub(text: str, replacements: Sequence[tuple[str, str]]) -> str:
    """Replace private absolute paths, longest first, also as JSON-escaped text."""
    pairs = []
    for old, new in replacements:
        if old:
            pairs.append((old, new))
            escaped = json.dumps(old)[1:-1]
            if escaped != old:
                pairs.append((escaped, new))
    for old, new in sorted(pairs, key=lambda pair: -len(pair[0])):
        text = text.replace(old, new)
    return text


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def release_runs(runs_dir: Path, prefixes: Sequence[str] = RUN_PREFIXES) -> list[Path]:
    """Real runs with a known prefix that did not end incomplete or failed."""
    selected = []
    for run_dir in sorted(Path(runs_dir).iterdir()) if Path(runs_dir).is_dir() else []:
        if not run_dir.is_dir() or not run_dir.name.startswith(tuple(prefixes)):
            continue
        provenance = _json(run_dir / "provenance.json")
        if not provenance or provenance.get("execution_kind") != "real":
            continue
        completion = _json(run_dir / "completion.json")
        if completion is not None and completion.get("status") != "completed":
            continue
        selected.append(run_dir)
    return selected


def plan(
    repo: Path, runs_dir: Path, datasets_dir: Path, sft_datasets: Iterable[str]
) -> list[Item]:
    """Every file of the release, in a stable order."""
    items: list[Item] = []
    for folder in ("data/handoff_tasks/v1", "data/manifests"):
        for path in sorted((repo / folder).glob("*")):
            if path.is_file():
                items.append(Item(path, f"tasks/{Path(folder).name}/{path.name}"))
    for run_dir in release_runs(runs_dir):
        for name in RUN_FILES:
            if (run_dir / name).is_file():
                items.append(Item(run_dir / name, f"runs/{run_dir.name}/{name}"))
        for sub in RUN_DIRS:
            # Rollout files are named <host>-<pid>.jsonl; publish them as parts.
            for k, path in enumerate(sorted((run_dir / sub).glob("*.jsonl"))):
                items.append(
                    Item(path, f"runs/{run_dir.name}/{sub}/part-{k:03d}.jsonl")
                )
    for name in sft_datasets:
        for file_name in SFT_FILES:
            path = datasets_dir / name / file_name
            if path.is_file():
                items.append(Item(path, f"sft/{name}/{file_name}"))
    return items


def hostnames(items: Sequence[Item]) -> set[str]:
    """Machine names recorded in the released runs (provenance, rollout files)."""
    found: set[str] = set()

    def visit(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "hostname" and isinstance(value, str) and value:
                    found.add(value)
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    for item in items:
        if item.source.name == "provenance.json":
            visit(_json(item.source))
        elif item.source.parent.name in RUN_DIRS and "-" in item.source.stem:
            found.add(item.source.stem.rsplit("-", 1)[0])
    return {name for name in found if len(name) >= 3}


def write(
    items: Sequence[Item],
    out: Path,
    replacements: Sequence[tuple[str, str]],
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Copy (and scrub) the items, plus ``extra`` text files, under ``out``.

    Returns SHA-256 per target; ``SHA256SUMS`` covers every written file.
    """
    digests: dict[str, str] = {}
    for item in items:
        target = out / item.target
        target.parent.mkdir(parents=True, exist_ok=True)
        if item.source.suffix in TEXT_SUFFIXES:
            text = item.source.read_text(encoding="utf-8")
            target.write_text(scrub(text, replacements), encoding="utf-8", newline="\n")
        else:
            shutil.copyfile(item.source, target)
        digests[item.target] = hashlib.sha256(target.read_bytes()).hexdigest()
    for name, text in (extra or {}).items():
        target = out / name
        target.write_text(scrub(text, replacements), encoding="utf-8", newline="\n")
        digests[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    lines = [f"{digest}  {name}\n" for name, digest in sorted(digests.items())]
    (out / "SHA256SUMS").write_text("".join(lines), encoding="utf-8", newline="\n")
    return digests


def run_index(items: Sequence[Item]) -> list[dict[str, object]]:
    """One row per released run: ID, files and, if present, headline metrics."""
    rows: dict[str, dict[str, object]] = {}
    for item in items:
        parts = item.target.split("/")
        if parts[0] != "runs":
            continue
        row = rows.setdefault(parts[1], {"run_id": parts[1], "files": 0})
        row["files"] = int(row["files"]) + 1
        if parts[-1] == "metrics.json":
            metrics = _json(item.source) or {}
            overall = metrics.get("overall", {})
            for key in ("success", "handoff_f1", "over_escalation_rate"):
                value = overall.get(key)
                if isinstance(value, dict):
                    value = value.get("point")
                if isinstance(value, int | float):
                    row[key] = round(float(value), 3)
    return list(rows.values())
