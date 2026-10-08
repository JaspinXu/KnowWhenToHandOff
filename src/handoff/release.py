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


# --------------------------------------------------------------------------- Kaggle
KAGGLE_SLUG = "knowwhentohandoff-data"
KAGGLE_TITLE = "KnowWhenToHandOff: hand-off tasks and runs"
KAGGLE_SUBTITLE = "tau2-bench hand-off task set, evaluation records and GRPO rollouts"
HF_DATASET_URL = "https://huggingface.co/datasets/JaspinXu/KnowWhenToHandOff-data"


def strip_front_matter(text: str) -> str:
    """Drop a leading YAML block (``---`` ... ``---``) from a Markdown card."""
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            return text[end + 5 :].lstrip("\n")
    return text


KAGGLE_LOADING = """## Loading

On Kaggle the files are under `/kaggle/input/knowwhentohandoff-data/`:

```python
import json
from pathlib import Path

import pandas as pd

root = Path("/kaggle/input/knowwhentohandoff-data")
runs = pd.read_json(root / "runs.jsonl", lines=True)            # run index
airline = json.loads((root / "tasks/v1/airline_tasks.json").read_text())
run = root / "runs" / "R2-final-test_handoff-s0-20261007"
dialogues = pd.read_json(run / "trajectories.jsonl", lines=True)
```
"""
KAGGLE_KEYWORDS = [
    "artificial intelligence",
    "nlp",
    "reinforcement learning",
    "text",
    "computer science",
    "deep learning",
]
KAGGLE_FILES = {
    "runs.jsonl": "One row per released run: run ID, file count and headline "
    "metrics (success, hand-off F1, over-escalation).",
    "tasks/v1/airline_tasks.json": "Airline hand-off tasks in tau2-bench's task "
    "schema (explicit request, out of scope, hard negative).",
    "tasks/v1/retail_tasks.json": "Retail hand-off tasks in tau2-bench's task "
    "schema (explicit request, out of scope, hard negative).",
    "tasks/manifests/labels_v1.json": "Should-transfer label and rule for every "
    "upstream tau2-bench task.",
    "tasks/manifests/splits_v1.json": "Train / dev / test task IDs per domain, "
    "with hashes.",
    "tasks/manifests/handoff_tasks_v1.json": "Every derived hand-off task with its "
    "source task, split, kind and label.",
    "tasks/manifests/label_overrides_v1.json": "Audited label corrections with "
    "reasons.",
    "sft/b2_v1/records_meta.jsonl": "Which teacher dialogues were kept for SFT "
    "and why.",
}
KAGGLE_SOURCES = (
    "Generated with the KnowWhenToHandOff code "
    "(https://github.com/JaspinXu/KnowWhenToHandOff) on the airline and retail "
    "domains of tau2-bench (https://github.com/sierra-research/tau2-bench, MIT); "
    "users simulated by Qwen3-30B-A3B-Instruct-2507, agents are Qwen3 models. "
    f"No real user data. Primary copy: {HF_DATASET_URL}."
)


def kaggle_metadata(user: str, card: str) -> dict[str, object]:
    """``dataset-metadata.json`` for the Kaggle mirror of the dataset release.

    The description is the dataset card without its YAML block and title (Kaggle
    shows the title itself), with a Kaggle-specific *Loading* section. Kaggle's
    licence list has no MIT entry, so the licence is ``other`` and the
    description states it. Tags that Kaggle does not know are ignored by Kaggle.
    """
    body = strip_front_matter(card)
    if body.startswith("# "):
        body = body.split("\n", 1)[1].lstrip("\n")
    if "## Loading" in body:
        head, rest = body.split("## Loading", 1)
        tail = rest[rest.find("\n## ") + 1 :] if "\n## " in rest else ""
        body = head + KAGGLE_LOADING + "\n" + tail
    description = body.rstrip() + (
        f"\n\nPrimary copy and model weights: {HF_DATASET_URL}. "
        "Licence: MIT (see *Licence and attribution* above).\n"
    )
    meta: dict[str, object] = {
        "title": KAGGLE_TITLE,
        "subtitle": KAGGLE_SUBTITLE,
        "id": f"{user}/{KAGGLE_SLUG}",
        "licenses": [{"name": "other"}],
        "keywords": KAGGLE_KEYWORDS,
        "description": description,
        "resources": [
            {"path": path, "description": text} for path, text in KAGGLE_FILES.items()
        ],
        "expectedUpdateFrequency": "never",
        "userSpecifiedSources": KAGGLE_SOURCES,
    }
    if not 6 <= len(KAGGLE_TITLE) <= 50 or not 20 <= len(KAGGLE_SUBTITLE) <= 80:
        raise ValueError("Kaggle title must be 6-50 and subtitle 20-80 characters")
    return meta


KAGGLE_NOTEBOOK = "knowwhentohandoff-quickstart"
KAGGLE_NOTEBOOK_TITLE = "KnowWhenToHandOff quickstart"  # Kaggle derives the slug


def kaggle_kernel_metadata(user: str, code_file: str) -> dict[str, object]:
    """``kernel-metadata.json`` for the example notebook (private, CPU, offline)."""
    return {
        "id": f"{user}/{KAGGLE_NOTEBOOK}",
        "title": KAGGLE_NOTEBOOK_TITLE,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [f"{user}/{KAGGLE_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
