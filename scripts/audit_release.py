# ruff: noqa: E501  (regexes and report lines are kept on one line)
"""Audit the published release for leaks and file problems (docs/release.md).

Downloads what is public-facing, except model weights, into a temporary
folder on the cluster: every non-weight file of the three model repositories,
the Hugging Face dataset, and the Kaggle mirror. Then it checks

- structure: file counts and types, weights or checkpoints where they should
  not be, very large files, model config keys that carry local paths;
- integrity: every dataset file against ``SHA256SUMS``, and the Kaggle copy
  against the Hugging Face copy;
- leaks: absolute paths, cluster host names, personal or course identifiers,
  student IDs, real e-mail addresses, tokens and keys, private URLs.

Run it as a CPU job after ``hf auth login`` and with the Kaggle token in place;
it prints a short report (hits are shown with secrets masked).

Example:
    python scripts/audit_release.py --kaggle "$HANDOFF_SCRATCH/kaggle-venv/bin/kaggle"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

HF_USER = "JaspinXu"
MODELS = ["R2", "R1", "SFT"]
KAGGLE_ID = "zhaobinxu/knowwhentohandoff-data"
TEXT = {".json", ".jsonl", ".md", ".txt", ".jinja", ".csv", ".yaml", ".yml", ""}
WEIGHTS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".npy", ".npz"}

PATTERNS = {
    "absolute path": r"(?:/home/|/Users/|/scratch/|/hpctmp/|/mnt/|[A-Za-z]:\\\\)[^\s\"']*",
    "cluster host": r"\bxlogin\d*\b|\bxcn[a-z]*\d+\b|\bxgp[a-z]\d+\b|comp\.nus\.edu\.sg|\bnus\.edu\.sg",
    "personal name": r"(?i)zhaobin(?!xu/knowwhentohandoff)|徐肇斌",  # the public Kaggle handle is expected
    "student id": r"\b[Ee]0\d{6}\b|\bA0\d{6}[A-Z]\b",
    "course marker": r"(?i)\bCS ?4246\b|\bCS ?5446\b|\bgroup ?21\b|course project|课程",
    "e-mail (non-example)": r"[A-Za-z0-9._%+-]+@(?!example\.(?:com|org|net)\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "token or key": r"(?i)\bhf_[A-Za-z0-9]{30,}|\bsk-[A-Za-z0-9_-]{20,}|\bKGAT_[A-Za-z0-9]+|api[_-]?key\s*[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_-]{8,}|bearer\s+[A-Za-z0-9._-]{20,}",
    "private URL": r"https?://(?!(?:www\.)?(?:github\.com|huggingface\.co|kaggle\.com|arxiv\.org|docs\.astral\.sh|example\.(?:com|org))\b)[^\s\"'<>)\]]+",
    "IP with port": r"\b(?:\d{1,3}\.){3}\d{1,3}:\d{2,5}\b",
}
SECRET = {"token or key"}


def mask(text: str) -> str:
    return f"{text[:3]}… ({len(text)} chars)"


def scan(root: Path, label: str, report: list[str]) -> None:
    files = [
        p for p in sorted(root.rglob("*")) if p.is_file() and ".cache" not in p.parts
    ]
    kinds = Counter(p.suffix or "(none)" for p in files)
    size = sum(p.stat().st_size for p in files)
    report.append(
        f"\n## {label}: {len(files)} files, {size / 1e6:.1f} MB, types {dict(kinds)}"
    )
    weights = [p for p in files if p.suffix in WEIGHTS]
    if weights and not label.startswith("model"):
        report.append(
            f"  !! weight-like files: {[str(p.relative_to(root)) for p in weights][:10]}"
        )
    big = [(p, p.stat().st_size) for p in files if p.stat().st_size > 100e6]
    for path, n in big:
        report.append(f"  note: large file {path.relative_to(root)} ({n / 1e6:.0f} MB)")
    hits: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    for path in files:
        if path.suffix not in TEXT or path.stat().st_size > 400e6:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = str(path.relative_to(root))
        for name, pattern in PATTERNS.items():
            for match in re.finditer(pattern, text):
                hits[name][rel] += 1
                if len(examples[name]) < 4:
                    value = (
                        mask(match.group(0)) if name in SECRET else match.group(0)[:90]
                    )
                    examples[name].append(f"{rel}: {value}")
    for name in PATTERNS:
        total = sum(hits[name].values())
        if total:
            files_hit = len(hits[name])
            report.append(f"  !! {name}: {total} hits in {files_hit} files")
            report.extend(f"     e.g. {line}" for line in examples[name])
        else:
            report.append(f"  ok {name}: none")


def check_models(api, workdir: Path, report: list[str]) -> None:
    from huggingface_hub import snapshot_download

    for name in MODELS:
        repo = f"{HF_USER}/KnowWhenToHandOff-Qwen3-4B-{name}"
        info = api.model_info(repo, files_metadata=True)
        listing = {s.rfilename: s.size or 0 for s in info.siblings}
        weights = sum(v for k, v in listing.items() if k.endswith(".safetensors"))
        report.append(
            f"\n# model {repo}: {'private' if info.private else 'PUBLIC'}, {len(listing)} files, weights {weights / 1e9:.2f} GB, commits {len(api.list_repo_commits(repo))}"
        )
        report.append(f"  files: {sorted(listing)}")
        local = Path(
            snapshot_download(
                repo, local_dir=workdir / name, ignore_patterns=["*.safetensors"]
            )
        )
        config = json.loads((local / "config.json").read_text())
        for key in ("_name_or_path", "name_or_path"):
            if key in config:
                report.append(f"  !! config.json has {key} = {config[key]!r}")
        tok = json.loads((local / "tokenizer_config.json").read_text())
        if "name_or_path" in tok:
            report.append(
                f"  !! tokenizer_config.json has name_or_path = {tok['name_or_path']!r}"
            )
        report.append(
            f"  config: architectures={config.get('architectures')} dtype={config.get('torch_dtype') or config.get('dtype')} transformers={config.get('transformers_version')}"
        )
        if not (local / "README.md").read_text().startswith("---"):
            report.append("  !! README.md has no YAML metadata")
        scan(local, f"model {name} (non-weight files)", report)


def sums_ok(root: Path, report: list[str], label: str) -> dict[str, str]:
    sums = {}
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        sums[name] = digest
    bad = [
        n
        for n, d in sums.items()
        if not (root / n).is_file()
        or hashlib.sha256((root / n).read_bytes()).hexdigest() != d
    ]
    report.append(
        f"  {'!!' if bad else 'ok'} {label} SHA256SUMS: {len(sums) - len(bad)}/{len(sums)} files match"
        + (f"; bad: {bad[:5]}" if bad else "")
    )
    return sums


def main() -> int:
    """Download, check and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kaggle", default="kaggle", help="path to the Kaggle CLI")
    parser.add_argument("--skip-kaggle", action="store_true")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    report: list[str] = ["# Release audit"]
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        check_models(api, work / "models", report)

        repo = f"{HF_USER}/KnowWhenToHandOff-data"
        info = api.dataset_info(repo)
        report.append(
            f"\n# dataset {repo}: {'private' if info.private else 'PUBLIC'}, {len(info.siblings)} files, commits {len(api.list_repo_commits(repo, repo_type='dataset'))}"
        )
        hf = Path(
            snapshot_download(repo, repo_type="dataset", local_dir=work / "hf-data")
        )
        sums = sums_ok(hf, report, "Hugging Face")
        extra = sorted(
            {
                str(p.relative_to(hf))
                for p in hf.rglob("*")
                if p.is_file() and ".cache" not in p.parts
            }
            - set(sums)
            - {"SHA256SUMS", "README.md", ".gitattributes"}
        )
        if extra:
            report.append(f"  note: files outside SHA256SUMS: {extra}")
        scan(hf, "Hugging Face dataset", report)

        if not args.skip_kaggle:
            kg = work / "kaggle"
            subprocess.run(
                [
                    args.kaggle,
                    "datasets",
                    "download",
                    KAGGLE_ID,
                    "-p",
                    str(kg),
                    "--unzip",
                    "-q",
                ],
                check=True,
            )
            nested = sorted(kg.rglob("*.zip"))
            for archive in nested:
                with zipfile.ZipFile(archive) as z:
                    z.extractall(archive.with_suffix(""))
                archive.unlink()
            report.append(
                f"\n# Kaggle {KAGGLE_ID}: nested archives on download: {[a.name for a in nested] or 'none (folders extracted by Kaggle)'}"
            )
            kaggle_files = {
                str(p.relative_to(kg)): p for p in kg.rglob("*") if p.is_file()
            }
            missing = [n for n in sums if n not in kaggle_files]
            report.append(
                f"  {'!!' if missing else 'ok'} Kaggle has {len(sums) - len(missing)}/{len(sums)} files listed in SHA256SUMS"
                + (f"; missing e.g. {missing[:5]}" if missing else "")
            )
            if not missing:
                sums_ok(kg, report, "Kaggle")
            scan(kg, "Kaggle dataset", report)
    text = "\n".join(report)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
