"""Download model snapshots to scratch and record their immutable revisions.

Run on the cluster (never on a laptop): snapshots go to ``$HF_HOME`` on scratch
and ``<scratch>/state/models.lock.json`` maps each model config name to its
repository, resolved commit SHA and local path. Serving and provenance read the
lock, so every run records exactly which weights it used.

Example:
    python scripts/download_models.py qwen3_4b qwen3_30b_a3b
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import OmegaConf

from handoff.cli import CliError, compose_config, fail, repo_root, require_scratch
from handoff.ops.runs import read_models_lock

ROOT = repo_root(__file__)
IGNORE = ["original/*", "*.gguf", "*.pth", "*.pt"]


def main() -> int:
    """Download each requested model and update the lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="model config names (configs/model)")
    parser.add_argument("--overrides", nargs="*", default=[], help="paths overrides")
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi, snapshot_download

        config = compose_config(ROOT, "base", args.overrides)
        require_scratch(config)
        lock_path = Path(config.paths.models_lock)
        lock = read_models_lock(lock_path)
        api = HfApi()
        for name in args.models:
            model_file = ROOT / "configs" / "model" / f"{name}.yaml"
            if not model_file.exists():
                raise CliError(f"unknown model config {name!r}")
            model = OmegaConf.load(model_file)
            repository = str(model.repository)
            if repository == "???":
                raise CliError(f"model {name!r} has no repository (use local_path)")
            requested = str(model.get("download_revision", "main"))
            ignore = list(model.get("download_ignore") or IGNORE)
            sha = api.model_info(repository, revision=requested).sha
            local = snapshot_download(
                repo_id=repository, revision=sha, ignore_patterns=ignore
            )
            lock[name] = {
                "repository": repository,
                "requested_revision": requested,
                "revision": sha,
                "local_path": str(local),
                "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "hf_home": os.environ.get("HF_HOME", ""),
            }
            print(f"{name}: {repository}@{sha} -> {local}")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = lock_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        temporary.replace(lock_path)
    except CliError as error:
        fail(str(error))
    print(f"models lock: {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
