"""Supervised warm start of the policy on teacher token records (T2.2, run B2).

Single GPU: ``python scripts/train_sft.py dataset_name=b2_v1``.
Several GPUs: ``torchrun --nproc_per_node=N scripts/train_sft.py ...
sft.fsdp="full_shard auto_wrap"``.
The final Hugging Face checkpoint is ``<scratch>/runs/<run_id>/final`` and can be
served by vLLM directly for the B2 evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from handoff.cli import (
    CliError,
    compose_config,
    container,
    fail,
    missing_keys,
    repo_root,
    require_scratch,
    setup_logging,
)
from handoff.ops.provenance import create_run_directory, finish_run
from handoff.ops.runs import git_state, hardware_info, make_run_id, read_models_lock
from handoff.train.sft_trainer import SFTSettings, run_sft

ROOT = repo_root(__file__)


def main() -> int:
    """Resolve settings, record provenance on rank 0 and train."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    setup_logging()
    rank = int(os.environ.get("RANK", "0"))
    try:
        config = compose_config(ROOT, "train_sft", args.overrides)
        require_scratch(config)
        missing = missing_keys(config, ["dataset_name"])
        if missing:
            raise CliError("missing settings: " + ", ".join(missing))
        paths = config.paths
        dataset = Path(paths.scratch) / "datasets" / str(config.dataset_name)
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        lock = read_models_lock(Path(paths.models_lock))
        base = lock.get(str(config.base_model))
        if base is None:
            raise CliError(f"{config.base_model} is not in the models lock")
        # One process picks a free ID for a same-day rerun; with several ranks
        # all must agree without coordination, so an existing ID is an error.
        single = int(os.environ.get("WORLD_SIZE", "1")) == 1
        run_id = make_run_id(
            str(config.experiment),
            seed=int(config.seed),
            tags=tuple(str(t) for t in config.tags),
            runs_dir=Path(paths.runs_dir) if single else None,
        )
        run_dir = Path(paths.runs_dir) / run_id
        settings = SFTSettings.from_mapping(
            {
                **container(config.sft),
                "model_path": base["local_path"],
                "train_records": str(dataset / "records.jsonl"),
                "output_dir": str(run_dir),
            }
        )
        if rank == 0:
            git_sha, dirty = git_state(ROOT)
            create_run_directory(
                Path(paths.runs_dir),
                run_id=run_id,
                config={
                    "sft": container(config.sft),
                    "base_model": {"name": config.base_model, **base},
                    "dataset": {
                        "name": str(config.dataset_name),
                        "records_sha256": manifest["records_sha256"],
                        "teacher_runs": manifest["teacher_runs"],
                    },
                },
                provenance={
                    "purpose": "B2 supervised warm start",
                    "execution_kind": str(config.execution_kind),
                    "git_sha": git_sha,
                    "git_dirty": dirty,
                    "tau2_sha": _tau2_sha(paths),
                    "model_revisions": {
                        "policy": base["revision"],
                        "user_sim": _teacher_revision(manifest, paths),
                    },
                    "seeds": {"run": int(config.seed)},
                    "hardware": hardware_info(),
                },
            )
        else:
            _wait_for(run_dir / "provenance.json")
        try:
            final = run_sft(settings)
        except BaseException:
            if rank == 0:
                finish_run(run_dir, status="failed", error_code="SFT_FAILED")
            raise
        if rank == 0:
            finish_run(run_dir, status="completed")
            print(f"checkpoint: {final}")
    except CliError as error:
        fail(str(error))
    return 0


def _wait_for(path: Path, timeout_s: float = 600.0) -> None:
    """Non-zero ranks wait until rank 0 has created the run records."""
    import time

    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() > deadline:
            raise CliError(f"rank 0 did not create {path}")
        time.sleep(2)


def _tau2_sha(paths) -> str:
    manifest = json.loads(Path(paths.splits_manifest).read_text(encoding="utf-8"))
    return str(manifest["tau2_commit"])


def _teacher_revision(manifest: dict, paths) -> str:
    """User-simulator revision of the first teacher run (for provenance)."""
    run_id = manifest["teacher_runs"][0]["run_id"]
    record = json.loads(
        (Path(paths.runs_dir) / run_id / "provenance.json").read_text(encoding="utf-8")
    )
    return str(record["model_revisions"]["user_sim"])


if __name__ == "__main__":
    raise SystemExit(main())
