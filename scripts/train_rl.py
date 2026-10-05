"""Launch GRPO training with tau2 episodes (T3.3–T3.4).

Builds the train (train split) and validation (dev split) datasets from the
frozen manifests, writes the agent-loop settings, validates the verl overrides
against verl's own config, records provenance and runs
``python -m handoff.train.verl_main`` inside the run directory. The user
simulator must already be served (``slurm/jobs/serve.sbatch``).

Examples:
    python scripts/train_rl.py rl=grpo reward=task_only experiment=R1 \\
        policy.init_path=$SCRATCH/runs/B2-s0-20261016/final
    python scripts/train_rl.py rl=grpo reward=r2 experiment=R2 policy.init_path=...
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
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
from handoff.envs.personas import personas_from_config
from handoff.envs.tau2_adapter import load_frozen_tasks
from handoff.ops.provenance import (
    append_attempt,
    check_resume_code,
    create_run_directory,
    finish_run,
    reopen_run,
    verify_run_directory,
)
from handoff.ops.runs import (
    checkpoint_fingerprint,
    git_state,
    hardware_info,
    make_run_id,
)
from handoff.rewards.composer import RewardConfig
from handoff.sim.endpoints import read_endpoint
from handoff.train.rl_data import rl_rows, rows_summary
from handoff.train.rl_launch import (
    agent_loop_yaml,
    build_agent_loop_settings,
    build_verl_overrides,
    validate_overrides,
    verl_environment,
)

ROOT = repo_root(__file__)


def _write_parquet(rows: list[dict], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _run_verl(run_dir: Path, command: list[str], settings_path: Path) -> int:
    """Run verl in the run directory and record how it ended."""
    environment = verl_environment(os.environ, settings_path)
    result = subprocess.run(command, cwd=run_dir, env=environment, check=False)
    if result.returncode == 0:
        finish_run(run_dir, status="completed")
    else:
        finish_run(run_dir, status="failed", error_code="VERL_EXIT_NONZERO")
    return result.returncode


def _resume(run_dir: Path) -> int:
    """Re-run a run's saved verl command; verl resumes from its latest checkpoint.

    The run must not have completed, and the checkout must be the commit the run
    started from (the saved command and settings are reused unchanged).
    """
    command_file = run_dir / "verl_command.txt"
    if not command_file.exists():
        raise CliError(f"{run_dir} has no verl_command.txt to resume")
    git_sha, dirty = git_state(ROOT)
    try:
        check_resume_code(
            verify_run_directory(run_dir), git_sha=git_sha, git_dirty=dirty
        )
        reopen_run(run_dir)
    except ValueError as error:
        raise CliError(str(error)) from None
    append_attempt(
        run_dir,
        {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_sha": git_sha,
            "git_dirty": dirty,
            "resumed": True,
        },
    )
    command = shlex.split(command_file.read_text(encoding="utf-8"))
    return _run_verl(run_dir, command, run_dir / "agent_loop_settings.json")


def main() -> int:
    """Prepare everything for one RL run and start verl."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    setup_logging()
    try:
        config = compose_config(ROOT, "train_rl", args.overrides)
        require_scratch(config)
        if config.resume_run_id:
            return _resume(Path(config.paths.runs_dir) / str(config.resume_run_id))
        missing = missing_keys(config, ["policy.init_path"])
        if missing:
            raise CliError("missing settings: " + ", ".join(missing))
        resolved = container(config)
        RewardConfig.from_mapping(resolved["reward"])  # validate before any work
        paths = config.paths
        runs_dir = Path(paths.runs_dir)
        # A dry run gets its own ID, so the real run of the day keeps the plain one.
        tags = tuple(str(t) for t in config.tags) + (("dry",) if config.dry_run else ())
        run_id = make_run_id(
            str(config.experiment), seed=int(config.seed), tags=tags, runs_dir=runs_dir
        )
        user = read_endpoint(Path(config.user_sim.endpoint_file))
        registry, store = load_frozen_tasks(paths, list(config.data.domains))
        if registry.handoff_manifest is None:
            raise CliError("no hand-off manifest; build the task sets first (tasks)")
        commit = registry.split_manifest["tau2_commit"]
        personas = personas_from_config(resolved["persona"])

        def refs(split: str, kinds: list[str]):
            selected = []
            for domain in config.data.domains:
                selected += registry.load(
                    domain, split, include_handoff_set=True, kinds=kinds
                )
            return selected

        tasks = {}
        train_refs = refs("train", list(config.data.train_kinds))
        val_refs = refs("dev", list(config.data.val_kinds))
        for ref in [*train_refs, *val_refs]:
            tasks[ref.key] = store.get(ref)
        train_rows = rl_rows(
            train_refs,
            tasks,
            personas,
            list(config.data.train_personas),
            allowed_splits=["train"],
        )
        val_rows = rl_rows(
            val_refs,
            tasks,
            personas,
            list(config.data.val_personas),
            allowed_splits=["dev"],
        )
        init_path = Path(str(config.policy.init_path))
        policy_revision = checkpoint_fingerprint(init_path)
        git_sha, dirty = git_state(ROOT)
        run_dir = create_run_directory(
            runs_dir,
            run_id=run_id,
            config={k: v for k, v in resolved.items() if k != "paths"},
            provenance={
                "purpose": f"{config.experiment} RL ({config.rl.name})",
                "execution_kind": str(config.execution_kind),
                "git_sha": git_sha,
                "git_dirty": dirty,
                "tau2_sha": commit,
                "model_revisions": {
                    "policy": policy_revision,
                    "user_sim": user.model_revision,
                },
                "seeds": {"run": int(config.seed)},
                "hardware": hardware_info(),
            },
        )
        data_dir = run_dir / "data"
        _write_parquet(train_rows, data_dir / "train.parquet")
        _write_parquet(val_rows, data_dir / "val.parquet")
        (data_dir / "summary.json").write_text(
            json.dumps(
                {"train": rows_summary(train_rows), "val": rows_summary(val_rows)},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        settings_path = run_dir / "agent_loop_settings.json"
        settings_path.write_text(
            json.dumps(
                build_agent_loop_settings(
                    resolved,
                    run_id=run_id,
                    run_dir=run_dir,
                    policy_ckpt=f"{config.policy.label}@{policy_revision}",
                    user_sim=f"{user.served_model_name}@{user.model_revision}",
                ),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        loop_file = run_dir / "agent_loop.yaml"
        loop_file.write_text(agent_loop_yaml(settings_path), encoding="utf-8")
        guard = ROOT / "src" / "handoff" / "rollout" / "verl_reward.py"
        overrides = build_verl_overrides(
            resolved,
            run_id=run_id,
            run_dir=run_dir,
            train_file=data_dir / "train.parquet",
            val_file=data_dir / "val.parquet",
            agent_loop_file=loop_file,
            reward_guard_file=guard,
        )
        validate_overrides(overrides)
        command = [sys.executable, "-m", str(config.rl.entrypoint), *overrides]
        (run_dir / "verl_command.txt").write_text(shlex.join(command) + "\n")
        print(shlex.join(command))
        if config.dry_run:
            finish_run(run_dir, status="interrupted", error_code="DRY_RUN")
            return 0
        append_attempt(
            run_dir,
            {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "git_sha": git_sha,
                "git_dirty": dirty,
                "resumed": False,
                "user_endpoint": user.base_url,
                "user_server_job": user.slurm_job_id,
            },
        )
        return _run_verl(run_dir, command, settings_path)
    except (CliError, ValueError, RuntimeError) as error:
        fail(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
