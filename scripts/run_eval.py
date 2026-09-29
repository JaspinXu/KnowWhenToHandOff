"""Evaluate a served policy on the frozen protocol (T0.2, T0.5, T1.6, T2.1, T3.6).

The policy and user-simulator servers must already be running (see
``slurm/jobs/eval.sbatch``, which starts the policy server itself). Results go to
``<scratch>/runs/<run_id>/``: tau2 result files, ``trajectories.jsonl``,
``metrics.json``, ``summary.md`` and immutable provenance records.

A run completes only if at most ``eval.max_infra_failure_share`` of its episodes
were lost to infrastructure failures (default: none); otherwise it exits with
status 3 and can be resumed with ``resume_run_id=<run id>``. A resumed run keeps
its directory and settings, re-runs only the missing episodes (tau2's own
checkpoint), and needs the same commit, tau2 data, manifests and model
revisions; the servers may be new.

Examples:
    python scripts/run_eval.py eval=smoke experiment=SMOKE policy.label=qwen3_4b
    python scripts/run_eval.py eval=test_main experiment=B0 policy.label=qwen3_4b
    python scripts/run_eval.py resume_run_id=B0-test_main-s0-20261008 \\
        policy.label=qwen3_4b
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

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
from handoff.envs.registry import TaskRegistry
from handoff.envs.tau2_adapter import load_frozen_tasks
from handoff.eval.runner import (
    EvalSettings,
    run_evaluation,
    select_refs,
    settings_record,
)
from handoff.ops.provenance import (
    append_attempt,
    check_resume_code,
    create_run_directory,
    finish_run,
    reopen_run,
    verify_run_directory,
)
from handoff.ops.runs import (
    LedgerScope,
    git_state,
    hardware_info,
    ledger_conflicts,
    make_run_id,
    record_in_ledger,
)
from handoff.rewards.composer import RewardConfig
from handoff.sim.endpoints import (
    Endpoint,
    load_api_key_into_env,
    read_endpoint,
    server_identity,
)

ROOT = repo_root(__file__)
INCOMPLETE_EXIT = 3
logger = logging.getLogger("run_eval")


def _manifest_hashes(registry: TaskRegistry) -> dict[str, str | None]:
    handoff = registry.handoff_manifest
    return {
        "splits": registry.split_manifest["manifest_sha256"],
        "handoff": handoff["manifest_sha256"] if handoff else None,
    }


def _ledger_scope(
    registry: TaskRegistry, settings: EvalSettings, policy: Endpoint, user: Endpoint
) -> LedgerScope:
    tasks = sorted(
        f"{ref.domain}/{ref.task_id}"
        for ref in select_refs(registry, settings)
        if ref.split == "test"
    )
    return LedgerScope(
        policy_revision=policy.model_revision,
        user_sim=f"{user.served_model_name}@{user.model_revision}",
        tasks=tuple(tasks),
        personas=tuple(sorted(settings.personas)),
    )


def _check_ledger(config: Any, scope: LedgerScope, run_id: str) -> str | None:
    """Refuse a second test evaluation of a checkpoint unless forced with a reason."""
    conflicts = ledger_conflicts(Path(config.paths.test_ledger), scope, run_id=run_id)
    if not conflicts:
        return None
    runs = ", ".join(str(record.get("run_id")) for record in conflicts)
    if not config.force_test_rerun:
        raise CliError(
            f"this checkpoint was already evaluated on these test tasks and styles "
            f"({runs}); test is evaluated once per checkpoint (AGENTS hard rule 1). "
            "Resume an unfinished run with resume_run_id=<run id>, or pass "
            "force_test_rerun=true test_rerun_reason='...'"
        )
    reason = str(config.test_rerun_reason or "").strip()
    if not reason:
        raise CliError("force_test_rerun needs test_rerun_reason (ADR-005)")
    logger.warning("forced test re-run over %s: %s", runs, reason)
    return reason


def _attempt(run_dir: Path, policy: Endpoint, user: Endpoint, resumed: bool) -> None:
    git_sha, dirty = git_state(ROOT)
    append_attempt(
        run_dir,
        {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_sha": git_sha,
            "git_dirty": dirty,
            "resumed": resumed,
            "policy_endpoint": policy.base_url,
            "policy_server_job": policy.slurm_job_id,
            "user_endpoint": user.base_url,
            "user_server_job": user.slurm_job_id,
        },
    )


def _new_run(config: Any) -> tuple[str, Path, EvalSettings, Any, Any]:
    """Create the run directory of a new evaluation."""
    missing = missing_keys(config, ["policy.label"])
    if missing:
        raise CliError("missing settings: " + ", ".join(missing))
    paths = config.paths
    run_id = make_run_id(
        str(config.experiment),
        seed=int(config.seed),
        tags=tuple(str(t) for t in config.tags) + (str(config.eval.name),),
        runs_dir=Path(paths.runs_dir),
    )
    settings = EvalSettings.from_mapping(
        run_id, {k: v for k, v in container(config.eval).items() if k != "name"}
    )
    policy = read_endpoint(Path(config.policy.endpoint_file))
    user = read_endpoint(Path(config.user_sim.endpoint_file))
    registry, store = load_frozen_tasks(config.paths, settings.domains)
    personas = personas_from_config(container(config.persona))
    scope = (
        _ledger_scope(registry, settings, policy, user) if settings.allow_test else None
    )
    forced_reason = _check_ledger(config, scope, run_id) if scope else None
    git_sha, dirty = git_state(ROOT)
    run_dir = create_run_directory(
        Path(paths.runs_dir),
        run_id=run_id,
        config={
            "eval_name": str(config.eval.name),
            "eval": settings_record(settings),
            "reward": container(config.reward),
            "persona": {p.id: p.sha256 for p in personas.values()},
            "policy": {"label": str(config.policy.label), **server_identity(policy)},
            "user_sim": {"label": str(config.user_sim.label), **server_identity(user)},
            "manifests": _manifest_hashes(registry),
            "test_rerun_reason": forced_reason,
        },
        provenance={
            "purpose": f"evaluation {config.eval.name} of {config.policy.label}",
            "execution_kind": str(config.execution_kind),
            "git_sha": git_sha,
            "git_dirty": dirty,
            "tau2_sha": registry.split_manifest["tau2_commit"],
            "model_revisions": {
                "policy": policy.model_revision,
                "user_sim": user.model_revision,
            },
            "seeds": {"run": int(config.seed), "tau2": settings.seed},
            "hardware": hardware_info(),
        },
    )
    if scope is not None:
        record_in_ledger(
            Path(paths.test_ledger),
            scope,
            run_id=run_id,
            eval_name=str(config.eval.name),
            forced_reason=forced_reason,
        )
    _attempt(run_dir, policy, user, resumed=False)
    return run_id, run_dir, settings, (registry, store, personas), (policy, user)


def _resumed_run(config: Any) -> tuple[str, Path, EvalSettings, Any, Any]:
    """Reopen an unfinished evaluation after checking nothing it depends on changed."""
    run_id = str(config.resume_run_id)
    run_dir = Path(config.paths.runs_dir) / run_id
    if not run_dir.is_dir():
        raise CliError(f"no run directory {run_dir}")
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if "eval" not in saved:
        raise CliError(f"{run_id} is not an evaluation run")
    completion = run_dir / "completion.json"
    if completion.exists() and (
        json.loads(completion.read_text(encoding="utf-8")).get("status") == "completed"
    ):
        raise CliError(f"{run_id} already completed; start a new run")
    git_sha, dirty = git_state(ROOT)
    try:
        record = verify_run_directory(run_dir)
        check_resume_code(record, git_sha=git_sha, git_dirty=dirty)
    except ValueError as error:
        raise CliError(str(error)) from None
    settings = EvalSettings.from_mapping(run_id, saved["eval"])
    policy = read_endpoint(Path(config.policy.endpoint_file))
    user = read_endpoint(Path(config.user_sim.endpoint_file))
    registry, store = load_frozen_tasks(config.paths, settings.domains)
    personas = personas_from_config(container(config.persona))
    problems = []
    if registry.split_manifest["tau2_commit"] != record["tau2_sha"]:
        problems.append("tau2 commit")
    if _manifest_hashes(registry) != saved["manifests"]:
        problems.append("task manifests")
    if {p.id: p.sha256 for p in personas.values()} != saved["persona"]:
        problems.append("style cards")
    for role, endpoint in (("policy", policy), ("user_sim", user)):
        expected = {k: v for k, v in saved[role].items() if k != "label"}
        if server_identity(endpoint) != expected:
            problems.append(f"{role} server (model, revision or vLLM flags)")
    if problems:
        raise CliError(f"cannot resume {run_id}: changed {', '.join(problems)}")
    reopen_run(run_dir)
    if settings.allow_test:
        record_in_ledger(
            Path(config.paths.test_ledger),
            _ledger_scope(registry, settings, policy, user),
            run_id=run_id,
            eval_name=str(saved.get("eval_name", "")),
            forced_reason=saved.get("test_rerun_reason"),
        )
    _attempt(run_dir, policy, user, resumed=True)
    return run_id, run_dir, settings, (registry, store, personas), (policy, user)


def main() -> int:
    """Compose the configuration, run (or resume) the evaluation, record the outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    setup_logging()
    try:
        config = compose_config(ROOT, "run_eval", args.overrides)
        require_scratch(config)
        load_api_key_into_env(Path(config.paths.api_key_file))
        prepare = _resumed_run if config.resume_run_id else _new_run
        run_id, run_dir, settings, inputs, endpoints = prepare(config)
        registry, store, personas = inputs
        policy, user = endpoints
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        logger.info("run directory: %s", run_dir)
        try:
            result = run_evaluation(
                settings,
                registry=registry,
                store=store,
                personas=personas,
                policy=policy,
                user=user,
                policy_ckpt=f"{saved['policy']['label']}@{policy.model_revision}",
                user_sim=f"{user.served_model_name}@{user.model_revision}",
                out_dir=run_dir,
                reward_config=RewardConfig.from_mapping(saved["reward"]),
            )
        except BaseException:
            finish_run(run_dir, status="failed", error_code="EVALUATION_FAILED")
            raise
        threshold = settings.max_infra_failure_share
        if config.accept_infra_failure_share is not None:
            threshold = float(config.accept_infra_failure_share)
            append_attempt(run_dir, {"accepted_infra_failure_share": threshold})
        complete = result.infra_failure_share <= threshold
        if complete:
            finish_run(run_dir, status="completed")
        else:
            finish_run(run_dir, status="failed", error_code="INFRA_FAILURES")
    except (CliError, ValueError, RuntimeError) as error:
        fail(str(error))
    print((run_dir / "summary.md").read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "episodes": result.n_episodes,
                "infra_failures": result.n_infra_failures,
                "complete": complete,
            }
        )
    )
    if not complete:
        print(
            f"INCOMPLETE: {result.n_infra_failures} episodes lost to infrastructure "
            f"failures (share {result.infra_failure_share:.3f} > {threshold}); see "
            f"infra_failures.jsonl, check the servers, then resume with "
            f"resume_run_id={run_id}"
        )
        return INCOMPLETE_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
