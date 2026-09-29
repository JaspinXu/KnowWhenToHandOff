"""Evaluation driver (tasks T0.2, T0.5, T1.6, T2.1, T3.6).

Runs frozen tasks × communication styles × trials through tau2's own batch
runner (``tau2.runner.run_tasks``) with the project agent and a local user
simulator, then converts every simulation into a :class:`Trajectory`, applies
rules and labels, and writes metrics with bootstrap CIs.

Nothing here selects models: endpoints come from endpoint files written by the
serving jobs. The test split can only be evaluated with ``allow_test=True``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from handoff.agents.tau2_agents import agent_name, register_agents
from handoff.envs.personas import Persona
from handoff.envs.registry import TaskRef, TaskRegistry
from handoff.envs.tau2_adapter import (
    TaskStore,
    split_episode_task_id,
    styled_task,
    task_from_dict,
)
from handoff.eval.metrics import (
    format_interval,
    hacking_indicators,
    style_breakdown,
    summarize,
)
from handoff.rewards.composer import RewardComposer, RewardConfig
from handoff.rollout.schema import Trajectory, write_trajectories
from handoff.rollout.scoring import EpisodeContext
from handoff.rollout.tau2_convert import (
    assert_deterministic_reward_basis,
    domain_tool_sets,
    finalize_simulation,
    gold_actions_from_task,
    trajectory_from_simulation,
)
from handoff.sim.endpoints import Endpoint, litellm_args, register_local_models

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalSettings:
    """Resolved evaluation protocol for one run (from the ``eval`` config group)."""

    run_id: str
    domains: tuple[str, ...]
    splits: tuple[str, ...]
    kinds: tuple[str, ...]
    personas: tuple[str, ...]
    num_trials: int
    seed: int
    max_steps: int = 200
    max_errors: int = 10
    max_concurrency: int = 16
    max_tasks_per_domain: int | None = None
    task_ids: tuple[str, ...] | None = None
    prompt_variant: str = "handoff_v1"
    agent_temperature: float = 0.0
    agent_max_tokens: int | None = 2048
    agent_extra_body: Mapping[str, Any] | None = None
    user_temperature: float = 0.7
    user_max_tokens: int | None = 1024
    user_extra_body: Mapping[str, Any] | None = None
    simulation_timeout_s: float | None = 1800.0
    max_retries: int = 3
    allow_test: bool = False
    bootstrap_resamples: int = 1000
    pass_k: int = 4
    max_infra_failure_share: float = 0.0

    def __post_init__(self) -> None:
        if "test" in self.splits and not self.allow_test:
            raise ValueError(
                "test split requested without allow_test; test is evaluated once "
                "per frozen checkpoint (AGENTS hard rule 1)"
            )
        if self.num_trials < 1:
            raise ValueError("num_trials must be >= 1")
        if not self.personas:
            raise ValueError("at least one persona is required")
        if not 0.0 <= self.max_infra_failure_share < 1.0:
            raise ValueError("max_infra_failure_share must be in [0, 1)")

    @classmethod
    def from_mapping(cls, run_id: str, config: Mapping[str, Any]) -> EvalSettings:
        """Build from a resolved Hydra ``eval`` mapping (or a saved settings record)."""
        values = {key: value for key, value in dict(config).items() if key != "run_id"}
        for key in ("domains", "splits", "kinds", "personas"):
            values[key] = tuple(values[key])
        if values.get("task_ids") is not None:
            values["task_ids"] = tuple(str(i) for i in values["task_ids"])
        return cls(run_id=run_id, **values)


@dataclass
class EvalResult:
    """Paths and counts produced by :func:`run_evaluation`."""

    out_dir: Path
    n_episodes: int
    n_infra_failures: int
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def infra_failure_share(self) -> float:
        """Share of the planned episodes lost to infrastructure failures."""
        total = self.n_episodes + self.n_infra_failures
        return self.n_infra_failures / total if total else 0.0


def select_refs(registry: TaskRegistry, settings: EvalSettings) -> list[TaskRef]:
    """Pick task references for the run according to the settings.

    Hand-off kinds need the hand-off manifest: without it the protocol would
    silently shrink to upstream tasks.
    """
    refs: list[TaskRef] = []
    include_derived = any(kind != "upstream" for kind in settings.kinds)
    if include_derived and registry.handoff_manifest is None:
        raise ValueError(
            "hand-off task kinds requested but no hand-off manifest was loaded; "
            "build the task sets first (slurm/submit.sh tasks)"
        )
    for domain in settings.domains:
        domain_refs: list[TaskRef] = []
        for split in settings.splits:
            domain_refs.extend(
                registry.load(
                    domain,
                    split,
                    include_handoff_set=include_derived,
                    kinds=settings.kinds,
                )
            )
        if settings.task_ids is not None:
            wanted = set(settings.task_ids)
            domain_refs = [ref for ref in domain_refs if ref.task_id in wanted]
        if settings.max_tasks_per_domain is not None:
            domain_refs = domain_refs[: settings.max_tasks_per_domain]
        refs.extend(domain_refs)
    if not refs:
        raise ValueError("the evaluation selected no tasks")
    return refs


def build_run_config(
    settings: EvalSettings,
    domain: str,
    policy: Endpoint,
    user: Endpoint,
) -> Any:
    """Create the tau2 ``TextRunConfig`` for one domain."""
    from tau2.data_model.simulation import TextRunConfig

    return TextRunConfig(
        domain=domain,
        agent=agent_name(settings.prompt_variant),
        llm_agent=policy.litellm_model,
        llm_args_agent=litellm_args(
            policy,
            temperature=settings.agent_temperature,
            max_tokens=settings.agent_max_tokens,
            extra_body=settings.agent_extra_body,
        ),
        user="user_simulator",
        llm_user=user.litellm_model,
        llm_args_user=litellm_args(
            user,
            temperature=settings.user_temperature,
            max_tokens=settings.user_max_tokens,
            extra_body=settings.user_extra_body,
        ),
        num_trials=settings.num_trials,
        max_steps=settings.max_steps,
        max_errors=settings.max_errors,
        max_concurrency=settings.max_concurrency,
        seed=settings.seed,
        log_level="WARNING",
        auto_resume=True,
        hallucination_retries=0,
        max_retries=settings.max_retries,
        timeout=settings.simulation_timeout_s,
    )


def _episode_tasks(
    refs: Sequence[TaskRef],
    store: TaskStore,
    personas: Mapping[str, Persona],
    settings: EvalSettings,
) -> dict[str, list[tuple[Any, TaskRef, str]]]:
    by_domain: dict[str, list[tuple[Any, TaskRef, str]]] = {}
    for ref in refs:
        for persona_id in settings.personas:
            styled = styled_task(store.get(ref), personas[persona_id])
            task = task_from_dict(styled)
            assert_deterministic_reward_basis(task)
            by_domain.setdefault(ref.domain, []).append((task, ref, persona_id))
    return by_domain


def _infra_record(
    simulation: Any, domain: str, ref: TaskRef | None, persona: str | None
) -> dict[str, Any]:
    info = getattr(simulation, "info", None) or {}
    return {
        "domain": domain,
        "task_id": simulation.task_id,
        "kind": None if ref is None else ref.kind,
        "persona": persona,
        "trial": getattr(simulation, "trial", None),
        "termination_reason": str(
            getattr(
                simulation.termination_reason, "value", simulation.termination_reason
            )
        ),
        "error_type": info.get("error_type"),
    }


def infra_breakdown(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Counts of lost episodes by style, kind and error type."""
    breakdown: dict[str, dict[str, int]] = {}
    for key in ("persona", "kind", "error_type"):
        counts: dict[str, int] = {}
        for record in records:
            value = str(record.get(key))
            counts[value] = counts.get(value, 0) + 1
        breakdown[f"by_{key}"] = dict(sorted(counts.items()))
    return breakdown


def run_evaluation(
    settings: EvalSettings,
    *,
    registry: TaskRegistry,
    store: TaskStore,
    personas: Mapping[str, Persona],
    policy: Endpoint,
    user: Endpoint,
    policy_ckpt: str,
    user_sim: str,
    out_dir: Path,
    reward_config: RewardConfig | None = None,
) -> EvalResult:
    """Run the evaluation and write trajectories and metrics into ``out_dir``."""
    from tau2.evaluator.evaluator import EvaluationType
    from tau2.runner import build_environment, run_tasks

    register_agents()
    register_local_models([policy.served_model_name, user.served_model_name])
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = select_refs(registry, settings)
    grouped = _episode_tasks(refs, store, personas, settings)
    trajectories: list[Trajectory] = []
    infra: list[dict[str, Any]] = []
    for domain, entries in grouped.items():
        tasks = [task for task, _, _ in entries]
        meta = {task.id: (task, ref, persona) for task, ref, persona in entries}
        config = build_run_config(settings, domain, policy, user)
        environment = build_environment(domain)
        write_tools, known_tools = domain_tool_sets(environment)
        logger.info("running %d %s episodes per trial", len(tasks), domain)
        results = run_tasks(
            config,
            tasks,
            save_path=out_dir / f"tau2_{domain}.json",
            evaluation_type=EvaluationType.ALL,
            console_display=False,
        )
        for simulation in results.simulations:
            task, ref, persona = meta.get(simulation.task_id, (None, None, None))
            if simulation.reward_info is None or not simulation.messages:
                infra.append(_infra_record(simulation, domain, ref, persona))
                continue
            if task is None:
                raise ValueError(f"tau2 returned unknown task {simulation.task_id}")
            format_errors = finalize_simulation(simulation, task, domain)
            task_id, _ = split_episode_task_id(simulation.task_id)
            ctx = EpisodeContext(
                run_id=settings.run_id,
                domain=domain,
                task_id=task_id,
                source_task_id=ref.source_task_id,
                split=ref.split,
                kind=ref.kind,
                persona=persona,
                should_transfer=ref.should_transfer,
                trial=int(simulation.trial or 0),
                seed=simulation.seed,
                policy_ckpt=policy_ckpt,
                user_sim=user_sim,
                gold_actions=gold_actions_from_task(task),
                write_tools=write_tools,
                known_tools=known_tools,
            )
            trajectories.append(
                trajectory_from_simulation(
                    ctx, simulation, extra_format_errors=format_errors
                )
            )
    trajectories.sort(key=lambda t: (t.domain, t.task_id, t.persona, t.trial))
    write_trajectories(out_dir / "trajectories.jsonl", trajectories)
    with (out_dir / "infra_failures.jsonl").open("w", encoding="utf-8") as stream:
        for record in infra:
            stream.write(json.dumps(record) + "\n")
    metrics = compute_run_metrics(
        trajectories,
        n_infra_failures=len(infra),
        reward_config=reward_config,
        pass_k=settings.pass_k,
        n_resamples=settings.bootstrap_resamples,
        seed=settings.seed,
    )
    metrics["infra_failures"] = infra_breakdown(infra)
    with (out_dir / "metrics.json").open("w", encoding="utf-8", newline="\n") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    (out_dir / "summary.md").write_text(render_summary(metrics), encoding="utf-8")
    return EvalResult(out_dir, len(trajectories), len(infra), metrics)


def compute_run_metrics(
    trajectories: Sequence[Trajectory],
    *,
    n_infra_failures: int = 0,
    reward_config: RewardConfig | None = None,
    pass_k: int = 4,
    n_resamples: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Metrics for a run: overall, by domain, by kind, by style, and reward terms."""

    def block(subset: Sequence[Trajectory]) -> dict[str, Any]:
        return {
            name: interval.as_dict()
            for name, interval in summarize(
                subset, k=pass_k, n_resamples=n_resamples, seed=seed
            ).items()
        }

    result: dict[str, Any] = {
        "n_episodes": len(trajectories),
        "n_infra_failures": n_infra_failures,
        "infra_failure_share": (
            n_infra_failures / (len(trajectories) + n_infra_failures)
            if trajectories or n_infra_failures
            else None
        ),
        "overall": block(trajectories) if trajectories else {},
        "by_domain": {},
        "by_kind": {},
        "by_split": {},
        "styles": {},
        "hacking": hacking_indicators(trajectories) if trajectories else {},
    }
    for attribute, target in (
        ("domain", "by_domain"),
        ("kind", "by_kind"),
        ("split", "by_split"),
    ):
        values = sorted({getattr(t, attribute) for t in trajectories})
        if len(values) > 1:
            for value in values:
                subset = [t for t in trajectories if getattr(t, attribute) == value]
                result[target][value] = block(subset)
    if len({t.persona for t in trajectories}) > 1:
        result["styles"] = {
            name: interval.as_dict()
            for name, interval in style_breakdown(
                trajectories, n_resamples=n_resamples, seed=seed
            ).items()
        }
    if reward_config is not None and trajectories:
        composer = RewardComposer(reward_config)
        terms = [composer(t).as_metrics(prefix="") for t in trajectories]
        result["reward"] = {
            "config": reward_config.to_dict(),
            "mean": {key: sum(t[key] for t in terms) / len(terms) for key in terms[0]},
        }
    return result


def render_summary(metrics: Mapping[str, Any]) -> str:
    """Markdown table of the headline metrics with 95% CIs."""
    overall = metrics.get("overall", {})
    lines = [
        f"Episodes: {metrics.get('n_episodes')}  ·  infrastructure failures: "
        f"{metrics.get('n_infra_failures')}",
        "",
        "| Metric | Value [95% CI] |",
        "| --- | --- |",
    ]
    for name in sorted(overall):
        lines.append(f"| {name} | {format_interval(overall[name])} |")
    if metrics.get("styles"):
        lines += ["", "| Style metric | Value [95% CI] |", "| --- | --- |"]
        for name, interval in sorted(metrics["styles"].items()):
            lines.append(f"| {name} | {format_interval(interval)} |")
    return "\n".join(lines) + "\n"


def settings_record(settings: EvalSettings) -> dict[str, Any]:
    """JSON-compatible record of the settings for run provenance."""
    return json.loads(json.dumps(asdict(settings), default=list))
