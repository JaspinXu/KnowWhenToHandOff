"""Write Markdown review packets for the human checks in the plan.

Packets (seeded, stratified, reproducible):

``tasks``     hand-off tasks to hand-check (T1.3; at least 30).
``styles``    dialogues per communication style (T1.4; 10 per style).
``audit``     dialogues for the reward-hacking / final audit (T3.5, T3.6; 20).
``failures``  failed episodes for error analysis (T2.3; 10).
``compare``   per-task improvements and regressions between two runs (T2.3).

Examples:
    python scripts/sample_review.py tasks --n 40
    python scripts/sample_review.py styles --run B0-test_styles-s0-20261008
    python scripts/sample_review.py failures --run B2-dev-s0-20261017 --n 10
    python scripts/sample_review.py compare --run B2-dev-... --baseline B0-dev-...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from handoff.cli import (
    CliError,
    compose_config,
    fail,
    repo_root,
    require_scratch,
    run_trajectories_path,
)
from handoff.envs.registry import read_json
from handoff.envs.tau2_adapter import read_task_file
from handoff.eval.analysis import (
    dialogue_markdown,
    is_failure,
    per_task_changes,
    stratified_sample,
    task_markdown,
)
from handoff.rollout.schema import iter_trajectories

ROOT = repo_root(__file__)


def _trajectories(runs_dir: Path, run_id: str):
    return list(iter_trajectories(run_trajectories_path(runs_dir, run_id)))


def main() -> int:
    """Write the requested packet to ``reports/review/``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "packet", choices=["tasks", "styles", "audit", "failures", "compare"]
    )
    parser.add_argument("--run", default=None)
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None, help="default: reports/review")
    parser.add_argument("--paths", nargs="*", default=[])
    args = parser.parse_args()
    try:
        config = compose_config(ROOT, "base", args.paths)
        out_dir = Path(args.out_dir) if args.out_dir else ROOT / "reports" / "review"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.packet == "tasks":
            manifest = read_json(Path(config.paths.handoff_manifest))
            items = []
            for domain, entry in manifest["domains"].items():
                tasks = {
                    t["id"]: t
                    for t in read_task_file(
                        Path(config.paths.handoff_tasks_dir) / f"{domain}_tasks.json"
                    )
                }
                for meta in entry["tasks"]:
                    items.append((tasks[meta["task_id"]], {**meta, "domain": domain}))
            picked = stratified_sample(
                items,
                lambda item: (item[1]["domain"], item[1]["kind"]),
                args.n or 40,
                seed=args.seed,
            )
            body = "\n".join(task_markdown(task, meta) for task, meta in picked)
            name = "handoff_tasks"
        else:
            require_scratch(config)
            runs_dir = Path(config.paths.runs_dir)
            if not args.run:
                raise CliError("--run is required for dialogue packets")
            trajectories = _trajectories(runs_dir, args.run)
            if args.packet == "compare":
                if not args.baseline:
                    raise CliError("--baseline is required for compare")
                changes = per_task_changes(
                    trajectories, _trajectories(runs_dir, args.baseline)
                )
                body = "```json\n" + json.dumps(changes, indent=2) + "\n```\n"
                name = f"compare_{args.run}_vs_{args.baseline}"
            else:
                if args.packet == "styles":
                    per_style = args.n or 10
                    picked = []
                    for style in sorted({t.persona for t in trajectories}):
                        picked += stratified_sample(
                            [t for t in trajectories if t.persona == style],
                            lambda t: t.domain,
                            per_style,
                            seed=args.seed,
                        )
                elif args.packet == "failures":
                    picked = stratified_sample(
                        [t for t in trajectories if is_failure(t)],
                        lambda t: (t.domain, t.kind),
                        args.n or 10,
                        seed=args.seed,
                    )
                else:
                    picked = stratified_sample(
                        trajectories,
                        lambda t: (t.domain, t.kind, t.persona),
                        args.n or 20,
                        seed=args.seed,
                    )
                body = "\n".join(dialogue_markdown(t) for t in picked)
                name = f"{args.packet}_{args.run}"
        path = out_dir / f"{name}.md"
        path.write_text(f"# Review packet: {name}\n\n{body}", encoding="utf-8")
    except (CliError, KeyError, OSError) as error:
        fail(str(error))
    print(f"review packet: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
