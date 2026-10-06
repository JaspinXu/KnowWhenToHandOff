"""Check an RL run's reward-hacking watch-list against the B2 baseline (T3.5).

Reads ``<run_dir>/rollouts/*.jsonl`` written by the agent loop and the B2
evaluation's ``metrics.json``. Exits with status 3 when an indicator drifts
more than ``--factor`` times its baseline (AGENTS §7: stop the run), and with
status 4 when more than ``--max-infra`` of recent episodes were lost to
infrastructure failures (check the user-simulator server).

Example:
    python scripts/monitor_rl.py --run R2-s1-20261024 \\
        --baseline B2-dev-s0-20261017
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from handoff.cli import CliError, compose_config, fail, repo_root, require_scratch
from handoff.eval.metrics import drifted_indicators
from handoff.train.monitors import (
    read_rollout_records,
    reference_max,
    rollout_indicators,
    split_self_baseline,
)

ROOT = repo_root(__file__)


def main() -> int:
    """Print the watch-list and flag drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--baseline", required=True, help="B2 evaluation run ID")
    parser.add_argument(
        "--self-baseline",
        type=int,
        default=0,
        help="also use the run's own first N training episodes (same sampling) as"
        " a reference; an indicator is flagged only above factor x the larger of"
        " the two references",
    )
    parser.add_argument("--last", type=int, default=512)
    parser.add_argument("--factor", type=float, default=2.0)
    parser.add_argument("--max-infra", type=float, default=0.05)
    parser.add_argument("--paths", nargs="*", default=[])
    args = parser.parse_args()
    try:
        config = compose_config(ROOT, "base", args.paths)
        require_scratch(config)
        runs = Path(config.paths.runs_dir)
        records = read_rollout_records(runs / args.run / "rollouts")
        if not records:
            raise CliError(f"no rollout records for {args.run} yet")
        current = rollout_indicators(records, last=args.last)
        baseline_metrics = json.loads(
            (runs / args.baseline / "metrics.json").read_text(encoding="utf-8")
        )
        b2 = baseline_metrics.get("hacking", {})
        baseline = b2
        if args.self_baseline > 0:
            first, rest = split_self_baseline(records, args.self_baseline)
            if not rest:
                raise CliError(f"fewer than {args.self_baseline} training episodes yet")
            own = rollout_indicators(first, last=None)
            baseline = {k: reference_max(own.get(k), b2.get(k)) for k in b2}
            current = rollout_indicators(rest, last=args.last)
        watched = {k: current.get(k) for k in baseline}
        flagged = drifted_indicators(watched, baseline, factor=args.factor)
    except (CliError, OSError, ValueError) as error:
        fail(str(error))
    print(
        json.dumps(
            {"current": current, "baseline": baseline, "b2": b2, "drifted": flagged},
            indent=2,
        )
    )
    if flagged:
        source = "the run's first episodes" if args.self_baseline else "the B2 baseline"
        print(f"DRIFT: {flagged} exceed {args.factor}x {source}; stop the run")
        return 3
    infra = current.get("infra_failure_rate") or 0.0
    if infra > args.max_infra:
        print(f"INFRA: {infra:.1%} of recent episodes failed; check the servers")
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
