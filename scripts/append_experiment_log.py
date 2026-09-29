"""Append one row to docs/experiment-log.md from a finished evaluation run.

The headline is built from the run's ``metrics.json`` (success, pass^k,
violation rate, hand-off F1 with 95% CIs), so the log never contains a number
without its run ID.

Example:
    python scripts/append_experiment_log.py B0-test_main-s0-20261008 \\
        --purpose "B0 baseline audit" --decision "continue with Qwen3-4B"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from handoff.cli import CliError, compose_config, fail, repo_root, require_scratch
from handoff.eval.report import format_interval
from handoff.ops.runs import append_experiment_log

ROOT = repo_root(__file__)
HEADLINE = ("success", "pass^4", "violation_rate", "handoff_f1")


def main() -> int:
    """Build the headline and append the row."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--paths", nargs="*", default=[])
    args = parser.parse_args()
    try:
        config = compose_config(ROOT, "base", args.paths)
        require_scratch(config)
        metrics_path = Path(config.paths.runs_dir) / args.run_id / "metrics.json"
        if not metrics_path.exists():
            raise CliError(f"{args.run_id} has no metrics.json")
        overall = json.loads(metrics_path.read_text(encoding="utf-8"))["overall"]
        headline = "; ".join(
            f"{name} {format_interval(overall.get(name), digits=2)}"
            for name in HEADLINE
            if name in overall
        )
        append_experiment_log(
            Path(config.paths.experiment_log),
            run_id=args.run_id,
            purpose=args.purpose,
            headline=headline,
            decision=args.decision,
        )
    except (CliError, ValueError) as error:
        fail(str(error))
    print(f"logged {args.run_id}: {headline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
