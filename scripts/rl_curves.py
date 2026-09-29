"""Learning curves and reward-term statistics of RL runs (T3.2–T3.4).

Reads verl's per-step dumps of each run (``rollout_dumps/`` for training,
``validation_dumps/`` for the greedy dev validation) and writes, to
``reports/<name>/``:

- ``learning_curves.{md,csv,tex}``: mean reward terms per step, train and dev;
- ``dev_curves.{md,csv,tex}``: the dev rows only (checkpoint selection);
- ``learning_curves.png`` when matplotlib is available;
- ``sources.json``: each run's provenance (git SHA, model revisions, seeds).

Works on a running job too, so it doubles as a progress check.

Example:
    python scripts/rl_curves.py --name rl --runs R1=R1-s1-20261020 R2=R2-s1-20261024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from handoff.cli import CliError, compose_config, fail, repo_root, require_scratch
from handoff.eval.report import (
    ReportLabel,
    RunSource,
    curve_rows,
    plot_learning_curves,
    sources_record,
    write_table,
)
from handoff.train.monitors import step_curve

ROOT = repo_root(__file__)


def main() -> int:
    """Collect the dumps of every run and write the tables and figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="report sub-directory")
    parser.add_argument("--runs", nargs="+", required=True, help="LABEL=RUN_ID")
    parser.add_argument("--out-dir", default=None, help="default: reports/<name>")
    parser.add_argument("--paths", nargs="*", default=[])
    args = parser.parse_args()
    try:
        config = compose_config(ROOT, "base", args.paths)
        require_scratch(config)
        curves = {}
        labels = []
        for item in args.runs:
            label, separator, run_id = item.partition("=")
            if not separator or not run_id:
                raise CliError(f"expected LABEL=RUN_ID, got {item!r}")
            run_dir = Path(config.paths.runs_dir) / run_id
            if not (run_dir / "provenance.json").exists():
                raise CliError(f"no run directory with provenance for {run_id}")
            labels.append(ReportLabel(label, [RunSource(run_id, run_dir)]))
            curves[label] = {
                "train": step_curve(run_dir / "rollout_dumps"),
                "dev": step_curve(run_dir / "validation_dumps"),
            }
            if not curves[label]["train"] and not curves[label]["dev"]:
                raise CliError(f"{run_id} has no verl step dumps yet")
        out_dir = Path(args.out_dir) if args.out_dir else ROOT / "reports" / args.name
        write_table(
            out_dir, "learning_curves", curve_rows(curves), "RL learning curves"
        )
        dev = curve_rows(curves, phases=("dev",))
        write_table(out_dir, "dev_curves", dev, "Dev validation during RL")
        (out_dir / "sources.json").write_text(
            json.dumps(sources_record(labels), indent=2) + "\n", encoding="utf-8"
        )
        try:
            plot_learning_curves(curves, out_dir / "learning_curves.png")
        except ImportError:
            print("matplotlib not installed: tables written, figure skipped")
    except (CliError, OSError, ValueError) as error:
        fail(str(error))
    print((out_dir / "dev_curves.md").read_text(encoding="utf-8"))
    print(f"curves: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
