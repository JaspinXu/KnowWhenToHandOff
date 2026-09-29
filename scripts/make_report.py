"""Generate report tables and figures from evaluation run IDs (T4.1).

Each ``--runs`` entry is ``LABEL=RUN_ID`` or, for a checkpoint whose test
evaluation is split over several protocols, ``LABEL=RUN_ID+RUN_ID+...``
(``test_main``, ``test_styles`` and ``test_handoff`` of the same checkpoint).

Example (dates shortened to <d>):
    python scripts/make_report.py --name main --baseline B2 --runs \\
        B0=B0-test_main-s0-<d>+B0-test_styles-s0-<d>+B0-test_handoff-s0-<d> \\
        B2=B2-test_main-s0-<d>+B2-test_styles-s0-<d>+B2-test_handoff-s0-<d>

Outputs go to ``reports/<name>/`` (tables as Markdown, CSV and LaTeX, PNG
figures when matplotlib is available, and ``sources.json`` with provenance).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from handoff.cli import CliError, compose_config, fail, repo_root, require_scratch
from handoff.eval.report import (
    RunSource,
    label_metrics,
    load_label,
    main_table_rows,
    paired_rows,
    plot_success,
    plot_tradeoff,
    sources_record,
    style_rows,
    write_table,
)

ROOT = repo_root(__file__)


def main() -> int:
    """Collect runs and write every table and figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="report sub-directory")
    parser.add_argument(
        "--runs", nargs="+", required=True, help="LABEL=RUN_ID[+RUN_ID...]"
    )
    parser.add_argument("--baseline", default=None, help="label for paired diffs")
    parser.add_argument("--pass-k", type=int, default=4)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-synthetic",
        action="store_true",
        help="accept synthetic_test runs (pipeline tests only, never for results)",
    )
    parser.add_argument("--out-dir", default=None, help="default: reports/<name>")
    parser.add_argument("--paths", nargs="*", default=[])
    args = parser.parse_args()
    try:
        config = compose_config(ROOT, "base", args.paths)
        require_scratch(config)
        labels = []
        for item in args.runs:
            label, separator, run_ids = item.partition("=")
            if not separator or not run_ids:
                raise CliError(f"expected LABEL=RUN_ID[+RUN_ID...], got {item!r}")
            sources = []
            for run_id in run_ids.split("+"):
                run_dir = Path(config.paths.runs_dir) / run_id
                if not (run_dir / "trajectories.jsonl").exists():
                    raise CliError(f"{run_id} has no trajectories (not an evaluation)")
                sources.append(RunSource(run_id, run_dir))
            labels.append(
                load_label(label, sources, allow_synthetic=args.allow_synthetic)
            )
        metrics = {
            data.label: label_metrics(
                data, k=args.pass_k, n_resamples=args.resamples, seed=args.seed
            )
            for data in labels
        }
        out_dir = Path(args.out_dir) if args.out_dir else ROOT / "reports" / args.name
        write_table(
            out_dir,
            "main_table",
            main_table_rows(labels, metrics, k=args.pass_k),
            "Main results",
        )
        styles = style_rows(labels, n_resamples=args.resamples, seed=args.seed)
        if len(styles) > 1:
            write_table(out_dir, "styles", styles, "Success by communication style")
        if args.baseline:
            write_table(
                out_dir,
                "paired_differences",
                paired_rows(labels, args.baseline, seed=args.seed),
                f"Paired differences vs {args.baseline}",
            )
        (out_dir / "sources.json").write_text(
            json.dumps(sources_record(labels), indent=2) + "\n", encoding="utf-8"
        )
        try:
            plot_success(metrics, out_dir / "success.png")
            plot_tradeoff(metrics, out_dir / "tradeoff.png")
        except ImportError:
            print("matplotlib not installed: tables written, figures skipped")
    except (CliError, ValueError) as error:
        fail(str(error))
    print((out_dir / "main_table.md").read_text(encoding="utf-8"))
    print(f"report: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
