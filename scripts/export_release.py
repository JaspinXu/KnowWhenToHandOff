"""Package the hand-off task set and every run record for a dataset release.

Run on the cluster (as a CPU job), check the listing, then upload the folder
as a Hugging Face dataset; see docs/release.md. Model weights are uploaded
separately.

Example:
    python scripts/export_release.py --out $HANDOFF_SCRATCH/release/data --dry-run
    python scripts/export_release.py --out $HANDOFF_SCRATCH/release/data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from handoff.cli import CliError, compose_config, fail, repo_root, require_scratch
from handoff.release import plan, run_index, write

ROOT = repo_root(__file__)


def main() -> int:
    """Write (or list) the release folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="empty or missing folder")
    parser.add_argument("--sft-datasets", nargs="*", default=["b2_v1"])
    parser.add_argument("--dry-run", action="store_true", help="list files only")
    parser.add_argument("--paths", nargs="*", default=[])
    args = parser.parse_args()
    try:
        config = compose_config(ROOT, "base", args.paths)
        require_scratch(config)
        scratch = Path(config.paths.scratch)
        items = plan(
            ROOT, Path(config.paths.runs_dir), scratch / "datasets", args.sft_datasets
        )
        if not items:
            raise CliError("nothing to release")
        total = sum(item.size for item in items)
        runs = run_index(items)
        print(f"{len(items)} files, {total / 1e6:.1f} MB, {len(runs)} runs")
        for row in runs:
            print(json.dumps(row))
        if args.dry_run:
            return 0
        out = Path(args.out)
        if out.exists() and any(out.iterdir()):
            raise CliError(f"{out} is not empty")
        out.mkdir(parents=True, exist_ok=True)
        replacements = [
            (str(scratch), "<scratch>"),
            (str(ROOT), "<repo>"),
            (str(Path.home()), "~"),
        ]
        write(items, out, replacements)
        (out / "runs.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in runs), encoding="utf-8"
        )
    except CliError as error:
        fail(str(error))
    print(f"release folder: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
