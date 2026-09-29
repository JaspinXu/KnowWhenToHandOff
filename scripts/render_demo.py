"""Render selected dialogues of a run into one offline HTML page (T4.3).

By default picks, per domain, one correct hand-off, one hard negative handled
without a transfer and one failure, so the page shows the behaviours the
project is about.

Example:
    python scripts/render_demo.py --run R2-final-test_handoff-s0-20261031 \\
        --out reports/demo/handoffs.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

from handoff.cli import (
    CliError,
    compose_config,
    fail,
    repo_root,
    require_scratch,
    run_trajectories_path,
)
from handoff.demo.viewer import render_html
from handoff.rollout.schema import iter_trajectories

ROOT = repo_root(__file__)


def _pick(trajectories, per_domain: int):
    picked = []
    for domain in sorted({t.domain for t in trajectories}):
        pool = [t for t in trajectories if t.domain == domain]
        selectors = [
            lambda t: t.should_transfer and t.transferred and t.success,
            lambda t: t.kind == "hard_negative" and not t.transferred and t.success,
            lambda t: not t.success,
        ]
        for selector in selectors[:per_domain]:
            match = next((t for t in pool if selector(t) and t not in picked), None)
            if match is not None:
                picked.append(match)
    return picked


def main() -> int:
    """Write the HTML page."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--episodes", nargs="*", default=None, help="episode ids")
    parser.add_argument("--per-domain", type=int, default=3)
    parser.add_argument("--paths", nargs="*", default=[])
    args = parser.parse_args()
    try:
        config = compose_config(ROOT, "base", args.paths)
        require_scratch(config)
        path = run_trajectories_path(Path(config.paths.runs_dir), args.run)
        trajectories = list(iter_trajectories(path))
        if args.episodes:
            wanted = set(args.episodes)
            chosen = [t for t in trajectories if t.episode_id in wanted]
        else:
            chosen = _pick(trajectories, args.per_domain)
        if not chosen:
            raise CliError("no dialogues selected")
        out = Path(args.out or ROOT / "reports" / "demo" / f"{args.run}.html")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            render_html(chosen, f"KnowWhenToHandOff · {args.run}"), encoding="utf-8"
        )
    except CliError as error:
        fail(str(error))
    print(f"demo page: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
