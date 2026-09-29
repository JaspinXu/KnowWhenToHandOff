"""Turn a verl actor checkpoint into a Hugging Face directory vLLM can serve.

RL runs export every saved step automatically (``<run_dir>/export/global_step_N``,
see ``handoff.train.export``). This script prints the export of one step and
creates it if it is missing (for example after a job was killed during a save).
``--step all`` exports every saved step that has none yet.

Example:
    python scripts/export_checkpoint.py R1-s1-20261020 --step latest
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from handoff.cli import CliError, compose_config, fail, repo_root, require_scratch
from handoff.train.export import export_step, is_exported, step_dirs

ROOT = repo_root(__file__)


def _select(run_dir: Path, step: str) -> list[Path]:
    """Step directories named by ``step``: ``latest``, ``all`` or a number."""
    steps = step_dirs(run_dir / "checkpoints")
    exported = sorted((run_dir / "export").glob("global_step_*"))
    if step == "all":
        return [path for _, path in steps]
    if step == "latest":
        candidates = [path for _, path in steps]
        if not candidates and exported:
            return [run_dir / "checkpoints" / exported[-1].name]
        if not candidates:
            raise CliError(f"no checkpoints in {run_dir / 'checkpoints'}")
        return candidates[-1:]
    wanted = f"global_step_{int(step)}"
    matches = [path for _, path in steps if path.name == wanted]
    if not matches and (run_dir / "export" / wanted).exists():
        return [run_dir / "checkpoints" / wanted]
    if not matches:
        raise CliError(f"step {step} not found; available: {[n for n, _ in steps]}")
    return matches


def main() -> int:
    """Locate or build the Hugging Face exports and print their paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--step", default="latest", help="latest, all or a number")
    parser.add_argument("--paths", nargs="*", default=[])
    args = parser.parse_args()
    exports: list[Path] = []
    try:
        config = compose_config(ROOT, "base", args.paths)
        require_scratch(config)
        run_dir = Path(config.paths.runs_dir) / args.run_id
        for step_dir in _select(run_dir, args.step):
            target = run_dir / "export" / step_dir.name
            if not is_exported(target):
                export_step(step_dir, target)
            exports.append(target)
    except (CliError, FileNotFoundError, RuntimeError) as error:
        fail(str(error))
    except subprocess.CalledProcessError as error:
        fail(f"verl model merger failed: {error}")
    for target in exports:
        print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
