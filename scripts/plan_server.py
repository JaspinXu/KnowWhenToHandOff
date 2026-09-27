"""Generate an offline Slurm server plan; never submit or run it."""

import argparse
import subprocess
from pathlib import Path
from typing import NoReturn

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from handoff.sim.plan_io import write_plan
from handoff.sim.server_plan import build_server_plan

ROOT = Path(__file__).resolve().parents[1]


class _SafeArgumentParser(argparse.ArgumentParser):
    """Keep rejected argument values out of terminal output."""

    def error(self, message: str) -> NoReturn:
        self.exit(2, "Invalid command-line arguments; use --help.\n")


def main() -> int:
    """Compose deployment settings and print or save an offline plan."""
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    try:
        with initialize_config_dir(
            version_base="1.3", config_dir=str(ROOT / "configs")
        ):
            config = compose(config_name="serve", overrides=args.overrides)
        missing = OmegaConf.missing_keys(config)
        if missing:
            parser.exit(
                2,
                "Missing required configuration: " + ", ".join(sorted(missing)) + "\n",
            )
        resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        if not isinstance(resolved, dict):
            raise ValueError("Expected config mapping")
        output = resolved.pop("output_dir")
        plan = build_server_plan(resolved)
        if output is None:
            print(plan.script, end="")
        else:
            if not isinstance(output, str) or not output:
                raise ValueError("output_dir must be a nonempty path")
            git_sha = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, timeout=10
            ).strip()
            dirty = bool(
                subprocess.check_output(
                    ["git", "-C", str(ROOT), "status", "--porcelain"], timeout=10
                )
            )
            path = write_plan(plan, Path(output), git_sha=git_sha, git_dirty=dirty)
            print(f"Offline plan saved: {path} (not executed)")
    except FileExistsError:
        parser.exit(2, "Output directory exists; choose a new plan directory.\n")
    except (
        ValueError,
        OSError,
        OmegaConfBaseException,
        HydraException,
        subprocess.SubprocessError,
    ):
        parser.exit(
            2, "Cannot create plan: check configuration, paths and Git availability.\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
