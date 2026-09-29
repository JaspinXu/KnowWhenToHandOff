"""Shared helpers for the thin CLI scripts in ``scripts/``.

Scripts compose Hydra configs from the checkout's ``configs/`` directory (never
from the shell's working directory), inject the repository path, and fail with
a short message instead of a traceback when a required setting is missing.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NoReturn

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


class CliError(RuntimeError):
    """A user-facing error with a short explanation."""


def repo_root(script_file: str) -> Path:
    """Repository root for a script living in ``<repo>/scripts``."""
    return Path(script_file).resolve().parents[1]


def compose_config(root: Path, name: str, overrides: Sequence[str]) -> DictConfig:
    """Compose ``configs/<name>.yaml`` and inject ``paths.repo`` if present."""
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
        config = compose(config_name=name, overrides=list(overrides))
    if "paths" in config:
        config.paths.repo = str(root)
    return config


def require_scratch(config: DictConfig) -> None:
    """Fail clearly when HANDOFF_SCRATCH is not configured."""
    scratch = str(OmegaConf.select(config, "paths.scratch", default="???"))
    if "???" in scratch or not Path(scratch).is_absolute():
        raise CliError(
            "HANDOFF_SCRATCH is not set; source slurm/cluster.env (see "
            "docs/cluster-runbook.md) or pass paths.scratch=/abs/path"
        )


def container(node: Any) -> Any:
    """Resolve an OmegaConf node into plain Python containers."""
    if isinstance(node, DictConfig) or OmegaConf.is_list(node):
        return OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
    return node


def missing_keys(config: DictConfig, keys: Sequence[str]) -> list[str]:
    """Return which of the dotted ``keys`` are absent or still ``???``."""
    missing = []
    for key in keys:
        parent_path, _, leaf = key.rpartition(".")
        parent = OmegaConf.select(config, parent_path) if parent_path else config
        absent = not isinstance(parent, DictConfig) or leaf not in parent
        if absent or OmegaConf.is_missing(parent, leaf):
            missing.append(key)
    return missing


def run_trajectories_path(runs_dir: Path, run_id: str) -> Path:
    """``<runs_dir>/<run_id>/trajectories.jsonl``; fail clearly if it is missing."""
    path = Path(runs_dir) / str(run_id) / "trajectories.jsonl"
    if not path.exists():
        raise CliError(f"no trajectories for {run_id} (not an evaluation run?)")
    return path


def setup_logging(verbose: bool = False) -> None:
    """Plain logging to stderr for scripts."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def fail(message: str, code: int = 2) -> NoReturn:
    """Print an error without a traceback and exit."""
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)
