"""Persist offline planning bundles, never experiment results."""

import hashlib
import json
import re
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from handoff.sim.server_plan import ServerPlan


def write_plan(
    plan: ServerPlan, destination: Path, *, git_sha: str, git_dirty: bool = False
) -> Path:
    """Create a new plan bundle exclusively and refuse existing destinations."""
    if re.fullmatch(r"[0-9a-f]{40}", git_sha) is None:
        raise ValueError("git_sha: expected full commit SHA")
    if type(git_dirty) is not bool:
        raise ValueError("git_dirty: expected boolean")
    destination = destination.absolute()
    destination.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "kind": "offline_server_plan",
        "status": "not_executed",
        "created_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "config_sha256": plan.config_sha256,
        "script_sha256": hashlib.sha256(plan.script.encode("utf-8")).hexdigest(),
    }
    files = {
        "serve.sbatch": plan.script,
        "resolved_config.json": plan.config_json,
        "plan.json": json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    }
    created: list[Path] = []
    try:
        for name, content in files.items():
            path = destination / name
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                created.append(path)
                stream.write(content)
    except BaseException:
        for path in reversed(created):
            with suppress(OSError):
                path.unlink()
        with suppress(OSError):
            destination.rmdir()
        raise
    return destination
