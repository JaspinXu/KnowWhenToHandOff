"""Pin plan provenance and non-destructive output behavior."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from handoff.sim.plan_io import write_plan
from handoff.sim.server_plan import ServerPlan


@pytest.fixture
def plan() -> ServerPlan:
    return ServerPlan(
        command=("python", "example"), script="#!/bin/bash\n", config_json="{}"
    )


def test_bundle_is_explicitly_not_a_run(plan: ServerPlan, tmp_path: Path) -> None:
    target = tmp_path / "plan"
    write_plan(plan, target, git_sha="b" * 40)
    metadata = json.loads((target / "plan.json").read_text())
    assert metadata["kind"] == "offline_server_plan"
    assert metadata["status"] == "not_executed"
    assert metadata["git_sha"] == "b" * 40
    assert metadata["config_sha256"] == hashlib.sha256(b"{}").hexdigest()
    assert (
        metadata["script_sha256"]
        == hashlib.sha256((target / "serve.sbatch").read_bytes()).hexdigest()
    )
    assert (target / "resolved_config.json").read_text() == "{}"
    assert (target / "serve.sbatch").read_bytes() == b"#!/bin/bash\n"


def test_never_overwrites_existing_output(plan: ServerPlan, tmp_path: Path) -> None:
    target = tmp_path / "plan"
    target.mkdir()
    sentinel = target / "serve.sbatch"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        write_plan(plan, target, git_sha="b" * 40)
    assert sentinel.read_text() == "keep"


def test_invalid_provenance_does_not_create_output(
    plan: ServerPlan, tmp_path: Path
) -> None:
    target = tmp_path / "plan"
    with pytest.raises(ValueError, match="git_sha"):
        write_plan(plan, target, git_sha="main")
    assert not target.exists()


def test_partial_write_cleans_only_its_own_files(
    plan: ServerPlan, tmp_path: Path
) -> None:
    target = tmp_path / "plan"
    sibling = tmp_path / "other"
    sibling.mkdir()
    original = Path.open

    def failing_open(path: Path, *args: object, **kwargs: object) -> object:
        if path.name == "plan.json":
            raise OSError("synthetic disk failure")
        return original(path, *args, **kwargs)

    with (
        patch.object(Path, "open", failing_open),
        pytest.raises(OSError, match="synthetic disk failure"),
    ):
        write_plan(plan, target, git_sha="b" * 40)
    assert not target.exists()
    assert sibling.is_dir()
