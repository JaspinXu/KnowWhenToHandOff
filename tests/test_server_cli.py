"""Exercise the real command from outside the repository, without a GPU."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "plan_server.py"


@pytest.mark.parametrize(
    "override",
    [
        "server.api_key=sentinel-test-secret",
        "server.host=[sentinel-test-secret",
        "+server.api_key=sentinel-test-secret",
        "--api-key=sentinel-test-secret",
    ],
)
def test_cli_never_echoes_rejected_values(tmp_path: Path, override: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), override],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2
    assert "sentinel-test-secret" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_cli_reports_missing_deployment_details(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 2
    assert "Missing required configuration" in result.stderr
    for key in (
        "cluster.scratch",
        "cluster.python",
        "model.revision",
        "server.vllm_version",
    ):
        assert key in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_cli_creates_plan_and_refuses_to_overwrite(tmp_path: Path) -> None:
    args = [
        sys.executable,
        str(SCRIPT),
        "model=custom",
        "model.repository=fixture/synthetic-model",
        "model.served_name=fixture-model",
        "model.revision=" + "a" * 40,
        "server.vllm_version=0.10.2",
        "cluster.scratch=/scratch/test",
        "cluster.workdir=/projects/handoff",
        "cluster.python=/opt/serving/bin/python",
        "output_dir=offline-plan",
    ]
    result = subprocess.run(
        args, cwd=tmp_path, capture_output=True, text=True, timeout=20, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "(not executed)" in result.stdout
    target = tmp_path / "offline-plan"
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    metadata = json.loads(before["plan.json"])
    assert metadata["status"] == "not_executed"
    assert len(metadata["git_sha"]) == 40
    assert b"\r" not in before["serve.sbatch"]
    assert sorted(before) == ["plan.json", "resolved_config.json", "serve.sbatch"]
    result = subprocess.run(
        args, cwd=tmp_path, capture_output=True, text=True, timeout=20, check=False
    )
    assert result.returncode == 2
    assert "Output directory exists" in result.stderr
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before
