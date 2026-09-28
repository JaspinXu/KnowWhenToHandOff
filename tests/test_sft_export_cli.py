"""Run the actual offline preparation command from outside the checkout."""

import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_sft.py"


def run_cli(cwd: Path, *overrides: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *overrides],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def source(cwd: Path) -> Path:
    path = cwd / "records with spaces.jsonl"
    record = {
        "split": "train",
        "prompt_ids": [10],
        "response_ids": [20, 30],
        "response_mask": [1, 0],
    }
    path.write_text((json.dumps(record) + "\n") * 3, encoding="utf-8")
    return path


def test_cli_exports_from_another_cwd_using_relative_paths(tmp_path: Path) -> None:
    source(tmp_path)
    result = run_cli(
        tmp_path,
        'input_path="records with spaces.jsonl"',
        'output_dir="bundle with spaces"',
        "train.pad_token_id=0",
        "batch_size=2",
    )
    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (tmp_path / "bundle with spaces" / "manifest.json").read_text()
    )
    assert manifest["record_count"] == 3 and manifest["batch_count"] == 2
    assert manifest["policy_tokens"] == 3
    assert manifest["kind"] == "offline_sft_batch_export"
    assert len(manifest["git_sha"]) == 40
    assert type(manifest["git_dirty"]) is bool
    assert "3 records" in result.stdout
    assert "not a training run" in result.stdout


def test_cli_requires_paths_and_real_pad_configuration(tmp_path: Path) -> None:
    result = run_cli(tmp_path)
    assert result.returncode == 2
    assert "Unresolved configuration" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "override",
    [
        "train.api_key=private-sentinel",
        "+api_key=private-sentinel",
        "train.pad_token_id=[private-sentinel",
        "--api-key=private-sentinel",
    ],
)
def test_cli_does_not_print_unknown_or_malformed_values(
    tmp_path: Path, override: str
) -> None:
    result = run_cli(tmp_path, override)
    assert result.returncode == 2
    assert "private-sentinel" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert (
        "Cannot prepare" in result.stderr
        or "Invalid command-line arguments" in result.stderr
    )
    assert not list(tmp_path.iterdir())


def test_cli_refuses_output_in_repository(tmp_path: Path) -> None:
    source(tmp_path)
    target = ROOT / ".superpowers" / f"forbidden-sft-{uuid.uuid4().hex}"
    result = run_cli(
        tmp_path,
        'input_path="records with spaces.jsonl"',
        f'output_dir="{target.as_posix()}"',
        "train.pad_token_id=0",
    )
    assert result.returncode == 2
    assert "scratch outside" in result.stderr
    assert not target.exists()


def test_cli_existing_bundle_is_preserved(tmp_path: Path) -> None:
    source(tmp_path)
    target = tmp_path / "existing"
    target.mkdir()
    marker = target / "manifest.json"
    marker.write_text("keep")
    result = run_cli(
        tmp_path,
        'input_path="records with spaces.jsonl"',
        "output_dir=existing",
        "train.pad_token_id=0",
    )
    assert result.returncode == 2
    assert "exists" in result.stderr
    assert marker.read_text() == "keep"


def test_cli_invalid_input_is_sanitized_and_output_cleaned(tmp_path: Path) -> None:
    (tmp_path / "bad.jsonl").write_text("private-sentinel invalid json")
    result = run_cli(
        tmp_path, "input_path=bad.jsonl", "output_dir=bundle", "train.pad_token_id=0"
    )
    assert result.returncode == 2
    assert "private-sentinel" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert "Cannot prepare" in result.stderr
    assert not (tmp_path / "bundle").exists()
