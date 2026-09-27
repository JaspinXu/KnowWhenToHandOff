"""Exercise run records using explicitly synthetic provenance on CPU."""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from handoff.ops.provenance import (
    create_run_directory,
    finish_run,
    verify_run_directory,
)

RUN_ID = "R2-fp05-s7-20260928"


@pytest.fixture
def provenance() -> dict:
    return {
        "purpose": "synthetic unit test, not an experiment",
        "execution_kind": "synthetic_test",
        "git_sha": "a" * 40,
        "git_dirty": False,
        "tau2_sha": "b" * 40,
        "model_revisions": {"policy": "c" * 40, "user_sim": "d" * 40},
        "seeds": {"run": 7, "policy": 7, "user_sim": 8},
        "hardware": {"device": "synthetic CPU fixture"},
    }


def test_records_config_hash_and_explicit_noncompleted_status(
    tmp_path: Path, provenance: dict
) -> None:
    run = create_run_directory(
        tmp_path, run_id=RUN_ID, config={"seed": 7}, provenance=provenance
    )
    record = verify_run_directory(run)
    assert record["execution_kind"] == "synthetic_test"
    assert record["status"] == "initialized"
    assert record["run_id"] == RUN_ID
    assert (
        record["config_sha256"]
        == hashlib.sha256((run / "config.json").read_bytes()).hexdigest()
    )
    assert not (run / "completion.json").exists()


def test_redacts_nested_credentials_without_mutating_inputs(
    tmp_path: Path, provenance: dict
) -> None:
    config = {
        "client": {"api_key": "sentinel-one", "Authorization": "sentinel-two"},
        "servers": [{"accessToken": "sentinel-three", "max_tokens": 100}],
        "normal": {"tokenizer": "my-tokenizer"},
    }
    original = copy.deepcopy(config)
    run = create_run_directory(
        tmp_path, run_id=RUN_ID, config=config, provenance=provenance
    )
    data = (run / "config.json").read_text()
    assert "sentinel" not in data
    parsed = json.loads(data)
    assert parsed["servers"][0]["accessToken"] == "[REDACTED]"
    assert parsed["servers"][0]["max_tokens"] == 100
    assert parsed["normal"]["tokenizer"] == "my-tokenizer"
    assert config == original


@pytest.mark.parametrize(
    "config",
    [
        {"seed": float("nan")},
        {"seed": "???"},
        {"endpoint": "${oc.env:ENDPOINT}"},
        {"bad": object()},
        {"endpoint": "https://user:password@example.invalid"},
        {"endpoint": "https://example.invalid?api_key=sentinel-secret"},
        {"endpoint": "HTTPS://user:sentinel-secret@example.invalid"},
        {"endpoint": "https://example.invalid?sig=sentinel-secret"},
        {"endpoint": " https://user:sentinel-secret@example.invalid"},
        {"endpoint": "\t\x00HTTPS://user:sentinel-secret@example.invalid"},
    ],
)
def test_rejects_nonreproducible_or_credential_urls_before_writing(
    tmp_path: Path, provenance: dict, config: dict
) -> None:
    with pytest.raises(ValueError) as exc:
        create_run_directory(
            tmp_path, run_id=RUN_ID, config=config, provenance=provenance
        )
    assert "sentinel-secret" not in str(exc.value)
    assert not (tmp_path / RUN_ID).exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tau2_sha", "main"),
        ("git_sha", ""),
        ("model_revisions", {"policy": "c" * 40}),
        ("seeds", {}),
        ("seeds", {"run": 8}),
        ("seeds", {"run": True}),
        ("hardware", {}),
        ("execution_kind", "offline_plan"),
    ],
)
def test_requires_complete_explicit_provenance(
    tmp_path: Path, provenance: dict, field: str, value: object
) -> None:
    provenance[field] = value
    with pytest.raises(ValueError):
        create_run_directory(tmp_path, run_id=RUN_ID, config={}, provenance=provenance)
    assert not (tmp_path / RUN_ID).exists()


def test_real_runs_reject_dirty_source(tmp_path: Path, provenance: dict) -> None:
    provenance.update(execution_kind="real", git_dirty=True)
    with pytest.raises(ValueError, match="git_dirty"):
        create_run_directory(tmp_path, run_id=RUN_ID, config={}, provenance=provenance)
    assert not (tmp_path / RUN_ID).exists()


def test_sanitized_provenance_must_still_validate(
    tmp_path: Path, provenance: dict
) -> None:
    provenance["seeds"]["access_token"] = 123
    with pytest.raises(ValueError):
        create_run_directory(tmp_path, run_id=RUN_ID, config={}, provenance=provenance)
    assert not (tmp_path / RUN_ID).exists()


@pytest.mark.parametrize("run_id", ["../escape", "B0-s7-20260230", "freeform"])
def test_rejects_invalid_run_ids(tmp_path: Path, provenance: dict, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        create_run_directory(tmp_path, run_id=run_id, config={}, provenance=provenance)
    assert list(tmp_path.iterdir()) == []


def test_never_overwrites_run_or_terminal_record(
    tmp_path: Path, provenance: dict
) -> None:
    run = create_run_directory(
        tmp_path, run_id=RUN_ID, config={}, provenance=provenance
    )
    original = (run / "provenance.json").read_bytes()
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, run_id=RUN_ID, config={}, provenance=provenance)
    finish_run(run, status="completed")
    completion = json.loads((run / "completion.json").read_text())
    assert completion["status"] == "completed"
    with pytest.raises(FileExistsError):
        finish_run(run, status="failed", error_code="SERVER_TIMEOUT")
    assert (run / "provenance.json").read_bytes() == original


def test_completion_requires_valid_config_and_safe_failure_code(
    tmp_path: Path, provenance: dict
) -> None:
    run = create_run_directory(
        tmp_path, run_id=RUN_ID, config={}, provenance=provenance
    )
    with pytest.raises(ValueError):
        finish_run(run, status="failed", error_code="secret in raw exception")
    with pytest.raises(ValueError):
        finish_run(run, status="completed", error_code="IMPOSSIBLE")
    with pytest.raises(ValueError):
        finish_run(run, status="pending")
    assert not (run / "completion.json").exists()
    (run / "config.json").write_text('{"tampered":true}')
    with pytest.raises(ValueError, match="hash"):
        finish_run(run, status="completed")
    assert not (run / "completion.json").exists()


def test_failed_write_preserves_siblings(tmp_path: Path, provenance: dict) -> None:
    sibling = tmp_path / "keep.txt"
    sibling.write_text("keep")
    original = Path.open

    def failed_open(path: Path, *args: object, **kwargs: object) -> object:
        if path.name == "provenance.json":
            raise OSError("synthetic disk error")
        return original(path, *args, **kwargs)

    with patch.object(Path, "open", failed_open), pytest.raises(OSError):
        create_run_directory(tmp_path, run_id=RUN_ID, config={}, provenance=provenance)
    assert not (tmp_path / RUN_ID).exists()
    assert sibling.read_text() == "keep"


@pytest.mark.parametrize(
    "filename", ["config.json", "provenance.json", "completion.json"]
)
def test_flush_failure_removes_only_new_file_and_allows_retry(
    tmp_path: Path, provenance: dict, filename: str
) -> None:
    original_open = Path.open
    run = tmp_path / RUN_ID
    if filename == "completion.json":
        create_run_directory(tmp_path, run_id=RUN_ID, config={}, provenance=provenance)

    class FailingClose:
        def __init__(self, stream: object) -> None:
            self.stream = stream

        def __enter__(self) -> object:
            return self.stream

        def __exit__(self, *args: object) -> None:
            self.stream.close()
            raise OSError("synthetic flush failure")

    def open_with_failed_close(path: Path, *args: object, **kwargs: object) -> object:
        stream = original_open(path, *args, **kwargs)
        if path.name == filename and args and args[0] == "x":
            return FailingClose(stream)
        return stream

    with (
        patch.object(Path, "open", open_with_failed_close),
        pytest.raises(OSError, match="synthetic flush failure"),
    ):
        if filename == "completion.json":
            finish_run(run, status="completed")
        else:
            create_run_directory(
                tmp_path, run_id=RUN_ID, config={}, provenance=provenance
            )
    assert not (run / filename).exists()
    if filename == "completion.json":
        assert (run / "provenance.json").exists()
        finish_run(run, status="completed")
    else:
        assert not run.exists()
        create_run_directory(tmp_path, run_id=RUN_ID, config={}, provenance=provenance)
