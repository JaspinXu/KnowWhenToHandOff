"""Exercise bounded, non-overwriting export of synthetic train token records."""

import hashlib
import json
from pathlib import Path

import pytest

from handoff.train.sft_batch import SFTBatchConfig


def record(number: int = 1) -> dict:
    return {
        "split": "train",
        "prompt_ids": [10],
        "response_ids": [number, 30, number + 1],
        "response_mask": [1, 0, 1],
    }


def source_file(tmp_path: Path, count: int = 3) -> Path:
    path = tmp_path / "synthetic.jsonl"
    path.write_bytes(
        b"".join(json.dumps(record(n)).encode() + b"\r\n" for n in range(count))
    )
    return path


def export(source: Path, target: Path, **overrides: object) -> dict:
    from handoff.train.sft_export import export_sft_batches

    kwargs = {
        "batch_config": SFTBatchConfig(16, 0, -100, 4),
        "batch_size": 2,
        "max_record_bytes": 1024,
        "git_sha": "a" * 40,
        "git_dirty": True,
        **overrides,
    }
    return export_sft_batches(source, target, **kwargs)


def test_exports_remainder_with_exact_hashes_counts_and_source_order(
    tmp_path: Path,
) -> None:
    source = source_file(tmp_path)
    original = source.read_bytes()
    destination = tmp_path / "bundle"
    manifest = export(source, destination)
    batches = [
        json.loads(line)
        for line in (destination / "batches.jsonl").read_bytes().splitlines()
    ]
    assert [len(batch["input_ids"]) for batch in batches] == [2, 1]
    assert [row for batch in batches for row in batch["input_ids"]] == [
        [10, 0, 30, 1],
        [10, 1, 30, 2],
        [10, 2, 30, 3],
    ]
    assert [row for batch in batches for row in batch["labels"]] == [
        [-100, 0, -100, 1],
        [-100, 1, -100, 2],
        [-100, 2, -100, 3],
    ]
    assert manifest["record_count"] == 3
    assert manifest["batch_count"] == 2
    assert manifest["input_tokens"] == 12
    assert manifest["policy_tokens"] == 6
    assert manifest["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert (
        manifest["batches_sha256"]
        == hashlib.sha256((destination / "batches.jsonl").read_bytes()).hexdigest()
    )
    assert (
        manifest["config_sha256"]
        == hashlib.sha256((destination / "config.json").read_bytes()).hexdigest()
    )
    assert manifest["kind"] == "offline_sft_batch_export"
    assert manifest["git_sha"] == "a" * 40 and manifest["git_dirty"] is True
    assert json.loads((destination / "manifest.json").read_text()) == manifest
    assert source.read_bytes() == original
    assert b"\r" not in (destination / "batches.jsonl").read_bytes()


def test_accepts_final_record_without_newline_and_exact_byte_limit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "input.jsonl"
    raw = json.dumps(record()).encode()
    source.write_bytes(raw)
    manifest = export(source, tmp_path / "bundle", max_record_bytes=len(raw))
    assert manifest["record_count"] == 1


@pytest.mark.parametrize(
    "suffix",
    [
        b"\n",
        b"{invalid private-input}\n",
        b"[]\n",
        b"\xff\n",
        b'{"split":"test","split":"train"}\n',
        b'{"x":NaN}\n',
        b"",
    ],
)
def test_bad_input_never_leaves_a_completed_or_partial_bundle(
    tmp_path: Path, suffix: bytes
) -> None:
    source = tmp_path / "input.jsonl"
    prefix = b"" if suffix == b"" else json.dumps(record()).encode() + b"\n"
    source.write_bytes(prefix + suffix)
    original = source.read_bytes()
    target = tmp_path / "bundle"
    with pytest.raises(ValueError) as error:
        export(source, target, batch_size=1)
    assert "private-input" not in str(error.value)
    assert not target.exists()
    assert source.read_bytes() == original


@pytest.mark.parametrize("split", ["test", "dev"])
def test_late_nontrain_row_removes_earlier_written_batches(
    tmp_path: Path, split: str
) -> None:
    source = source_file(tmp_path, 2)
    bad = {**record(), "split": split}
    with source.open("ab") as stream:
        stream.write(json.dumps(bad).encode())
    with pytest.raises(ValueError, match="batch"):
        export(source, tmp_path / "bundle", batch_size=1)
    assert not (tmp_path / "bundle").exists()


def test_oversize_record_is_rejected_without_splitting_or_truncating(
    tmp_path: Path,
) -> None:
    source = source_file(tmp_path, 1)
    with pytest.raises(ValueError, match="max_record_bytes"):
        export(source, tmp_path / "bundle", max_record_bytes=8)
    assert not (tmp_path / "bundle").exists()


def test_existing_destination_and_input_are_never_modified(tmp_path: Path) -> None:
    source = source_file(tmp_path)
    target = tmp_path / "bundle"
    target.mkdir()
    marker = target / "manifest.json"
    marker.write_text("keep this")
    with pytest.raises(FileExistsError):
        export(source, target)
    assert marker.read_text() == "keep this"
    with pytest.raises(FileExistsError):
        export(source, source)
    assert source.read_bytes().startswith(b'{"split"')


@pytest.mark.parametrize(
    "field,value",
    [
        ("batch_size", 0),
        ("batch_size", True),
        ("max_record_bytes", -1),
        ("max_record_bytes", "large"),
        ("git_sha", "main"),
        ("git_dirty", 1),
    ],
)
def test_invalid_export_settings_fail_before_creating_output(
    tmp_path: Path, field: str, value: object
) -> None:
    source = source_file(tmp_path)
    with pytest.raises(ValueError):
        export(source, tmp_path / "bundle", **{field: value})
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize(
    "failed_name", ["batches.jsonl", "config.json", "manifest.json"]
)
def test_close_failure_cleans_own_files_and_allows_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_name: str
) -> None:
    source = source_file(tmp_path)
    target = tmp_path / "bundle"
    real_open = Path.open

    class FailingClose:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, *args):
            self.stream.close()
            raise OSError("synthetic disk flush failure")

    def failing_open(path: Path, mode: str = "r", *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        return (
            FailingClose(stream) if path.name == failed_name and "x" in mode else stream
        )

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "open", failing_open)
        with pytest.raises(OSError):
            export(source, target)
    assert not target.exists()
    assert export(source, target)["record_count"] == 3


def test_reader_never_requests_an_unbounded_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_file(tmp_path)
    real_open = Path.open

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def readline(self, size):
            assert 0 < size <= 1025
            return self.stream.readline(size)

    def guarded_open(path: Path, mode: str = "r", *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        return BoundedReader(stream) if path == source and mode == "rb" else stream

    monkeypatch.setattr(Path, "open", guarded_open)
    assert export(source, tmp_path / "bundle")["record_count"] == 3


def test_cleanup_preserves_an_unrelated_file_created_during_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = source_file(tmp_path)
    target = tmp_path / "bundle"
    real_open = Path.open

    def failing_open(path: Path, mode: str = "r", *args, **kwargs):
        if path == target / "config.json" and "x" in mode:
            with real_open(target / "unrelated.txt", "w") as stream:
                stream.write("preserve this")
            raise OSError("synthetic write failure")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError):
        export(source, target)
    assert sorted(path.name for path in target.iterdir()) == ["unrelated.txt"]
    assert (target / "unrelated.txt").read_text() == "preserve this"
