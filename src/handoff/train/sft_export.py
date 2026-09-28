"""Stream preserved token records into offline, hash-addressable SFT batches."""

import hashlib
import json
import re
from contextlib import suppress
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from handoff.train.sft_batch import SFTBatchConfig, build_sft_batch


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("nonstandard JSON constant")


def _record(raw: bytes, line: int) -> dict[str, Any]:
    try:
        result = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
        if not isinstance(result, dict):
            raise ValueError("expected JSON object")
        return result
    except (ValueError, RecursionError):
        raise ValueError(f"invalid JSON record at line {line}") from None


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def export_sft_batches(
    source: Path,
    destination: Path,
    *,
    batch_config: SFTBatchConfig,
    batch_size: int,
    max_record_bytes: int,
    git_sha: str,
    git_dirty: bool,
) -> dict[str, Any]:
    """Prepare a new offline bundle; never overwrite data or claim training ran.

    Keep source/output on scratch. Input records must already be tokenized and
    selected by an approved data pipeline. Only their declared train split is
    checked here; source membership and teacher eligibility are not certified.
    The manifest is written last. Ordinary failures remove our created files;
    process termination can leave an unfinished directory without a manifest.
    """
    if not isinstance(batch_config, SFTBatchConfig):
        raise ValueError("batch_config must be a validated SFTBatchConfig")
    for name, value in (
        ("batch_size", batch_size),
        ("max_record_bytes", max_record_bytes),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(git_sha, str) or re.fullmatch(r"[a-f0-9]{40}", git_sha) is None:
        raise ValueError("git_sha must be a full commit SHA")
    if type(git_dirty) is not bool:
        raise ValueError("git_dirty must be a boolean")
    config_bytes = _json_bytes(
        {
            "train": asdict(batch_config),
            "batch_size": batch_size,
            "max_record_bytes": max_record_bytes,
        }
    )
    destination = destination.absolute()
    destination.mkdir(parents=True, exist_ok=False)
    created: list[Path] = []
    source_hash, batches_hash = hashlib.sha256(), hashlib.sha256()
    records = batches = input_tokens = policy_tokens = 0
    try:
        output_path = destination / "batches.jsonl"
        with source.open("rb") as reader, output_path.open("xb") as writer:
            created.append(output_path)
            pending: list[dict[str, Any]] = []

            def flush() -> None:
                nonlocal batches, input_tokens, policy_tokens
                try:
                    batch = build_sft_batch(pending, config=batch_config)
                except ValueError:
                    start = records - len(pending) + 1
                    raise ValueError(
                        f"invalid sample batch at input lines {start}-{records}"
                    ) from None
                data = _json_bytes(batch)
                writer.write(data)
                batches_hash.update(data)
                batches += 1
                input_tokens += sum(sum(row) for row in batch["attention_mask"])
                policy_tokens += sum(
                    token != batch_config.ignore_index
                    for row in batch["labels"]
                    for token in row
                )
                pending.clear()

            while raw := reader.readline(max_record_bytes + 1):
                if len(raw) > max_record_bytes:
                    raise ValueError(
                        f"record exceeds max_record_bytes at line {records + 1}"
                    )
                source_hash.update(raw)
                records += 1
                pending.append(_record(raw, records))
                if len(pending) == batch_size:
                    flush()
            if pending:
                flush()
            if not records:
                raise ValueError("source must contain at least one record")
        manifest = {
            "schema_version": 1,
            "kind": "offline_sft_batch_export",
            "created_at": datetime.now(UTC).isoformat(),
            "git_sha": git_sha,
            "git_dirty": git_dirty,
            "source_sha256": source_hash.hexdigest(),
            "batches_sha256": batches_hash.hexdigest(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "record_count": records,
            "batch_count": batches,
            "input_tokens": input_tokens,
            "policy_tokens": policy_tokens,
        }
        for name, content in (
            ("config.json", config_bytes),
            ("manifest.json", _json_bytes(manifest)),
        ):
            path = destination / name
            with path.open("xb") as writer:
                created.append(path)
                writer.write(content)
    except BaseException:
        for path in reversed(created):
            with suppress(OSError):
                path.unlink()
        with suppress(OSError):
            destination.rmdir()
        raise
    return manifest
