"""Write immutable, credential-sanitized run provenance with explicit completion."""

import hashlib
import json
import math
import re
from collections.abc import Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)

_PROVENANCE_KEYS = {
    "purpose",
    "execution_kind",
    "git_sha",
    "git_dirty",
    "tau2_sha",
    "model_revisions",
    "seeds",
    "hardware",
}


def _secret_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in {
        "token",
        "authorization",
        "cookie",
        "hftoken",
        "key",
        "sig",
        "signature",
        "sessiontoken",
        "xamzsecuritytoken",
        "xamzsignature",
    } or normalized.endswith(
        (
            "apikey",
            "password",
            "secret",
            "credentials",
            "credential",
            "accesstoken",
            "refreshtoken",
            "authtoken",
        )
    )


def _sanitize(value: object) -> JSONValue:
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("configuration contains nonfinite values")
        return value
    if isinstance(value, str):
        if value == "???" or "${" in value:
            raise ValueError("configuration must be fully resolved")
        try:
            parsed = urlsplit(value)
        except ValueError:
            raise ValueError("configuration contains an invalid URL") from None
        if parsed.scheme.lower() in ("http", "https"):
            unsafe = (
                parsed.username
                or parsed.password
                or any(_secret_key(key) for key, _ in parse_qsl(parsed.query))
            )
            if unsafe:
                raise ValueError("configuration contains credential-bearing URL")
        return value
    if isinstance(value, Mapping):
        result: dict[str, JSONValue] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("configuration keys must be strings")
            result[key] = "[REDACTED]" if _secret_key(key) else _sanitize(child)
        return result
    if isinstance(value, list | tuple):
        return [_sanitize(item) for item in value]
    raise ValueError("configuration must contain JSON-compatible values")


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{40}", value) is None:
        raise ValueError(f"{field} must be an immutable full commit SHA")
    return value


def _revision(value: object, field: str) -> str:
    """Accept a Hugging Face commit SHA or a local checkpoint fingerprint."""
    if isinstance(value, str) and (
        re.fullmatch(r"[a-f0-9]{40}", value)
        or re.fullmatch(r"sha256:[a-f0-9]{64}", value)
    ):
        return value
    raise ValueError(
        f"{field} must be a full commit SHA or a sha256: checkpoint fingerprint"
    )


RUN_ID_PATTERN = re.compile(
    r"(?P<experiment>B[012]|R[12]|SMOKE|TEACH|SPIKE)"
    r"(?:-[A-Za-z0-9_]+)*-s(?P<seed>[0-9]+)-(?P<date>[0-9]{8})"
)


def _validate(run_id: str, provenance: Mapping[str, object]) -> None:
    match = RUN_ID_PATTERN.fullmatch(run_id)
    if match is None:
        raise ValueError("run_id must follow the README experiment naming convention")
    try:
        datetime.strptime(match["date"], "%Y%m%d")
    except ValueError:
        raise ValueError("run_id contains an invalid date") from None
    if set(provenance) != _PROVENANCE_KEYS:
        raise ValueError("provenance has missing or unknown fields")
    _sha(provenance["git_sha"], "git_sha")
    _sha(provenance["tau2_sha"], "tau2_sha")
    if type(provenance["git_dirty"]) is not bool:
        raise ValueError("git_dirty must be explicitly recorded as a boolean")
    if provenance["execution_kind"] not in ("real", "synthetic_test"):
        raise ValueError("execution_kind must be real or synthetic_test")
    if provenance["execution_kind"] == "real" and provenance["git_dirty"]:
        raise ValueError("git_dirty must be false for a reproducible real run")
    purpose = provenance["purpose"]
    if not isinstance(purpose, str) or not purpose.strip():
        raise ValueError("purpose must be nonempty")
    models = provenance["model_revisions"]
    if not isinstance(models, Mapping) or not {"policy", "user_sim"} <= models.keys():
        raise ValueError("model_revisions must include policy and user_sim")
    for revision in models.values():
        _revision(revision, "model_revisions")
    seeds = provenance["seeds"]
    if (
        not isinstance(seeds, Mapping)
        or "run" not in seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds.values())
        or seeds["run"] != int(match["seed"])
    ):
        raise ValueError("seeds must be nonnegative integers consistent with run_id")
    if not isinstance(provenance["hardware"], Mapping) or not provenance["hardware"]:
        raise ValueError("hardware must be explicitly recorded")


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _write_new(path: Path, text: str) -> None:
    created = False
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            created = True
            stream.write(text)
    except BaseException:
        if created:
            with suppress(OSError):
                path.unlink()
        raise


def create_run_directory(
    root: Path,
    *,
    run_id: str,
    config: Mapping[str, object],
    provenance: Mapping[str, object],
) -> Path:
    """Create immutable run records; creating a record is not completing a run.

    Pass deliberately selected metadata, never the whole process environment.
    Known credential keys are redacted recursively as a secondary safeguard.
    The caller is responsible for supplying observed revisions and hardware.
    """
    _validate(run_id, provenance)
    if not isinstance(config, Mapping):
        raise ValueError("configuration must be a mapping")
    clean_provenance = _sanitize(provenance)
    if not isinstance(clean_provenance, dict):
        raise ValueError("provenance must be a mapping")
    _validate(run_id, clean_provenance)
    config_text = _json(_sanitize(config))
    record = {
        **clean_provenance,
        "schema_version": 1,
        "run_id": run_id,
        "status": "initialized",
        "created_at": datetime.now(UTC).isoformat(),
        "config_sha256": hashlib.sha256(config_text.encode("utf-8")).hexdigest(),
    }
    record_text = _json(record)
    destination = root.absolute() / run_id
    destination.mkdir(parents=True, exist_ok=False)
    created: list[Path] = []
    try:
        for name, text in (
            ("config.json", config_text),
            ("provenance.json", record_text),
        ):
            path = destination / name
            _write_new(path, text)
            created.append(path)
    except BaseException:
        for path in reversed(created):
            with suppress(OSError):
                path.unlink()
        with suppress(OSError):
            destination.rmdir()
        raise
    return destination


def verify_run_directory(run_dir: Path) -> dict[str, JSONValue]:
    """Validate immutable provenance and the saved config hash before reuse."""
    try:
        record = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record["schema_version"] != 1:
            raise ValueError("invalid run record")
        _validate(record["run_id"], {key: record[key] for key in _PROVENANCE_KEYS})
        if record["run_id"] != run_dir.name or record["status"] != "initialized":
            raise ValueError("run identity or initialization state mismatch")
        digest = hashlib.sha256((run_dir / "config.json").read_bytes()).hexdigest()
        if digest != record["config_sha256"]:
            raise ValueError("configuration hash mismatch")
    except (KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("invalid run provenance") from None
    return record


def reopen_run(run_dir: Path) -> dict[str, JSONValue]:
    """Prepare an unfinished, failed or interrupted run to be resumed.

    Verifies the immutable records and refuses a completed run. An earlier
    terminal marker is kept as ``completion.attempt<N>.json`` so that the
    resumed attempt can write its own. Returns the provenance record.
    """
    record = verify_run_directory(run_dir)
    completion = run_dir / "completion.json"
    if completion.exists():
        status = json.loads(completion.read_text(encoding="utf-8")).get("status")
        if status == "completed":
            raise ValueError(f"{run_dir.name} already completed; start a new run")
        attempt = 1
        while (run_dir / f"completion.attempt{attempt}.json").exists():
            attempt += 1
        completion.replace(run_dir / f"completion.attempt{attempt}.json")
    return record


def check_resume_code(
    record: Mapping[str, object], *, git_sha: str, git_dirty: bool
) -> None:
    """A resumed run must run the code (commit) it started with."""
    if git_sha != record["git_sha"]:
        raise ValueError(
            f"{record['run_id']} started at commit {record['git_sha']} but the "
            f"checkout is at {git_sha}; check out that commit to resume it"
        )
    if record["execution_kind"] == "real" and git_dirty:
        raise ValueError("git_dirty must be false for a reproducible real run")


def append_attempt(run_dir: Path, entry: Mapping[str, object]) -> None:
    """Append one attempt (job, endpoints, code version) to ``attempts.jsonl``.

    Facts that change when a run is resumed, such as server addresses, live here
    rather than in the immutable config. Credential keys are redacted.
    """
    clean = _sanitize({**entry, "time": datetime.now(UTC).isoformat()})
    path = run_dir / "attempts.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(clean, sort_keys=True, allow_nan=False) + "\n")


def finish_run(run_dir: Path, *, status: str, error_code: str | None = None) -> None:
    """Write a terminal marker once, after caller-confirmed execution ends."""
    if status not in ("completed", "failed", "interrupted"):
        raise ValueError("invalid terminal status")
    if error_code is not None and (
        status == "completed"
        or re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", error_code) is None
    ):
        raise ValueError(
            "error_code must be a bounded identifier, never raw exception text"
        )
    record = verify_run_directory(run_dir)
    terminal = {
        "schema_version": 1,
        "run_id": record["run_id"],
        "execution_kind": record["execution_kind"],
        "status": status,
        "error_code": error_code,
        "finished_at": datetime.now(UTC).isoformat(),
        "config_sha256": record["config_sha256"],
    }
    _write_new(run_dir / "completion.json", _json(terminal))
