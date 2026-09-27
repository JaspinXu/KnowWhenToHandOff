"""Pure validation and Slurm rendering without starting a model or scheduler."""

import base64
import hashlib
import ipaddress
import json
import math
import re
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class ServerPlan:
    """An offline plan, with canonical config text and no execution claims."""

    command: tuple[str, ...]
    script: str
    config_json: str

    @property
    def config_sha256(self) -> str:
        """Return the hash of the exact canonical configuration bytes."""
        return hashlib.sha256(self.config_json.encode("utf-8")).hexdigest()


def _section(value: object, keys: set[str], field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{field}: missing or unknown configuration fields")
    return value


def _text(value: object, pattern: str, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
        raise ValueError(f"{field}: invalid value")
    return value


def _integer(value: object, field: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field}: expected integer >= {minimum}")
    return value


def _path(value: object, field: str) -> str:
    text = _text(value, r"/[A-Za-z0-9_./-]+", field)
    if text == "/" or any(part in {".", "..", ""} for part in text[1:].split("/")):
        raise ValueError(f"{field}: expected normalized absolute non-root POSIX path")
    return text


def build_server_plan(config: Mapping[str, object]) -> ServerPlan:
    """Validate a deployment config and render it without I/O or model access."""
    _section(config, {"seed", "model", "server", "cluster"}, "config")
    model = _section(
        config["model"],
        {"repository", "revision", "served_name", "tool_call_parser"},
        "model",
    )
    server = _section(
        config["server"],
        {
            "vllm_version",
            "host",
            "port",
            "dtype",
            "max_model_len",
            "gpu_memory_utilization",
            "tensor_parallel_size",
        },
        "server",
    )
    cluster = _section(
        config["cluster"],
        {
            "job_name",
            "scratch",
            "workdir",
            "python",
            "gpus",
            "cpus",
            "memory_gb",
            "time_limit",
            "partition",
            "account",
            "constraint",
        },
        "cluster",
    )
    seed = _integer(config["seed"], "seed", minimum=0)
    repository = _text(
        model["repository"],
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*",
        "repository",
    )
    revision = _text(model["revision"], r"[0-9a-f]{40}", "revision")
    token = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
    served_name = _text(model["served_name"], token, "served_name")
    parser = model["tool_call_parser"]
    if parser is not None:
        parser = _text(parser, token, "tool_call_parser")
    version = _text(
        server["vllm_version"],
        r"[0-9]+\.[0-9]+\.[0-9]+(?:[a-z0-9.+-]+)?",
        "vllm_version",
    )
    try:
        host = str(ipaddress.ip_address(str(server["host"])))
    except ValueError:
        raise ValueError("host: expected an IP binding address") from None
    port = _integer(server["port"], "port")
    if port > 65535:
        raise ValueError("port: must be <= 65535")
    dtype = server["dtype"]
    if dtype not in ("auto", "half", "float16", "bfloat16", "float", "float32"):
        raise ValueError("dtype: unsupported value")
    length = _integer(server["max_model_len"], "max_model_len")
    tp = _integer(server["tensor_parallel_size"], "tensor_parallel_size")
    utilization = server["gpu_memory_utilization"]
    if (
        type(utilization) not in (float, int)
        or not math.isfinite(utilization)
        or not 0 < utilization <= 1
    ):
        raise ValueError("gpu_memory_utilization: expected finite number in (0, 1]")
    job = _text(cluster["job_name"], token, "job_name")
    scratch = _path(cluster["scratch"], "scratch")
    workdir = _path(cluster["workdir"], "workdir")
    python = _path(cluster["python"], "python")
    if PurePosixPath(scratch).is_relative_to(PurePosixPath(workdir)):
        raise ValueError("scratch: must be outside the repository workdir")
    gpus = _integer(cluster["gpus"], "gpus")
    cpus = _integer(cluster["cpus"], "cpus")
    memory = _integer(cluster["memory_gb"], "memory_gb")
    if tp != gpus:
        raise ValueError("tensor_parallel_size: must equal the single-node GPU count")
    time_limit = _text(
        cluster["time_limit"], r"[0-9]+:[0-5][0-9]:[0-5][0-9]", "time_limit"
    )
    if not any(int(part) for part in time_limit.split(":")):
        raise ValueError("time_limit: must be positive")
    directives = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --gpus-per-node={gpus}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory}G",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --chdir={workdir}",
        f"#SBATCH --output={scratch}/%x-%j.out",
    ]
    for key in ("partition", "account", "constraint"):
        value = cluster[key]
        if value is not None:
            directives.append(f"#SBATCH --{key}={_text(value, token, key)}")
    command = [
        python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        repository,
        "--revision",
        revision,
        "--tokenizer-revision",
        revision,
        "--served-model-name",
        served_name,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        str(dtype),
        "--max-model-len",
        str(length),
        "--gpu-memory-utilization",
        str(utilization),
        "--tensor-parallel-size",
        str(tp),
        "--seed",
        str(seed),
    ]
    if parser is not None:
        command.extend(["--enable-auto-tool-choice", "--tool-call-parser", parser])
    config_json = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    encoded = base64.b64encode(config_json.encode()).decode("ascii")
    version_check = shlex.join(
        [
            python,
            "-c",
            "import sys; from importlib.metadata import version; "
            "actual = version('vllm'); "
            "sys.exit(0 if actual == sys.argv[1] else 'vLLM version mismatch')",
            version,
        ]
    )
    save_config = shlex.join(
        [
            python,
            "-c",
            "import base64,sys; from pathlib import Path; "
            "Path(sys.argv[1]).write_bytes(base64.b64decode(sys.argv[2]))",
        ]
    )
    body = [
        "",
        "# Generated offline; cluster compatibility and model behavior are unverified.",
        "set -euo pipefail",
        "umask 077",
        'if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_JOB_GPUS:-}" ]]; then',
        '  echo "A Slurm GPU allocation is required." >&2',
        "  exit 2",
        "fi",
        f"test -d {shlex.quote(scratch)}",
        f"test -x {shlex.quote(python)}",
        version_check,
        f"export HF_HOME={shlex.quote(scratch + '/huggingface')}",
        f"job_root={shlex.quote(scratch + '/handoff-jobs')}",
        'mkdir -p -- "$HF_HOME" "$job_root"',
        'job_dir="$job_root/${SLURM_JOB_ID}"',
        'mkdir -- "$job_dir"',
        f'{save_config} "$job_dir/resolved_config.json" {shlex.quote(encoded)}',
        'cp -- "${BASH_SOURCE[0]}" "$job_dir/submitted.sbatch"',
        f'git -C {shlex.quote(workdir)} rev-parse HEAD > "$job_dir/git-sha.txt"',
        f'git -C {shlex.quote(workdir)} status --porcelain > "$job_dir/git-status.txt"',
        f'{shlex.quote(python)} --version > "$job_dir/python-version.txt" 2>&1',
        f"printf '%s\\n' {shlex.quote(version)} > \"$job_dir/vllm-version.txt\"",
        "nvidia-smi --query-gpu=name,memory.total,driver_version "
        '--format=csv > "$job_dir/hardware.csv"',
        "exec srun --kill-on-bad-exit=1 " + shlex.join(command),
        "",
    ]
    return ServerPlan(tuple(command), "\n".join(directives + body), config_json)
