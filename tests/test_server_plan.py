"""Validate cluster plans offline; never load a model or contact a cluster."""

import copy
import hashlib
import json
import shlex

import pytest

from handoff.sim.server_plan import build_server_plan


@pytest.fixture
def server_config() -> dict:
    return {
        "seed": 7,
        "model": {
            "repository": "example/synthetic-test-model",
            "revision": "a" * 40,
            "served_name": "fixture-policy",
            "tool_call_parser": "hermes",
        },
        "server": {
            "vllm_version": "0.10.2",
            "host": "127.0.0.1",
            "port": 8123,
            "dtype": "bfloat16",
            "max_model_len": 4096,
            "gpu_memory_utilization": 0.85,
            "tensor_parallel_size": 1,
        },
        "cluster": {
            "job_name": "handoff-policy",
            "scratch": "/scratch/test-owner",
            "workdir": "/projects/handoff",
            "python": "/opt/serving/bin/python",
            "gpus": 1,
            "cpus": 4,
            "memory_gb": 32,
            "time_limit": "01:00:00",
            "partition": None,
            "account": None,
            "constraint": None,
        },
    }


def test_plan_pins_model_and_tokenizer_and_preserves_seed(server_config: dict) -> None:
    plan = build_server_plan(server_config)
    args = list(plan.command)
    for flag, value in [
        ("--revision", "a" * 40),
        ("--tokenizer-revision", "a" * 40),
        ("--seed", "7"),
        ("--port", "8123"),
        ("--tool-call-parser", "hermes"),
        ("--tensor-parallel-size", "1"),
    ]:
        assert args[args.index(flag) + 1] == value
    assert "--enable-auto-tool-choice" in args
    assert "--trust-remote-code" not in args
    assert shlex.join(plan.command) in plan.script


def test_script_guards_allocation_and_uses_scratch(server_config: dict) -> None:
    script = build_server_plan(server_config).script
    assert script.startswith("#!/usr/bin/env bash\n#SBATCH")
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --gpus-per-node=1" in script
    assert "#SBATCH --time=01:00:00" in script
    assert "SLURM_JOB_ID" in script
    assert "SLURM_JOB_GPUS" in script
    assert "export HF_HOME=/scratch/test-owner/huggingface" in script
    assert "umask 077" in script
    assert "srun --kill-on-bad-exit=1" in script
    version_line = next(
        line for line in script.splitlines() if "version mismatch" in line
    )
    assert "version('vllm')" in shlex.split(version_line)[2]
    assert "nvidia-smi" in script
    assert "set -x" not in script
    assert "pip install" not in script
    assert "printenv" not in script


def test_render_is_deterministic_and_does_not_mutate(server_config: dict) -> None:
    original = copy.deepcopy(server_config)
    first = build_server_plan(server_config)
    second = build_server_plan(server_config)
    assert first == second
    assert json.loads(first.config_json) == original
    assert first.config_sha256 == hashlib.sha256(first.config_json.encode()).hexdigest()
    assert server_config == original


@pytest.mark.parametrize(
    ("group", "key", "value"),
    [
        ("model", "revision", "main"),
        ("model", "revision", None),
        ("model", "repository", "$(echo injected)"),
        ("model", "repository", "--help"),
        ("model", "served_name", "name\n#SBATCH --nodes=100"),
        ("model", "tool_call_parser", "hermes; whoami"),
        ("server", "vllm_version", "latest"),
        ("server", "port", 0),
        ("server", "port", 65536),
        ("server", "port", True),
        ("server", "host", "0.0.0.0; echo bad"),
        ("server", "dtype", "something-invalid"),
        ("server", "max_model_len", 0),
        ("server", "tensor_parallel_size", 2),
        ("server", "gpu_memory_utilization", 0),
        ("server", "gpu_memory_utilization", 1.1),
        ("server", "gpu_memory_utilization", float("nan")),
        ("server", "gpu_memory_utilization", True),
        ("cluster", "scratch", "/"),
        ("cluster", "scratch", "relative"),
        ("cluster", "scratch", "/scratch/../home"),
        ("cluster", "scratch", "/projects/handoff/cache"),
        ("cluster", "workdir", "/repo\nmalicious"),
        ("cluster", "python", "python"),
        ("cluster", "gpus", 0),
        ("cluster", "cpus", -1),
        ("cluster", "memory_gb", 0),
        ("cluster", "time_limit", "00:00:00"),
        ("cluster", "time_limit", "01:99:00"),
        ("cluster", "partition", "gpu --nodes=99"),
        ("cluster", "account", "x\n#SBATCH --nodes=99"),
    ],
)
def test_reject_invalid_settings(
    server_config: dict, group: str, key: str, value: object
) -> None:
    server_config[group][key] = value
    with pytest.raises(ValueError, match=key):
        build_server_plan(server_config)


def test_rejects_missing_unknown_and_secret_fields(server_config: dict) -> None:
    for group, key in [("model", "revision"), ("cluster", "scratch")]:
        config = copy.deepcopy(server_config)
        del config[group][key]
        with pytest.raises(ValueError):
            build_server_plan(config)
    server_config["server"]["api_key"] = "sentinel-must-not-appear"
    with pytest.raises(ValueError) as exc:
        build_server_plan(server_config)
    assert "sentinel-must-not-appear" not in str(exc.value)


@pytest.mark.parametrize("seed", [-1, True, "7"])
def test_rejects_invalid_seed(server_config: dict, seed: object) -> None:
    server_config["seed"] = seed
    with pytest.raises(ValueError, match="seed"):
        build_server_plan(server_config)


def test_optional_scheduler_fields_and_no_tool_parser(server_config: dict) -> None:
    server_config["cluster"].update(
        partition="gpu", account="course", constraint="h200"
    )
    server_config["model"]["tool_call_parser"] = None
    plan = build_server_plan(server_config)
    assert "#SBATCH --partition=gpu" in plan.script
    assert "#SBATCH --account=course" in plan.script
    assert "#SBATCH --constraint=h200" in plan.script
    assert "--enable-auto-tool-choice" not in plan.command
    assert "--tool-call-parser" not in plan.command
