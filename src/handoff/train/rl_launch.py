"""Build the verl command line and agent-loop settings for one RL run.

Kept separate from ``scripts/train_rl.py`` so the override flattening and the
settings format are unit-tested without verl or GPUs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SAFE = re.compile(r"[A-Za-z0-9_./\-]+")
AGENT_LOOP_NAME = "tau2_agent"


def format_value(value: Any) -> str:
    """Render a Python value in Hydra's command-line override grammar."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        if value and _SAFE.fullmatch(value):
            return value
        escaped = value.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"
    if isinstance(value, Sequence):
        return "[" + ",".join(format_value(item) for item in value) + "]"
    raise ValueError(f"cannot render override value of type {type(value).__name__}")


def flatten(tree: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten nested mappings into ``(dotted.key, value)`` pairs (lists kept)."""
    items: list[tuple[str, Any]] = []
    for key, value in tree.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            items.extend(flatten(value, dotted))
        else:
            items.append((dotted, value))
    return items


def render_overrides(pairs: Sequence[tuple[str, Any]]) -> list[str]:
    """Turn pairs into ``key=value`` strings (keys starting with ``+`` add keys)."""
    return [f"{key}={format_value(value)}" for key, value in pairs]


def agent_loop_yaml(settings_path: Path) -> str:
    """The verl agent-loop registry file pointing at the settings JSON."""
    return (
        f"- name: {AGENT_LOOP_NAME}\n"
        "  _target_: handoff.rollout.verl_loop.Tau2AgentLoop\n"
        f"  handoff_config_path: {json.dumps(str(settings_path))}\n"
    )


def build_agent_loop_settings(
    config: Mapping[str, Any],
    *,
    run_id: str,
    run_dir: Path,
    policy_ckpt: str,
    user_sim: str,
) -> dict[str, Any]:
    """Settings consumed by ``Tau2AgentLoop`` in every rollout worker."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "user_endpoint_file": str(config["user_sim"]["endpoint_file"]),
        "api_key_file": str(config["paths"]["api_key_file"]),
        "user_temperature": float(config["user_sim"]["temperature"]),
        "user_max_tokens": config["user_sim"].get("max_tokens"),
        "user_outage_timeout_s": float(
            config["user_sim"].get("outage_timeout_s", 1800)
        ),
        "episode": dict(config["episode"]),
        "reward": dict(config["reward"]),
        "policy_ckpt": policy_ckpt,
        "user_sim": user_sim,
        "max_concurrent_episodes": int(config["max_concurrent_episodes"]),
    }


def build_verl_overrides(
    config: Mapping[str, Any],
    *,
    run_id: str,
    run_dir: Path,
    train_file: Path,
    val_file: Path,
    agent_loop_file: Path,
    reward_guard_file: Path,
) -> list[str]:
    """All verl overrides for the run: preset values plus run-specific paths."""
    rl = config["rl"]
    episode = config["episode"]
    pairs = flatten(rl["verl"])
    run_specific: list[tuple[str, Any]] = [
        ("data.train_files", str(train_file)),
        ("data.val_files", str(val_file)),
        ("data.max_prompt_length", int(episode["max_prompt_tokens"])),
        ("data.max_response_length", int(episode["max_response_tokens"])),
        ("data.seed", int(config["seed"])),
        ("actor_rollout_ref.model.path", str(config["policy"]["init_path"])),
        (
            "actor_rollout_ref.rollout.agent.agent_loop_config_path",
            str(agent_loop_file),
        ),
        ("reward.custom_reward_function.path", str(reward_guard_file)),
        ("reward.custom_reward_function.name", "compute_score"),
        ("trainer.experiment_name", run_id),
        ("trainer.default_local_dir", str(run_dir / "checkpoints")),
        ("trainer.rollout_data_dir", str(run_dir / "rollout_dumps")),
        ("trainer.validation_data_dir", str(run_dir / "validation_dumps")),
    ]
    attention = str(rl.get("attn_implementation", "sdpa"))
    run_specific.append(
        ("+actor_rollout_ref.model.override_config.attn_implementation", attention)
    )
    if rl.get("use_lora"):
        lora = rl["lora"]
        run_specific += [
            ("actor_rollout_ref.model.lora_rank", int(lora["rank"])),
            ("actor_rollout_ref.model.lora_alpha", int(lora["alpha"])),
            ("actor_rollout_ref.model.target_modules", str(lora["target_modules"])),
            ("actor_rollout_ref.actor.optim.lr", float(lora["lr"])),
            ("actor_rollout_ref.rollout.load_format", "safetensors"),
            ("actor_rollout_ref.rollout.layered_summon", True),
        ]
    keys = {key for key, _ in run_specific}
    merged = [(key, value) for key, value in pairs if key not in keys]
    return render_overrides([*merged, *run_specific])


def verl_config_dir() -> Path:
    """Location of verl's Hydra configs without importing verl (or Ray)."""
    import importlib.util

    spec = importlib.util.find_spec("verl")
    if spec is None or spec.origin is None:
        raise RuntimeError("verl is not installed in this environment")
    return Path(spec.origin).parent / "trainer" / "config"


def validate_overrides(overrides: Sequence[str]) -> None:
    """Compose verl's own config with the overrides (catches unknown keys)."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(version_base=None, config_dir=str(verl_config_dir())):
        compose(config_name="ppo_trainer", overrides=list(overrides))


# Slurm exports ROCR_VISIBLE_DEVICES next to CUDA_VISIBLE_DEVICES for GPU gres
# on some clusters (NUS SoC does). verl's workers refuse to start when both are
# set, and on NVIDIA GPUs the ROCm variable has no effect, so it is dropped.
_CONFLICTING_GPU_VARS = ("ROCR_VISIBLE_DEVICES",)
_GPU_SELECTORS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")


def verl_environment(base: Mapping[str, str], settings_path: Path) -> dict[str, str]:
    """Environment for the verl subprocess: agent-loop settings, no GPU clash."""
    environment = dict(base, HANDOFF_AGENT_LOOP_CONFIG=str(settings_path))
    if any(environment.get(name) for name in _GPU_SELECTORS):
        for name in _CONFLICTING_GPU_VARS:
            environment.pop(name, None)
    return environment
