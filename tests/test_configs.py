"""Every shipped config composes and validates against the code that reads it."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from handoff.cli import (
    CliError,
    compose_config,
    container,
    missing_keys,
    require_scratch,
)
from handoff.envs.personas import personas_from_config
from handoff.eval.runner import EvalSettings
from handoff.rewards.composer import RewardConfig
from handoff.rollout.episode import EpisodeSettings
from handoff.train.rl_launch import build_verl_overrides
from handoff.train.sft_trainer import SFTSettings

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def names(group: str) -> list[str]:
    return sorted(p.stem for p in (CONFIGS / group).glob("*.yaml"))


@pytest.mark.parametrize("name", names("eval"))
def test_eval_presets_validate(name):
    config = compose_config(ROOT, "run_eval", [f"eval={name}", "policy.label=x"])
    values = container(config.eval)
    assert values.pop("name") == name
    settings = EvalSettings.from_mapping("SMOKE-s0-20261001", values)
    assert ("test" in settings.splits) == settings.allow_test
    if settings.allow_test:
        assert name.startswith("test_")
    if "train" in settings.splits:
        assert name in {"smoke", "teacher_train"}


@pytest.mark.parametrize("name", names("reward"))
def test_reward_presets_validate(name):
    config = compose_config(ROOT, "run_eval", [f"reward={name}", "policy.label=x"])
    reward = RewardConfig.from_mapping(container(config.reward))
    if name == "task_only":
        assert reward.lambda_violation == reward.lambda_fp == reward.lambda_fn == 0.0
    if name == "r2":
        assert min(reward.lambda_fp, reward.lambda_fn, reward.lambda_violation) > 0


@pytest.mark.parametrize("name", names("rl"))
def test_rl_presets_build_overrides(name, tmp_path, monkeypatch):
    monkeypatch.setenv("HANDOFF_SCRATCH", str(tmp_path))
    config = compose_config(
        ROOT, "train_rl", [f"rl={name}", f"policy.init_path='{tmp_path.as_posix()}/b2'"]
    )
    require_scratch(config)
    resolved = container(config)
    EpisodeSettings.from_mapping(resolved["episode"])
    overrides = build_verl_overrides(
        resolved,
        run_id="R1-s1-20261020",
        run_dir=tmp_path / "run",
        train_file=tmp_path / "t.parquet",
        val_file=tmp_path / "v.parquet",
        agent_loop_file=tmp_path / "loop.yaml",
        reward_guard_file=tmp_path / "guard.py",
    )
    assert len({o.split("=", 1)[0] for o in overrides}) == len(overrides)
    assert "actor_rollout_ref.rollout.agent.default_agent_loop=tau2_agent" in overrides
    assert "actor_rollout_ref.rollout.name=vllm" in overrides


def test_personas_and_sft_and_task_set_configs(tmp_path, monkeypatch):
    monkeypatch.setenv("HANDOFF_SCRATCH", str(tmp_path))
    run_eval = compose_config(ROOT, "run_eval", ["policy.label=x"])
    cards = personas_from_config(container(run_eval.persona))
    assert set(cards) == {
        "neutral",
        "terse",
        "verbose",
        "impatient",
        "non-native",
        "low-tech-literacy",
    }
    sft = compose_config(ROOT, "train_sft", ["dataset_name=d"])
    SFTSettings.from_mapping(
        {
            **container(sft.sft),
            "model_path": "m",
            "train_records": "r",
            "output_dir": "o",
        }
    )
    tasks = compose_config(ROOT, "task_sets", [])
    counts = container(tasks.handoff_tasks)["counts"]
    assert counts["airline"]["test"]["explicit_request"] >= 1
    assert tasks.paths.repo == str(ROOT)


def test_scratch_guard_and_missing_keys(monkeypatch):
    monkeypatch.delenv("HANDOFF_SCRATCH", raising=False)
    config = compose_config(ROOT, "run_eval", [])
    with pytest.raises(CliError):
        require_scratch(config)
    assert missing_keys(config, ["policy.label"]) == ["policy.label"]


def test_model_configs_have_serving_fields():
    for path in (CONFIGS / "model").glob("*.yaml"):
        model = OmegaConf.load(path)
        assert {
            "repository",
            "served_name",
            "tool_call_parser",
            "name",
        } <= set(model)
