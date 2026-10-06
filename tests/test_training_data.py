"""SFT teacher filtering/tokenization, RL rows, launcher overrides and monitors."""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
from fakes import (
    FakeChatTokenizer,
    build_all,
    call,
    conversation,
    result,
    say,
    synthetic_task,
    trajectory,
)

from handoff.envs.personas import Persona
from handoff.envs.registry import TaskRegistry
from handoff.rollout.chat_delta import ChatDeltaTokenizer
from handoff.rollout.schema import FormatError, Violation
from handoff.train.monitors import (
    read_rollout_records,
    reference_max,
    rollout_indicators,
    split_self_baseline,
    step_curve,
)
from handoff.train.rl_data import rl_rows, rows_summary, stable_seed
from handoff.train.rl_launch import (
    agent_loop_yaml,
    build_agent_loop_settings,
    build_verl_overrides,
    flatten,
    format_value,
    verl_environment,
)
from handoff.train.sft_batch import SFTBatchConfig, build_sft_batch
from handoff.train.sft_data import (
    TeacherFilter,
    chat_messages,
    dataset_statistics,
    rejection_reason,
    select_teacher_episodes,
    tokenize_episode,
    write_sft_records,
)


@pytest.fixture
def registry():
    tasks = [synthetic_task(str(i), transfer=(i in (3, 13))) for i in range(20)]
    splits = {
        "train": [str(i) for i in range(12)],
        "test": [str(i) for i in range(12, 20)],
    }
    _, labels, manifest, _, handoff = build_all((tasks, splits))
    return TaskRegistry(manifest, labels, handoff)


def train_id(registry, kind="upstream", should=None):
    for ref in registry.all_refs():
        if (
            ref.split == "train"
            and ref.kind == kind
            and (should is None or ref.should_transfer == should)
        ):
            return ref
    raise AssertionError("no such task")


def teacher_messages():
    return conversation(
        say("user", "Please cancel R1, my id is ada_1"),
        say("assistant", None, call("c1", "get_user_details", user_id="ada_1")),
        result("c1", '{"user_id": "ada_1"}'),
        say("assistant", "Cancel R1? Please confirm."),
        say("user", "yes"),
        say("assistant", "Done."),
    )


def test_teacher_filter_reasons(registry):
    ref = train_id(registry, should=False)
    good = trajectory(
        domain="airline",
        task_id=ref.task_id,
        source_task_id=ref.source_task_id,
        split="train",
        messages=teacher_messages(),
    )
    rules = TeacherFilter()
    assert rejection_reason(good, registry, rules) is None
    test_ref = next(r for r in registry.all_refs() if r.split == "test")
    leaked = trajectory(
        task_id=test_ref.task_id, source_task_id=test_ref.source_task_id, split="train"
    )
    assert rejection_reason(leaked, registry, rules) == "split_test"
    assert (
        rejection_reason(trajectory(task_id="nope"), registry, rules) == "unknown_task"
    )
    cases = {
        "task_failed": {"task_reward": 0.0},
        "policy_violation": {"violations": [Violation("off_reference_write", 1)]},
        "format_error": {"format_errors": [FormatError("unknown_tool", 1)]},
        "wrong_transfer_decision": {"transferred": True},
        # the simulated user ended the conversation before the policy spoke
        "no_policy_turn": {"messages": conversation(say("user", "Never mind."))},
    }
    for reason, change in cases.items():
        bad = trajectory(
            domain="airline",
            task_id=ref.task_id,
            source_task_id=ref.source_task_id,
            **change,
        )
        assert rejection_reason(bad, registry, rules) == reason
    kept, reasons = select_teacher_episodes(
        [good, good, good], registry, TeacherFilter(max_per_task_persona=2)
    )
    assert len(kept) == 2 and reasons["group_cap"] == 1


def test_chat_messages_and_tokenization(registry, tmp_path):
    ref = train_id(registry, should=False)
    episode = trajectory(
        domain="airline",
        task_id=ref.task_id,
        source_task_id=ref.source_task_id,
        messages=teacher_messages(),
    )
    messages = chat_messages(episode, "POLICY")
    assert [m["role"] for m in messages][:4] == [
        "system",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[3]["tool_calls"][0]["function"]["arguments"] == {"user_id": "ada_1"}
    delta = ChatDeltaTokenizer(FakeChatTokenizer())
    record = tokenize_episode(episode, delta, "POLICY")
    assert set(record) == {"split", "prompt_ids", "response_ids", "response_mask"}
    batch = build_sft_batch(
        [record],
        config=SFTBatchConfig(
            max_length=100000, pad_token_id=12, ignore_index=-100, pad_to_multiple_of=8
        ),
    )
    targets = [t for t in batch["labels"][0] if t != -100]
    decoded = FakeChatTokenizer().decode(targets)
    assert "Done.<|im_end|>" in decoded and "cancel R1" not in decoded
    records, meta = tmp_path / "r.jsonl", tmp_path / "m.jsonl"
    skipped = write_sft_records(
        [episode, episode],
        delta_for=lambda d: delta,
        system_prompt_for=lambda d: "POLICY",
        records_path=records,
        meta_path=meta,
        max_length=10,
    )
    assert skipped["too_long"] == 2 and records.read_text() == ""
    write_sft_records(
        [episode],
        delta_for=lambda d: delta,
        system_prompt_for=lambda d: "POLICY",
        records_path=records,
        meta_path=meta,
    )
    stats = dataset_statistics([episode], {"task_failed": 3}, {}, meta)
    assert stats["n_records"] == 1 and stats["n_candidates"] == 4
    assert stats["splits"] == ["train"]


def personas():
    return {
        "neutral": Persona("neutral", ""),
        "terse": Persona("terse", "Be terse."),
    }


def test_rl_rows_carry_styled_tasks_and_refuse_other_splits(registry):
    refs = [r for r in registry.all_refs() if r.split == "train"][:3]
    tasks = {r.key: synthetic_task(r.source_task_id) | {"id": r.task_id} for r in refs}
    rows = rl_rows(
        refs, tasks, personas(), ["neutral", "terse"], allowed_splits=["train"]
    )
    assert len(rows) == 6
    info = rows[1]["extra_info"]
    task = json.loads(info["task_json"])
    assert task["id"].endswith("::terse")
    assert (
        task["user_scenario"]["persona"].endswith("Be terse.")
        or "Be terse." in task["user_scenario"]["persona"]
    )
    assert rows[0]["agent_name"] == "tau2_agent"
    assert rows[0]["data_source"].startswith("tau2/airline/")
    assert info["eval_seed"] == stable_seed("airline", refs[0].task_id, "terse")
    assert rows_summary(rows)["persona"] == {"neutral": 3, "terse": 3}
    test_ref = next(r for r in registry.all_refs() if r.split == "test")
    with pytest.raises(ValueError, match="test"):
        rl_rows(
            [test_ref],
            {test_ref.key: synthetic_task(test_ref.source_task_id)},
            personas(),
            ["neutral"],
            allowed_splits=["train"],
        )


def launch_config(tmp_path, **rl_changes):
    rl = {
        "name": "grpo",
        "entrypoint": "handoff.train.verl_main",
        "use_lora": False,
        "attn_implementation": "sdpa",
        "lora": {"rank": 64, "alpha": 128, "target_modules": "all-linear", "lr": 1e-5},
        "verl": {
            "algorithm": {"adv_estimator": "grpo"},
            "trainer": {"logger": ["console"], "total_training_steps": 60},
            "data": {"train_batch_size": 8},
        },
    }
    rl.update(rl_changes)
    return {
        "rl": rl,
        "seed": 1,
        "episode": {"max_prompt_tokens": 8192, "max_response_tokens": 24576},
        "policy": {"init_path": str(tmp_path / "b2")},
        "paths": {"api_key_file": str(tmp_path / "key")},
        "user_sim": {
            "endpoint_file": str(tmp_path / "usersim.json"),
            "temperature": 0.7,
            "max_tokens": 1024,
        },
        "reward": {"lambda_fp": 0.5},
        "max_concurrent_episodes": 64,
    }


def test_override_rendering():
    assert format_value(True) == "true" and format_value(None) == "null"
    assert format_value(1e-6) == "1e-06" and format_value(8) == "8"
    assert format_value("/a/b-c.parquet") == "/a/b-c.parquet"
    assert format_value("full_shard auto_wrap") == "'full_shard auto_wrap'"
    assert format_value("it's") == "'it\\'s'"
    assert format_value(["model", "hf_model"]) == "[model,hf_model]"
    assert flatten({"a": {"b": 1, "c": {"d": [1]}}}) == [("a.b", 1), ("a.c.d", [1])]


def test_verl_overrides_include_run_paths_and_optional_features(tmp_path):
    config = launch_config(tmp_path)
    overrides = build_verl_overrides(
        config,
        run_id="R1-s1-20261020",
        run_dir=tmp_path / "run",
        train_file=tmp_path / "train.parquet",
        val_file=tmp_path / "val.parquet",
        agent_loop_file=tmp_path / "loop.yaml",
        reward_guard_file=tmp_path / "guard.py",
    )
    joined = "\n".join(overrides)
    assert "algorithm.adv_estimator=grpo" in joined
    assert "trainer.logger=[console]" in joined
    init = format_value(str(tmp_path / "b2"))  # Windows paths get quoted
    assert f"actor_rollout_ref.model.path={init}" in joined
    assert "data.max_response_length=24576" in joined
    assert "actor_rollout_ref.rollout.max_model_len=32768" in joined
    assert "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" in joined
    assert "lora_rank" not in joined
    fancy = launch_config(tmp_path, use_lora=True)
    joined = "\n".join(
        build_verl_overrides(
            fancy,
            run_id="R2-s1-20261020",
            run_dir=tmp_path / "run",
            train_file=tmp_path / "t",
            val_file=tmp_path / "v",
            agent_loop_file=tmp_path / "l",
            reward_guard_file=tmp_path / "g",
        )
    )
    assert "actor_rollout_ref.model.lora_rank=64" in joined
    assert "actor_rollout_ref.actor.optim.lr=1e-05" in joined


def test_agent_loop_settings_and_yaml(tmp_path):
    settings = build_agent_loop_settings(
        launch_config(tmp_path),
        run_id="R1-s1-20261020",
        run_dir=tmp_path,
        policy_ckpt="p",
        user_sim="u",
    )
    assert settings["reward"] == {"lambda_fp": 0.5}
    assert settings["user_temperature"] == 0.7
    text = agent_loop_yaml(tmp_path / "settings.json")
    assert "handoff.rollout.verl_loop.Tau2AgentLoop" in text
    assert "tau2_agent" in text


def test_rollout_records_skip_partial_lines(tmp_path):
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    lines = [{"persona": "terse", "success": 0.0}] * 5
    (rollouts / "w.jsonl").write_text(
        "\n".join(json.dumps(r) for r in lines) + "\n{partial"
    )
    assert len(read_rollout_records(rollouts)) == 5
    assert read_rollout_records(tmp_path / "missing") == []


def test_rollout_records_are_read_in_time_order(tmp_path):
    """After a resume, new worker files can sort before old ones by name."""
    rollouts = tmp_path / "rollouts"
    rollouts.mkdir()
    old = [{"time": 1.0 + i, "persona": "terse", "success": 0.0} for i in range(3)]
    new = [{"time": 100.0 + i, "persona": "terse", "success": 1.0} for i in range(3)]
    (rollouts / "zz-node-1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in old) + "\n"
    )
    (rollouts / "aa-node-2.jsonl").write_text(
        "\n".join(json.dumps(r) for r in new) + "\n"
    )
    records = read_rollout_records(rollouts)
    assert [r["time"] for r in records] == [1.0, 2.0, 3.0, 100.0, 101.0, 102.0]
    assert rollout_indicators(records, last=3)["success"] == 1.0


def test_lora_sft_refuses_fsdp():
    from handoff.train.sft_trainer import SFTSettings

    base = {"model_path": "m", "train_records": "r", "output_dir": "o"}
    SFTSettings.from_mapping({**base, "lora": {"rank": 8}})
    SFTSettings.from_mapping({**base, "fsdp": "full_shard auto_wrap"})
    with pytest.raises(ValueError, match="LoRA with FSDP"):
        SFTSettings.from_mapping(
            {**base, "lora": {"rank": 8}, "fsdp": "full_shard auto_wrap"}
        )


def test_rollout_indicators():
    rng = random.Random(0)
    records = [
        {
            "should_transfer": 0.0,
            "transferred": float(rng.random() < 0.5),
            "n_messages": 20.0,
            "n_turns": 5.0,
            "n_format_errors": 0.0,
            "user_stop_failure": 0.0,
            "success": 1.0,
        }
        for _ in range(10)
    ]
    records.append({"validate": True, "transferred": 1.0})
    records += [{"infra_failure": 1.0, "n_messages": 0.0}] * 2
    indicators = rollout_indicators(records)
    assert indicators["n_episodes"] == 10.0
    assert indicators["mean_messages"] == 20.0
    assert 0.0 <= indicators["transfer_rate_on_negatives"] <= 1.0
    assert indicators["infra_failure_rate"] == pytest.approx(2 / 12)


def _dump(directory, step, records):
    """Write one verl generation dump (``<step>.jsonl``) with text columns."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps({"input": "...", "output": "...", "step": step, **record})
        for record in records
    ]
    (directory / f"{step}.jsonl").write_text("\n".join(lines) + "\n")


def test_step_curve_averages_reward_terms_per_step(tmp_path):
    dumps = tmp_path / "rollout_dumps"
    _dump(dumps, 10, [{"total": 1.0, "task": 1.0, "fp_penalty": 0.0}] * 3)
    _dump(dumps, 2, [{"total": 0.5, "task": 1.0}, {"total": -0.5, "task": 0.0}])
    (dumps / "notes.jsonl").write_text("{}\n")  # not a step file
    rows = step_curve(dumps)
    assert [row["step"] for row in rows] == [2, 10]  # numeric, not lexical
    assert rows[0] == {"step": 2, "n_episodes": 2, "total": 0.0, "task": 0.5}
    assert rows[1]["fp_penalty"] == 0.0  # zero means are kept
    assert step_curve(tmp_path / "missing") == []


def test_rl_curves_script_writes_tables(tmp_path, monkeypatch, capsys):
    run_dir = tmp_path / "runs" / "R2-s1-20261024"
    _dump(run_dir / "rollout_dumps", 1, [{"total": 0.25, "success": 1.0}])
    _dump(run_dir / "validation_dumps", 0, [{"total": 0.0, "success": 0.0}])
    _dump(run_dir / "validation_dumps", 10, [{"total": 1.0, "success": 1.0}])
    (run_dir / "config.json").write_text("{}")
    (run_dir / "provenance.json").write_text('{"git_sha": "abc"}')
    monkeypatch.setenv("HANDOFF_SCRATCH", str(tmp_path))
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "script_rl_curves", root / "scripts" / "rl_curves.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path / "reports"
    argv = ["rl_curves", "--name", "rl", "--runs", "R2=R2-s1-20261024"]
    monkeypatch.setattr(sys, "argv", [*argv, "--out-dir", str(out)])
    assert module.main() == 0
    csv_rows = (out / "learning_curves.csv").read_text().splitlines()
    assert csv_rows[0].startswith("Run,Phase,Step,Episodes,total,task")
    assert [row.split(",")[1:3] for row in csv_rows[1:]] == [
        ["train", "1"],
        ["dev", "0"],
        ["dev", "10"],
    ]
    dev = (out / "dev_curves.md").read_text()
    assert "| R2 | dev | 10 | 1 | 1.000 |" in dev and "train" not in dev
    assert "curves:" in capsys.readouterr().out
    sources = json.loads((out / "sources.json").read_text())
    assert sources[0]["run_id"] == "R2-s1-20261024"
    assert sources[0]["git_sha"] == "abc"
    monkeypatch.setattr(sys, "argv", ["rl_curves", "--name", "rl", "--runs", "R2=x"])
    with pytest.raises(SystemExit):
        module.main()


def test_verl_environment_drops_rocr_when_cuda_is_set(tmp_path):
    settings = tmp_path / "agent_loop.json"
    slurm_env = {"CUDA_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "0"}
    environment = verl_environment(slurm_env, settings)
    assert "ROCR_VISIBLE_DEVICES" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["HANDOFF_AGENT_LOOP_CONFIG"] == str(settings)
    assert slurm_env["ROCR_VISIBLE_DEVICES"] == "0"  # caller's mapping untouched


def test_verl_environment_keeps_rocr_without_cuda(tmp_path):
    environment = verl_environment({"ROCR_VISIBLE_DEVICES": "1"}, tmp_path / "s.json")
    assert environment["ROCR_VISIBLE_DEVICES"] == "1"


def test_split_self_baseline_drops_validation_and_keeps_order():
    records = [{"i": 0}, {"i": 1, "validate": True}, {"i": 2}, {"i": 3}, {"i": 4}]
    first, rest = split_self_baseline(records, 2)
    assert [r["i"] for r in first] == [0, 2]
    assert [r["i"] for r in rest] == [3, 4]


def test_reference_max_ignores_missing_values():
    assert reference_max(0.0, 0.02) == 0.02
    assert reference_max(0.125, 0.04) == 0.125
    assert reference_max(None, 0.3) == 0.3
    assert reference_max(None, None) is None
