"""End-to-end checks against the installed tau2 package (skipped without it).

A toy domain (``toy_tau2``) replaces the benchmark data, and scripted models
replace LiteLLM, so no network, GPU, dataset or model is needed. The tests run
the real evaluation driver (tau2's batch runner with the project agent) and the
real training driver (tau2's orchestrator with the token-in/token-out bridge)
on the same conversations and require identical scoring: the parity test of
AGENTS.md section 6.
"""

import asyncio
import os

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
pytest.importorskip("tau2")

import tau2.user.user_simulator as user_simulator  # noqa: E402
import tau2.utils.llm_utils as llm_utils  # noqa: E402
import toy_tau2  # noqa: E402
from fakes import COMMIT, FakeChatTokenizer  # noqa: E402

from handoff.agents.tau2_agents import agent_name, register_agents  # noqa: E402
from handoff.envs.labels import build_label_file, labels_by_task  # noqa: E402
from handoff.envs.personas import Persona  # noqa: E402
from handoff.envs.registry import (  # noqa: E402
    TaskRef,
    TaskRegistry,
    build_handoff_manifest,
    build_split_manifest,
)
from handoff.envs.tau2_adapter import (  # noqa: E402
    TaskStore,
    styled_task,
    task_from_dict,
)
from handoff.eval.runner import EvalSettings, run_evaluation  # noqa: E402
from handoff.rewards.composer import RewardComposer, RewardConfig  # noqa: E402
from handoff.rollout.episode import EpisodeSettings, Tau2EpisodeRunner  # noqa: E402
from handoff.rollout.schema import read_trajectories  # noqa: E402
from handoff.rollout.scoring import EpisodeContext  # noqa: E402
from handoff.rollout.tau2_convert import (  # noqa: E402
    assert_deterministic_reward_basis,
    domain_tool_sets,
)
from handoff.sim.endpoints import Endpoint  # noqa: E402

# asyncio event loops and tau2's thread pool use local socketpairs only.
pytestmark = pytest.mark.enable_socket

DOMAIN = toy_tau2.DOMAIN
PERSONAS = {
    "neutral": Persona("neutral", ""),
    "terse": Persona("terse", "You write very short messages."),
}
REWARD = RewardConfig(
    lambda_violation=0.25, lambda_fp=0.5, lambda_fn=1.0, lambda_format=0.1
)


@pytest.fixture(autouse=True)
def scripted_models(monkeypatch, tmp_path_factory):
    toy_tau2.register_toy_domain()
    monkeypatch.setattr(llm_utils, "completion", toy_tau2.fake_completion)
    # The user-simulator guidelines are a tau2 data file; tests need no content.
    guidelines = tmp_path_factory.mktemp("tau2") / "simulation_guidelines.md"
    guidelines.write_text("Simulate the user.<PERSONA_GUIDELINES>", encoding="utf-8")
    monkeypatch.setattr(user_simulator, "GLOBAL_USER_SIM_GUIDELINES_PATH", guidelines)


def protocol():
    """Registry and task store for the toy domain (all tasks in train)."""
    tasks = toy_tau2.UPSTREAM_TASKS
    label_file = build_label_file({DOMAIN: tasks}, tau2_commit=COMMIT)
    labels = labels_by_task(label_file)
    manifest = build_split_manifest(
        {
            DOMAIN: {
                "official_splits": toy_tau2.OFFICIAL_SPLITS,
                "labels": {t: v for (_, t), v in labels.items()},
                "file_sha256": {"tasks": "a" * 64},
            }
        },
        dev_fraction=0.0,
        seed=0,
        tau2_commit=COMMIT,
        labels_sha256=label_file["labels_sha256"],
    )
    derived = toy_tau2.derived_tasks()
    refs = [
        TaskRef(
            domain=DOMAIN,
            task_id=task["id"],
            split="train",
            source_task_id="0",
            kind="explicit_request",
            should_transfer=True,
            template_id="xr_direct",
        )
        for task in derived
    ]
    handoff = build_handoff_manifest(
        {DOMAIN: refs},
        generator_version="handoff_tasks_v1",
        split_manifest_sha256=manifest["manifest_sha256"],
        tasks_file_sha256={DOMAIN: "b" * 64},
        config={},
    )
    registry = TaskRegistry(manifest, labels, handoff)
    store = TaskStore({DOMAIN: tasks}, {DOMAIN: derived})
    return registry, store


def endpoint(role, name):
    return Endpoint(
        role=role, base_url="http://toy.invalid:1/v1", served_model_name=name
    )


def evaluate(tmp_path, personas=("neutral", "terse")):
    registry, store = protocol()
    settings = EvalSettings(
        run_id="SMOKE-toy-s0-20261001",
        domains=(DOMAIN,),
        splits=("train",),
        kinds=("upstream", "explicit_request"),
        personas=personas,
        num_trials=1,
        seed=0,
        max_concurrency=3,
        max_retries=0,
        bootstrap_resamples=20,
        pass_k=1,
    )
    result = run_evaluation(
        settings,
        registry=registry,
        store=store,
        personas=PERSONAS,
        policy=endpoint("policy", toy_tau2.POLICY_MODEL),
        user=endpoint("usersim", toy_tau2.USER_MODEL),
        policy_ckpt="toy-policy@1",
        user_sim="toy-user@1",
        out_dir=tmp_path / "eval",
        reward_config=REWARD,
    )
    return result, registry, store


def train_episode(registry, store, task_id, persona, seed=7):
    ref = registry.get(DOMAIN, task_id)
    task = styled_task(store.get(ref), PERSONAS[persona])
    settings = EpisodeSettings(
        user_model=f"openai/{toy_tau2.USER_MODEL}",
        user_args={"api_base": "http://toy.invalid:1/v1", "temperature": 0.0},
        max_prompt_tokens=20000,
        max_response_tokens=20000,
    )
    runner = Tau2EpisodeRunner(FakeChatTokenizer(), settings, RewardComposer(REWARD))
    ctx = EpisodeContext(
        run_id="R2-toy-s0-20261001",
        domain=DOMAIN,
        task_id=task_id,
        source_task_id=ref.source_task_id,
        split=ref.split,
        kind=ref.kind,
        persona=persona,
        should_transfer=ref.should_transfer,
        trial=0,
        seed=seed,
        policy_ckpt="toy-policy@1",
        user_sim="toy-user@1",
    )
    policy = toy_tau2.ScriptedPolicy(task_id)
    return asyncio.run(
        runner.run(task=task, ctx=ctx, seed=seed, generator=policy)
    ), policy


def comparable(trajectory):
    """Everything that scoring depends on, without driver-specific IDs."""
    messages = []
    for message in trajectory.messages:
        item = {k: v for k, v in message.items() if k not in ("tool_call_id",)}
        if item.get("tool_calls"):
            item["tool_calls"] = [
                {"name": c["name"], "arguments": c["arguments"]}
                for c in item["tool_calls"]
            ]
        messages.append(item)
    return {
        "messages": messages,
        "termination_reason": trajectory.termination_reason,
        "task_reward": trajectory.task_reward,
        "reward_basis": trajectory.reward_basis,
        "db_match": trajectory.db_match,
        "transferred": trajectory.transferred,
        "transfer_turn": trajectory.transfer_turn,
        "violations": trajectory.violations,
        "format_errors": trajectory.format_errors,
        "n_turns": trajectory.n_turns,
        "n_tool_calls": trajectory.n_tool_calls,
        "n_tokens": trajectory.n_tokens,
    }


EXPECTED = {
    # task: (termination, task reward, violations, format errors, transferred)
    "0": ("user_stop", 1.0, [], [], False),
    "1": (
        "user_stop",
        1.0,
        ["write_without_confirmation", "multiple_actions_in_turn"],
        [],
        False,
    ),
    "2": ("context_window_exceeded", 0.0, [], ["truncated_generation"], False),
    "3": ("agent_error", 0.0, [], ["empty_message"], False),
    "4": ("user_stop", 1.0, [], ["unparsed_tool_call"], False),
    "5": ("agent_error", 0.0, [], ["invalid_arguments"], False),
    "0__xr0": ("user_stop", 1.0, [], [], True),
}


def test_toy_tasks_validate_and_tool_types():
    registry, store = protocol()
    for ref in registry.all_refs():
        for persona in PERSONAS.values():
            task = task_from_dict(styled_task(store.get(ref), persona))
            assert_deterministic_reward_basis(task)
    environment = toy_tau2.get_environment()
    writes, known = domain_tool_sets(environment)
    assert writes == {"cancel_order"}
    assert known == {"get_order", "cancel_order", "transfer_to_human_agents"}
    nl_task = dict(toy_tau2.UPSTREAM_TASKS[0])
    nl_task["evaluation_criteria"] = {
        **nl_task["evaluation_criteria"],
        "reward_basis": ["DB", "NL_ASSERTION"],
    }
    # NL_ASSERTION without assertions: tau2 scores it 1.0 without a judge
    assert_deterministic_reward_basis(task_from_dict(nl_task))
    nl_task["evaluation_criteria"]["nl_assertions"] = ["Agent should say why."]
    with pytest.raises(ValueError, match="NL_ASSERTION"):
        assert_deterministic_reward_basis(task_from_dict(nl_task))
    names = register_agents()
    assert register_agents() == names and agent_name("handoff_v1") in names


def test_evaluation_driver_end_to_end(tmp_path, monkeypatch):
    user_prompts = []

    def recording_completion(**kwargs):
        if kwargs["model"].endswith(toy_tau2.USER_MODEL):
            user_prompts.append(kwargs["messages"][0]["content"])
        return toy_tau2.fake_completion(**kwargs)

    monkeypatch.setattr(llm_utils, "completion", recording_completion)
    result, _, _ = evaluate(tmp_path)
    # The style card reaches the simulator through tau2's user scenario.
    styled = [p for p in user_prompts if PERSONAS["terse"].text in p]
    assert styled and len(styled) < len(user_prompts)
    assert result.n_infra_failures == 0
    assert result.n_episodes == len(EXPECTED) * 2
    out = result.out_dir
    assert {p.name for p in out.iterdir()} >= {
        f"tau2_{DOMAIN}.json",
        "trajectories.jsonl",
        "infra_failures.jsonl",
        "metrics.json",
        "summary.md",
    }
    trajectories = read_trajectories([out / "trajectories.jsonl"])
    for trajectory in trajectories:
        termination, reward, rules, formats, transferred = EXPECTED[trajectory.task_id]
        assert trajectory.termination_reason == termination, trajectory.task_id
        assert trajectory.task_reward == reward, trajectory.task_id
        assert [v.rule for v in trajectory.violations] == rules, trajectory.task_id
        assert [e.kind for e in trajectory.format_errors] == formats, trajectory.task_id
        assert trajectory.transferred is transferred
        assert not any(
            "###STOP###" == m.get("content")
            for m in trajectory.messages
            if m["role"] == "assistant"
        )
    kinds = {t.task_id: t.kind for t in trajectories}
    assert kinds["0__xr0"] == "explicit_request" and kinds["0"] == "upstream"
    assert {t.persona for t in trajectories} == {"neutral", "terse"}
    metrics = result.metrics
    assert metrics["overall"]["success"]["point"] == pytest.approx(4 / 7)
    assert metrics["reward"]["mean"]["n_violations"] == pytest.approx(2 / 7)
    assert set(metrics["by_kind"]) == {"upstream", "explicit_request"}


def test_eval_and_train_drivers_score_identically(tmp_path):
    result, registry, store = evaluate(tmp_path, personas=("neutral",))
    evaluated = {
        t.task_id: t for t in read_trajectories([result.out_dir / "trajectories.jsonl"])
    }
    composer = RewardComposer(REWARD)
    for task_id, eval_trajectory in evaluated.items():
        episode, policy = train_episode(registry, store, task_id, "neutral")
        train_trajectory = episode.trajectory
        assert comparable(train_trajectory) == comparable(eval_trajectory), task_id
        assert episode.reward == composer(eval_trajectory), task_id
        # Token bookkeeping: policy tokens are exactly what was generated.
        generated = sum(
            toy_tau2.completion_tokens(*toy_tau2.agent_turn(task_id, i))
            for i in range(policy.index)
        )
        assert sum(episode.response_mask) == generated
        assert len(episode.response_ids) == len(episode.response_mask)
        assert policy.prompts[0] == episode.prompt_ids
        assert episode.response_logprobs is not None


def test_training_prompt_matches_rendered_conversation(tmp_path):
    registry, store = protocol()
    episode, policy = train_episode(registry, store, "0", "terse")
    fake = FakeChatTokenizer()
    text = fake.decode(episode.prompt_ids + episode.response_ids)
    assert text.startswith("<|im_start|>system\n")
    assert "Hi! How can I help you today?" in text
    assert "Hi, I want to cancel my order o1." in text
    assert "cancelled" in text
    assert episode.trajectory.persona == "terse"
    assert episode.trajectory.success and episode.reward.total == 1.0
    # Every later prompt extends the previous context token by token.
    for earlier, later in zip(policy.prompts, policy.prompts[1:], strict=False):
        assert later[: len(earlier)] == earlier
