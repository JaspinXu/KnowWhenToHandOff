"""Checks against the pinned GPU stack (vLLM 0.12, verl 0.8); skipped without it.

These run on CPU inside the cluster environment (the setup job runs the whole
test suite) and whenever vLLM and verl are installed. They pin down what the
drivers assume about the stack:

- the Hermes parser of the training driver agrees with vLLM's own parser;
- every model config yields a server command that vLLM's CLI accepts;
- every RL preset composes with verl's own Hydra config;
- the verl agent loop runs a complete (toy) tau2 episode end to end.
"""

import asyncio
import importlib.util
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

ROOT = Path(__file__).resolve().parents[1]
HAS_VLLM = importlib.util.find_spec("vllm") is not None
HAS_VERL = importlib.util.find_spec("verl") is not None
HAS_TAU2 = importlib.util.find_spec("tau2") is not None
needs_vllm = pytest.mark.skipif(not HAS_VLLM, reason="vLLM is not installed")
needs_verl = pytest.mark.skipif(not HAS_VERL, reason="verl is not installed")

GENERATIONS = [
    "Hello! How can I help?",
    '<tool_call>\n{"name": "get_order", "arguments": {"order_id": "o1"}}\n</tool_call>',
    'Let me check.\n<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>',
    '<tool_call>{"name": "a", "arguments": {"x": [1, 2]}}</tool_call>'
    '<tool_call>{"name": "b", "arguments": {"y": "é"}}</tool_call>',
    '<tool_call>{"name": "a", "arguments": {"x": 1}}',
    "<tool_call>{not json}</tool_call>",
    "<tool_call></tool_call>",
    '<tool_call>{"arguments": {}}</tool_call>',
    '<tool_call>{"name": 3, "arguments": {}}</tool_call>',
    '<tool_call>{"name": "a"}</tool_call>',
    '<tool_call>{"name": "a", "arguments": [1]}</tool_call>',
    '<tool_call>{"name": "a", "arguments": "{\\"x\\": 1}"}</tool_call>',
    'Done <tool_call>{"name": "a", "arguments": {}}</tool_call> trailing text',
]


@needs_vllm
@pytest.mark.parametrize("text", GENERATIONS)
def test_hermes_parser_matches_vllm(text):
    from fakes import FakeChatTokenizer
    from vllm.entrypoints.openai.tool_parsers.hermes_tool_parser import (
        Hermes2ProToolParser,
    )

    from handoff.rollout.hermes import parse_hermes

    served = Hermes2ProToolParser(FakeChatTokenizer()).extract_tool_calls(text, None)
    ours = parse_hermes(text)
    if not served.tools_called:
        assert ours.tool_calls == () and ours.error in (None, "invalid_tool_json")
        assert ours.content == served.content
        return
    assert ours.content == served.content
    arguments = [json.loads(call.function.arguments) for call in served.tool_calls]
    if all(isinstance(value, dict) for value in arguments):
        # tau2 accepts these calls: both drivers see identical tool calls.
        assert [(c.name, c.arguments) for c in ours.tool_calls] == [
            (call.function.name, value)
            for call, value in zip(served.tool_calls, arguments, strict=True)
        ]
    else:
        # tau2 cannot build a ToolCall from non-object arguments: the turn is lost.
        assert ours.error == "invalid_arguments" and ours.tool_calls == ()


@pytest.fixture(scope="module")
def vllm_parser():
    import vllm.platforms as platforms
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    try:
        return make_arg_parser(FlexibleArgumentParser())
    except RuntimeError:
        # No GPU on this machine: parse the arguments as the CPU platform would.
        from vllm.platforms.cpu import CpuPlatform

        platforms._current_platform = CpuPlatform()
        return make_arg_parser(FlexibleArgumentParser())


@needs_vllm
@pytest.mark.parametrize(
    "name",
    sorted(
        p.stem
        for p in (ROOT / "configs" / "model").glob("*.yaml")
        if p.stem != "custom"
    ),
)
def test_serve_commands_parse_with_vllm(name, vllm_parser):
    from omegaconf import OmegaConf
    from vllm.entrypoints.openai.cli_args import validate_parsed_serve_args

    from handoff.sim.serving import serve_spec_from_config

    model = OmegaConf.to_container(
        OmegaConf.load(ROOT / "configs" / "model" / f"{name}.yaml")
    )
    lock = {model["name"]: {"local_path": f"/scratch/hf/{name}", "revision": "a" * 40}}
    spec = serve_spec_from_config(model, role="policy", port=8000, models_lock=lock)
    args = vllm_parser.parse_args(spec.command()[3:])
    validate_parsed_serve_args(args)
    assert args.served_model_name == [model["served_name"]]
    assert args.generation_config == "vllm" and args.enable_auto_tool_choice


@needs_verl
@pytest.mark.parametrize("lora", [False, True])
def test_rl_presets_compose_with_verl(lora, tmp_path, monkeypatch):
    from handoff.cli import compose_config, container
    from handoff.train.rl_launch import build_verl_overrides, validate_overrides

    monkeypatch.setenv("HANDOFF_SCRATCH", str(tmp_path))
    overrides = ["rl=grpo", f"policy.init_path='{tmp_path.as_posix()}/b2'"]
    overrides += [f"rl.use_lora={str(lora).lower()}"]
    config = container(compose_config(ROOT, "train_rl", overrides))
    validate_overrides(
        build_verl_overrides(
            config,
            run_id="R1-s1-20261020",
            run_dir=tmp_path / "run",
            train_file=tmp_path / "t.parquet",
            val_file=tmp_path / "v.parquet",
            agent_loop_file=tmp_path / "loop.yaml",
            reward_guard_file=tmp_path / "guard.py",
        )
    )


class ScriptedServer:
    """verl server-manager double replaying one task's scripted policy."""

    def __init__(self, task_id):
        from toy_tau2 import ScriptedPolicy

        self.policy = ScriptedPolicy(task_id)
        self.calls = []

    async def generate(self, request_id, *, prompt_ids, sampling_params, **_):
        from verl.workers.rollout.replica import TokenOutput

        self.calls.append((request_id, dict(sampling_params)))
        output = await self.policy.generate(
            prompt_ids, max_tokens=sampling_params["max_tokens"]
        )
        return TokenOutput(
            token_ids=output.token_ids,
            log_probs=output.log_probs,
            stop_reason="completed",
        )


def _toy_agent_loop(tmp_path, monkeypatch, completion=None, task_id="0"):
    """A Tau2AgentLoop on the toy domain with a scripted policy and user."""
    import tau2.user.user_simulator as user_simulator
    import tau2.utils.llm_utils as llm_utils
    import toy_tau2
    from fakes import FakeChatTokenizer
    from omegaconf import OmegaConf
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap

    from handoff.envs.personas import Persona
    from handoff.envs.tau2_adapter import styled_task
    from handoff.rollout.verl_loop import Tau2AgentLoop
    from handoff.sim.endpoints import Endpoint, write_endpoint

    toy_tau2.register_toy_domain()
    monkeypatch.setattr(llm_utils, "completion", completion or toy_tau2.fake_completion)
    guidelines = tmp_path / "guidelines.md"
    guidelines.write_text("Simulate the user.<PERSONA_GUIDELINES>", encoding="utf-8")
    monkeypatch.setattr(user_simulator, "GLOBAL_USER_SIM_GUIDELINES_PATH", guidelines)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    endpoint = tmp_path / "usersim.json"
    write_endpoint(
        endpoint,
        Endpoint(
            role="usersim",
            base_url="http://toy.invalid:1/v1",
            served_model_name=toy_tau2.USER_MODEL,
        ),
    )
    settings = tmp_path / "agent_loop_settings.json"
    settings.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "R2-toy-s1-20261020",
                "run_dir": str(tmp_path / "run"),
                "user_endpoint_file": str(endpoint),
                "api_key_file": None,
                "user_temperature": 0.0,
                "user_max_tokens": 64,
                "user_outage_timeout_s": 5,
                "episode": {"max_prompt_tokens": 4096, "max_response_tokens": 8192},
                "reward": {
                    "lambda_violation": 0.25,
                    "lambda_fp": 0.5,
                    "lambda_fn": 1.0,
                },
                "policy_ckpt": "toy@1",
                "user_sim": "toy-user@1",
                "max_concurrent_episodes": 4,
            }
        ),
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {"prompt_length": 4096, "response_length": 8192}
            },
            "data": {},
        }
    )
    server = ScriptedServer(task_id)
    loop = Tau2AgentLoop(
        trainer_config=DictConfigWrap(config),
        server_manager=server,
        tokenizer=FakeChatTokenizer(),
        processor=None,
        dataset_cls=None,
        data_config=DictConfigWrap(config.data),
        handoff_config_path=str(settings),
    )
    task = styled_task(toy_tau2.UPSTREAM_TASKS[int(task_id)], Persona("neutral", ""))
    extra_info = {
        "domain": toy_tau2.DOMAIN,
        "task_id": task_id,
        "source_task_id": task_id,
        "split": "train",
        "kind": "upstream",
        "persona": "neutral",
        "should_transfer": False,
        "eval_seed": 5,
        "task_json": json.dumps(task),
    }
    return loop, server, extra_info, endpoint


def _records(tmp_path):
    return [
        json.loads(line)
        for path in (tmp_path / "run" / "rollouts").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]


@needs_verl
@pytest.mark.skipif(not HAS_TAU2, reason="tau2 is not installed")
@pytest.mark.enable_socket
def test_verl_agent_loop_runs_a_toy_episode(tmp_path, monkeypatch):
    import toy_tau2

    loop, server, extra_info, _ = _toy_agent_loop(tmp_path, monkeypatch)
    output = asyncio.run(
        loop.run({"temperature": 1.0, "top_p": 1.0}, extra_info=extra_info)
    )
    assert output.reward_score == 1.0
    assert (
        len(output.response_ids)
        == len(output.response_mask)
        == len(output.response_logprobs)
    )
    assert sum(output.response_mask) == sum(
        toy_tau2.completion_tokens(*toy_tau2.agent_turn("0", i)) for i in range(4)
    )
    extra = output.extra_fields["reward_extra_info"]
    assert extra["success"] == 1.0 and extra["infra_failure"] == 0.0
    assert {request_id for request_id, _ in server.calls} == {server.calls[0][0]}
    assert all(params["max_tokens"] == 2048 for _, params in server.calls)
    lines = _records(tmp_path)
    assert lines and lines[-1]["task_id"] == "0" and lines[-1]["validate"] is False
    assert isinstance(lines[-1]["time"], float)


@needs_verl
@pytest.mark.skipif(not HAS_TAU2, reason="tau2 is not installed")
@pytest.mark.enable_socket
def test_episode_without_a_policy_turn_is_a_placeholder(tmp_path, monkeypatch):
    """The user ends the conversation at once: no policy tokens, no crash."""
    import toy_tau2

    def user_leaves(**kwargs):
        if kwargs["model"] == f"openai/{toy_tau2.USER_MODEL}":
            return toy_tau2._response(toy_tau2.USER_MODEL, "###STOP###", [], "stop", 3)
        return toy_tau2.fake_completion(**kwargs)

    loop, server, extra_info, _ = _toy_agent_loop(tmp_path, monkeypatch, user_leaves)
    output = asyncio.run(loop.run({"temperature": 1.0}, extra_info=extra_info))
    assert output.response_mask == [0] and output.prompt_ids and not server.calls
    extra = output.extra_fields["reward_extra_info"]
    assert extra["infra_failure"] == 0.0
    assert _records(tmp_path)[-1]["placeholder"] == "no_policy_turn"


@needs_verl
@pytest.mark.skipif(not HAS_TAU2, reason="tau2 is not installed")
@pytest.mark.enable_socket
def test_agent_loop_waits_out_a_simulator_restart(tmp_path, monkeypatch):
    """The simulator dies mid-run and comes back on another node."""
    import litellm
    import toy_tau2

    import handoff.rollout.verl_loop as verl_loop
    from handoff.sim.endpoints import Endpoint, write_endpoint

    state = {"down": True, "probes": 0}

    def flaky(**kwargs):
        if kwargs["model"] == f"openai/{toy_tau2.USER_MODEL}" and state["down"]:
            raise litellm.APIConnectionError(
                message="connection refused", llm_provider="openai", model="x"
            )
        return toy_tau2.fake_completion(**kwargs)

    loop, server, extra_info, endpoint = _toy_agent_loop(tmp_path, monkeypatch, flaky)

    def probe(endpoint_record, **_):
        state["probes"] += 1
        if state["probes"] >= 2:  # the restarted server has published itself
            state["down"] = False
            return "" if endpoint_record.base_url.startswith("http://node9") else "no"
        write_endpoint(
            endpoint,
            Endpoint(
                role="usersim",
                base_url="http://node9:2/v1",
                served_model_name=toy_tau2.USER_MODEL,
            ),
        )
        return "ConnectionRefusedError"

    monkeypatch.setattr(verl_loop, "probe", probe)
    monkeypatch.setattr(verl_loop, "PROBE_INTERVAL_S", 0.01)
    output = asyncio.run(loop.run({"temperature": 1.0}, extra_info=extra_info))
    assert output.reward_score == 1.0 and sum(output.response_mask) > 0
    assert loop.runner.settings.user_args["api_base"] == "http://node9:2/v1"

    # a simulator that stays down stops the job (resume later)
    state.update(down=True, probes=-1000)
    monkeypatch.setattr(verl_loop, "probe", lambda *a, **k: "ConnectionRefusedError")
    with pytest.raises(verl_loop.UserSimulatorUnavailable):
        asyncio.run(loop.run({"temperature": 1.0}, extra_info=extra_info))


@needs_verl
def test_every_saved_step_is_exported_before_rotation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from omegaconf import OmegaConf
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    import handoff.train.verl_main as verl_main

    checkpoints = tmp_path / "run" / "checkpoints"
    (checkpoints / "global_step_10" / "actor").mkdir(parents=True)
    exported = []

    def fake_export(step_dir, target):
        exported.append((step_dir.name, target))
        target.mkdir(parents=True, exist_ok=True)
        (target / "handoff_export.json").write_text("{}")
        return target

    saved = []
    monkeypatch.setattr(verl_main, "export_step", fake_export)
    monkeypatch.setattr("handoff.train.export.export_step", fake_export)
    monkeypatch.setattr(
        RayPPOTrainer, "_save_checkpoint", lambda self: saved.append(self.global_steps)
    )
    config = OmegaConf.create({"trainer": {"default_local_dir": str(checkpoints)}})
    verl_main._install_checkpoint_export(config)
    # a step saved before a crash is exported at start-up
    assert exported == [
        ("global_step_10", tmp_path / "run" / "export" / "global_step_10")
    ]
    trainer = SimpleNamespace(global_steps=20)
    RayPPOTrainer._save_checkpoint(trainer)
    assert saved == [20]
    assert exported[-1] == (
        "global_step_20",
        tmp_path / "run" / "export" / "global_step_20",
    )
    verl_main._install_checkpoint_export(config)  # installing twice wraps once
    RayPPOTrainer._save_checkpoint(SimpleNamespace(global_steps=30))
    assert saved == [20, 30] and len(exported) == 3


@needs_vllm
@pytest.mark.enable_socket  # the test client's event loop uses a local socketpair
def test_server_middleware_protects_every_route_but_health(monkeypatch):
    """vLLM's --api-key covers /v1 only; ours closes /pause, /resume, ..."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from handoff.sim.vllm_auth import require_api_key

    monkeypatch.setenv("VLLM_API_KEY", "secret")
    app = FastAPI()
    app.middleware("http")(require_api_key)
    for path in ("/pause", "/health", "/v1/models"):
        app.add_api_route(path, lambda: {"ok": True}, methods=["GET", "POST"])
    client = TestClient(app)

    def pause(key=None):
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        return client.post("/pause", headers=headers).status_code

    assert (pause(), pause("nope"), pause("secret")) == (401, 401, 200)
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
