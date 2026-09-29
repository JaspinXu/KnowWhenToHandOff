"""verl ``AgentLoopBase`` adapter for tau2 episodes (task T3.1).

Registered in verl through the agent-loop YAML written by ``scripts/train_rl.py``::

    - name: tau2_agent
      _target_: handoff.rollout.verl_loop.Tau2AgentLoop
      handoff_config_path: <run_dir>/agent_loop_settings.json

Each dataset row carries a fully serialized, persona-styled tau2 task in
``extra_info``. The loop simulates the episode with :class:`Tau2EpisodeRunner`,
returns token IDs with the policy mask, and sets ``reward_score`` to the
composite reward so verl uses it directly (no separate reward function runs).
Every reward term is exposed through ``reward_extra_info`` and written, with a
timestamp, to a per-process JSON-lines file for the reward-hacking monitor.

The user simulator runs in a separate Slurm job that may be restarted (it has a
time limit). The loop re-reads the simulator's endpoint file whenever it
changes, and when an episode fails while the simulator is unreachable it waits
for the simulator to come back (``user_outage_timeout_s``), retries the episode
once, and otherwise stops the training job so it can be resumed later.

Tested against verl 0.8.0 (pinned in ADR-001); only ``AgentLoopBase``,
``AgentLoopOutput``, ``register`` and ``server_manager.generate`` are used.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import random
import socket
import threading
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)

from handoff.rewards.composer import RewardBreakdown, RewardComposer, RewardConfig
from handoff.rollout.episode import (
    EpisodeResult,
    EpisodeSettings,
    GenerationOutput,
    InfrastructureError,
    PromptTooLongError,
    Tau2EpisodeRunner,
)
from handoff.rollout.scoring import EpisodeContext
from handoff.sim.endpoints import (
    Endpoint,
    litellm_args,
    load_api_key_into_env,
    probe,
    read_endpoint,
    register_local_models,
)

logger = logging.getLogger(__name__)
_RUNNERS: dict[str, tuple[Tau2EpisodeRunner, dict[str, Any], UserSimulatorLink]] = {}
_LOCK = threading.Lock()
EXTRA_KEYS = (
    *RewardBreakdown.__dataclass_fields__,
    "success",
    "transferred",
    "should_transfer",
    "n_turns",
    "n_messages",
    "user_stop_failure",
    "infra_failure",
)
PROBE_INTERVAL_S = 30.0


class UserSimulatorUnavailable(RuntimeError):
    """The user simulator stayed unreachable for longer than the outage timeout."""


class VerlGenerator:
    """Adapt verl's server manager to :class:`PolicyGenerator`."""

    def __init__(self, server_manager: Any, sampling_params: dict[str, Any]) -> None:
        """Use one sticky request ID per episode for prefix caching."""
        self.server_manager = server_manager
        self.sampling_params = dict(sampling_params)
        self.request_id = uuid4().hex

    async def generate(
        self, prompt_ids: list[int], *, max_tokens: int
    ) -> GenerationOutput:
        """Generate one turn; verl mutates the params dict, so pass a copy."""
        output = await self.server_manager.generate(
            request_id=self.request_id,
            prompt_ids=list(prompt_ids),
            sampling_params={**self.sampling_params, "max_tokens": int(max_tokens)},
        )
        if getattr(output, "stop_reason", None) == "aborted":
            return GenerationOutput(token_ids=[])
        return GenerationOutput(list(output.token_ids), output.log_probs)


def _user_settings(settings: dict[str, Any], endpoint: Endpoint) -> dict[str, Any]:
    """Episode fields that depend on the user simulator's endpoint."""
    return {
        "user_model": endpoint.litellm_model,
        "user_args": litellm_args(
            endpoint,
            temperature=float(settings.get("user_temperature", 0.7)),
            max_tokens=settings.get("user_max_tokens"),
        ),
    }


class UserSimulatorLink:
    """The user simulator's endpoint, re-read whenever its file changes."""

    def __init__(self, runner: Tau2EpisodeRunner, settings: dict[str, Any]) -> None:
        """Remember where the endpoint file is and when it was last read."""
        self.runner = runner
        self.settings = settings
        self.path = Path(settings["user_endpoint_file"])
        self.api_key_file = (
            Path(settings["api_key_file"]) if settings.get("api_key_file") else None
        )
        self.mtime_ns: int | None = None
        self.endpoint: Endpoint | None = None
        self.lock = threading.Lock()

    def refresh(self) -> Endpoint:
        """Re-read the endpoint file if it changed; update the runner's settings."""
        with self.lock:
            mtime = self.path.stat().st_mtime_ns
            if self.endpoint is None or mtime != self.mtime_ns:
                endpoint = read_endpoint(self.path)
                if self.endpoint is not None and endpoint != self.endpoint:
                    logger.warning("user simulator moved to %s", endpoint.base_url)
                load_api_key_into_env(self.api_key_file)
                register_local_models([endpoint.served_model_name])
                self.runner.settings = dataclasses.replace(
                    self.runner.settings, **_user_settings(self.settings, endpoint)
                )
                self.endpoint, self.mtime_ns = endpoint, mtime
            return self.endpoint

    def reachable(self) -> bool:
        """Whether the (possibly new) endpoint answers ``GET /models``."""
        try:
            endpoint = self.refresh()
        except (OSError, ValueError, TypeError):
            return False
        return not probe(
            endpoint, api_key=os.environ.get("OPENAI_API_KEY"), timeout_s=10.0
        )

    async def wait(self, timeout_s: float) -> bool:
        """Wait out a simulator outage after a failed episode.

        Returns ``False`` at once if the simulator answers (the failure had
        another cause) and ``True`` once it is back after an outage. Raises
        :class:`UserSimulatorUnavailable` when it stays down for ``timeout_s``:
        stopping the job then is cheaper than training on placeholder episodes,
        and the run resumes from its last checkpoint.
        """
        if await asyncio.to_thread(self.reachable):
            return False
        logger.error("user simulator unreachable; waiting up to %.0f s", timeout_s)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            await asyncio.sleep(PROBE_INTERVAL_S)
            if await asyncio.to_thread(self.reachable):
                logger.warning("user simulator reachable again")
                return True
        raise UserSimulatorUnavailable(
            f"user simulator ({self.path}) unreachable for {timeout_s:.0f} s; "
            "restart it (slurm/submit.sh serve) and resume this run"
        )


def _load_runner(
    path: str, tokenizer: Any
) -> tuple[Tau2EpisodeRunner, dict[str, Any], UserSimulatorLink]:
    with _LOCK:
        if path in _RUNNERS:
            return _RUNNERS[path]
        settings = json.loads(Path(path).read_text(encoding="utf-8"))
        load_api_key_into_env(
            Path(settings["api_key_file"]) if settings.get("api_key_file") else None
        )
        user = read_endpoint(Path(settings["user_endpoint_file"]))
        episode_config = {
            **dict(settings["episode"]),
            **_user_settings(settings, user),
        }
        runner = Tau2EpisodeRunner(
            tokenizer,
            EpisodeSettings.from_mapping(episode_config),
            RewardComposer(RewardConfig.from_mapping(settings["reward"])),
            max_workers=int(settings.get("max_concurrent_episodes", 64)),
        )
        _RUNNERS[path] = (runner, settings, UserSimulatorLink(runner, settings))
        return _RUNNERS[path]


def _record(settings: dict[str, Any], record: dict[str, Any]) -> None:
    directory = Path(settings["run_dir"]) / "rollouts"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{socket.gethostname()}-{os.getpid()}.jsonl"
    with (directory / name).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": time.time(), **record}) + "\n")


def episode_extra(result: EpisodeResult) -> dict[str, float]:
    """``reward_extra_info`` of a finished episode (keys: :data:`EXTRA_KEYS`)."""
    trajectory = result.trajectory
    return {
        **result.reward.as_metrics(prefix=""),
        "success": float(trajectory.success),
        "transferred": float(trajectory.transferred),
        "should_transfer": float(trajectory.should_transfer),
        "n_turns": float(trajectory.n_turns),
        "n_messages": float(len(trajectory.messages)),
        "user_stop_failure": float(
            trajectory.termination_reason == "user_stop"
            and not trajectory.success
            and not trajectory.transferred
        ),
        "infra_failure": 0.0,
    }


@register("tau2_agent")
class Tau2AgentLoop(AgentLoopBase):
    """One tau2 conversation per dataset row."""

    def __init__(self, *args: Any, handoff_config_path: str | None = None, **kwargs):
        """Load (once per process) the episode runner described by the settings."""
        super().__init__(*args, **kwargs)
        path = handoff_config_path or os.environ.get("HANDOFF_AGENT_LOOP_CONFIG")
        if not path:
            raise ValueError("Tau2AgentLoop needs handoff_config_path")
        self.runner, self.settings, self.user_link = _load_runner(path, self.tokenizer)
        self.response_length = int(self.rollout_config.response_length)
        episode = self.runner.settings
        if episode.max_response_tokens > self.response_length:
            raise ValueError(
                "episode.max_response_tokens exceeds rollout.response_length; "
                "they must match"
            )
        if episode.max_prompt_tokens > int(self.rollout_config.prompt_length):
            raise ValueError(
                "episode.max_prompt_tokens exceeds rollout.prompt_length; "
                "they must match (verl pads prompts to that length)"
            )

    def _context(self, info: dict[str, Any], seed: int) -> EpisodeContext:
        return EpisodeContext(
            run_id=str(self.settings["run_id"]),
            domain=str(info["domain"]),
            task_id=str(info["task_id"]),
            source_task_id=str(info["source_task_id"]),
            split=str(info["split"]),
            kind=str(info["kind"]),
            persona=str(info["persona"]),
            should_transfer=bool(info["should_transfer"]),
            trial=0,
            seed=seed,
            policy_ckpt=str(self.settings.get("policy_ckpt", "")),
            user_sim=str(self.settings.get("user_sim", "")),
        )

    async def run(
        self, sampling_params: dict[str, Any], **kwargs: Any
    ) -> AgentLoopOutput:
        """Simulate, score and tokenize one episode."""
        info = dict(kwargs.get("extra_info") or {})
        task = json.loads(info["task_json"])
        greedy = float(sampling_params.get("temperature", 1.0)) == 0.0
        seed = (
            int(info.get("eval_seed", 0))
            if greedy
            else random.SystemRandom().randrange(1 << 31)
        )
        ctx = self._context(info, seed)
        link = self.user_link
        outage_timeout = float(self.settings.get("user_outage_timeout_s", 1800.0))
        for attempt in (1, 2):
            try:
                link.refresh()
                result = await self.runner.run(
                    task=task,
                    ctx=ctx,
                    seed=seed,
                    generator=VerlGenerator(self.server_manager, sampling_params),
                )
                break
            except InfrastructureError:
                logger.warning("aborted generation for %s/%s", ctx.domain, ctx.task_id)
                return self._placeholder(ctx, greedy, "aborted_generation")
            except PromptTooLongError as error:
                logger.error("%s/%s: %s", ctx.domain, ctx.task_id, error)
                return self._placeholder(ctx, greedy, "prompt_too_long")
            except Exception:  # noqa: BLE001 - a lost episode must not stop training
                # User-simulator errors, timeouts and similar failures end this
                # episode only; an outage of the simulator is waited out first.
                logger.exception("episode %s/%s failed", ctx.domain, ctx.task_id)
                if attempt == 1 and await link.wait(outage_timeout):
                    continue
                return self._placeholder(ctx, greedy, "episode_error")
        extra = episode_extra(result)
        if not any(result.response_mask):
            # The user ended the conversation before the policy spoke: nothing
            # to train on, but the real reward still enters the group baseline.
            return self._placeholder(
                ctx,
                greedy,
                "no_policy_turn",
                extra=extra,
                reward=float(result.reward.total),
                termination=result.trajectory.termination_reason,
            )
        _record(
            self.settings,
            {
                "validate": greedy,
                "domain": ctx.domain,
                "task_id": ctx.task_id,
                "persona": ctx.persona,
                "kind": ctx.kind,
                "termination": result.trajectory.termination_reason,
                **extra,
            },
        )
        response_ids = result.response_ids[: self.response_length]
        return AgentLoopOutput(
            prompt_ids=result.prompt_ids,
            response_ids=response_ids,
            response_mask=result.response_mask[: self.response_length],
            response_logprobs=(
                None
                if result.response_logprobs is None
                else result.response_logprobs[: self.response_length]
            ),
            reward_score=float(result.reward.total),
            num_turns=len(result.trajectory.messages),
            metrics={
                "generate_sequences": result.timings["generate"],
                "tool_calls": result.timings["environment"],
                "num_preempted": -1,
            },
            extra_fields={"reward_extra_info": extra},
        )

    def _placeholder(
        self,
        ctx: EpisodeContext,
        greedy: bool,
        reason: str,
        *,
        extra: dict[str, float] | None = None,
        reward: float = 0.0,
        termination: str | None = None,
    ) -> AgentLoopOutput:
        """A no-gradient output for an episode without usable policy tokens.

        Its response mask is all zeros, so it contributes no policy gradient;
        its reward still enters the GRPO group baseline. Without ``extra`` the
        episode counts as lost to infrastructure (reward 0, recorded and
        watched through the infrastructure-failure share).
        """
        pad = int(self.tokenizer.pad_token_id or 0)
        if extra is None:
            extra = dict.fromkeys(EXTRA_KEYS, 0.0)
            extra["infra_failure"] = 1.0
        _record(
            self.settings,
            {
                "validate": greedy,
                "domain": ctx.domain,
                "task_id": ctx.task_id,
                "persona": ctx.persona,
                "kind": ctx.kind,
                "termination": termination or reason,
                "placeholder": reason,
                **extra,
            },
        )
        return AgentLoopOutput(
            prompt_ids=[pad],
            response_ids=[pad],
            response_mask=[0],
            reward_score=reward,
            num_turns=0,
            metrics={"generate_sequences": 0.0, "tool_calls": 0.0, "num_preempted": -1},
            extra_fields={"reward_extra_info": extra},
        )
