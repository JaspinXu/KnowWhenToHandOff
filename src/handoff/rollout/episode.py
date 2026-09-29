"""Training-side episode driver: tau2's orchestrator with token-in/token-out policy.

The evaluation driver and this driver share everything except how the policy
produces a message. Here tau2's own ``Orchestrator`` runs the conversation (user
simulator, tool execution, termination rules) in a worker thread, and a bridge
agent hands every policy turn to :class:`PolicySession`, which lives on the
asyncio event loop and talks to the rollout server with token IDs:

- sampled IDs are appended unchanged with mask 1;
- user/tool observations are rendered by :class:`ChatDeltaTokenizer` with mask 0;
- generations are parsed with the Hermes parser that vLLM uses at evaluation.

Turns the policy cannot complete end the episode through the same stop sentinel
as the evaluation agent (:mod:`handoff.rollout.tau2_convert`). The finished
``SimulationRun`` is finalized, scored by tau2's evaluator and turned into a
:class:`Trajectory` by the same code as in evaluation, so rewards are identical
for identical conversations (see ``tests/test_tau2_integration.py``).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from handoff.rewards.composer import RewardBreakdown, RewardComposer
from handoff.rollout.chat_delta import ChatDeltaTokenizer
from handoff.rollout.hermes import parse_hermes
from handoff.rollout.schema import DEFAULT_GREETING, Trajectory
from handoff.rollout.scoring import EpisodeContext


@dataclass(frozen=True)
class GenerationOutput:
    """Token IDs (and optional log-probabilities) of one policy generation."""

    token_ids: list[int]
    log_probs: list[float] | None = None


class PolicyGenerator(Protocol):
    """Token-in/token-out generation (verl's server manager implements this)."""

    async def generate(
        self, prompt_ids: list[int], *, max_tokens: int
    ) -> GenerationOutput:
        """Generate one assistant turn for the given prompt IDs."""


class InfrastructureError(RuntimeError):
    """The rollout server returned no tokens (aborted request)."""


def generation_eos_ids(model_path: str) -> list[int]:
    """``eos_token_id`` entries of a local model's ``generation_config.json``."""
    if not model_path:
        return []
    path = Path(model_path) / "generation_config.json"
    if not path.is_file():
        return []
    value = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id")
    if value is None:
        return []
    return [int(item) for item in (value if isinstance(value, list) else [value])]


@dataclass(frozen=True)
class EpisodeSettings:
    """Limits and simulator settings of one training episode.

    The turn budget mirrors evaluation: every policy turn may use
    ``max_tokens_per_turn`` tokens (the evaluation agent's ``max_tokens``), and
    the episode ends with ``context_window_exceeded`` (no format error) as soon
    as the conversation plus one full turn would exceed ``max_context_tokens``
    (the evaluation server's ``max_model_len``), where vLLM would reject the
    request. ``max_response_tokens`` (verl's response length) is a second limit
    for training only.
    """

    prompt_variant: str = "handoff_v1"
    max_steps: int = 200
    max_errors: int = 10
    max_prompt_tokens: int = 8192
    max_response_tokens: int = 24576
    max_tokens_per_turn: int = 2048
    max_context_tokens: int = 32768
    user_model: str = ""
    user_args: Mapping[str, Any] = field(default_factory=dict)
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    simulation_timeout_s: float | None = 1800.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> EpisodeSettings:
        """Build from a resolved config mapping, rejecting unknown keys."""
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown episode settings: {sorted(unknown)}")
        return cls(**dict(data))


@dataclass(frozen=True)
class Decision:
    """What the policy decided in one turn, in tau2-independent form.

    ``stop_reason`` is set when the turn cannot be used and the episode must
    end (``context_window_exceeded`` or ``agent_error``); ``format_errors``
    then names the defect of that turn.
    """

    content: str | None = None
    tool_calls: tuple[tuple[str, str, dict[str, Any]], ...] = ()
    stop_reason: str | None = None
    format_errors: tuple[str, ...] = ()
    completion_tokens: int = 0
    prompt_tokens: int = 0

    @property
    def stop(self) -> bool:
        """Whether the episode ends on this turn."""
        return self.stop_reason is not None


class PromptTooLongError(ValueError):
    """The initial prompt exceeds the configured prompt budget."""


class PolicySession:
    """Policy-side token state for one episode (runs on the event loop)."""

    def __init__(
        self,
        delta: ChatDeltaTokenizer,
        generator: PolicyGenerator,
        settings: EpisodeSettings,
        system_prompt: str,
        stop_token_ids: Sequence[int],
        decode: Any,
    ) -> None:
        """Keep the tokenizer, generator and budgets for one conversation."""
        self.delta = delta
        self.generator = generator
        self.settings = settings
        self.system_prompt = system_prompt
        self.stop_token_ids = frozenset(int(i) for i in stop_token_ids)
        self.decode = decode
        self.initial_history: list[dict[str, Any]] = [
            {"role": "assistant", "content": DEFAULT_GREETING}
        ]
        self.history: list[dict[str, Any]] = []
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] | None = []
        self.stop_reason: str | None = None
        self.turns = 0
        self.generation_seconds = 0.0

    @property
    def started(self) -> bool:
        """Whether the initial prompt has been built."""
        return bool(self.prompt_ids)

    def _append(self, ids: Sequence[int], owner: int, logprobs: Sequence[float] | None):
        self.response_ids.extend(ids)
        self.response_mask.extend([owner] * len(ids))
        if self.response_logprobs is not None:
            if logprobs is None and owner == 1:
                self.response_logprobs = None
            else:
                values = list(logprobs) if logprobs is not None else [0.0] * len(ids)
                self.response_logprobs.extend(values)

    def set_initial_history(self, messages: Sequence[Mapping[str, Any]]) -> None:
        """Use tau2's opening messages (normally its greeting) before the first turn."""
        if self.started:
            raise RuntimeError("the episode has already started")
        self.initial_history = [dict(message) for message in messages]

    def _stop(
        self,
        reason: str,
        errors: tuple[str, ...] = (),
        *,
        completion_tokens: int = 0,
        prompt_tokens: int = 0,
    ) -> Decision:
        self.stop_reason = reason
        return Decision(
            stop_reason=reason,
            format_errors=errors,
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
        )

    def _fits(self, observation_tokens: int) -> bool:
        """Whether a full turn still fits after ``observation_tokens`` more tokens.

        The context check is the one vLLM applies at evaluation (prompt plus
        ``max_tokens`` within ``max_model_len``); the response check keeps the
        episode inside verl's response length.
        """
        turn = self.settings.max_tokens_per_turn
        response = len(self.response_ids) + observation_tokens
        context = len(self.prompt_ids) + response
        return (
            context + turn <= self.settings.max_context_tokens
            and response + turn <= self.settings.max_response_tokens
        )

    async def step(self, new_messages: Sequence[Mapping[str, Any]]) -> Decision:
        """Consume observations, generate one turn and parse it."""
        new = [dict(message) for message in new_messages]
        if not self.started:
            self.history = [
                {"role": "system", "content": self.system_prompt},
                *self.initial_history,
                *new,
            ]
            self.prompt_ids = self.delta.prompt_ids(self.history)
            if len(self.prompt_ids) > self.settings.max_prompt_tokens:
                raise PromptTooLongError(
                    f"initial prompt has {len(self.prompt_ids)} tokens; raise "
                    "max_prompt_tokens (verl data.max_prompt_length)"
                )
            if not self._fits(0):
                return self._stop("context_window_exceeded")
        else:
            observation = self.delta.observation_ids(self.history, new)
            if not self._fits(len(observation)):
                return self._stop("context_window_exceeded")
            self._append(observation, 0, None)
            self.history.extend(new)
        max_tokens = self.settings.max_tokens_per_turn
        context = len(self.prompt_ids) + len(self.response_ids)
        started = time.perf_counter()
        output = await self.generator.generate(
            self.prompt_ids + self.response_ids, max_tokens=max_tokens
        )
        self.generation_seconds += time.perf_counter() - started
        ids = list(output.token_ids)
        if not ids:
            raise InfrastructureError("rollout server returned no tokens")
        logprobs = None if output.log_probs is None else list(output.log_probs)
        if logprobs is not None and len(logprobs) != len(ids):
            logprobs = None
        self._append(ids, 1, logprobs)
        self.turns += 1
        if ids[-1] not in self.stop_token_ids:
            return self._stop(
                "context_window_exceeded",
                ("truncated_generation",),
                completion_tokens=len(ids),
                prompt_tokens=context,
            )
        # vLLM's hermes parser decodes with special tokens kept and without the
        # final end-of-turn token; mirror that exactly.
        parsed = parse_hermes(self.decode(ids[:-1]))
        content = parsed.content
        if parsed.error == "invalid_arguments":
            # tau2 rejects this turn before it becomes a message, so evaluation
            # never sees its token usage; report none here either.
            return self._stop("agent_error", ("invalid_arguments",))
        if not parsed.tool_calls and not (content or "").strip():
            return self._stop(
                "agent_error",
                ("empty_message",),
                completion_tokens=len(ids),
                prompt_tokens=context,
            )
        calls = tuple(
            (f"call_{self.turns:03d}_{index}", call.name, dict(call.arguments))
            for index, call in enumerate(parsed.tool_calls)
        )
        message: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if calls:
            message["tool_calls"] = [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
                for call_id, name, arguments in calls
            ]
        self.history.append(message)
        return Decision(
            content=content,
            tool_calls=calls,
            completion_tokens=len(ids),
            prompt_tokens=context,
        )


def make_bridge_agent(
    tools: Sequence[Any],
    domain_policy: str,
    session: PolicySession,
    loop: asyncio.AbstractEventLoop,
) -> Any:
    """Create a tau2 half-duplex agent that forwards turns to ``session``.

    It behaves like tau2's ``LLMAgent`` (same opening history, same stop
    detection) except that generation happens token-in/token-out on ``loop``.
    """
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.agent.llm_agent import LLMAgent
    from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall
    from tau2.utils.llm_utils import to_litellm_messages

    from handoff.rollout.tau2_convert import sentinel_of, stop_message

    class BridgeAgent(HalfDuplexAgent):
        """Runs in the orchestrator thread; generation happens on the loop."""

        def get_init_state(self, message_history=None):
            history = list(message_history or [])
            session.set_initial_history(to_litellm_messages(history))
            return history

        def generate_next_message(self, message, state):
            incoming = (
                list(message.tool_messages)
                if isinstance(message, MultiToolMessage)
                else [message]
            )
            future = asyncio.run_coroutine_threadsafe(
                session.step(to_litellm_messages(incoming)), loop
            )
            decision: Decision = future.result()
            usage = (
                {
                    "completion_tokens": decision.completion_tokens,
                    "prompt_tokens": decision.prompt_tokens,
                }
                if decision.completion_tokens
                else None
            )
            if decision.stop:
                reply = stop_message(
                    decision.stop_reason, decision.format_errors, usage=usage
                )
            else:
                # Mirror vLLM's hermes parser: text before the first tool call is
                # kept as content next to the tool calls.
                reply = AssistantMessage(
                    role="assistant",
                    content=decision.content,
                    tool_calls=[
                        ToolCall(id=call_id, name=name, arguments=arguments)
                        for call_id, name, arguments in decision.tool_calls
                    ]
                    or None,
                    cost=0.0,
                    usage=usage,
                )
            return reply, [*state, *incoming, reply]

        @classmethod
        def is_stop(cls, message) -> bool:
            return sentinel_of(message) is not None or LLMAgent.is_stop(message)

        def set_seed(self, seed: int) -> None:
            return None

    return BridgeAgent(tools=list(tools), domain_policy=domain_policy)


@dataclass
class EpisodeResult:
    """Tokens, trajectory and reward of one finished training episode."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float] | None
    trajectory: Trajectory
    reward: RewardBreakdown
    timings: dict[str, float]


class Tau2EpisodeRunner:
    """Run and score training episodes; one instance per worker process."""

    def __init__(
        self,
        tokenizer: Any,
        settings: EpisodeSettings,
        composer: RewardComposer,
        *,
        executor: Executor | None = None,
        max_workers: int = 64,
    ) -> None:
        """Keep the tokenizer, limits, reward composer and a thread pool."""
        self.tokenizer = tokenizer
        self.settings = settings
        self.composer = composer
        self.executor = executor or ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="tau2-episode"
        )
        self.stop_token_ids = self._stop_ids(tokenizer)
        if settings.user_model.startswith("openai/"):
            from handoff.sim.endpoints import register_local_models

            register_local_models([settings.user_model.removeprefix("openai/")])

    @staticmethod
    def _stop_ids(tokenizer: Any) -> list[int]:
        """End-of-turn IDs: the tokenizer's EOS plus the generation config's.

        vLLM stops on every ``eos_token_id`` of the model's
        ``generation_config.json`` (Qwen lists ``<|im_end|>`` and
        ``<|endoftext|>``), in verl's rollout server and in the evaluation server
        alike, so a turn ending on any of them is complete.
        """
        ids = {int(tokenizer.convert_tokens_to_ids(tokenizer.eos_token))}
        if getattr(tokenizer, "eos_token_id", None) is not None:
            ids.add(int(tokenizer.eos_token_id))
        ids.update(generation_eos_ids(str(getattr(tokenizer, "name_or_path", ""))))
        return sorted(ids)

    def _decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    async def run(
        self,
        *,
        task: Mapping[str, Any],
        ctx: EpisodeContext,
        seed: int,
        generator: PolicyGenerator,
    ) -> EpisodeResult:
        """Simulate one conversation, evaluate it and compose the reward."""
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.runner import build_environment
        from tau2.user.user_simulator import UserSimulator

        from handoff.agents.prompts import agent_system_prompt
        from handoff.envs.tau2_adapter import task_from_dict
        from handoff.rollout.tau2_convert import (
            assert_deterministic_reward_basis,
            domain_tool_sets,
            finalize_simulation,
            gold_actions_from_task,
            trajectory_from_simulation,
        )

        loop = asyncio.get_running_loop()
        tau2_task = task_from_dict(task)
        assert_deterministic_reward_basis(tau2_task)
        started = time.perf_counter()
        environment = build_environment(ctx.domain)
        tools = environment.get_tools()
        policy = environment.get_policy()
        write_tools, known_tools = domain_tool_sets(environment)
        delta = ChatDeltaTokenizer(
            self.tokenizer,
            tools=[tool.openai_schema for tool in tools],
            chat_template_kwargs=self.settings.chat_template_kwargs,
        )
        session = PolicySession(
            delta,
            generator,
            self.settings,
            agent_system_prompt(policy, self.settings.prompt_variant),
            self.stop_token_ids,
            self._decode,
        )
        agent = make_bridge_agent(tools, policy, session, loop)
        try:
            user_tools = (
                environment.get_user_tools(include=tau2_task.user_tools) or None
            )
        except Exception:  # noqa: BLE001 - tau2 raises when a domain has no user tools
            user_tools = None
        user = UserSimulator(
            llm=self.settings.user_model,
            instructions=str(tau2_task.user_scenario),
            tools=user_tools,
            llm_args=dict(self.settings.user_args),
        )
        orchestrator = Orchestrator(
            domain=ctx.domain,
            agent=agent,
            user=user,
            environment=environment,
            task=tau2_task,
            max_steps=self.settings.max_steps,
            max_errors=self.settings.max_errors,
            seed=seed,
            validate_communication=False,
            timeout=self.settings.simulation_timeout_s,
        )
        simulation = await loop.run_in_executor(self.executor, orchestrator.run)
        format_errors = finalize_simulation(simulation, tau2_task, ctx.domain)
        full_ctx = dataclasses.replace(
            ctx,
            write_tools=write_tools,
            known_tools=known_tools,
            gold_actions=ctx.gold_actions or gold_actions_from_task(tau2_task),
        )
        trajectory = trajectory_from_simulation(
            full_ctx,
            simulation,
            extra_format_errors=format_errors,
            extra={"stop_reason": session.stop_reason, "policy_turns": session.turns},
        )
        total = time.perf_counter() - started
        return EpisodeResult(
            prompt_ids=session.prompt_ids,
            response_ids=session.response_ids,
            response_mask=session.response_mask,
            response_logprobs=session.response_logprobs,
            trajectory=trajectory,
            reward=self.composer(trajectory),
            timings={
                "generate": session.generation_seconds,
                "environment": max(total - session.generation_seconds, 0.0),
            },
        )
