"""Register the project's evaluation agent with the tau2 runner.

The agent is tau2's own ``LLMAgent`` (LiteLLM chat completions against a local
vLLM server) with the shared system prompt of :mod:`handoff.agents.prompts`.
Registration is idempotent and names encode the prompt variant, for example
``handoff_agent:handoff_v1``.

Turns the policy cannot complete end the episode with the same stop sentinel as
the training driver (see :mod:`handoff.rollout.tau2_convert`):

- a generation cut at the token limit (see :func:`hit_token_limit`) →
  ``context_window_exceeded`` with a ``truncated_generation`` format error;
- a message with neither text nor tool calls → ``agent_error`` with an
  ``empty_message`` format error (plain tau2 would raise and retry the episode,
  then count it as an infrastructure failure);
- a tool call whose arguments are not a JSON object (tau2 raises while building
  the call) → ``agent_error`` with an ``invalid_arguments`` format error;
- a request rejected because the conversation no longer fits the model's
  context → ``context_window_exceeded``.
"""

from __future__ import annotations

import json
from typing import Any

from handoff.agents.prompts import PROMPT_VARIANTS, agent_system_prompt

AGENT_NAME_PREFIX = "handoff_agent"


def agent_name(variant: str) -> str:
    """Registry name of the evaluation agent for a prompt variant."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}")
    return f"{AGENT_NAME_PREFIX}:{variant}"


def finish_reason(message: Any) -> str | None:
    """``finish_reason`` of the completion behind a tau2 assistant message."""
    raw = getattr(message, "raw_data", None)
    if not isinstance(raw, dict):
        return None
    choices = raw.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    return choices[0].get("finish_reason")


def hit_token_limit(message: Any, max_tokens: int | None) -> bool:
    """Whether a completion was cut at ``max_tokens``.

    vLLM reports ``finish_reason == "tool_calls"`` whenever it parsed a tool call,
    even from an unclosed ``<tool_call>`` block cut at the limit, so a tool-call
    turn that used every allowed token counts as cut too. (A turn whose end token
    fell exactly on the limit is indistinguishable here; the training driver,
    which sees the tokens, would accept it.)
    """
    reason = finish_reason(message)
    if reason == "length":
        return True
    usage = getattr(message, "usage", None) or {}
    tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return (
        reason == "tool_calls"
        and max_tokens is not None
        and isinstance(tokens, int)
        and tokens >= int(max_tokens)
    )


def is_context_overflow(error: BaseException) -> bool:
    """Whether a LiteLLM error means the prompt no longer fits the context."""
    import litellm

    if isinstance(error, litellm.ContextWindowExceededError):
        return True
    text = str(error).lower()
    return isinstance(error, litellm.BadRequestError) and (
        "maximum context length" in text or "context window" in text
    )


def _agent_class() -> type:
    import pydantic
    from tau2.agent.llm_agent import LLMAgent

    from handoff.rollout.tau2_convert import sentinel_of, stop_message

    class HandoffLLMAgent(LLMAgent):
        """tau2 LLM agent with the shared prompt and the shared stop sentinel."""

        def __init__(self, tools, domain_policy, llm, llm_args=None, variant=""):
            super().__init__(
                tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args
            )
            self.prompt_variant = variant

        @property
        def system_prompt(self) -> str:
            return agent_system_prompt(self.domain_policy, self.prompt_variant)

        def generate_next_message(self, message, state):
            try:
                reply, state = super().generate_next_message(message, state)
            except (json.JSONDecodeError, pydantic.ValidationError):
                reply = stop_message("agent_error", ("invalid_arguments",))
                state.messages.append(reply)
                return reply, state
            except Exception as error:
                if not is_context_overflow(error):
                    raise
                reply = stop_message("context_window_exceeded")
                state.messages.append(reply)
                return reply, state
            stop = None
            if hit_token_limit(reply, (self.llm_args or {}).get("max_tokens")):
                stop = stop_message(
                    "context_window_exceeded",
                    ("truncated_generation",),
                    usage=reply.usage,
                )
            elif not (reply.has_text_content() or reply.is_tool_call()):
                stop = stop_message(
                    "agent_error", ("empty_message",), usage=reply.usage
                )
            if stop is None:
                return reply, state
            state.messages[-1] = stop
            return stop, state

        @classmethod
        def is_stop(cls, message) -> bool:
            return sentinel_of(message) is not None or super().is_stop(message)

    return HandoffLLMAgent


def register_agents() -> list[str]:
    """Register one agent factory per prompt variant; return their names."""
    from tau2.registry import registry

    agent_class = _agent_class()
    names = []
    for variant in PROMPT_VARIANTS:
        name = agent_name(variant)
        names.append(name)
        if registry.get_agent_factory(name) is not None:
            continue

        def factory(tools: Any, domain_policy: str, _variant=variant, **kwargs: Any):
            return agent_class(
                tools=tools,
                domain_policy=domain_policy,
                llm=kwargs.get("llm"),
                llm_args=kwargs.get("llm_args"),
                variant=_variant,
            )

        registry.register_agent_factory(factory, name)
    return names
