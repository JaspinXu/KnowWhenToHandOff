"""Agent system prompt shared by the evaluation and training drivers.

Parity requires that a policy sees exactly the same system prompt when it is
trained (verl agent loop) and when it is evaluated (tau2 runner). Both drivers
call :func:`agent_system_prompt`. The instruction text comes from the pinned
tau2 ``LLMAgent`` so the ``tau2_default`` variant matches the public benchmark.
"""

from __future__ import annotations

PROMPT_VARIANTS = ("tau2_default", "handoff_v1")

TRANSPARENCY_ADDENDUM = (
    "When you refuse a request or transfer the user to a human agent, first tell "
    "the user briefly which part of the policy applies."
)


def agent_instruction(variant: str) -> str:
    """Return the instruction block for a prompt variant."""
    from tau2.agent.llm_agent import AGENT_INSTRUCTION

    if variant == "tau2_default":
        return AGENT_INSTRUCTION
    if variant == "handoff_v1":
        return f"{AGENT_INSTRUCTION}\n{TRANSPARENCY_ADDENDUM}"
    raise ValueError(f"unknown prompt variant {variant!r}; use {PROMPT_VARIANTS}")


def agent_system_prompt(domain_policy: str, variant: str) -> str:
    """Return the full system prompt: instructions plus the domain policy."""
    from tau2.agent.llm_agent import SYSTEM_PROMPT

    return SYSTEM_PROMPT.format(
        agent_instruction=agent_instruction(variant), domain_policy=domain_policy
    )
