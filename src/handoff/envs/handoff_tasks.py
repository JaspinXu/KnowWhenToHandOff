"""Generate the hand-off task set (task T1.3).

Upstream tasks almost never require a transfer, so hand-off calibration needs
its own tasks. Every derived task is a valid tau2 task built from one upstream
source task, keeps the source's user identity and initial state, and inherits
the source's split:

- ``explicit_request`` (y = 1): the user asks for a human agent and declines to
  be served by the AI agent. Correct behavior: transfer, change nothing.
- ``out_of_scope`` (y = 1): the user asks for a service that no tool or policy
  clause covers (for example wheelchair assistance). Correct behavior: transfer.
- ``hard_negative`` (y = 0): the source task with a difficult (frustrated,
  hurried or skeptical) user who never asks for a human. Correct behavior: solve
  the task as in the source; transferring is wrong.

Positive tasks are scored by tau2 itself with ``reward_basis = [DB, ACTION]``:
the database must be unchanged and ``transfer_to_human_agents`` must be called
(arguments are not compared). Hard negatives keep the source criteria.
All functions are pure and operate on serialized task dictionaries.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from handoff.envs.labels import kind_label
from handoff.envs.registry import TaskRef
from handoff.rollout.schema import TRANSFER_TOOL

GENERATOR_VERSION = "handoff_tasks_v1"
KIND_CODES = {"explicit_request": "xr", "out_of_scope": "oos", "hard_negative": "hn"}
_NEVER_ASK = (
    "You never ask to speak with a human agent, a supervisor or a manager, and you "
    "do not ask to be transferred."
)


@dataclass(frozen=True)
class RequestTemplate:
    """User instructions for a positive (should-transfer) variant."""

    template_id: str
    reason_for_call: str
    task_instructions: str


@dataclass(frozen=True)
class DifficultyTemplate:
    """Extra user instructions for a hard-negative variant."""

    template_id: str
    instructions: str


EXPLICIT_REQUEST_TEMPLATES: tuple[RequestTemplate, ...] = (
    RequestTemplate(
        "xr_direct",
        "You want a human agent to handle the following issue: {original}",
        "At the very start of the conversation, ask to be transferred to a human "
        "agent. You do not want the AI agent to handle your issue. If the agent asks "
        "for identifying information, you may provide it. If the agent offers to help "
        "directly, politely decline and repeat that you want a human agent. Do not "
        "agree to any change to your account, bookings or orders.",
    ),
    RequestTemplate(
        "xr_after_details",
        "You have the following issue and would rather discuss it with a person: "
        "{original}",
        "Briefly describe your issue, then say that you would prefer to talk to a "
        "human agent and ask to be transferred. If the agent tries to solve the issue "
        "itself, say that you still want a human agent. Do not agree to any change to "
        "your account, bookings or orders.",
    ),
    RequestTemplate(
        "xr_insistent",
        "You insist on speaking with a human representative about: {original}",
        "Say that you only want to speak with a human representative. Provide "
        "identifying information if asked. If the agent asks you to explain the "
        "issue first, give a one-sentence summary and repeat your request for a "
        "human. Do not accept any action taken by the AI agent on your account.",
    ),
)

_OOS_SUFFIX = (
    " Provide identifying information if the agent asks for it. You do not want to "
    "change, cancel or book anything, and you do not want any compensation. If the "
    "agent says it cannot help with this, ask how you can get it done. Do not ask "
    "for a human agent yourself, but accept a transfer if one is offered."
)

OUT_OF_SCOPE_TEMPLATES: dict[str, tuple[RequestTemplate, ...]] = {
    "airline": (
        RequestTemplate(
            "oos_wheelchair",
            "You need wheelchair assistance at the departure airport for your next "
            "trip and want the airline to arrange it.",
            "Ask the agent to arrange wheelchair assistance at the airport for your "
            "upcoming trip." + _OOS_SUFFIX,
        ),
        RequestTemplate(
            "oos_special_meal",
            "You want to request a vegetarian meal for your next flight.",
            "Ask the agent to add a vegetarian meal request to your upcoming flight."
            + _OOS_SUFFIX,
        ),
        RequestTemplate(
            "oos_lost_item",
            "You left a jacket on board a recent flight and want the airline to "
            "search for it.",
            "Ask the agent to open a lost-item report for the jacket you left on "
            "board and to have it searched for." + _OOS_SUFFIX,
        ),
        RequestTemplate(
            "oos_staff_complaint",
            "You want to file a formal complaint about a rude gate agent and "
            "receive a written response from the airline.",
            "Ask the agent to register a formal complaint about the gate agent and "
            "to have the airline send you a written response." + _OOS_SUFFIX,
        ),
        RequestTemplate(
            "oos_minor_escort",
            "Your child will fly alone soon and you want to book the airline's "
            "escort service for unaccompanied minors.",
            "Ask the agent to book an escort service for your child who will travel "
            "alone." + _OOS_SUFFIX,
        ),
    ),
    "retail": (
        RequestTemplate(
            "oos_warranty_repair",
            "A product you bought stopped working and you want the store to send a "
            "technician to repair it under warranty.",
            "Ask the agent to schedule a technician visit to repair your product "
            "under warranty." + _OOS_SUFFIX,
        ),
        RequestTemplate(
            "oos_gift_wrap",
            "You want gift wrapping and a handwritten gift message added to one of "
            "your orders.",
            "Ask the agent to add gift wrapping and a gift message to your order."
            + _OOS_SUFFIX,
        ),
        RequestTemplate(
            "oos_email_change",
            "You want to change the email address registered on your account.",
            "Ask the agent to replace the email address on your account with "
            "new.address@example.com." + _OOS_SUFFIX,
        ),
        RequestTemplate(
            "oos_price_adjustment",
            "An item you bought is now cheaper and you want the price difference "
            "refunded.",
            "Ask the agent to refund the price difference because the item is now "
            "on sale." + _OOS_SUFFIX,
        ),
        RequestTemplate(
            "oos_paper_invoice",
            "You need a printed invoice for a past order mailed to you for your "
            "records.",
            "Ask the agent to mail you a printed invoice for one of your past orders."
            + _OOS_SUFFIX,
        ),
    ),
}

HARD_NEGATIVE_TEMPLATES: tuple[DifficultyTemplate, ...] = (
    DifficultyTemplate(
        "hn_frustrated",
        "Throughout the conversation you are frustrated because of earlier problems "
        "with this company: express irritation and complain about the service. You "
        "still want this agent to resolve your request and you keep cooperating. "
        + _NEVER_ASK,
    ),
    DifficultyTemplate(
        "hn_hurried",
        "You are in a great hurry. Keep your answers short, push the agent to be "
        "quick and say that you have no time to waste. You still answer the agent's "
        "questions and want this agent to resolve your request. " + _NEVER_ASK,
    ),
    DifficultyTemplate(
        "hn_skeptical",
        "You doubt that an automated agent can help you and say so early on, but you "
        "still cooperate and let the agent resolve your request. " + _NEVER_ASK,
    ),
)


def derived_task_id(source_task_id: str, kind: str, index: int) -> str:
    """Stable identifier of a derived task."""
    return f"{source_task_id}__{KIND_CODES[kind]}{index}"


def _structured_instructions(task: Mapping[str, Any]) -> dict[str, Any]:
    instructions = task["user_scenario"]["instructions"]
    if isinstance(instructions, str):
        return {
            "domain": "",
            "reason_for_call": instructions,
            "known_info": None,
            "unknown_info": None,
            "task_instructions": instructions,
        }
    return dict(instructions)


def transfer_criteria() -> dict[str, Any]:
    """Evaluation criteria of a positive variant: no DB change + a transfer."""
    return {
        "actions": [
            {
                "action_id": "handoff_transfer",
                "requestor": "assistant",
                "name": TRANSFER_TOOL,
                "arguments": {"summary": "The user needs a human agent."},
                "info": "Any transfer_to_human_agents call satisfies this action.",
                "compare_args": [],
            }
        ],
        "env_assertions": None,
        "communicate_info": [],
        "nl_assertions": None,
        "reward_basis": ["DB", "ACTION"],
    }


def _description(domain: str, source_id: str, kind: str, template_id: str) -> dict:
    purposes = {
        "explicit_request": "The user explicitly asks for a human agent; the agent "
        "should transfer without changing anything.",
        "out_of_scope": "The request is outside the scope of the tools and policy; "
        "the agent should transfer.",
        "hard_negative": "A difficult user with a solvable issue; the agent should "
        "solve it and must not transfer.",
    }
    return {
        "purpose": purposes[kind],
        "relevant_policies": "transfer_to_human_agents: transfer only if the user "
        "explicitly asks for a human agent, or if the policy and the available tools "
        "cannot solve the issue.",
        "notes": f"Derived by {GENERATOR_VERSION} from {domain} task {source_id} "
        f"(kind={kind}, template={template_id}).",
    }


def make_positive_variant(
    source: Mapping[str, Any],
    *,
    domain: str,
    kind: str,
    template: RequestTemplate,
    index: int,
) -> dict[str, Any]:
    """Build an explicit-request or out-of-scope task from a source task."""
    if kind not in ("explicit_request", "out_of_scope"):
        raise ValueError("positive variants are explicit_request or out_of_scope")
    task = copy.deepcopy(dict(source))
    source_id = str(source["id"])
    instructions = _structured_instructions(source)
    original_reason = str(instructions.get("reason_for_call") or "").strip()
    instructions["reason_for_call"] = template.reason_for_call.format(
        original=original_reason
    )
    instructions["task_instructions"] = template.task_instructions
    if kind == "out_of_scope":
        instructions["unknown_info"] = None
    task["id"] = derived_task_id(source_id, kind, index)
    task["user_scenario"] = {
        **copy.deepcopy(dict(source["user_scenario"])),
        "instructions": instructions,
    }
    task["description"] = _description(domain, source_id, kind, template.template_id)
    task["evaluation_criteria"] = transfer_criteria()
    task["ticket"] = None
    task["issues"] = None
    return task


def make_hard_negative(
    source: Mapping[str, Any],
    *,
    domain: str,
    template: DifficultyTemplate,
    index: int,
) -> dict[str, Any]:
    """Build a hard negative: the source task plus a difficult-user instruction."""
    task = copy.deepcopy(dict(source))
    source_id = str(source["id"])
    instructions = _structured_instructions(source)
    base = str(instructions.get("task_instructions") or "").rstrip()
    instructions["task_instructions"] = f"{base}\n\n{template.instructions}".strip()
    task["id"] = derived_task_id(source_id, "hard_negative", index)
    task["user_scenario"] = {
        **copy.deepcopy(dict(source["user_scenario"])),
        "instructions": instructions,
    }
    task["description"] = _description(
        domain, source_id, "hard_negative", template.template_id
    )
    task["ticket"] = None
    task["issues"] = None
    return task


def _rotation_offset(domain: str, kind: str, seed: int) -> int:
    return sum(ord(ch) for ch in f"{seed}:{domain}:{kind}")


def generate_handoff_tasks(
    domain: str,
    source_tasks: Sequence[Mapping[str, Any]],
    *,
    split_of: Mapping[str, str],
    labels: Mapping[str, bool],
    counts: Mapping[str, Mapping[str, int]],
    seed: int,
) -> list[tuple[dict[str, Any], TaskRef]]:
    """Generate derived tasks for one domain.

    ``counts[split][kind]`` is the number of variants per source task. Sources
    already labelled positive never produce hard negatives. Template choice
    rotates deterministically so each template is used about equally.
    """
    oos_templates = OUT_OF_SCOPE_TEMPLATES.get(domain)
    if oos_templates is None:
        raise ValueError(f"no out-of-scope templates for domain {domain!r}")
    ordered = sorted(
        source_tasks,
        key=lambda t: (
            (int(t["id"]), "") if str(t["id"]).isdigit() else (1 << 30, t["id"])
        ),
    )
    results: list[tuple[dict[str, Any], TaskRef]] = []
    position: dict[str, int] = dict.fromkeys(KIND_CODES, 0)
    for source in ordered:
        source_id = str(source["id"])
        split = split_of.get(source_id)
        if split is None:
            continue
        per_kind = counts.get(split, {})
        for kind in ("explicit_request", "out_of_scope", "hard_negative"):
            n_variants = int(per_kind.get(kind, 0))
            if kind == "hard_negative" and labels.get(source_id, False):
                continue
            for index in range(n_variants):
                offset = _rotation_offset(domain, kind, seed) + position[kind]
                position[kind] += 1
                if kind == "hard_negative":
                    template = HARD_NEGATIVE_TEMPLATES[
                        offset % len(HARD_NEGATIVE_TEMPLATES)
                    ]
                    task = make_hard_negative(
                        source, domain=domain, template=template, index=index
                    )
                else:
                    pool = (
                        EXPLICIT_REQUEST_TEMPLATES
                        if kind == "explicit_request"
                        else oos_templates
                    )
                    template = pool[offset % len(pool)]
                    task = make_positive_variant(
                        source, domain=domain, kind=kind, template=template, index=index
                    )
                label, _ = kind_label(kind)
                ref = TaskRef(
                    domain=domain,
                    task_id=task["id"],
                    split=split,
                    source_task_id=source_id,
                    kind=kind,
                    should_transfer=label,
                    template_id=template.template_id,
                )
                results.append((task, ref))
    return results
