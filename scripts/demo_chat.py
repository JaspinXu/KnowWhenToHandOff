"""Interactive terminal demo: you play the customer, the served policy is the agent.

Runs a real tau2 environment (tools, database, policy) for one task and shows
every tool call; hand-offs and policy citations are highlighted. At the end the
conversation is scored with the same evaluator, rules and labels as training.
Type ``/quit`` to end the conversation.

Example (on a compute node with the policy served):
    python scripts/demo_chat.py --domain airline --task-id 5 \\
        --endpoint-file $HANDOFF_SCRATCH/state/endpoints/policy-<job>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from handoff.cli import CliError, compose_config, fail, repo_root, require_scratch
from handoff.eval.citations import CITATION_PATTERN
from handoff.rollout.schema import TRANSFER_TOOL

ROOT = repo_root(__file__)
ORANGE, YELLOW, RED, DIM, RESET = (
    "\033[33m",
    "\033[93m",
    "\033[31m",
    "\033[2m",
    "\033[0m",
)


def _human_user_class():
    from tau2.data_model.message import UserMessage
    from tau2.user.user_simulator_base import HalfDuplexUser, UserState

    class HumanUser(HalfDuplexUser):
        """A person typing the user's turns in the terminal."""

        def __init__(self) -> None:
            super().__init__(instructions=None, tools=None)

        def get_init_state(self, message_history=None):
            return UserState(system_messages=[], messages=list(message_history or []))

        def generate_next_message(self, message, state):
            content = getattr(message, "content", None)
            if content:
                shown = CITATION_PATTERN.sub(
                    lambda m: f"{YELLOW}{m.group(0)}{RESET}", content
                )
                print(f"\nagent: {shown}")
            text = input("you> ").strip() or "(no reply)"
            if text == "/quit":
                text = "###STOP###"
            reply = UserMessage(role="user", content=text)
            state.messages.append(reply)
            return reply, state

        @classmethod
        def is_stop(cls, message) -> bool:
            return "###STOP###" in (message.content or "")

        def set_seed(self, seed: int) -> None:
            return None

    return HumanUser


def main() -> int:
    """Run one interactive conversation and print its evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, help="airline or retail")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--endpoint-file", required=True)
    parser.add_argument("--prompt-variant", default="handoff_v1")
    parser.add_argument("--paths", nargs="*", default=[])
    args = parser.parse_args()
    try:
        from tau2.orchestrator.orchestrator import Orchestrator
        from tau2.registry import registry as tau2_registry
        from tau2.runner import build_agent, build_environment

        from handoff.agents.tau2_agents import agent_name, register_agents
        from handoff.envs.labels import upstream_label
        from handoff.envs.tau2_adapter import task_to_dict
        from handoff.rollout.scoring import EpisodeContext, build_trajectory
        from handoff.rollout.tau2_convert import (
            domain_tool_sets,
            finalize_simulation,
            gold_actions_from_task,
            normalize_tau2_messages,
        )
        from handoff.sim.endpoints import (
            litellm_args,
            load_api_key_into_env,
            read_endpoint,
            register_local_models,
        )

        config = compose_config(ROOT, "base", args.paths)
        require_scratch(config)
        load_api_key_into_env(Path(config.paths.api_key_file))
        endpoint = read_endpoint(Path(args.endpoint_file))
        register_local_models([endpoint.served_model_name])
        register_agents()
        if args.domain not in tau2_registry.get_domains():
            raise CliError(f"unknown domain {args.domain!r}")
        tasks = {t.id: t for t in tau2_registry.get_tasks_loader(args.domain)(None)}
        if args.task_id not in tasks:
            raise CliError(f"unknown {args.domain} task {args.task_id}")
        task = tasks[args.task_id]
        print(f"{DIM}Your scenario:{RESET}\n{task.user_scenario}\n")
        environment = build_environment(args.domain)
        respond = environment.get_response

        def traced(call):
            colour = ORANGE if call.name == TRANSFER_TOOL else DIM
            print(f"{colour}tool: {call.name}({json.dumps(call.arguments)}){RESET}")
            return respond(call)

        environment.get_response = traced
        agent = build_agent(
            agent_name(args.prompt_variant),
            environment,
            llm=endpoint.litellm_model,
            llm_args=litellm_args(endpoint, temperature=0.0, max_tokens=2048),
        )
        orchestrator = Orchestrator(
            domain=args.domain,
            agent=agent,
            user=_human_user_class()(),
            environment=environment,
            task=task,
            max_steps=200,
        )
        simulation = orchestrator.run()
        format_errors = finalize_simulation(simulation, task, args.domain)
        write_tools, known_tools = domain_tool_sets(environment)
        label, _ = upstream_label(task_to_dict(task))
        trajectory = build_trajectory(
            EpisodeContext(
                run_id="DEMO",
                domain=args.domain,
                task_id=str(task.id),
                source_task_id=str(task.id),
                split="train",
                kind="upstream",
                persona="neutral",
                should_transfer=label,
                trial=0,
                seed=None,
                policy_ckpt=f"{endpoint.served_model_name}@{endpoint.model_revision}",
                user_sim="human",
                gold_actions=gold_actions_from_task(task),
                write_tools=write_tools,
                known_tools=known_tools,
            ),
            messages=normalize_tau2_messages(simulation.messages),
            termination_reason=str(simulation.termination_reason.value),
            task_reward=float(simulation.reward_info.reward),
            driver_format_errors=format_errors,
        )
        print(
            f"\ntask reward: {trajectory.task_reward} · should transfer: {label} · "
            f"transferred: {trajectory.transferred}"
        )
        for violation in trajectory.violations:
            print(f"{RED}violation: {violation.rule} at turn {violation.turn}{RESET}")
        for error in trajectory.format_errors:
            print(f"{RED}format error: {error.kind} at turn {error.turn}{RESET}")
    except (CliError, KeyboardInterrupt, EOFError) as error:
        fail(str(error) or "interrupted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
