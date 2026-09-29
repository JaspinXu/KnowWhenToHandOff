"""Check that a policy tokenizer supports exact incremental tokenization (T3.1).

Run on the cluster before any RL job for the policy you will train. For each
domain it renders a synthetic conversation (tool calls, tool results, user
turns and a hand-off) with the real tau2 system prompt and tool schemas, and
verifies that tokens accumulated turn by turn equal a full encoding of the
conversation. It also checks that the initial RL prompt (system prompt, tools,
greeting and a first user message) leaves room within ``--max-prompt-tokens``
(the ``episode.max_prompt_tokens`` of ``configs/train_rl.yaml``). Optionally
re-checks real teacher trajectories.

Example:
    python scripts/check_tokenization.py --model qwen3_4b
    python scripts/check_tokenization.py --tokenizer /path/to/B2/final \\
        --trajectories $HANDOFF_SCRATCH/runs/TEACH-.../trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from handoff.cli import CliError, compose_config, fail, repo_root, require_scratch
from handoff.ops.runs import read_models_lock
from handoff.rollout.chat_delta import ChatDeltaTokenizer, check_template
from handoff.rollout.schema import iter_trajectories
from handoff.train.sft_data import PROMPT_MESSAGES, chat_messages

ROOT = repo_root(__file__)


def synthetic_conversation(system_prompt: str) -> list[dict]:
    """A short conversation that exercises every boundary the loop crosses."""
    call = {
        "id": "call_001_0",
        "type": "function",
        "function": {"name": "get_user_details", "arguments": {"user_id": "demo_1"}},
    }
    transfer = {
        "id": "call_003_0",
        "type": "function",
        "function": {
            "name": "transfer_to_human_agents",
            "arguments": {"summary": "User asks for a human agent."},
        },
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "Hello, my user id is demo_1."},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {
            "role": "tool",
            "content": json.dumps({"user_id": "demo_1"}),
            "tool_call_id": "call_001_0",
        },
        {"role": "assistant", "content": "Thanks. What can I do for you?"},
        {"role": "user", "content": "I want to talk to a human agent, please."},
        {"role": "assistant", "content": "", "tool_calls": [transfer]},
        {
            "role": "tool",
            "content": "Transfer successful",
            "tool_call_id": "call_003_0",
        },
        {
            "role": "assistant",
            "content": "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.",
        },
    ]


def main() -> int:
    """Run the checks and exit non-zero on any problem."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="qwen3_4b",
        help="models-lock entry; its config also supplies chat_template_kwargs",
    )
    parser.add_argument("--tokenizer", default=None, help="tokenizer directory")
    parser.add_argument("--domains", nargs="*", default=["airline", "retail"])
    parser.add_argument("--prompt-variant", default="handoff_v1")
    parser.add_argument("--trajectories", default=None, help="JSONL to re-check")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument(
        "--first-message-budget",
        type=int,
        default=1024,
        help="tokens reserved for the simulated user's first message",
    )
    parser.add_argument("--paths", nargs="*", default=[])
    args = parser.parse_args()
    problems: list[str] = []
    try:
        from tau2.runner import build_environment
        from transformers import AutoTokenizer

        from handoff.agents.prompts import agent_system_prompt

        path = args.tokenizer
        if path is None:
            config = compose_config(ROOT, "base", args.paths)
            require_scratch(config)
            lock = read_models_lock(Path(config.paths.models_lock))
            if args.model not in lock:
                raise CliError(f"{args.model} is not in the models lock")
            path = lock[args.model]["local_path"]
        tokenizer = AutoTokenizer.from_pretrained(path)
        model_config = ROOT / "configs" / "model" / f"{args.model}.yaml"
        template_kwargs = (
            OmegaConf.to_container(OmegaConf.load(model_config), resolve=True).get(
                "chat_template_kwargs"
            )
            if model_config.exists()
            else None
        )
        deltas = {}
        prompts = {}
        for domain in args.domains:
            environment = build_environment(domain)
            tools = [tool.openai_schema for tool in environment.get_tools()]
            deltas[domain] = ChatDeltaTokenizer(
                tokenizer, tools=tools, chat_template_kwargs=template_kwargs
            )
            prompts[domain] = agent_system_prompt(
                environment.get_policy(), args.prompt_variant
            )
            conversation = synthetic_conversation(prompts[domain])
            for problem in check_template(
                deltas[domain], conversation, prompt_messages=PROMPT_MESSAGES
            ):
                problems.append(f"{domain}/synthetic: {problem}")
            opening = len(deltas[domain].prompt_ids(conversation[:PROMPT_MESSAGES]))
            print(f"{domain}: initial prompt {opening} tokens")
            if opening + args.first_message_budget > args.max_prompt_tokens:
                problems.append(
                    f"{domain}: initial prompt of {opening} tokens leaves less than "
                    f"{args.first_message_budget} of {args.max_prompt_tokens} for the "
                    "first user message; raise episode.max_prompt_tokens"
                )
        if args.trajectories:
            checked = 0
            for trajectory in iter_trajectories(Path(args.trajectories)):
                if trajectory.domain not in deltas or checked >= args.limit:
                    continue
                messages = chat_messages(trajectory, prompts[trajectory.domain])
                # Episodes usually end on the user's reply; the check needs the
                # conversation up to the policy's last turn.
                while messages and messages[-1]["role"] != "assistant":
                    messages.pop()
                if len(messages) <= PROMPT_MESSAGES:
                    continue
                checked += 1
                for problem in check_template(
                    deltas[trajectory.domain], messages, prompt_messages=PROMPT_MESSAGES
                ):
                    problems.append(f"{trajectory.episode_id}: {problem}")
            print(f"re-checked {checked} recorded trajectories")
    except CliError as error:
        fail(str(error))
    if problems:
        print("FAIL")
        for problem in problems:
            print(" -", problem)
        return 1
    print(f"PASS: incremental tokenization is exact for {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
