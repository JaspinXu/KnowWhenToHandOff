"""Token-level rollout pieces: Hermes parsing, chat deltas and the policy session."""

import asyncio
import json

import pytest
from fakes import FakeChatTokenizer

from handoff.rollout.chat_delta import (
    ChatDeltaTokenizer,
    TemplateError,
    check_template,
    incremental_ids,
)
from handoff.rollout.episode import (
    EpisodeSettings,
    GenerationOutput,
    InfrastructureError,
    PolicySession,
    PromptTooLongError,
)
from handoff.rollout.hermes import parse_hermes

TOOLS = [
    {"type": "function", "function": {"name": "get_user_details", "parameters": {}}}
]
# asyncio event loops create a local socketpair; no network is used.
asyncio_loop = pytest.mark.enable_socket


def test_hermes_plain_text_single_and_multiple_calls():
    assert parse_hermes("Hello!").content == "Hello!"
    one = parse_hermes(
        '<tool_call>\n{"name": "a", "arguments": {"x": 1}}\n</tool_call>'
    )
    assert one.content is None and one.tool_calls[0].name == "a"
    assert one.tool_calls[0].arguments == {"x": 1}
    two = parse_hermes(
        'Checking.\n<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
    )
    assert two.content == "Checking.\n"
    assert [c.name for c in two.tool_calls] == ["a", "b"]
    assert two.tool_calls[1].arguments == {"y": 2}
    unclosed = parse_hermes('<tool_call>{"name": "a", "arguments": {}}')
    assert unclosed.tool_calls[0].name == "a"


@pytest.mark.parametrize(
    "text",
    [
        "<tool_call>{not json}</tool_call>",
        "<tool_call></tool_call>",
        '<tool_call>{"arguments": {}}</tool_call>',
        '<tool_call>{"name": 1, "arguments": {}}</tool_call>',
        '<tool_call>{"name": "a"}</tool_call>',
        '<tool_call>["a"]</tool_call>',
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        "<tool_call>oops</tool_call>",
    ],
)
def test_hermes_invalid_calls_fall_back_to_text(text):
    parsed = parse_hermes(text)
    assert parsed.tool_calls == ()
    assert parsed.content == text
    assert parsed.error == "invalid_tool_json"


@pytest.mark.parametrize("arguments", ["[1]", '"{\\"y\\": 2}"', "null", "3"])
def test_hermes_non_object_arguments_lose_the_turn(arguments):
    parsed = parse_hermes(
        f'Hi\n<tool_call>{{"name": "a", "arguments": {arguments}}}</tool_call>'
    )
    assert parsed.tool_calls == () and parsed.error == "invalid_arguments"
    assert parsed.content == "Hi\n"


def delta(**kwargs) -> ChatDeltaTokenizer:
    return ChatDeltaTokenizer(FakeChatTokenizer(**kwargs), tools=TOOLS)


def conversation():
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "get_user_details", "arguments": {"user_id": "u1"}},
    }
    return [
        {"role": "system", "content": "POLICY"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "I am u1"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "content": '{"id": "u1"}', "tool_call_id": "c1"},
        {"role": "assistant", "content": "What do you need?"},
        {"role": "user", "content": "A human, please."},
        {"role": "assistant", "content": "Transferring you."},
    ]


def test_probe_finds_end_of_turn_and_suffix():
    tokenizer = delta()
    assert tokenizer.end_of_turn == "<|im_end|>"
    assert tokenizer.post_eos_suffix == "\n"


def test_incremental_tokens_equal_full_encoding():
    tokenizer = delta()
    assert check_template(tokenizer, conversation(), prompt_messages=3) == []
    ids, full = incremental_ids(tokenizer, conversation(), prompt_messages=3)
    assert ids + tokenizer.encode("\n") == full


def test_non_prefix_stable_template_is_rejected():
    problems = check_template(
        delta(drop_old_tool_results=True), conversation(), prompt_messages=3
    )
    assert problems and "prefix-stable" in problems[0]
    with pytest.raises(TemplateError):
        delta().observation_ids([{"role": "user", "content": "x"}], [])


def test_mask_conversation_targets_only_assistant_turns():
    tokenizer = delta()
    masked = tokenizer.mask_conversation(conversation(), prompt_messages=3)
    fake = FakeChatTokenizer()
    targets = fake.decode(
        [t for t, m in zip(masked.response_ids, masked.response_mask, strict=True) if m]
    )
    assert "What do you need?<|im_end|>" in targets
    assert "Transferring you.<|im_end|>" in targets
    assert "A human" not in targets and "<|im_start|>" not in targets
    assert '"user_id": "u1"' in targets
    full = fake.apply_chat_template(conversation(), tools=TOOLS)
    assert fake.decode(masked.prompt_ids + masked.response_ids) == full


class ScriptedGenerator:
    """Returns pre-tokenized generations in order and records the prompts."""

    def __init__(self, texts, *, logprobs=True, truncate_last=False):
        self.tokenizer = FakeChatTokenizer()
        self.texts = list(texts)
        self.prompts = []
        self.logprobs = logprobs
        self.truncate_last = truncate_last

    async def generate(self, prompt_ids, *, max_tokens):
        self.prompts.append((list(prompt_ids), max_tokens))
        text = self.texts.pop(0)
        ids = self.tokenizer.encode(text)
        if not (self.truncate_last and not self.texts):
            ids.append(self.tokenizer.eos_token_id)
        lps = [-0.5] * len(ids) if self.logprobs else None
        return GenerationOutput(ids, lps)


def session_for(generator, **settings):
    fake = FakeChatTokenizer()
    return PolicySession(
        ChatDeltaTokenizer(fake, tools=TOOLS),
        generator,
        EpisodeSettings(**settings),
        "POLICY",
        [fake.eos_token_id],
        lambda ids: fake.decode(ids),
    )


def run(coroutine):
    return asyncio.run(coroutine)


@asyncio_loop
def test_session_accumulates_tokens_with_masks():
    generator = ScriptedGenerator(
        [
            "<tool_call>\n"
            '{"name": "get_user_details", "arguments": {"user_id": "u1"}}'
            "\n</tool_call>",
            "What do you need?",
        ]
    )
    session = session_for(generator, max_response_tokens=4000)
    first = run(session.step([{"role": "user", "content": "I am u1"}]))
    assert first.tool_calls[0][1:] == ("get_user_details", {"user_id": "u1"})
    call_id = first.tool_calls[0][0]
    second = run(
        session.step(
            [{"role": "tool", "content": '{"id": "u1"}', "tool_call_id": call_id}]
        )
    )
    assert second.content == "What do you need?" and not second.stop
    fake = FakeChatTokenizer()
    text = fake.decode(session.prompt_ids + session.response_ids)
    assert text.startswith("<|im_start|>system\nPOLICY")
    policy = fake.decode(
        [
            t
            for t, m in zip(session.response_ids, session.response_mask, strict=True)
            if m
        ]
    )
    assert "What do you need?<|im_end|>" in policy
    assert "<tool_response>" not in policy
    assert len(session.response_logprobs) == len(session.response_ids)
    assert all(
        lp == 0.0
        for lp, m in zip(session.response_logprobs, session.response_mask, strict=True)
        if not m
    )
    assert (
        generator.prompts[1][0]
        == session.prompt_ids
        + session.response_ids[: len(generator.prompts[1][0]) - len(session.prompt_ids)]
    )


@asyncio_loop
def test_session_turn_budget_mirrors_evaluation():
    """Every turn gets the full max_tokens; the episode ends cleanly (no format
    error) when the conversation plus one turn exceeds the context, as vLLM
    rejects such a request at evaluation."""
    probe = session_for(ScriptedGenerator(["ok"]))
    run(probe.step([{"role": "user", "content": "hi"}]))
    prompt = len(probe.prompt_ids)

    generator = ScriptedGenerator(["ok", "never used"])
    limited = session_for(
        generator, max_tokens_per_turn=8, max_context_tokens=prompt + 8
    )
    first = run(limited.step([{"role": "user", "content": "hi"}]))
    assert not first.stop and generator.prompts[0][1] == 8
    stop = run(limited.step([{"role": "user", "content": "more"}]))
    assert stop.stop and stop.stop_reason == "context_window_exceeded"
    assert stop.format_errors == () and len(generator.prompts) == 1

    budget = session_for(
        ScriptedGenerator(["ok"]), max_response_tokens=10, max_tokens_per_turn=5
    )
    run(budget.step([{"role": "user", "content": "hi"}]))
    stop = run(
        budget.step([{"role": "user", "content": "a much longer reply that exceeds"}])
    )
    assert stop.stop and budget.stop_reason == "context_window_exceeded"
    assert stop.format_errors == () and len(budget.response_ids) <= 10

    idle = ScriptedGenerator(["x"])
    too_long = session_for(idle, max_tokens_per_turn=8, max_context_tokens=prompt + 7)
    decision = run(too_long.step([{"role": "user", "content": "hi"}]))
    assert decision.stop_reason == "context_window_exceeded" and idle.prompts == []
    assert too_long.response_ids == []


@asyncio_loop
def test_session_truncation_and_bad_output():

    truncated = session_for(ScriptedGenerator(["never ends"], truncate_last=True))
    decision = run(truncated.step([{"role": "user", "content": "hi"}]))
    assert decision.stop and decision.format_errors == ("truncated_generation",)
    assert decision.stop_reason == "context_window_exceeded"
    assert decision.completion_tokens == len("never ends")

    empty = session_for(ScriptedGenerator(["   "]))
    decision = run(empty.step([{"role": "user", "content": "hi"}]))
    assert decision.stop and "empty_message" in decision.format_errors
    assert empty.stop_reason == "agent_error"

    arguments = session_for(
        ScriptedGenerator(['<tool_call>{"name": "a", "arguments": "x"}</tool_call>'])
    )
    decision = run(arguments.step([{"role": "user", "content": "hi"}]))
    assert decision.stop_reason == "agent_error"
    assert decision.format_errors == ("invalid_arguments",)

    # Unparsable markup goes to the user as text, as with vLLM's parser; the
    # format check counts it once from the message (unparsed_tool_call).
    invalid = session_for(ScriptedGenerator(["<tool_call>{bad}</tool_call>"]))
    decision = run(invalid.step([{"role": "user", "content": "hi"}]))
    assert not decision.stop and decision.format_errors == ()
    assert decision.content == "<tool_call>{bad}</tool_call>"
    bad_ids = FakeChatTokenizer().encode("<tool_call>{bad}</tool_call>")
    assert decision.completion_tokens == len(bad_ids) + 1  # plus end-of-turn


@asyncio_loop
def test_session_prompt_limit_and_aborted_generation():
    long_prompt = session_for(ScriptedGenerator(["x"]), max_prompt_tokens=5)
    with pytest.raises(PromptTooLongError):
        run(long_prompt.step([{"role": "user", "content": "hi"}]))

    class Aborted:
        async def generate(self, prompt_ids, *, max_tokens):
            return GenerationOutput([])

    with pytest.raises(InfrastructureError):
        run(session_for(Aborted()).step([{"role": "user", "content": "hi"}]))


@asyncio_loop
def test_session_drops_logprobs_when_server_omits_them():
    session = session_for(ScriptedGenerator(["a", "b"], logprobs=False))
    run(session.step([{"role": "user", "content": "hi"}]))
    assert session.response_logprobs is None
    run(session.step([{"role": "user", "content": "again"}]))
    assert session.response_logprobs is None


def test_episode_settings_reject_unknown_keys():
    with pytest.raises(ValueError):
        EpisodeSettings.from_mapping({"max_steps": 10, "surprise": 1})
    settings = EpisodeSettings.from_mapping({"max_steps": 10, "user_args": {"a": 1}})
    assert settings.max_steps == 10 and json.dumps(dict(settings.user_args))


def test_sft_masking_handles_parallel_tool_results():
    """Two tool results render as one template block (Qwen groups them)."""
    delta = ChatDeltaTokenizer(FakeChatTokenizer(), tools=TOOLS)
    calls = [
        {"id": "a", "type": "function", "function": {"name": "x", "arguments": {}}},
        {"id": "b", "type": "function", "function": {"name": "y", "arguments": {}}},
    ]
    messages = [
        {"role": "system", "content": "POLICY"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": calls},
        {"role": "tool", "content": "1", "tool_call_id": "a"},
        {"role": "tool", "content": "2", "tool_call_id": "b"},
        {"role": "assistant", "content": "Done."},
    ]
    masked = delta.mask_conversation(messages, prompt_messages=3)
    ids, _ = incremental_ids(delta, messages, prompt_messages=3)
    assert masked.prompt_ids + masked.response_ids == ids + delta.encode("\n")
    fake = FakeChatTokenizer()
    targets = fake.decode(
        [t for t, m in zip(masked.response_ids, masked.response_mask, strict=True) if m]
    )
    assert "Done.<|im_end|>" in targets and "<tool_response>" not in targets


def test_generation_config_end_tokens_complete_a_turn(tmp_path):
    from handoff.rollout.episode import Tau2EpisodeRunner, generation_eos_ids

    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [151645, 151643]})
    )
    assert generation_eos_ids(str(tmp_path)) == [151645, 151643]
    assert generation_eos_ids(str(tmp_path / "missing")) == []
    assert generation_eos_ids("") == []

    class Tokenizer(FakeChatTokenizer):
        name_or_path = str(tmp_path)

    ids = Tau2EpisodeRunner._stop_ids(Tokenizer())
    assert {151645, 151643, FakeChatTokenizer().eos_token_id} <= set(ids)


def test_evaluation_detects_tool_calls_cut_at_the_token_limit():
    from types import SimpleNamespace

    from handoff.agents.tau2_agents import hit_token_limit

    def reply(reason, tokens):
        return SimpleNamespace(
            raw_data={"choices": [{"finish_reason": reason}]},
            usage={"completion_tokens": tokens, "prompt_tokens": 10},
        )

    assert hit_token_limit(reply("length", 5), 2048)
    # vLLM reports "tool_calls" for a call parsed from an unclosed block
    assert hit_token_limit(reply("tool_calls", 2048), 2048)
    assert not hit_token_limit(reply("tool_calls", 120), 2048)
    assert not hit_token_limit(reply("stop", 2048), 2048)
    assert not hit_token_limit(reply("tool_calls", 2048), None)
