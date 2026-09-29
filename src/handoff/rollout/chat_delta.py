"""Token accounting for multi-turn rollouts without re-tokenizing generations.

The policy's sampled token IDs are kept exactly as generated. Only the tokens
*between* two generations (user replies, tool results and the chat-template
boundaries around them) are produced by rendering the chat template:

    delta = post_eos_suffix + render(history + new, gen_prompt=True)
                            - render(history, gen_prompt=False)

where ``post_eos_suffix`` is what the template writes after the end-of-turn
token (``"\\n"`` for Qwen). This requires a prefix-stable template: rendering a
longer conversation must start with the rendering of the shorter one. The check
is explicit and fails loudly (``scripts/check_tokenization.py`` runs it on the
real tokenizer before any RL job).

The same class renders complete teacher conversations for SFT, where assistant
turns do come from text; there the per-token mask marks assistant content plus
the end-of-turn token as trainable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class ChatTokenizer(Protocol):
    """The subset of a Hugging Face tokenizer used here."""

    eos_token: str | None

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any:
        """Render a conversation."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]:
        """Encode text to token IDs."""

    def convert_tokens_to_ids(self, token: str) -> int:
        """Look up a token ID."""


class TemplateError(ValueError):
    """The chat template cannot be used for incremental tokenization."""


@dataclass(frozen=True)
class MaskedSequence:
    """Tokens of one conversation with a policy-target mask (1 = train on it)."""

    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]


class ChatDeltaTokenizer:
    """Render chat-template deltas for one conversation's tools and settings."""

    def __init__(
        self,
        tokenizer: ChatTokenizer,
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        end_of_turn_token: str | None = None,
    ) -> None:
        """Probe the template once to find the end-of-turn token and its suffix."""
        self.tokenizer = tokenizer
        self.tools = [dict(tool) for tool in tools] if tools else None
        self.kwargs = dict(chat_template_kwargs or {})
        token = end_of_turn_token or tokenizer.eos_token
        if not token:
            raise TemplateError("tokenizer has no end-of-turn token")
        self.end_of_turn = token
        self.end_of_turn_id = int(tokenizer.convert_tokens_to_ids(token))
        probe = self.render(
            [
                {"role": "user", "content": "PROBE_USER"},
                {"role": "assistant", "content": "PROBE_ASSISTANT"},
            ],
            add_generation_prompt=False,
        )
        position = probe.rfind("PROBE_ASSISTANT")
        after = probe[position + len("PROBE_ASSISTANT") :]
        if position < 0 or not after.startswith(token):
            raise TemplateError(
                "assistant turns do not end with the end-of-turn token directly"
            )
        self.post_eos_suffix = after[len(token) :]

    def render(
        self, messages: Sequence[Mapping[str, Any]], add_generation_prompt: bool
    ) -> str:
        """Render messages with the tokenizer's chat template (text, not IDs)."""
        rendered = self.tokenizer.apply_chat_template(
            [dict(m) for m in messages],
            tools=self.tools,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **self.kwargs,
        )
        if not isinstance(rendered, str):
            raise TemplateError("apply_chat_template(tokenize=False) must return text")
        return rendered

    def encode(self, text: str) -> list[int]:
        """Encode template text; special tokens in the text are kept as specials."""
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def prompt_ids(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        """IDs of the initial prompt, ending with the assistant generation prompt."""
        return self.encode(self.render(messages, add_generation_prompt=True))

    def observation_ids(
        self,
        history: Sequence[Mapping[str, Any]],
        new_messages: Sequence[Mapping[str, Any]],
    ) -> list[int]:
        """IDs between the end of the last generation and the next generation.

        ``history`` must end with the assistant turn whose sampled tokens (ending
        with the end-of-turn token) are already in the sequence.
        """
        if not history or history[-1].get("role") != "assistant":
            raise TemplateError("history must end with an assistant turn")
        before = self.render(history, add_generation_prompt=False)
        after = self.render([*history, *new_messages], add_generation_prompt=True)
        if not after.startswith(before):
            raise TemplateError(
                "chat template is not prefix-stable; incremental tokenization would "
                "not match the text the policy sees at evaluation time"
            )
        if not before.endswith(self.end_of_turn + self.post_eos_suffix):
            raise TemplateError("rendered history does not end with an end of turn")
        return self.encode(self.post_eos_suffix + after[len(before) :])

    def mask_conversation(
        self, messages: Sequence[Mapping[str, Any]], *, prompt_messages: int
    ) -> MaskedSequence:
        """Tokenize a finished text conversation for SFT with assistant-only targets.

        The first ``prompt_messages`` messages form the prompt. Every later
        assistant turn is a target (content and end-of-turn token); everything
        else (user and tool turns, template scaffolding, the generation prompt)
        is masked out. Consecutive non-assistant messages (for example the
        results of two tool calls) are rendered as one block, because templates
        such as Qwen's group them into a single turn.
        """
        if prompt_messages < 1 or prompt_messages > len(messages):
            raise ValueError("prompt_messages must select a non-empty prefix")
        prompt_text = self.render(
            messages[:prompt_messages], add_generation_prompt=False
        )
        response_ids: list[int] = []
        mask: list[int] = []
        rendered_so_far = prompt_text
        index = prompt_messages
        while index < len(messages):
            if messages[index].get("role") == "assistant":
                with_prompt = self.render(messages[:index], add_generation_prompt=True)
                full = self.render(messages[: index + 1], add_generation_prompt=False)
                self._require_prefix(rendered_so_far, with_prompt)
                self._require_prefix(with_prompt, full)
                scaffold = with_prompt[len(rendered_so_far) :]
                turn = full[len(with_prompt) :]
                if not turn.endswith(self.end_of_turn + self.post_eos_suffix):
                    raise TemplateError("assistant turn does not end with end of turn")
                target = turn[: len(turn) - len(self.post_eos_suffix)]
                self._extend(response_ids, mask, scaffold, 0)
                self._extend(response_ids, mask, target, 1)
                self._extend(response_ids, mask, self.post_eos_suffix, 0)
                rendered_so_far = full
                index += 1
                continue
            end = index
            while end < len(messages) and messages[end].get("role") != "assistant":
                end += 1
            full = self.render(messages[:end], add_generation_prompt=False)
            self._require_prefix(rendered_so_far, full)
            self._extend(response_ids, mask, full[len(rendered_so_far) :], 0)
            rendered_so_far = full
            index = end
        return MaskedSequence(self.encode(prompt_text), response_ids, mask)

    def _extend(self, ids: list[int], mask: list[int], text: str, value: int) -> None:
        if not text:
            return
        encoded = self.encode(text)
        ids.extend(encoded)
        mask.extend([value] * len(encoded))

    @staticmethod
    def _require_prefix(prefix: str, text: str) -> None:
        if not text.startswith(prefix):
            raise TemplateError(
                "chat template is not prefix-stable for this conversation"
            )


def incremental_ids(
    delta: ChatDeltaTokenizer,
    messages: Sequence[Mapping[str, Any]],
    *,
    prompt_messages: int,
) -> tuple[list[int], list[int]]:
    """Token IDs a rollout would accumulate, and the IDs of the full rendering.

    Assistant turns are encoded as a model would generate them (content through
    the end-of-turn token); everything else goes through
    :meth:`ChatDeltaTokenizer.observation_ids`. For a usable template the first
    list plus the encoded ``post_eos_suffix`` equals the second list.
    """
    history = [dict(m) for m in messages[:prompt_messages]]
    ids = delta.prompt_ids(history)
    index = prompt_messages
    while index < len(messages):
        message = dict(messages[index])
        if message.get("role") == "assistant":
            with_prompt = delta.render(history, add_generation_prompt=True)
            full = delta.render([*history, message], add_generation_prompt=False)
            turn = full[len(with_prompt) :]
            generated = turn[: len(turn) - len(delta.post_eos_suffix)]
            ids = ids + delta.encode(generated)
            history.append(message)
            index += 1
            continue
        end = index
        while end < len(messages) and messages[end].get("role") != "assistant":
            end += 1
        block = [dict(m) for m in messages[index:end]]
        ids = ids + delta.observation_ids(history, block)
        history.extend(block)
        index = end
    full_ids = delta.encode(delta.render(history, add_generation_prompt=False))
    return ids, full_ids


def check_template(
    delta: ChatDeltaTokenizer,
    messages: Sequence[Mapping[str, Any]],
    *,
    prompt_messages: int,
) -> list[str]:
    """Return problems found when tokenizing ``messages`` incrementally (empty = ok)."""
    if messages[-1].get("role") != "assistant":
        return ["the check conversation must end with an assistant turn"]
    try:
        ids, full_ids = incremental_ids(
            delta, messages, prompt_messages=prompt_messages
        )
    except TemplateError as error:
        return [str(error)]
    expected = ids + delta.encode(delta.post_eos_suffix)
    if expected != full_ids:
        first = next(
            (
                i
                for i, (a, b) in enumerate(zip(expected, full_ids, strict=False))
                if a != b
            ),
            min(len(expected), len(full_ids)),
        )
        return [
            f"incremental tokens differ from a full encoding at position {first} "
            f"({len(expected)} vs {len(full_ids)} tokens)"
        ]
    return []
