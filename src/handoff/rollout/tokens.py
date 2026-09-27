"""Accumulate actual generation IDs without re-tokenizing conversation text."""

from collections.abc import Sequence
from typing import Literal

Role = Literal["assistant", "user", "tool"]


def _copy_ids(ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(ids, str | bytes | bytearray) or not isinstance(ids, Sequence):
        raise ValueError("token_ids must be a sequence of integer IDs, not text")
    copied = tuple(ids)
    if any(type(token) is not int or token < 0 for token in copied):
        raise ValueError("token_ids must contain nonnegative integers")
    return copied


class TokenAccumulator:
    """Keep sampled policy IDs and supplied observation IDs aligned with masks.

    The caller supplies already encoded user/tool observations. This class never
    invokes a tokenizer. Capacity includes both policy and observation tokens
    after the initial prompt; exceeding it fails without partial mutation.
    """

    def __init__(self, prompt_ids: Sequence[int], *, max_response_tokens: int) -> None:
        """Copy an initial prompt and establish an explicit response capacity."""
        if type(max_response_tokens) is not int or max_response_tokens <= 0:
            raise ValueError("max_response_tokens must be a positive integer")
        self._prompt = _copy_ids(prompt_ids)
        if not self._prompt:
            raise ValueError("prompt_ids must not be empty")
        self._capacity = max_response_tokens
        self._response: list[int] = []
        self._mask: list[int] = []

    def append(self, role: Role, token_ids: Sequence[int]) -> None:
        """Append one turn atomically; policy tokens are 1, observations are 0."""
        if role not in ("assistant", "user", "tool"):
            raise ValueError("role must be assistant, user or tool")
        ids = _copy_ids(token_ids)
        if len(self._response) + len(ids) > self._capacity:
            raise ValueError("response capacity exceeded; tokens were not appended")
        self._response.extend(ids)
        self._mask.extend([int(role == "assistant")] * len(ids))

    def as_output(self) -> dict[str, list[int]]:
        """Return independent copies of the three README training-token fields."""
        return {
            "prompt_ids": list(self._prompt),
            "response_ids": self._response.copy(),
            "response_mask": self._mask.copy(),
        }
