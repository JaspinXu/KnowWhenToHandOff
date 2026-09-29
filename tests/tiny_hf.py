# ruff: noqa: E501 - the chat template below is kept one Jinja statement per line.
"""A tiny Hugging Face tokenizer and model built locally (nothing is downloaded).

The tokenizer is a byte-level BPE trained on a few sentences, with Qwen's
special tokens and the Qwen2.5-Instruct chat template (tools, tool calls and
tool responses), so the real ``transformers`` chat-template and training code
paths run on CPU. Import only when ``transformers`` and ``tokenizers`` exist.
"""

from __future__ import annotations

from pathlib import Path

QWEN25_TEMPLATE = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\\n' }}"
    "{%- if messages[0]['role'] == 'system' %}{{- messages[0]['content'] }}"
    "{%- else %}{{- 'You are a helpful assistant.' }}{%- endif %}"
    "{{- '\\n\\n# Tools\\n\\nYou are provided with function signatures within "
    "<tools></tools> XML tags:\\n<tools>' }}"
    "{%- for tool in tools %}{{- '\\n' }}{{- tool | tojson }}{%- endfor %}"
    "{{- '\\n</tools>\\n\\nFor each function call, return a json object with "
    "function name and arguments within <tool_call></tool_call> XML tags.<|im_end|>\\n' }}"
    "{%- else %}"
    "{%- if messages[0]['role'] == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}"
    "{%- else %}{{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if (message.role == 'user') or (message.role == 'system' and not loop.first) "
    "or (message.role == 'assistant' and not message.tool_calls) %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>' + message.role }}"
    "{%- if message.content %}{{- '\\n' + message.content }}{%- endif %}"
    "{%- for tool_call in message.tool_calls %}"
    "{%- if tool_call.function is defined %}{%- set tool_call = tool_call.function %}{%- endif %}"
    '{{- \'\\n<tool_call>\\n{"name": "\' }}{{- tool_call.name }}'
    "{{- '\", \"arguments\": ' }}{{- tool_call.arguments | tojson }}{{- '}\\n</tool_call>' }}"
    "{%- endfor %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- elif message.role == 'tool' %}"
    "{%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != 'tool') %}"
    "{{- '<|im_start|>user' }}{%- endif %}"
    "{{- '\\n<tool_response>\\n' }}{{- message.content }}{{- '\\n</tool_response>' }}"
    "{%- if loop.last or (messages[loop.index0 + 1].role != 'tool') %}"
    "{{- '<|im_end|>\\n' }}{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"
)

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
ADDED = ["<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]
CORPUS = [
    "Hi! How can I help you today? I want to cancel my order o1.",
    "The order is pending. Shall I cancel it? Yes, please cancel it.",
    '{"name": "get_order", "arguments": {"order_id": "o1"}} {"status": "cancelled"}',
    "You are a helpful assistant. Tools, functions, policy, transfer to a human agent.",
]


def build_tokenizer():
    """A byte-level BPE tokenizer with Qwen's specials and chat template."""
    from tokenizers import (
        AddedToken,
        Tokenizer,
        decoders,
        models,
        pre_tokenizers,
        trainers,
    )
    from transformers import PreTrainedTokenizerFast

    core = Tokenizer(models.BPE())
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    core.train_from_iterator(CORPUS * 4, trainer=trainer)
    core.add_tokens(
        [AddedToken(token, special=False, normalized=False) for token in ADDED]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=core,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tokenizer.chat_template = QWEN25_TEMPLATE
    return tokenizer


def save_tiny_model(directory: Path) -> Path:
    """Write a 2-layer Qwen2 model with the tiny tokenizer; return its path."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    tokenizer = build_tokenizer()
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=2048,
        tie_word_embeddings=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = Qwen2ForCausalLM(config)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory, safe_serialization=True)
    tokenizer.save_pretrained(directory)
    return directory
