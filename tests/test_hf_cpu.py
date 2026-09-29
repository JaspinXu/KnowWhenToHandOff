"""Hugging Face code paths on CPU with a tiny locally built model.

Skipped unless ``torch``, ``transformers``, ``tokenizers`` and ``peft`` are
installed (they are in the cluster environment, whose setup job runs this).
Covers the real chat-template tokenization used by both drivers, the cluster's
tokenization check, and a short SFT run (full fine-tuning and LoRA) that must
produce a checkpoint that loads again.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
for module in ("torch", "transformers", "tokenizers", "peft"):
    pytest.importorskip(module)

import tiny_hf  # noqa: E402
from fakes import call, conversation, result, say, trajectory  # noqa: E402

from handoff.rollout.chat_delta import ChatDeltaTokenizer, check_template  # noqa: E402
from handoff.train.sft_data import (  # noqa: E402
    PROMPT_MESSAGES,
    chat_messages,
    tokenize_episode,
)
from handoff.train.sft_trainer import SFTSettings, run_sft  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]


def teacher_episode():
    return trajectory(
        messages=conversation(
            say("user", "Hi, I want to cancel my order o1."),
            say("assistant", "", call("c1", "get_order", order_id="o1")),
            result("c1", '{"order_id": "o1", "status": "pending"}'),
            say("assistant", "The order is pending. Shall I cancel it?"),
            say("user", "Yes, please cancel it."),
            say("assistant", "Order o1 is cancelled."),
        )
    )


def test_real_template_supports_incremental_tokenization():
    delta = ChatDeltaTokenizer(tiny_hf.build_tokenizer(), tools=TOOLS)
    assert delta.end_of_turn == "<|im_end|>" and delta.post_eos_suffix == "\n"
    messages = chat_messages(teacher_episode(), "POLICY")
    assert check_template(delta, messages, prompt_messages=PROMPT_MESSAGES) == []
    record = tokenize_episode(teacher_episode(), delta, "POLICY")
    tokenizer = delta.tokenizer
    targets = tokenizer.decode(
        [
            t
            for t, m in zip(
                record["response_ids"], record["response_mask"], strict=True
            )
            if m
        ]
    )
    assert "Order o1 is cancelled.<|im_end|>" in targets
    assert "Yes, please" not in targets and "<tool_response>" not in targets
    rendered = tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False)
    assert tokenizer.decode(record["prompt_ids"] + record["response_ids"]) == rendered


@pytest.mark.enable_socket
def test_check_tokenization_script_on_tiny_model(tmp_path, monkeypatch, capsys):
    pytest.importorskip("tau2")
    import toy_tau2

    toy_tau2.register_toy_domain()
    model_dir = tiny_hf.save_tiny_model(tmp_path / "tiny")
    spec = importlib.util.spec_from_file_location(
        "check_tokenization", ROOT / "scripts" / "check_tokenization.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = [
        "check_tokenization",
        "--tokenizer",
        str(model_dir),
        "--domains",
        toy_tau2.DOMAIN,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0
    assert "PASS" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [*argv, "--max-prompt-tokens", "64"])
    assert module.main() == 1
    assert "raise episode.max_prompt_tokens" in capsys.readouterr().out


@pytest.mark.enable_socket
@pytest.mark.parametrize("lora_rank", [0, 4])
def test_sft_trains_and_saves_a_loadable_checkpoint(tmp_path, lora_rank):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = tiny_hf.save_tiny_model(tmp_path / "base")
    delta = ChatDeltaTokenizer(AutoTokenizer.from_pretrained(model_dir), tools=TOOLS)
    records = tmp_path / "records.jsonl"
    with records.open("w", encoding="utf-8") as stream:
        for _ in range(4):
            stream.write(
                json.dumps(tokenize_episode(teacher_episode(), delta, "POLICY")) + "\n"
            )
    settings = SFTSettings.from_mapping(
        {
            "model_path": str(model_dir),
            "train_records": str(records),
            "output_dir": str(tmp_path / "run"),
            "max_length": 2048,
            "num_train_epochs": 1.0,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-3,
            "bf16": False,
            "logging_steps": 1,
            "save_strategy": "no",
            "lora": {"rank": lora_rank, "alpha": 8, "target_modules": ["all-linear"]},
        }
    )
    final = run_sft(settings)
    assert (final / "config.json").exists() and (final / "handoff_sft.json").exists()
    assert not (final / "adapter_config.json").exists()  # LoRA is merged
    model = AutoModelForCausalLM.from_pretrained(final)
    tokenizer = AutoTokenizer.from_pretrained(final)
    ids = tokenizer("Hi", return_tensors="pt").input_ids
    assert (
        model.generate(ids, max_new_tokens=2, do_sample=False).shape[1]
        == ids.shape[1] + 2
    )


def test_export_merges_a_lora_adapter_into_full_weights(tmp_path, monkeypatch):
    """A LoRA RL step exports as plain weights that vLLM (and HF) can load."""
    import shutil

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM

    import handoff.train.export as export

    base_dir = tiny_hf.save_tiny_model(tmp_path / "base")
    adapter_dir = tmp_path / "adapter"
    torch.manual_seed(1)
    peft_model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(base_dir),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]),
    )
    for name, parameter in peft_model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.5)
    peft_model.save_pretrained(adapter_dir)

    def fake_merger(command, check):
        """verl's merger writes the base weights plus lora_adapter/."""
        work = Path(command[command.index("--target_dir") + 1])
        shutil.copytree(base_dir, work)
        shutil.copytree(adapter_dir, work / "lora_adapter")

    monkeypatch.setattr(export.subprocess, "run", fake_merger)
    step = tmp_path / "run" / "checkpoints" / "global_step_5"
    (step / "actor" / "huggingface").mkdir(parents=True)
    target = export.export_step(step, tmp_path / "run" / "export" / "global_step_5")
    assert json.loads((target / "handoff_export.json").read_text())["lora_merged"]
    assert not (target / "lora_adapter").exists()
    merged = AutoModelForCausalLM.from_pretrained(target)
    reference = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(base_dir, torch_dtype=torch.bfloat16),
        adapter_dir,
    )
    base = AutoModelForCausalLM.from_pretrained(base_dir)
    inputs = torch.tensor([[1, 5, 9, 12, 7]])
    with torch.no_grad():
        got = merged(inputs).logits.float()
        want = reference(inputs).logits.float()
        unchanged = base(inputs).logits.float()
    assert torch.allclose(got, want, atol=5e-2)
    assert not torch.allclose(got, unchanged, atol=1e-3)
