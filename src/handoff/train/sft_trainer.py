"""Assistant-only supervised fine-tuning on token records (task T2.2).

A deliberately small trainer around Hugging Face ``Trainer``: it reads the
records written by :mod:`handoff.train.sft_data`, pads them with
:func:`handoff.train.sft_batch.build_sft_batch` (loss only on policy tokens),
and saves a Hugging Face checkpoint that vLLM can serve directly. Multi-GPU runs
use ``torchrun`` (DDP or FSDP through ``TrainingArguments``). LoRA adapters are
merged before saving so evaluation never needs adapter support.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from handoff.train.sft_batch import SFTBatchConfig, build_sft_batch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoraSettings:
    """LoRA hyper-parameters; ``rank == 0`` disables LoRA (full fine-tuning)."""

    rank: int = 0
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("all-linear",)


@dataclass(frozen=True)
class SFTSettings:
    """Everything the SFT run needs; built from the ``sft`` Hydra group."""

    model_path: str
    train_records: str
    output_dir: str
    eval_records: str | None = None
    max_length: int = 32768
    pad_to_multiple_of: int = 8
    learning_rate: float = 1e-5
    num_train_epochs: float = 2.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    bf16: bool = True
    attn_implementation: str = "sdpa"
    use_liger_kernel: bool = False
    fsdp: str = ""
    seed: int = 0
    logging_steps: int = 5
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    report_to: tuple[str, ...] = ()
    lora: LoraSettings = field(default_factory=LoraSettings)

    def __post_init__(self) -> None:
        if self.lora.rank > 0 and self.fsdp:
            # Merging adapters needs the full model on one process; under FSDP
            # the trainer only holds shards.
            raise ValueError(
                "LoRA with FSDP is not supported: use LoRA on one GPU, or full "
                "fine-tuning (lora.rank=0) with FSDP"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> SFTSettings:
        """Validate a resolved mapping (unknown keys are errors)."""
        values = dict(data)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown sft settings: {sorted(unknown)}")
        lora = dict(values.pop("lora", {}) or {})
        if "target_modules" in lora:
            lora["target_modules"] = tuple(lora["target_modules"])
        values["report_to"] = tuple(values.get("report_to") or ())
        return cls(lora=LoraSettings(**lora), **values)


def read_records(path: Path, max_length: int) -> tuple[list[dict[str, Any]], int]:
    """Load token records; drop (and count) records longer than ``max_length``."""
    records: list[dict[str, Any]] = []
    dropped = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != "train":
                raise ValueError("SFT records must be declared train samples")
            if len(record["prompt_ids"]) + len(record["response_ids"]) > max_length:
                dropped += 1
                continue
            records.append(record)
    return records, dropped


class RecordCollator:
    """Turn token records into padded tensors with assistant-only labels."""

    def __init__(self, config: SFTBatchConfig) -> None:
        """Keep the batch configuration."""
        self.config = config

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Pad and convert one batch to tensors."""
        import torch

        rows = [
            {
                "split": sample["split"],
                "prompt_ids": sample["prompt_ids"],
                "response_ids": sample["response_ids"],
                "response_mask": sample["response_mask"],
            }
            for sample in samples
        ]
        batch = build_sft_batch(rows, config=self.config)
        return {
            key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()
        }


def run_sft(settings: SFTSettings) -> Path:
    """Train and save a Hugging Face checkpoint; return its directory."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(settings.seed)
    random.seed(settings.seed)
    output_dir = Path(settings.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    batch_config = SFTBatchConfig(
        max_length=settings.max_length,
        pad_token_id=int(pad_id),
        ignore_index=-100,
        pad_to_multiple_of=settings.pad_to_multiple_of,
    )
    train, dropped = read_records(Path(settings.train_records), settings.max_length)
    if not train:
        raise ValueError("no SFT records left after the length filter")
    evaluation = None
    if settings.eval_records:
        evaluation, _ = read_records(Path(settings.eval_records), settings.max_length)
    logger.info("SFT records: %d train (%d too long dropped)", len(train), dropped)
    # Full fine-tuning keeps float32 master weights (bf16 autocast computes);
    # LoRA trains small adapters on a frozen bf16 base.
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_path,
        torch_dtype=torch.float32 if settings.lora.rank == 0 else torch.bfloat16,
        attn_implementation=settings.attn_implementation,
    )
    if settings.lora.rank > 0:
        from peft import LoraConfig, get_peft_model

        targets = list(settings.lora.target_modules)
        model = get_peft_model(
            model,
            LoraConfig(
                r=settings.lora.rank,
                lora_alpha=settings.lora.alpha,
                lora_dropout=settings.lora.dropout,
                target_modules=targets[0] if targets == ["all-linear"] else targets,
                task_type="CAUSAL_LM",
            ),
        )
    arguments = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.num_train_epochs,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        lr_scheduler_type=settings.lr_scheduler_type,
        max_grad_norm=settings.max_grad_norm,
        bf16=settings.bf16,
        gradient_checkpointing=settings.gradient_checkpointing,
        # Non-reentrant checkpointing also works when the inputs need no
        # gradient (frozen embeddings under LoRA).
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_liger_kernel=settings.use_liger_kernel,
        fsdp=settings.fsdp,
        fsdp_config=(
            {"state_dict_type": "FULL_STATE_DICT", "use_orig_params": True}
            if settings.fsdp
            else None
        ),
        logging_steps=settings.logging_steps,
        save_strategy=settings.save_strategy,
        save_total_limit=settings.save_total_limit,
        eval_strategy="epoch" if evaluation else "no",
        seed=settings.seed,
        data_seed=settings.seed,
        report_to=list(settings.report_to) or "none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train,
        eval_dataset=evaluation,
        data_collator=RecordCollator(batch_config),
    )
    trainer.train()
    final = output_dir / "final"
    if settings.lora.rank > 0:
        if trainer.is_world_process_zero():
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(str(final), safe_serialization=True)
    else:
        trainer.save_model(str(final))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final))
        (final / "handoff_sft.json").write_text(
            json.dumps(
                {
                    "settings": asdict(settings),
                    "n_train": len(train),
                    "dropped": dropped,
                },
                indent=2,
                default=list,
            ),
            encoding="utf-8",
        )
    trainer.accelerator.wait_for_everyone()
    return final
