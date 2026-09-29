"""Export verl actor checkpoints as Hugging Face directories vLLM can serve (T3.3).

verl saves ``<run_dir>/checkpoints/global_step_N/actor/`` (FSDP shards plus the
model config, tokenizer and generation config in ``huggingface/``). Our RL
presets do not save a separate Hugging Face copy of the weights (it would be
float32, twice the size); instead every saved step is exported right after it
is written (see :mod:`handoff.train.verl_main`), so verl can keep only the
latest resumable checkpoint.

An export runs verl's model merger (bfloat16 output). For a LoRA run the merger
writes the base weights plus ``lora_adapter/``, and the adapter is merged into
full weights, so evaluation never needs adapter support.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

EXPORT_RECORD = "handoff_export.json"
_STEP = re.compile(r"global_step_(\d+)")


def step_dirs(checkpoints: Path) -> list[tuple[int, Path]]:
    """``(step, directory)`` of every saved step, in step order."""
    found = [
        (int(match.group(1)), path)
        for path in checkpoints.glob("global_step_*")
        if (match := _STEP.fullmatch(path.name)) and (path / "actor").is_dir()
    ]
    return sorted(found)


def is_exported(target: Path) -> bool:
    """Whether ``target`` holds a finished export."""
    return (target / EXPORT_RECORD).is_file()


def copy_missing_metadata(source: Path, target: Path) -> None:
    """Copy config, tokenizer and chat-template files the export lacks."""
    if not source.is_dir():
        return
    for path in source.iterdir():
        weights = path.suffix in (".safetensors", ".bin") or path.name.endswith(
            ".index.json"
        )
        if path.is_file() and not weights and not (target / path.name).exists():
            shutil.copy2(path, target / path.name)


def _merge_lora(unmerged: Path, target: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(unmerged, torch_dtype=torch.bfloat16)
    merged = PeftModel.from_pretrained(base, unmerged / "lora_adapter")
    merged.merge_and_unload().save_pretrained(target, safe_serialization=True)


def export_step(step_dir: Path, target: Path, *, python: str = sys.executable) -> Path:
    """Write a servable bfloat16 Hugging Face directory for one saved step.

    Idempotent: an existing finished export is returned unchanged. Work happens
    in ``<target>.tmp`` so an interrupted export never looks finished.
    """
    if is_exported(target):
        return target
    actor = step_dir / "actor"
    if not actor.is_dir():
        raise FileNotFoundError(f"no actor checkpoint in {step_dir}")
    work = target.with_name(target.name + ".tmp")
    if work.exists():
        shutil.rmtree(work)
    if target.exists():
        shutil.rmtree(target)
    work.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            python,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(actor),
            "--target_dir",
            str(work),
        ],
        check=True,
    )
    lora = (work / "lora_adapter").is_dir()
    if lora:
        merged = target.with_name(target.name + ".merged")
        if merged.exists():
            shutil.rmtree(merged)
        _merge_lora(work, merged)
        shutil.rmtree(work)
        work = merged
    if not (any(work.glob("*.safetensors")) or any(work.glob("*.bin"))):
        raise RuntimeError(f"export of {step_dir} produced no weights")
    copy_missing_metadata(actor / "huggingface", work)
    record = {"source": str(step_dir), "step": step_dir.name, "lora_merged": lora}
    (work / EXPORT_RECORD).write_text(json.dumps(record, indent=2) + "\n")
    work.replace(target)
    return target


def export_pending(checkpoints: Path, exports: Path) -> list[Path]:
    """Export every saved step that has no finished export yet."""
    return [
        export_step(step_dir, exports / step_dir.name)
        for _, step_dir in step_dirs(checkpoints)
        if not is_exported(exports / step_dir.name)
    ]
