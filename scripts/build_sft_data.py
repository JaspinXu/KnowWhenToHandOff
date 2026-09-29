"""Build the B2 imitation dataset from teacher rollouts (T2.1).

Keeps only train-split (per the frozen registry), successful, violation-free,
format-clean episodes with the correct transfer decision, renders them with the
student tokenizer's chat template and writes token records, per-record
metadata, statistics and a manifest to ``<scratch>/datasets/<dataset_name>/``.

Example:
    python scripts/build_sft_data.py \\
        teacher_runs=[TEACH-teacher_train-s0-20261012] dataset_name=b2_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

from handoff.cli import (
    CliError,
    compose_config,
    container,
    fail,
    missing_keys,
    repo_root,
    require_scratch,
    setup_logging,
)
from handoff.envs.registry import TaskRegistry
from handoff.ops.runs import git_state, read_models_lock
from handoff.rollout.chat_delta import ChatDeltaTokenizer
from handoff.rollout.schema import iter_trajectories
from handoff.train.sft_data import (
    TeacherFilter,
    dataset_statistics,
    select_teacher_episodes,
    write_sft_records,
)

ROOT = repo_root(__file__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    """Filter, tokenize and describe the teacher data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    setup_logging()
    try:
        from tau2.runner import build_environment
        from transformers import AutoTokenizer

        from handoff.agents.prompts import agent_system_prompt

        config = compose_config(ROOT, "sft_data", args.overrides)
        require_scratch(config)
        missing = missing_keys(config, ["teacher_runs", "dataset_name"])
        if missing:
            raise CliError("missing settings: " + ", ".join(missing))
        paths = config.paths
        out_dir = Path(paths.scratch) / "datasets" / str(config.dataset_name)
        if out_dir.exists():
            raise CliError(f"{out_dir} exists; choose a new dataset_name")
        registry = TaskRegistry.from_files(
            Path(paths.splits_manifest),
            Path(paths.labels_file),
            Path(paths.handoff_manifest),
        )
        trajectories = []
        run_files = []
        for run_id in config.teacher_runs:
            path = Path(paths.runs_dir) / str(run_id) / "trajectories.jsonl"
            if not path.exists():
                raise CliError(f"no trajectories for teacher run {run_id}")
            run_files.append({"run_id": str(run_id), "sha256": _sha256(path)})
            trajectories.extend(iter_trajectories(path))
        kept, rejected = select_teacher_episodes(
            trajectories, registry, TeacherFilter(**container(config.filter))
        )
        if not kept:
            raise CliError(f"no teacher episode passed the filter: {dict(rejected)}")
        tokenizer_path = config.tokenizer_path
        if not tokenizer_path:
            lock = read_models_lock(Path(paths.models_lock))
            if str(config.student_model) not in lock:
                raise CliError(f"{config.student_model} is not in the models lock")
            tokenizer_path = lock[str(config.student_model)]["local_path"]
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        template_kwargs = container(config.chat_template_kwargs)

        @cache
        def environment(domain: str):
            return build_environment(domain)

        @cache
        def delta_for(domain: str) -> ChatDeltaTokenizer:
            tools = [tool.openai_schema for tool in environment(domain).get_tools()]
            return ChatDeltaTokenizer(
                tokenizer, tools=tools, chat_template_kwargs=template_kwargs
            )

        @cache
        def system_prompt_for(domain: str) -> str:
            return agent_system_prompt(
                environment(domain).get_policy(), str(config.prompt_variant)
            )

        out_dir.mkdir(parents=True)
        records = out_dir / "records.jsonl"
        meta = out_dir / "records_meta.jsonl"
        skipped = write_sft_records(
            kept,
            delta_for=delta_for,
            system_prompt_for=system_prompt_for,
            records_path=records,
            meta_path=meta,
            max_length=int(config.max_length),
        )
        stats = dataset_statistics(kept, rejected, skipped, meta)
        git_sha, dirty = git_state(ROOT)
        manifest = {
            "schema_version": 1,
            "kind": "sft_token_records",
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": git_sha,
            "git_dirty": dirty,
            "teacher_runs": run_files,
            "tokenizer_path": str(tokenizer_path),
            "prompt_variant": str(config.prompt_variant),
            "records_sha256": _sha256(records),
            "splits_manifest_sha256": registry.split_manifest["manifest_sha256"],
            "statistics": stats,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except CliError as error:
        fail(str(error))
    print(json.dumps(stats, indent=2))
    print(f"dataset: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
