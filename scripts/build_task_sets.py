"""Build labels (T1.2), frozen splits (T1.1) and the hand-off task set (T1.3).

Reads the pinned tau2 data, writes small JSON files under ``data/`` that are
committed to git, and prints the counts. Refuses to overwrite existing frozen
files unless ``overwrite=true`` (changing them after M1 needs an ADR).

Example:
    python scripts/build_task_sets.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from handoff.cli import CliError, compose_config, container, fail, repo_root
from handoff.envs.handoff_tasks import GENERATOR_VERSION, generate_handoff_tasks
from handoff.envs.labels import build_label_file, labels_by_task
from handoff.envs.personas import personas_from_config
from handoff.envs.registry import (
    TaskRegistry,
    build_handoff_manifest,
    build_split_manifest,
    read_json,
    write_json,
)
from handoff.envs.tau2_adapter import (
    load_domain_source,
    tau2_commit,
    validate_task_set,
    write_task_file,
)
from handoff.rollout.tau2_convert import needs_llm_judge

ROOT = repo_root(__file__)


def main() -> int:
    """Generate and validate every task-set artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    try:
        config = compose_config(ROOT, "task_sets", args.overrides)
        paths = config.paths
        outputs = [
            Path(paths.labels_file),
            Path(paths.splits_manifest),
            Path(paths.handoff_manifest),
        ]
        existing = [str(p) for p in outputs if p.exists()]
        if existing and not config.overwrite:
            raise CliError(
                "frozen task-set files exist: " + ", ".join(existing) + "; pass "
                "overwrite=true only with an ADR (AGENTS hard rule 4)"
            )
        personas = personas_from_config(container(config.persona))
        commit = tau2_commit()
        sources = {domain: load_domain_source(domain) for domain in config.domains}
        # Upstream tasks whose reward needs tau2's LLM judge stay out of every
        # split (AGENTS hard rule 3); the split manifest lists them.
        excluded = {
            domain: {str(t["id"]) for t in source.tasks if needs_llm_judge(t)}
            for domain, source in sources.items()
        }
        overrides_path = Path(paths.label_overrides)
        overrides = read_json(overrides_path) if overrides_path.exists() else {}
        label_file = build_label_file(
            {d: s.tasks for d, s in sources.items()},
            tau2_commit=commit,
            overrides=overrides.get("domains", {}),
        )
        labels = labels_by_task(label_file)
        split_manifest = build_split_manifest(
            {
                domain: {
                    "official_splits": source.official_splits,
                    "labels": {
                        task_id: value
                        for (d, task_id), value in labels.items()
                        if d == domain
                    },
                    "file_sha256": source.file_sha256,
                    "excluded": sorted(excluded[domain]),
                }
                for domain, source in sources.items()
            },
            dev_fraction=float(config.dev_fraction),
            seed=int(config.split_seed),
            tau2_commit=commit,
            labels_sha256=label_file["labels_sha256"],
        )
        generated: dict[str, list] = {}
        task_hashes: dict[str, str] = {}
        handoff_config = container(config.handoff_tasks)
        for domain, source in sources.items():
            splits = split_manifest["domains"][domain]["splits"]
            split_of = {tid: name for name, ids in splits.items() for tid in ids}
            pairs = generate_handoff_tasks(
                domain,
                source.tasks,
                split_of=split_of,
                labels={t: v for (d, t), v in labels.items() if d == domain},
                counts=handoff_config["counts"][domain],
                seed=int(handoff_config["seed"]),
            )
            usable = [t for t in source.tasks if str(t["id"]) in split_of]
            problems = validate_task_set(
                [*usable, *(task for task, _ in pairs)], personas
            )
            if problems:
                raise CliError(
                    f"{len(problems)} {domain} task(s) cannot be used:\n  "
                    + "\n  ".join(problems[:20])
                )
            task_hashes[domain] = write_task_file(
                Path(paths.handoff_tasks_dir) / f"{domain}_tasks.json",
                [task for task, _ in pairs],
            )
            generated[domain] = [ref for _, ref in pairs]
        handoff_manifest = build_handoff_manifest(
            generated,
            generator_version=GENERATOR_VERSION,
            split_manifest_sha256=split_manifest["manifest_sha256"],
            tasks_file_sha256=task_hashes,
            config=handoff_config,
        )
        write_json(Path(paths.labels_file), label_file)
        write_json(Path(paths.splits_manifest), split_manifest)
        write_json(Path(paths.handoff_manifest), handoff_manifest)
        registry = TaskRegistry(split_manifest, labels, handoff_manifest)
        registry.assert_lineage_disjoint()
    except CliError as error:
        fail(str(error))
    summary = {
        "tau2_commit": commit,
        "labels": {
            d: {k: v for k, v in e.items() if k != "labels"}
            for d, e in label_file["domains"].items()
        },
        "splits": {d: e["counts"] for d, e in split_manifest["domains"].items()},
        "excluded_llm_judged": {
            d: len(e["excluded"]) for d, e in split_manifest["domains"].items()
        },
        "positives": {d: e["positives"] for d, e in split_manifest["domains"].items()},
        "handoff": {d: e["counts"] for d, e in handoff_manifest["domains"].items()},
    }
    print(json.dumps(summary, indent=2))
    print(
        "Wrote frozen task sets; commit data/manifests and data/handoff_tasks, "
        "then hand-check a sample (python scripts/sample_review.py tasks)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
