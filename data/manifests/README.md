# Frozen manifests

Written once on the cluster by `bash slurm/submit.sh tasks`
(`scripts/build_task_sets.py`) and committed; regenerating them after M1 needs an
ADR (AGENTS hard rule 4). The script refuses to overwrite existing files.

| File | Content |
| --- | --- |
| `labels_v1.json` | Should-transfer label, rule and tau2 commit for every upstream task ([labels](../../docs/labels.md)) |
| `label_overrides_v1.json` | Audited label corrections, each with a reason (starts empty) |
| `splits_v1.json` | Train / dev / test task IDs per domain, dev seed and fraction, label and data-file hashes, `manifest_sha256` |
| `handoff_tasks_v1.json` | Every derived hand-off task with its source task, split, kind, template and label, plus the task-file hash ([hand-off tasks](../../docs/handoff-tasks.md)) |

The task files themselves are in `data/handoff_tasks/v1/`. Every evaluation and
training run records the manifest hashes it used.
