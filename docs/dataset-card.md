---
license: mit
language:
- en
pretty_name: KnowWhenToHandOff — hand-off tasks and run records
tags:
- agents
- tool-use
- customer-service
- reinforcement-learning
- human-handoff
- tau2-bench
- reward-hacking
size_categories:
- 10K<n<100K
configs:
- config_name: runs
  data_files: runs.jsonl
  default: true
- config_name: airline_tasks
  data_files: tasks/v1/airline_tasks.json
- config_name: retail_tasks
  data_files: tasks/v1/retail_tasks.json
---

# KnowWhenToHandOff — hand-off tasks and run records

Everything behind the KnowWhenToHandOff study except model weights: a task set
for deciding **when a tool-using customer-service agent should hand off to a
human**, and the complete records of every run (baselines, SFT, two GRPO
variants). Code: <https://github.com/JaspinXu/KnowWhenToHandOff>. Models:
[`KnowWhenToHandOff-Qwen3-4B-R2`](https://huggingface.co/JaspinXu/KnowWhenToHandOff-Qwen3-4B-R2),
[`-R1`](https://huggingface.co/JaspinXu/KnowWhenToHandOff-Qwen3-4B-R1),
[`-SFT`](https://huggingface.co/JaspinXu/KnowWhenToHandOff-Qwen3-4B-SFT).
Kaggle mirror: <https://www.kaggle.com/datasets/zhaobinxu/knowwhentohandoff-data>.

> **Synthetic data only.** All users are simulated (Qwen3-30B-A3B-Instruct-2507)
> on τ²-bench's synthetic airline and retail databases; names, IDs and
> e-mail addresses (`@example.com`) are fictional. No real customer data.

## Contents

| Folder | What | Use |
| --- | --- | --- |
| `tasks/v1/` | 461 hand-off tasks (`airline_tasks.json`, `retail_tasks.json`) in τ²-bench's task schema: *explicit request* and *out of scope* (should transfer), *hard negative* (frustrated user, solvable, should not) | Train or evaluate hand-off decisions with τ²-bench |
| `tasks/manifests/` | Train/dev/test split of the upstream and derived tasks, should-transfer labels with overrides, generator manifest with hashes | Reproduce the splits; never train on `test` |
| `runs/<run id>/` | Per run: `trajectories.jsonl` (every conversation, tool call, violation and reward), `metrics.json` (with 95 % bootstrap intervals), `summary.md`, `provenance.json` (git SHA, τ²-bench commit, model revisions, seeds), `config.json`, `completion.json` | Re-analyse results without a GPU |
| `runs/R1-s1-*/rollouts/`, `runs/R2-s1-*/rollouts/` | Every GRPO training episode (60 steps × 64) with its reward terms | Study how over-escalation emerged during RL |
| `runs/TEACH-*` | Teacher rollouts used for SFT | Imitation data source |
| `sft/b2_v1/` | Which teacher episodes were kept for SFT and why (`records_meta.jsonl`, `manifest.json`) | Rebuild the SFT set |
| `runs.jsonl` | One row per run with headline metrics | Index |
| `SHA256SUMS` | Checksums of every file | Integrity |

## Loading

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download

runs = load_dataset("JaspinXu/KnowWhenToHandOff-data", "runs", split="train")       # run index
airline = load_dataset("JaspinXu/KnowWhenToHandOff-data", "airline_tasks", split="train")

# Everything (trajectories, rollouts, manifests), about 125 MB:
path = snapshot_download("JaspinXu/KnowWhenToHandOff-data", repo_type="dataset")
```

Each line of a `trajectories.jsonl` is one conversation: `messages` (with tool
calls and results), `task_reward`, `db_match`, `transferred` and `transfer_turn`
against the `should_transfer` label, `violations`, `format_errors`, plus the
task, domain, split, kind (`upstream` for official
tasks, or `explicit_request`, `out_of_scope`, `hard_negative`), user style and seed. The code that writes and scores them is in
the [code repository](https://github.com/JaspinXu/KnowWhenToHandOff).

Run IDs: `B0` prompted Qwen3-4B, `B1` prompted Qwen3-30B-A3B, `B2` SFT,
`R1` GRPO with the task reward, `R2` GRPO with the Responsible-AI reward;
`*-test_main-*`, `*-test_styles-*`, `*-test_handoff-*` and `*-dev-*` are the
evaluation protocols. Absolute cluster paths are replaced by `<scratch>`,
`<repo>` and `~`.

## Headline results (test)

| | Success | Δ vs SFT | Hand-off F1 | Over-escalation |
| --- | --- | --- | --- | --- |
| B2 SFT | 0.327 | — | 0.768 | 0.016 |
| R1 GRPO, task reward | 0.347 | +0.020 [−0.066, 0.107] | 0.802 | 0.462 |
| R2 GRPO, Responsible-AI reward | 0.342 | +0.015 [−0.056, 0.082] | 0.812 | 0.103 |

Under the task-only reward a transfer scores 1.0 whenever the reference
solution makes no database change, so "transfer when unsure" becomes weakly
dominant; R1's rollouts show it emerging in the last 20 steps.

## Licence and attribution

MIT. The tasks are derived from [τ²-bench](https://github.com/sierra-research/tau2-bench)
(MIT, © 2025 Sierra Research; pinned commit
`fc0055dc4e0a316c3f83133267fbd6faaa770992`); keep its licence notice when
redistributing. Conversations were generated with Qwen3 models (Apache-2.0).
Research use only; not for deploying customer-facing systems.
