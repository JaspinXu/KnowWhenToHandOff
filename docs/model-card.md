# Model card template (T4.4)

Fill this in for each released checkpoint (R1, R2; B2 if released) and paste it
into the Hugging Face repository's `README.md`. Every number must come from a
committed report table and name its run ID (AGENTS hard rule 8); leave a
placeholder until the table exists.

---

# KnowWhenToHandOff-<run> (Qwen3-4B-Instruct-2507, <SFT | GRPO task-only | GRPO + Responsible-AI reward>)

> **Research artefact — not for deployment.** Trained and evaluated only with
> simulated users on a public benchmark. It has not been tested with real
> customers, real data or real business systems.

## Model

- Base model: Qwen/Qwen3-4B-Instruct-2507 (revision `<sha>`), Apache-2.0.
- Training: `<SFT on filtered teacher episodes | GRPO from B2, reward preset task_only | r2>`
  ([README](https://github.com/JaspinXu/KnowWhenToHandOff#training-pipeline)).
- Run ID, git SHA, τ²-bench commit and seeds: `<run id>`, `<sha>`,
  `fc0055dc4e0a316c3f83133267fbd6faaa770992`, see `reports/main/sources.json`.

## Intended use

Studying when a tool-using customer-service agent should hand off to a human,
in the τ²-bench airline and retail domains. Out of scope: any production or
customer-facing use, other domains, and decisions about real people.

## Training data

Simulated conversations on τ²-bench v1.0.1 train-split tasks (dev carved out
for selection) and hand-off variants derived from them; the user simulator is
Qwen3-30B-A3B-Instruct-2507. No real user data. Test tasks were never used for
training or selection.

## Evaluation

Test split, protocol frozen in ADR-005; 95 % bootstrap CIs over tasks.

| Metric | This model | B2 | Source |
| --- | --- | --- | --- |
| Success (pass^1) | `<…>` | `<…>` | `reports/main/main_table.md` |
| pass^4 | `<…>` | `<…>` | same |
| Violation rate | `<…>` | `<…>` | same |
| Hand-off F1 / over-escalation | `<…>` | `<…>` | same |
| Style gap (max − min success) | `<…>` | `<…>` | `reports/main/styles.md` |

## Limitations

- One user simulator and one seed per run; results may not transfer to real
  users or other simulators.
- Communication styles describe how users talk, not who they are; no claim is
  made about any demographic group.
- Policy compliance is checked by deterministic rules that cover only part of
  each domain policy; citation rate is a keyword rule.
- Observed reward-hacking behaviours: `<link to the RQ4 section of the report>`.

## Licences

Model weights inherit the base model's licence; τ²-bench code and data are used
under their licence (check the pinned commit's `LICENSE`). Released trajectory samples are limited to what that
licence allows.
