# Configuration

Hydra configuration lives in this checkout. Scripts compose it from their own
location (`handoff.cli.compose_config`), never from the shell's working
directory, inject `paths.repo`, and record the resolved config in every run.
Templates contain no credentials; the model servers' shared API key lives on
scratch (`state/secrets/`), other secrets in the ignored `.env` or the cluster
environment.

## Entry-point configs (one per script)

| Config | Script | Selects |
| --- | --- | --- |
| `run_eval.yaml` | `scripts/run_eval.py` | `eval`, `reward`, `persona`; policy and simulator endpoint files; `resume_run_id`, test re-run settings |
| `task_sets.yaml` | `scripts/build_task_sets.py` | domains, dev fraction and seed, `handoff_tasks`, `persona` |
| `sft_data.yaml` | `scripts/build_sft_data.py` | teacher runs, student tokenizer, filter |
| `train_sft.yaml` | `scripts/train_sft.py` | base model (models lock), SFT hyper-parameters, LoRA |
| `train_rl.yaml` | `scripts/train_rl.py` | `rl`, `reward`, `persona`, episode limits (the evaluation's turn and context budget), simulator outage timeout, data kinds and styles |
| `base.yaml` | download, serving, reports, monitors | `paths` only |

## Groups

| Group | Presets | Notes |
| --- | --- | --- |
| `paths/` | `default` | Scratch from `HANDOFF_SCRATCH`; manifests and hand-off tasks in the repo |
| `eval/` | `smoke`, `dev`, `test_main`, `test_styles`, `test_handoff`, `teacher_train` | Protocols of [ADR-005](../docs/adr/ADR-005-evaluation-protocol.md); `test_*` set `allow_test`; `max_infra_failure_share` decides when a run is complete |
| `reward/` | `task_only` (R1), `r2` (R2) | The two reward settings of [ADR-004](../docs/adr/ADR-004-reward-composition.md) |
| `rl/` | `grpo` | verl 0.8 overrides under `verl:`, flattened by `scripts/train_rl.py`; resumable checkpoints only, the latest one kept (every saved step is exported) |
| `model/` | `qwen3_4b` (policy), `qwen3_30b_a3b` (simulator, teacher, B1), `custom` | Repository, served name, tool-call parser, vLLM arguments, chat-template arguments (sent with every request); revisions come from the models lock |
| `persona/` | `styles_v1` | The six communication-style cards ([personas](../docs/personas.md)) |
| `handoff_tasks/` | `v1` | Variants per source task, by split and kind |

Override anything on the command line, for example
`python scripts/run_eval.py eval=dev eval.max_concurrency=8 policy.label=b2`.
