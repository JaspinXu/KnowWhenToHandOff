# Implementation status

Updated: 30 September 2026. This file separates code verification from research
results. **No benchmark score, trained checkpoint or measured cluster throughput
exists yet**; CPU fakes and toy domains must never be cited as model results.
Owner decisions and scope changes are in [AGENTS.md §17](../AGENTS.md#17-decision-log).

## Task status

"Code" means implemented and tested on CPU; "Run" is what the cluster or a
person still has to do for the README definition of done.

| Task | Code | Run / review still needed |
| --- | --- | --- |
| T0.1 | Done; CI (Ubuntu + Windows) green | — |
| T0.2 | tau2 v1.0.1 pin, smoke protocol, result parsing | `eval=smoke` on the cluster |
| T0.3 | Slurm jobs, serving, endpoint files | Servers answering tool calls; throughput |
| T0.4 | verl stack pinned and verified on CPU (ADR-001) | GPU spike: step time, go/no-go |
| T0.5 | LiteLLM `openai/<served>` + `api_base` for agent and simulator | Full 5-task run on local models |
| T1.1 | Registry, stratified dev carve-out, frozen manifests, disjointness tests | Build and commit the manifests |
| T1.2 | Label rules and file, overrides with reasons ([labels](labels.md)) | Commit the label file |
| T1.3 | Generator, tau2 validation, lineage ([hand-off tasks](handoff-tasks.md)) | Hand-check 30+ tasks |
| T1.4 | Style cards and injection through tau2 ([personas](personas.md)) | Review 10 dialogues per style |
| T1.5 | All metrics with CIs ([metrics](metrics.md)) | Freeze with ADR-005 at M1 |
| T1.6 | Evaluation driver, protocols, reports | B0 and B1 runs, audit write-up |
| T2.1 | Teacher protocol and filter | Teacher rollouts, dataset statistics |
| T2.2 | SFT trainer (full, LoRA, FSDP) | SFT run and B2 evaluation |
| T2.3 | Comparison and failure packets | Annotate 10 failures |
| T3.1 | `Tau2AgentLoop`, parity test with the evaluation driver | GPU validation |
| T3.2 | Reward composer, per-term logging, the two reward presets, per-step reward-term statistics (`rl_curves.py`) | Reward statistics from a real run |
| T3.3–T3.4 | GRPO launcher, checkpoint export, resume, learning curves (`rl_curves.py`) | The R1 and R2 runs |
| T3.5 | Reward-hacking monitor, review packet | Review 20 dialogues |
| T3.6 | Test protocols, test ledger | Final evaluation |
| T4.1 | Tables, figures and sources from run IDs | Real results |
| T4.3 | HTML demo page and terminal demo | Slides, recorded demo |
| T4.2, T4.4 | Model card template ([model card](model-card.md)) | Report, release |

## Verification

Only model servers, model weights and benchmark data are replaced by test
doubles; the drivers, rewards, metrics and scripts under test are the real ones.
The cluster `setup` job runs every tier on the real environment before any GPU
job. Tiers are defined in [AGENTS.md §12](../AGENTS.md#12-testing).

| Tier | What it covers | Last result |
| --- | --- | --- |
| 1 Laptop / CI | Every pure module; integration tests skip | 30 Sep: 275 passed, 28 skipped |
| 2 + tau2 v1.0.1 | Evaluation driver through tau2's batch runner, training driver through tau2's orchestrator, their parity (messages, rewards, violations, format errors, token counts on seven scripted conversations), and the CLI pipeline (task sets → evaluation → report, review packet, demo) on a toy domain | 29 Sep: 286 passed |
| 3 + cluster lock on CPU | verl Hydra composition of the RL preset (full and LoRA), vLLM's CLI accepting every server command, Hermes parser parity with vLLM, a `Tau2AgentLoop` episode through verl's `AgentLoopBase`, tiny-model SFT (full and LoRA), the tokenization check, the RL dry run, teacher → SFT data → SFT | 29 Sep: 315 passed |

The 30 Sep changes touch code covered by tiers 2–3 (`run_eval.py` and
`train_rl.py` task loading, `episode.py`); the next `setup` job re-runs them.
