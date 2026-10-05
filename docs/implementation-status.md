# Implementation status

Updated: 5 October 2026. This file separates code verification from research
results. Real results come only from cluster runs (run IDs in
[experiment-log.md](experiment-log.md)); CPU fakes and toy domains must never be
cited as model results.
Owner decisions and scope changes are in [AGENTS.md §17](../AGENTS.md#17-decision-log).

## Task status

"Code" means implemented and tested on CPU; "Run" is what the cluster or a
person still has to do for the README definition of done.

| Task | Code | Run / review still needed |
| --- | --- | --- |
| T0.1 | Done; CI (Ubuntu + Windows) green | — |
| T0.2 | tau2 v1.0.1 pin, smoke protocol, result parsing | Done 4 Oct: smoke run on local models (runs live in `$HANDOFF_SCRATCH/runs`, not `data/simulations/`) |
| T0.3 | Slurm jobs, serving, endpoint files | Done 3–4 Oct: both servers answer tool calls; 490 episodes in 23 min |
| T0.4 | verl stack pinned and verified on CPU (ADR-001) | Done 5 Oct: spike recorded in ADR-001 (go, LoRA) |
| T0.5 | LiteLLM `openai/<served>` + `api_base` for agent and simulator | Done 4 Oct: smoke and B0 ran with no external API |
| T1.1 | Registry, stratified dev carve-out, frozen manifests, disjointness tests | Done 3 Oct (`ef34428`) |
| T1.2 | Label rules and file, overrides with reasons ([labels](labels.md)) | Done 3 Oct (`ef34428`) |
| T1.3 | Generator, tau2 validation, lineage ([hand-off tasks](handoff-tasks.md)) | Done 3 Oct: 40/40 checked |
| T1.4 | Style cards and injection through tau2 ([personas](personas.md)) | Done 4 Oct: 50 dialogues reviewed |
| T1.5 | All metrics with CIs ([metrics](metrics.md)) | Done 4 Oct: frozen with ADR-005 |
| T1.6 | Evaluation driver, protocols, reports | Done 5 Oct: B0, B1, [audit](responsible-ai-audit.md), `reports/m1/` |
| T2.1 | Teacher protocol and filter | Done 4–5 Oct: 3,432 teacher episodes → 790 SFT dialogues (`b2_v1`) |
| T2.2 | SFT trainer (full, LoRA, FSDP) | Done 5 Oct: LoRA SFT (full fine-tuning OOM), B2 evaluated |
| T2.3 | Comparison and failure packets | Done 5 Oct: [error analysis](error-analysis.md) |
| T3.1 | `Tau2AgentLoop`, parity test with the evaluation driver | GPU-validated in the spike (5 Oct) |
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
| 1 Laptop / CI | Every pure module; integration tests skip | 5 Oct: 281 passed, 28 skipped |
| 2 + tau2 v1.0.1 | Evaluation driver through tau2's batch runner, training driver through tau2's orchestrator, their parity (messages, rewards, violations, format errors, token counts on seven scripted conversations), and the CLI pipeline (task sets → evaluation → report, review packet, demo) on a toy domain | 29 Sep: 286 passed |
| 3 + cluster lock on CPU | verl Hydra composition of the RL preset (full and LoRA), vLLM's CLI accepting every server command, Hermes parser parity with vLLM, a `Tau2AgentLoop` episode through verl's `AgentLoopBase`, tiny-model SFT (full and LoRA), the tokenization check, the RL dry run, teacher → SFT data → SFT | 29 Sep: 315 passed |

The 30 Sep changes touch code covered by tiers 2–3 (`run_eval.py` and
`train_rl.py` task loading, `episode.py`); the next `setup` job re-runs them.
