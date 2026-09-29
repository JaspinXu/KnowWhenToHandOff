# Code readiness before GPU experiments

Re-audited on 29 September 2026. **The pipeline is code-complete**: every stage of
the study has an executable path from the cluster's login node
([cluster runbook](cluster-runbook.md)). What remains are cluster runs, human
reviews and owner decisions, none of which code can replace.

## How the code was verified without GPUs

| Tier | Environment | What runs |
| --- | --- | --- |
| Laptop / CI | `uv sync --locked` (no tau2, vLLM or verl) | Unit tests of every pure module; integration tests are skipped |
| tau2 integration | + tau2 v1.0.1 | Evaluation driver end to end through tau2's batch runner, training driver through tau2's orchestrator, **parity** between them on seven scripted conversations, and the command-line pipeline (task sets → evaluation → report, review packet, demo) on an in-memory toy domain |
| Cluster replica | + the full `requirements/cluster-gpu.txt` on CPU | verl Hydra composition of the RL preset (full and LoRA), vLLM's CLI accepting every server command, Hermes parser parity with vLLM, a `Tau2AgentLoop` episode through verl's `AgentLoopBase`, a tiny-model SFT run (full and LoRA), the tokenization check, the RL launch dry run, and the teacher → SFT-data → SFT pipeline |

The setup job runs the whole suite on the cluster, so the third tier is repeated
on the real environment before any GPU job. Only model servers, model weights
and benchmark data are replaced by test doubles; the drivers, rewards, metrics
and scripts are the real ones.

## Workflow status

| Workflow | Code | Outstanding (not code) |
| --- | --- | --- |
| Environment and quality | uv lock, ruff, pytest, CI (Ubuntu + Windows), cluster lock and setup job | Green remote CI run; first `setup` job on the cluster |
| Models and servers | Download with revision lock, vLLM serving with endpoint files and shared key, waiting and stale-endpoint detection | Server responses and throughput on the cluster (T0.3) |
| Benchmark and tasks | tau2 adapter, labels, splits, hand-off generator, persona injection, frozen-manifest builder | Running `tasks`, hand-checking 30+ tasks, committing the manifests |
| Evaluation | tau2 runner with the project agent, stop sentinel, metrics with CIs, test ledger, reports | B0/B1 runs, the style-card review, freezing ADR-005 at M1 |
| Reward | Rules, format checks, composer, the R1 and R2 presets, parity with evaluation | R2 vs R1 on dev (T3.4) |
| SFT | Teacher filtering by registry split, assistant-only records, HF Trainer (full / LoRA / FSDP) | Teacher rollouts, SFT run, B2 evaluation |
| Multi-turn RL | verl agent loop (token-in/token-out), GRPO preset, reward-hacking monitor, learning curves, resume, checkpoint export | T0.4 spike (step time), R1 and R2 runs |
| Reproducibility | Run records (config, git SHA, tau2 commit, model revisions, seeds, hardware), experiment log, report provenance | Filling the experiment log |
| Presentation | Report tables and figures from run IDs, review packets, HTML and terminal demos | Report, slides, model card |

## Decisions still open

ADR-001 is accepted for implementation; its T0.4 measurements are pending.
ADR-002 to ADR-005 are proposed and should be accepted (or amended) at M1,
after the B0 dev runs. See [implementation status](implementation-status.md).
