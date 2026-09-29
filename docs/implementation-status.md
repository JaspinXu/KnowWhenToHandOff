# Implementation status

Updated: 29 September 2026. This file separates code verification from research
results. **No benchmark score, trained checkpoint or measured cluster throughput
exists yet**; CPU fakes and toy domains must never be cited as model results.

## Owner decisions

- 27 Sep: implement and CPU-test code before cluster access.
- One A100 and one H200 are available; VRAM, scheduler topology, account,
  partition and scratch path are configured in `slurm/cluster.env` on the cluster.
- 29 Sep: complete the whole pipeline so it can be cloned on the cluster and run
  with Slurm; download no datasets or models on the laptop. The deferred points
  of ADR-006/ADR-007 were therefore decided in ADR-001 to ADR-005 (ADR-001
  accepted for implementation; ADR-002 to ADR-005 proposed for M1).
- 29 Sep: it is a course project; keep code simple and the experiments minimal.
  The plan is five runs (B0, B1 optional, B2, R1, R2; one seed each); the λ
  grid, fairness sampler (R3), PPO (P1), ablations (A1, A2) and the held-out
  simulator were removed, together with the standalone tools written before the
  pipeline (server planner, completion parser, offline SFT export).

## Task status

"Code" means implemented and tested on CPU; "Run" is what the cluster or a
person still has to do for the README definition of done.

| Task | Code | Run / review still needed |
| --- | --- | --- |
| T0.1 | Done | A green remote CI run |
| T0.2 | tau2 v1.0.1 pin, smoke protocol, result parsing | `eval=smoke` on the cluster |
| T0.3 | Slurm jobs, serving, endpoints | Servers answering tool calls; throughput |
| T0.4 | verl stack pinned and verified on CPU (ADR-001) | GPU spike: step time, go/no-go |
| T0.5 | LiteLLM `openai/<served>` + `api_base` for agent and simulator | Full 5-task run on local models |
| T1.1 | Registry, stratified dev carve-out, frozen manifests, disjointness tests | Build and commit the manifests |
| T1.2 | Label rules and file, overrides with reasons ([labels](labels.md)) | Commit the label file |
| T1.3 | Generator, tau2 validation, lineage ([hand-off tasks](handoff-tasks.md)) | Hand-check 30+ tasks |
| T1.4 | Style cards and injection through tau2 | Review 10 dialogues per style |
| T1.5 | All metrics with CIs ([metrics](metrics.md)) | Freeze with ADR-005 at M1 |
| T1.6 | Evaluation driver, protocols, reports | B0 and B1 runs, audit write-up |
| T2.1 | Teacher protocol and filter | Teacher rollouts, dataset statistics |
| T2.2 | SFT trainer (full, LoRA, FSDP) | SFT run and B2 evaluation |
| T2.3 | Comparison and failure packets | Annotate 10 failures |
| T3.1 | `Tau2AgentLoop`, parity test with the evaluation driver | GPU validation |
| T3.2 | Reward composer, per-term logging, the two reward presets | Reward statistics from a real run |
| T3.3–T3.4 | RL launcher (GRPO), checkpoint export, resume | The R1 and R2 runs |
| T3.5 | Reward-hacking monitor, review packet | Review 20 dialogues |
| T3.6 | Test protocols, test ledger | Final evaluation |
| T4.1 | Tables, figures and sources from run IDs | Real results |
| T4.3 | HTML demo page and terminal demo | Slides, recorded demo |
| T4.2, T4.4 | — (writing) | Report, release |

## Verification evidence (29 Sep 2026)

- Laptop / CI environment (`uv run --locked pytest -q`): all tests pass;
  integration tests that need tau2, vLLM, verl or PyTorch are skipped.
- With tau2 v1.0.1 installed: all tests pass, including the evaluation and
  training drivers on a toy tau2 domain, their parity (same messages, rewards,
  violations, format errors and token counts on seven scripted conversations)
  and the command-line pipeline.
- CPU replica of the cluster lock (vLLM 0.12.0, verl 0.8.0, torch 2.9.0):
  all tests pass, including the stack-specific ones listed in
  [code readiness](code-readiness.md).
- Ruff lint and format pass; shell scripts pass `bash -n` and ShellCheck.

## Earlier slices (27–28 Sep)

The foundation (T0.1), run provenance, persona injection (T1.4) and SFT batch
preparation were built first, each test-first with an independent review, and
are part of the pipeline ([personas](personas.md)).
