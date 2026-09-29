# AGENTS.md

Working agreement for every AI coding agent (Codex, Claude, Copilot, Gemini, …)
and every human who works on **KnowWhenToHandOff**. The owner switches between
agents, so this file is the shared memory of the project: read all of it before
you change anything, and update it before you stop (§16).

- Design and research plan: [README.md](README.md) (Chinese mirror:
  [README.zh-CN.md](README.zh-CN.md)).
- Cluster commands, step by step: [docs/cluster-runbook.md](docs/cluster-runbook.md).
- Claude reads this file through [CLAUDE.md](CLAUDE.md); Codex reads it directly.
- If anything here conflicts with a direct instruction from the owner, follow
  the owner, then record the new instruction in §17 so the next agent knows.

---

## Contents

1. [The project in brief](#1-the-project-in-brief)
2. [Status board](#2-status-board)
3. [Hard rules](#3-hard-rules)
4. [Where things run](#4-where-things-run)
5. [Repository map](#5-repository-map)
6. [Pipeline and data flow](#6-pipeline-and-data-flow)
7. [Contracts and on-disk layouts](#7-contracts-and-on-disk-layouts)
8. [Configuration](#8-configuration)
9. [Commands](#9-commands)
10. [Experiments](#10-experiments)
11. [Coding conventions](#11-coding-conventions)
12. [Testing](#12-testing)
13. [Keeping docs in sync](#13-keeping-docs-in-sync)
14. [Git, CI and delivery](#14-git-ci-and-delivery)
15. [Known pitfalls](#15-known-pitfalls)
16. [Session protocol for switching agents](#16-session-protocol-for-switching-agents)
17. [Decision log](#17-decision-log)
18. [When unsure](#18-when-unsure)
19. [Glossary](#19-glossary)

---

## 1. The project in brief

A solo-built course project for NUS CS4246/5446 (*Reinforcement Learning and
Sequential Decision Making*, AY2026/27 Semester 1). We train
**Qwen3-4B-Instruct-2507** with multi-turn reinforcement learning (GRPO, verl) on
**τ²-bench v1.0.1** (airline and retail domains) to act as a customer-service
agent that:

- completes the task (τ²-bench database-state reward),
- follows the domain policy (no violations),
- calls `transfer_to_human_agents` only when the user asks for a human or the
  policy and tools cannot solve the issue,
- serves users with different communication styles equally well (measured),
- cites the policy behind refusals and hand-offs (measured).

**Pipeline:** baseline audit (B0, B1) → imitation warm start (B2, SFT) → GRPO
with the task reward only (R1) and with Responsible-AI reward terms (R2).

**Scope — keep it small.** It is a course project with a hard deadline, not a
research paper. The owner asked (29 Sep) for simple code and a minimal
experiment set: five runs, one seed each, no hyper-parameter sweeps, no
ablations, no extra algorithms. Do not add experiments, models, frameworks or
abstractions unless the owner asks. Prefer deleting over adding.

**Dates:** M1 baseline audit 11 Oct · M2 SFT 18 Oct · M3 RL + experiment freeze
1 Nov · presentation week of 10 Nov · report due 15 Nov 2026 23:59
(≤ 10 pages, Vancouver references, contribution and AI-use statements).

---

## 2. Status board

> Update this section at the end of every session (§16). Newest first; keep it
> short and factual. Code verification is not a research result.

**As of 30 Sep 2026** (`main` after the completeness audit, see §17):

- **Code:** the whole pipeline is implemented and CPU-verified. 30 Sep audit
  against the README DoDs added the one missing piece, RL learning curves and
  per-step reward-term statistics (`scripts/rl_curves.py`, T3.2–T3.4), fixed
  `export_checkpoint.py --step latest` (it sorted step names as text), and
  removed duplicated helpers. Laptop tier: 277 passed, 28 skipped; ruff,
  `bash -n` and `uv build` pass. Tiers 2–3 were last run on 29 Sep (286 and 315
  passed); the 30 Sep changes touch `run_eval.py`/`train_rl.py` task loading
  (now `tau2_adapter.load_frozen_tasks`) and `episode.py`, so the next cluster
  `setup` job re-runs them. ShellCheck runs in CI.
- **Experiments:** none yet. No benchmark score, checkpoint or throughput
  measurement exists. Toy/fake results must never be cited.
- **Next steps (in order):** push and check CI; on the cluster, `setup` →
  `download` → `tasks` (hand-check ≥ 30 hand-off tasks, commit `data/`) →
  `serve` → `check-tokens`
  → smoke eval → B0 (dev + three test presets) → optional B1 → teacher →
  `sft-data` → `sft` (B2) → B2 eval → RL spike (record in ADR-001) → R1 → R2 →
  final evaluations → report. Runbook §9 has the order with GPUs.
- **Open risks:** whether `/temp` on the SoC cluster is shared across nodes is
  unconfirmed (`df -T /temp` on two nodes; the setup job refuses node-local
  storage). The cluster's NVIDIA driver must be ≥ 525.
- **Owner decisions pending:** accept ADR-002 to ADR-005 at M1; confirm the
  course group number and team contribution statement for the report.

Per-task detail: [docs/implementation-status.md](docs/implementation-status.md)
and [docs/code-readiness.md](docs/code-readiness.md).

---

## 3. Hard rules

Non-negotiable. If a task seems to need breaking one, stop and ask the owner.

1. **The test split is sacred.**
    - Never train on, build SFT data from, tune on, or select checkpoints with
      test tasks — official test split and hand-off test set alike.
    - Use the dev split (carved from train) for every selection decision.
    - Each frozen checkpoint is evaluated on test once per simulator; the test
      ledger enforces this. Never bypass it without an ADR.
2. **Do not edit upstream τ²-bench.** Wrap it in `src/handoff/`. The version is
   pinned (commit `fc0055dc4e0a316c3f83133267fbd6faaa770992`, v1.0.1:
   `TAU2_COMMIT` in `slurm/cluster.env.example`, the split manifest and
   ADR-005). Upgrading needs an ADR.
3. **Rewards and metrics are deterministic code.** They come from the τ²-bench
   evaluator, rule checks and labels. No LLM judge anywhere; tasks whose reward
   needs an NL assertion are rejected.
4. **The evaluation protocol is frozen after M1 (11 Oct 2026).** Changing a
   metric definition, split, trial count, simulator or setting for reported
   numbers needs a new ADR in `docs/adr/`.
5. **Never download datasets or models on the owner's laptop.** The laptop
   environment is CPU-only and never installs τ²-bench (its git checkout
   contains the benchmark data), vLLM or verl. Downloads happen only in cluster
   jobs (`setup`, `download`). Installing the locked dev dependencies with
   `uv sync` is fine.
6. **No secrets in the repo or in output.** `.env` and `slurm/cluster.env` are
   gitignored. The vLLM API key lives on scratch (`state/secrets/`, mode 600)
   and travels only through environment variables. Never print keys, never put
   them in endpoint files, configs, logs or run records.
7. **No large artefacts in git.** Checkpoints, trajectories, datasets, HF
   caches and review packets stay on cluster scratch (or gitignored folders).
   Git holds code, configs, small manifests and final report assets.
8. **No result without provenance.** Every number in the README, report or
   slides traces to a run ID with its config, git SHA, τ²-bench commit, model
   revisions and seeds (`reports/<name>/sources.json`, `docs/experiment-log.md`).
9. **Real runs only from a clean, committed checkout.** Provenance refuses
   dirty code, configs or data (changes under `docs/` and `reports/` are
   allowed). Commit and push before submitting cluster jobs.
10. **Stay inside the task.** Implement only what was asked. A change to a
    shared contract (§7) is proposed in an ADR draft or the commit body and
    updates every caller in the same change.
11. **No real user data; no stereotypes.** All users are simulated. Style cards
    describe communication style only, never a demographic group.

---

## 4. Where things run

| Place | What runs there | What never runs there |
| --- | --- | --- |
| Owner's laptop (Windows 11, PowerShell, git + GitHub CLI; clone at `G:\KnowWhenToHandOff\KnowWhenToHandOff`) | `uv sync --locked`, ruff, CPU unit tests, `uv build`, git push | τ²-bench, vLLM, verl, torch; any model or dataset download |
| GitHub Actions (`.github/workflows/ci.yml`) | Ubuntu + Windows: lint, format check, pytest, build; Ubuntu also `bash -n` + ShellCheck on Slurm scripts | anything needing tau2/GPU |
| NUS SoC Slurm cluster (Linux) | environment build, downloads, task sets, model servers, evaluation, SFT, RL, reports | nothing heavy on the login node (only `slurm/run.sh` light commands) |
| An agent's cloud sandbox (if it has one) | all three test tiers, if the agent can install tau2/verl there | pushing with the owner's credentials |

**Hardware:** one A100 80 GB (`GPU_TYPE_SIM`) keeps the user simulator
(Qwen3-30B-A3B-Instruct-2507, bf16, ~61 GB) up for a whole campaign; one H200
141 GB (`GPU_TYPE_BIG`) runs evaluations, SFT and RL one job at a time.

**Pinned stack** (ADR-001, `requirements/cluster-gpu.txt`, generated by
`scripts/lock_cluster_env.sh`): Python 3.12, verl 0.8.0, vLLM 0.12.0 (wheels need
glibc ≥ 2.31), PyTorch 2.9.0 + CUDA 12.8 (driver ≥ 525), transformers 4.57.6,
LiteLLM 1.82.6, peft 0.19.1. The laptop/CI environment is `pyproject.toml` +
`uv.lock` (hydra-core, pytest, pytest-socket, ruff, pre-commit).

---

## 5. Repository map

The package is `handoff` (src layout). Each file's docstring says what it owns.

| Path | Owns |
| --- | --- |
| `pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` | Laptop/CI tooling and pins |
| `requirements/cluster-gpu.in`, `cluster-gpu.txt` | Cluster GPU lock; regenerate only with `scripts/lock_cluster_env.sh` + ADR |
| `configs/` | Hydra configs (§8) |
| `data/manifests/` | Frozen labels, splits and hand-off manifests with hashes (`label_overrides_v1.json` is hand-written); built by `tasks` on the cluster, then committed |
| `data/handoff_tasks/v1/` | Generated hand-off task JSON (committed after the hand check) |
| `src/handoff/cli.py` | Shared CLI helpers: compose Hydra config from the repo, scratch guard, errors |
| `src/handoff/envs/registry.py` | `TaskRegistry`, split manifests, stratified dev carve-out (T1.1) |
| `src/handoff/envs/labels.py` | Should-transfer labels (T1.2) |
| `src/handoff/envs/handoff_tasks.py` | Hand-off task generator: `explicit_request`, `out_of_scope` (y = 1), `hard_negative` (y = 0) (T1.3) |
| `src/handoff/envs/personas.py` | Style cards and injection into `user_scenario.persona` (T1.4) |
| `src/handoff/envs/tau2_adapter.py` | Manifests → tau2 `Task` objects; tau2 commit and data-hash checks |
| `src/handoff/agents/prompts.py`, `tau2_agents.py` | Shared system prompt; evaluation agent registered with the tau2 runner |
| `src/handoff/sim/serving.py`, `endpoints.py`, `vllm_auth.py` | vLLM server launch, endpoint files, LiteLLM settings, API-key middleware |
| `src/handoff/rollout/schema.py` | `Trajectory`, `Violation`, `FormatError` — the shared episode record |
| `src/handoff/rollout/tau2_convert.py`, `scoring.py` | tau2 objects → normalized messages → scored `Trajectory` |
| `src/handoff/rollout/episode.py`, `chat_delta.py`, `hermes.py` | Training-side episode driver (token-in/token-out), incremental tokenization, Hermes tool-call parsing |
| `src/handoff/rollout/verl_loop.py`, `verl_reward.py` | verl `Tau2AgentLoop` (`tau2_agent`), simulator-outage handling, guard reward fn |
| `src/handoff/rewards/rules.py`, `format.py`, `composer.py` | Violation rules, format errors, `RewardComposer` (T3.2) |
| `src/handoff/eval/runner.py`, `metrics.py`, `citations.py`, `report.py`, `analysis.py` | Evaluation driver, metrics with bootstrap CIs, citation rule, report tables/figures, review packets |
| `src/handoff/train/sft_data.py`, `sft_batch.py`, `sft_trainer.py` | Teacher filter, assistant-only SFT records/labels, HF Trainer (T2.1–T2.2) |
| `src/handoff/train/rl_data.py`, `rl_launch.py`, `verl_main.py`, `export.py`, `monitors.py` | RL dataset rows, verl command, entry point with checkpoint export, reward-hacking monitor and learning curves (T3.2–T3.5) |
| `src/handoff/ops/runs.py`, `provenance.py`, `slurm.py` | Run IDs, model identity, test ledger; immutable run records; Slurm queries |
| `src/handoff/demo/viewer.py` | Offline HTML dialogue viewer (T4.3) |
| `scripts/` | Thin CLI entry points only (no business logic): `build_task_sets`, `download_models`, `model_server`, `check_tokenization`, `run_eval`, `build_sft_data`, `train_sft`, `train_rl`, `monitor_rl`, `rl_curves`, `export_checkpoint`, `make_report`, `sample_review`, `append_experiment_log`, `render_demo`, `demo_chat`, `lock_cluster_env.sh` |
| `slurm/submit.sh`, `run.sh`, `lib/common.sh`, `jobs/*.sbatch`, `cluster.env.example` | Cluster wrapper, login-node helper, shared shell functions, one job per step |
| `tests/` | CPU tests (§12); `fakes.py`, `toy_tau2.py`, `tiny_hf.py` are shared doubles |
| `docs/` | Runbook, metrics, labels, hand-off tasks, personas, ADRs, experiment log, status |
| `reports/` | Generated tables and figures (final versions committed); `reports/review/` is not committed |

---

## 6. Pipeline and data flow

| Step | Job / script | Reads | Writes |
| --- | --- | --- | --- |
| Environment | `setup` → `slurm/jobs/setup_env.sbatch` | lock, tau2 @ pinned commit | `$HANDOFF_VENV`, `$TAU2_REPO_DIR`; runs all test tiers |
| Models | `download` → `download_models.py` | model configs | `$HF_HOME` snapshots, `state/models.lock.json` (repo, SHA, path) |
| Task sets | `tasks` → `build_task_sets.py` | tau2 data, `configs/handoff_tasks/v1.yaml`, persona cards | `data/manifests/{labels_v1,splits_v1,handoff_tasks_v1}.json`, `data/handoff_tasks/v1/`, review packet |
| Simulator | `serve` → `model_server.py serve` | models lock | `state/endpoints/usersim.json`, `state/secrets/vllm_api_key` |
| Tokenizer check | `check-tokens` → `check_tokenization.py` | tokenizer, domains | PASS/FAIL |
| Evaluation | `eval` / `teacher` → `run_eval.py` | manifests, endpoints | `runs/<id>/` (§7); exit 3 = incomplete |
| SFT data | `sft-data` → `build_sft_data.py` | teacher runs (train split only) | `datasets/<name>/` + `manifest.json` |
| SFT | `sft` → `train_sft.py` | dataset, base model | `runs/B2-.../final` (HF checkpoint, LoRA merged) |
| RL | `rl` → `train_rl.py` → `python -m handoff.train.verl_main` | B2 checkpoint, manifests, simulator | `runs/R*-.../checkpoints` (latest only), `export/global_step_N`, `rollouts/*.jsonl`, `rollout_dumps/`, `validation_dumps/` (verl per-step dumps) |
| Monitor | `slurm/run.sh python scripts/monitor_rl.py` | rollouts, B2 dev metrics | exit 3 = hacking drift, 4 = infra losses |
| Curves | `slurm/run.sh python scripts/rl_curves.py` | verl step dumps | `reports/<name>/learning_curves.*`, `dev_curves.*`, `sources.json` |
| Report | `make_report.py`, `sample_review.py`, `render_demo.py`, `append_experiment_log.py` | completed runs | `reports/<name>/`, `reports/review/`, `docs/experiment-log.md` |

**Two drivers, one semantics.** Evaluation uses tau2's own runner with the
project agent through LiteLLM → vLLM; training uses tau2's orchestrator inside
`Tau2AgentLoop` with token-in/token-out generation. Both share the task store,
persona injection, prompt, rules, format checks, labels and `RewardComposer`; a
parity test on scripted conversations keeps them identical (messages, rewards,
violations, format errors, token counts).

**Episode budget (both drivers):** 200 tau2 steps, 10 tool errors, 30 min wall
clock, 2,048 tokens per policy turn, 32,768-token context. The episode ends
cleanly (`context_window_exceeded`, no format error) once one more full turn
would not fit. Training additionally caps the prompt at 8,192 and the policy
response at 24,576 tokens.

---

## 7. Contracts and on-disk layouts

Changing any of these is a shared-contract change (§3 rule 10).

- **`Trajectory`** (`rollout/schema.py`): `run_id, task_id, source_task_id,
  domain, split, kind, persona, trial, seed, policy_ckpt, user_sim, messages,
  termination_reason, task_reward, reward_basis, db_match, transferred,
  transfer_turn, should_transfer, violations, format_errors, n_turns, …`.
  `task_reward` always comes from the tau2 evaluator. "Transferred" = the first
  `transfer_to_human_agents` call, whatever the tool returned.
- **Violation rules** (`RULE_IDS`): `write_without_confirmation`,
  `write_before_authentication`, `multiple_actions_in_turn`,
  `action_after_transfer`, `off_reference_write` ([labels](docs/labels.md)).
- **Format errors** (`FORMAT_ERROR_KINDS`): `unparsed_tool_call`,
  `unknown_tool`, `invalid_arguments`, `empty_message`, `protocol_error`,
  `truncated_generation`.
- **`RewardComposer`** → `RewardBreakdown` (every term logged):
  `R = R_task − λv·Σ w_r N_r − λFP·1[transfer ∧ ¬y] − λFN·1[¬transfer ∧ y] − λf·N_format`.
  `task_only`: all λ = 0 (R1). `r2`: λv 0.25, λFP 0.5, λFN 1.0, λf 0.1 (R2).
- **Run IDs:** `<experiment>[-<tags>]-s<seed>-<YYYYMMDD>`, experiment ∈
  `B0 B1 B2 R1 R2 SMOKE TEACH SPIKE` (`RUN_ID_PATTERN` in `ops/provenance.py`).
  Evaluation IDs include the eval preset (`B0-test_main-s0-20261008`); a second
  run of the same ID on one day gets `tryN`; dry RL runs get `dry`.
- **Scratch layout** (`configs/paths/default.yaml`): `$HANDOFF_SCRATCH/`
  `state/{endpoints/,secrets/vllm_api_key,models.lock.json,test_ledger.jsonl}`,
  `runs/<run_id>/`, `datasets/<name>/`, `hf_cache/`, `venv/`,
  `third_party/tau2-bench/`.
- **Run directory:** `config.json` (resolved, no URLs or keys),
  `provenance.json` (git SHA and dirty flag, tau2 commit, model revisions,
  seeds, hardware; immutable), `attempts.jsonl` (each job attempt),
  `completion.json` (written only when the run is complete; earlier ones are
  kept as `completion.attemptN.json` on resume), plus for evaluations
  `trajectories.jsonl`, `metrics.json`, `summary.md`, `infra_failures.jsonl`.
- **Endpoint file** (`state/endpoints/<name>.json`): role, base URL, served name,
  model path/repo/revision, Slurm job ID, start time, `max_model_len`,
  `chat_template_kwargs`, server arguments. Never the key. Files of ended jobs
  are ignored.
- **Test ledger:** one record per test run (policy revision, simulator, test
  tasks, styles). A new run conflicts if it shares policy + simulator + a task +
  a style with another run ID. Forcing needs `force_test_rerun=true` and a
  `test_rerun_reason`.
- **Complete runs:** an evaluation with more lost episodes than
  `eval.max_infra_failure_share` (0 by default, 0.05 for teacher rollouts) exits
  3 and is resumed in place with `RESUME=<run id>`; only completed real runs
  enter reports.

---

## 8. Configuration

Scripts compose Hydra configs from `configs/` relative to the repo (never the
shell's working directory) and record the resolved config in every run. No
hard-coded paths, hosts, ports, model names or λ values in library code.

| Entry config | Script | Main groups / keys |
| --- | --- | --- |
| `task_sets.yaml` | `build_task_sets.py` | domains, dev fraction, `split_seed` 20261011, `handoff_tasks`, `persona` |
| `run_eval.yaml` | `run_eval.py` | `eval`, `reward`, `persona`, `experiment`, `policy.*`, `user_sim.*`, `resume_run_id`, test re-run keys |
| `sft_data.yaml` | `build_sft_data.py` | teacher runs, student tokenizer, filter |
| `train_sft.yaml` | `train_sft.py` | `base_model`, `dataset_name`, `sft.*` (LR 1e-5, 2 epochs, LoRA off) |
| `train_rl.yaml` | `train_rl.py` | `rl`, `reward`, `persona`, `policy.init_path`, `episode.*`, `user_sim.*`, `data.*`, `dry_run`, `resume_run_id` |
| `base.yaml` | download, serving, reports, monitors | `paths` only |

| Group | Presets |
| --- | --- |
| `eval/` | `smoke` (5 airline train tasks), `dev` (4 trials), `test_main` (neutral, 4 trials), `test_styles` (5 styles, 2 trials), `test_handoff` (hand-off kinds, 2 trials), `teacher_train` (train, 6 styles, 2 trials, T 0.6) |
| `reward/` | `task_only` (R1), `r2` (R2) |
| `rl/` | `grpo` (verl overrides under `verl:`: group 8, batch 8, 60 steps, save/validate every 10, LR 1e-6, temperature 1.0, no KL, keep 1 checkpoint) |
| `model/` | `qwen3_4b` (policy), `qwen3_30b_a3b` (simulator, teacher, B1), `custom` |
| `persona/` | `styles_v1` (neutral, terse, verbose, impatient, non-native, low-tech-literacy) |
| `handoff_tasks/` | `v1` |
| `paths/` | `default` (§7) |

Override on the command line, e.g. `python scripts/run_eval.py eval=dev
eval.max_concurrency=8 policy.label=b2`. Through Slurm, extra overrides go in
`EXTRA="..."`, split like a shell command line: a value with spaces goes in
single quotes inside the double quotes; a value with commas must be quoted for
Hydra.

**Environment variables** (set in `slurm/cluster.env`, sourced by every job):
`HANDOFF_SCRATCH`, `HANDOFF_VENV`, `HF_HOME`, `UV_CACHE_DIR`, `TAU2_REPO_DIR`,
`TAU2_DATA_DIR`, `TAU2_COMMIT`, `HANDOFF_PYTHON`, `HANDOFF_LOGS` (optional),
`SLURM_ACCOUNT`, `PARTITION_GPU`, `PARTITION_CPU`, `GPU_TYPE_ANY`,
`GPU_TYPE_SIM`, `GPU_TYPE_BIG`, `TIME_*`, `CPUS_GPU_JOB`, `MEM_GPU_JOB`;
optional `HF_TOKEN`, `WANDB_API_KEY`. `HANDOFF_ALLOW_LOCAL_SCRATCH=1` overrides
the shared-scratch check (only if the storage really is shared).

---

## 9. Commands

> Laptop commands run on CPU with sockets disabled in tests. Everything that
> needs GPUs, models or τ²-bench data runs on the cluster through
> `slurm/submit.sh`; the full sequence is in the runbook.

```bash
# laptop / CI (Python 3.12, uv)
uv sync --locked
uv run --locked ruff check . && uv run --locked ruff format --check .
uv run --locked pytest -q                 # tau2/vLLM/verl tests skip here
uv build
for f in slurm/*.sh slurm/lib/common.sh slurm/jobs/*.sbatch; do bash -n "$f" || exit 1; done
shellcheck -x -P slurm slurm/*.sh slurm/lib/common.sh slurm/jobs/*.sbatch   # if installed
# PowerShell: Copy-Item .env.example .env   (optional; tests need no credentials)

# cluster: one-time
cp slurm/cluster.env.example slurm/cluster.env   # edit it; source it in your shell
bash slurm/submit.sh setup                # venv, pinned stack, tau2 v1.0.1, all test tiers
bash slurm/submit.sh download MODELS="qwen3_4b qwen3_30b_a3b"
bash slurm/submit.sh tasks                # labels, splits, hand-off tasks (hand-check, then commit data/)
bash slurm/submit.sh check-tokens MODEL=qwen3_4b

# cluster: evaluation (the simulator server must be running)
bash slurm/submit.sh serve                # ROLE=usersim MODEL=qwen3_30b_a3b on GPU_TYPE_SIM
bash slurm/submit.sh eval EVAL=smoke EXPERIMENT=SMOKE POLICY_MODEL=qwen3_4b
bash slurm/submit.sh eval EVAL=dev EXPERIMENT=B0 POLICY_MODEL=qwen3_4b
bash slurm/submit.sh eval EVAL=test_main EXPERIMENT=B0 POLICY_MODEL=qwen3_4b   # also test_styles, test_handoff
bash slurm/submit.sh eval EVAL=test_main EXPERIMENT=B1 POLICY_ENDPOINT=$HANDOFF_SCRATCH/state/endpoints/usersim.json POLICY_LABEL=qwen3_30b_a3b
bash slurm/submit.sh eval RESUME=<run id> POLICY_MODEL=qwen3_4b    # after exit 3 (lost episodes)

# cluster: training
bash slurm/submit.sh teacher POLICY_ENDPOINT=$HANDOFF_SCRATCH/state/endpoints/usersim.json POLICY_LABEL=qwen3_30b_a3b
bash slurm/submit.sh sft-data TEACHER_RUNS=<TEACH run id> DATASET=b2_v1
bash slurm/submit.sh sft DATASET=b2_v1
bash slurm/submit.sh eval EVAL=dev EXPERIMENT=B2 POLICY_PATH=<B2 final dir> POLICY_LABEL=b2
bash slurm/submit.sh rl REWARD=task_only EXPERIMENT=SPIKE INIT=<B2 dir> EXTRA="rl.verl.trainer.total_training_steps=3 rl.verl.trainer.val_before_train=false"
bash slurm/submit.sh rl REWARD=task_only EXPERIMENT=R1 INIT=<B2 dir>
bash slurm/submit.sh rl REWARD=r2 EXPERIMENT=R2 INIT=<B2 dir>
bash slurm/submit.sh rl ... EXTRA="dry_run=true"          # build data + validate the verl command only
bash slurm/submit.sh rl RESUME=<RL run id>                # time limit, crash, simulator outage
bash slurm/run.sh python scripts/monitor_rl.py --run <RL run id> --baseline <B2 dev run id>
bash slurm/run.sh python scripts/rl_curves.py --name rl --runs R1=<R1 run id> R2=<R2 run id>
bash slurm/run.sh python scripts/export_checkpoint.py <RL run id> --step latest

# results (login node, inside the environment)
bash slurm/run.sh python scripts/append_experiment_log.py <run id> --purpose "..." --decision "..."
bash slurm/run.sh python scripts/make_report.py --name main --baseline B2 \
  --runs B0=<test_main id>+<test_styles id>+<test_handoff id> B2=<...> R1=<...> R2=<...>
bash slurm/submit.sh cpu CMD="python scripts/sample_review.py audit --run <final test run> --n 20"
bash slurm/submit.sh cpu CMD="python scripts/render_demo.py --run <test_handoff run>"
```

Job logs: `logs/handoff-<job>-<jobid>.out` in the checkout. Every `submit.sh`
job accepts `KEY=VALUE` pairs and `-- <extra sbatch flags>`; config names and
`INIT` are checked before anything is queued. Always run the smoke evaluation
after changing the evaluation path, and a dry run before a long RL run.

---

## 10. Experiments

| Run | Policy init | Training | Reward | Question |
| --- | --- | --- | --- | --- |
| B0 | Qwen3-4B, prompted | — | — | Starting point; RQ2/RQ3 baseline audit |
| B1 (optional) | Qwen3-30B-A3B, prompted | — | — | Reference ceiling (`test_main` only) |
| B2 | B0 | SFT on filtered 30B teacher episodes (train split) | — | RQ1: what imitation buys |
| R1 | B2 | GRPO | `task_only` | RQ1: what RL buys |
| R2 | B2 | GRPO | `r2` | RQ2: compliance and hand-offs vs success |

- RQ3 (style gap) and RQ4 (reward hacking) are measured on these runs; no
  extra training.
- Not in scope (removed 29 Sep): λ grid, fairness sampler (R3), PPO (P1),
  ablations (A1/A2), held-out simulator, a second smoke model, an LLM citation
  judge. Do not reintroduce them without the owner.
- **Selection:** dev only. The RL checkpoint is the final step unless the dev
  validation curve (`rl_curves.py` → `dev_curves.md`) shows a clearly better
  earlier export.
- **Final evaluation per checkpoint:** `dev`, then `test_main`, `test_styles`,
  `test_handoff` once each. 95 % bootstrap CIs over tasks (1,000 resamples),
  paired comparisons on shared tasks.
- **Seeds:** one per run; state it as a limitation in the report.
- **If time runs short:** drop B1 first, then reduce RL steps; never cut
  evaluation.
- **Reward-hacking watch-list** (`monitor_rl.py`, last 512 episodes vs B2 dev;
  stop the run with `scancel` if one exceeds max(2 × B2, 0.02)): transfer rate
  on should-not-transfer episodes, mean messages, mean turns, format-error rate,
  share of episodes ending on a user stop without a completed task.
- **Experiment log:** after every run with numbers, append a row to
  `docs/experiment-log.md` with `append_experiment_log.py` and commit it.
- **Human checks:** ≥ 30 hand-off tasks (packet `tasks`), 10 dialogues per style
  (`styles`, from B0 `test_styles`), 10 B2 failures (`failures`), 20 dialogues
  per final checkpoint (`audit`). Review packets are not committed.

---

## 11. Coding conventions

- Python 3.12 with full type hints (most modules use
  `from __future__ import annotations`). Frozen dataclasses for records and settings; validate settings in `from_mapping`.
- Pure functions for labels, rules, rewards and metrics: input in, value out,
  no I/O or global state.
- `scripts/` are thin: parse args/config, call library code, print; raise
  `CliError` for user errors. `logging` in library code, `print` only in scripts.
- Log every reward term separately, never only the total.
- Seeds are explicit everywhere and recorded.
- Async code in the verl agent loop never blocks the event loop (tau2 runs in a
  worker thread).
- Training tokens: keep the IDs vLLM sampled and append observation tokens
  incrementally; never re-tokenize the final messages. `response_mask` = 1 for
  policy tokens, 0 for user/tool tokens.
- English identifiers, comments and docstrings; a docstring on every public
  function and module (with the task ID it serves); no dead code or
  commented-out blocks. Line length 88 (ruff: E, F, I, UP, B).
- New heavy dependencies (trainer, serving stack) only through an ADR; the
  cluster lock changes only through `scripts/lock_cluster_env.sh`.
- Simplicity over generality: the smallest change that meets the definition of
  done. No options nobody uses.

---

## 12. Testing

Tests run on CPU with sockets disabled (`pytest-socket`); model servers,
weights and benchmark data are replaced by doubles (`tests/fakes.py`,
`tests/toy_tau2.py` — an in-memory tau2 domain with scripted policies and users,
`tests/tiny_hf.py` — a tiny Hugging Face model built on the fly). The drivers,
rewards, metrics and scripts under test are the real ones.

| Tier | Needs | Files | Where it runs |
| --- | --- | --- | --- |
| 1 Unit | `uv sync --locked` | everything not below | laptop, CI, cluster |
| 2 tau2 integration | + tau2 v1.0.1 | `test_tau2_integration.py`, `test_cli_pipeline.py` | cluster `setup` job (agent sandboxes may replicate it) |
| 3 Cluster stack on CPU | + `requirements/cluster-gpu.txt` | `test_gpu_stack.py`, `test_hf_cpu.py` | cluster `setup` job |

Each tier skips itself when its packages are missing. Before a commit: tier 1
plus ruff and the shell checks; when you touched tau2, verl, vLLM or
transformers code paths, also run tiers 2 and 3 where you can, and say in the
commit body if you could not.

**Required coverage:**

| Area | What the tests pin down |
| --- | --- |
| Splits | Train, dev and test (official and hand-off) are disjoint; manifest hashes match; changed tau2 data is refused |
| Labels | Should-transfer rules on hand-written examples, including upstream gold transfers |
| Violation rules | Every rule has a positive and a negative trajectory (incl. negated confirmations) |
| Reward composer | Each term in isolation; λ = 0 reduces to the task reward; FP/FN trigger correctly |
| Metrics | pass^k vs brute force; bootstrap CI shape; style gap; F1 edge cases (no positives) |
| Persona injection | Only `user_scenario.persona` changes; task facts and criteria untouched |
| Parity | Eval and train drivers give the same messages, reward and token counts |
| Ops | Run IDs, provenance immutability, resume checks, test ledger, endpoint handling |

When you fix a bug in rewards, rules or metrics, add the trajectory that exposed
it as a fixture. Windows CI runs tier 1: compare paths through the same
formatting code the library uses (backslashes get quoted in Hydra overrides).

---

## 13. Keeping docs in sync

| When you change… | Also update |
| --- | --- |
| Anything in `README.md` | `README.zh-CN.md` in the same commit (or a `TODO(zh-CN)` note at its top) |
| Commands, jobs, env vars or script flags | §9 here, `docs/cluster-runbook.md`, `slurm/README.md` |
| Config groups or presets | §8 here, `configs/README.md` |
| Experiments or scope | README experiment matrix + zh-CN, §10 here, runbook §7/§9, `docs/implementation-status.md` |
| A metric, split, trial count or simulator | ADR-005, `docs/metrics.md` (after M1: a new ADR) |
| Reward terms or λ | ADR-004, README problem formulation, §7 here |
| Labels, rules or format errors | `docs/labels.md`, §7 here |
| A shared contract (§7) | every caller, its tests, §7 here |
| Task completion | the README checkbox (only when the whole DoD is met), `docs/implementation-status.md` |
| Anything at session end | §2 status board and §17 decision log here |

The claude.ai Project "KnowWhenToHandOff" keeps copies (`claude/README.md`,
`claude/AGENTS.md`, `claude/HANDOFF.md`); agents with access refresh them after
significant changes.

---

## 14. Git, CI and delivery

- **Branch:** `main` is the only long-lived branch; it is what the cluster
  clones. Never force-push `main` or rewrite published history.
- **Commits:** small, one task step each, task-ID prefix:
  `T1.3: generate should-transfer variants for airline`. Body: what and why,
  assumptions, and a one-line reproduction command. Agents may add their
  co-author trailer.
- **Before pushing:** ruff check + format, pytest, `uv build`, shell checks
  green; after pushing, check CI (`gh run list --limit 3`; `gh run view <id>
  --log-failed` on failure) and fix it before moving on.
- **How pushes happen:** from the owner's clone on Windows with the owner's git
  credentials (commits authored `JaspinXu`). An agent working in a cloud sandbox
  without GitHub access hands over a git bundle (e.g. in
  `G:\KnowWhenToHandOff\_claude_transfer\`), fetches it into the owner's clone,
  fast-forwards `main` and pushes. Ask the owner before the first push of a
  session unless they already asked for it.
- **Line endings:** `.gitattributes` forces LF for `.py`, `.sh`, `.sbatch`,
  `.yaml`; the owner's git uses `core.autocrlf=true`. Shell scripts with CRLF
  break on the cluster.
- **Ignored:** `.env`, `slurm/cluster.env`, `logs/`, `runs/`, `wandb/`,
  `data/simulations/`, checkpoints, `reports/review/`.

---

## 15. Known pitfalls

Lessons already paid for; check here before debugging.

- **vLLM 0.12:** `--generation-config vllm` still stops on the model's
  `generation_config.json` EOS IDs (Qwen: `<|im_end|>`, `<|endoftext|>`), so
  training stops on the same set. There is no server-side default for
  `chat_template_kwargs`; send them per request (`extra_body`). `finish_reason`
  `tool_calls` hides a length cut (`hit_token_limit` checks completion tokens).
  A request with prompt + `max_tokens` > `max_model_len` is rejected. `--api-key`
  protects `/v1` only, hence the `handoff.sim.vllm_auth` middleware.
- **verl 0.8:** checkpoints hold a float32 FSDP state (~50 GB for 4B), so only
  the latest is kept and each saved step is exported at once to bf16 HF format
  (`handoff.train.export`, hooked into `RayPPOTrainer._save_checkpoint`).
  LoRA adapters are merged at export. `_save_checkpoint` runs in the TaskRunner
  process, which is why the hook is installed there.
- **Chat templates:** RL needs a prefix-stable template; Qwen3 hybrid-thinking
  templates are not (they drop earlier reasoning). `check-tokens` must PASS for
  any new policy or tokenizer.
- **tau2:** `run_tasks` with auto-resume re-runs infra failures and missing
  simulations in the same save path; tasks with `NL_ASSERTION` in their reward
  basis are refused; its data directory comes from `TAU2_DATA_DIR`.
- **User simulator outages:** the serve job has a time limit. Evaluations
  refuse a simulator that would end first (`ALLOW_SHORT_SIM=1` to override);
  RL episodes wait up to `user_sim.outage_timeout_s` (1,800 s) for a restarted
  simulator, then the job stops for `RESUME`.
- **Hydra overrides:** a value containing a comma must be quoted (otherwise
  Hydra reads a sweep); `rl_launch.format_value` quotes any value outside a safe
  character set (spaces, commas, Windows backslashes), so tests compare through it.
- **SFT:** LoRA with FSDP is refused (use one GPU for LoRA).
- **Scratch:** must be shared across nodes (the simulator publishes its address
  there); the setup job refuses node-local filesystems.
- **Tool fetching restrictions** (agents): if a web page cannot be fetched, do
  not work around it; ask the owner.

---

## 16. Session protocol for switching agents

**At the start of a session**

1. Read this file completely, then the parts of README and the runbook that
   the task touches.
2. `git status`, `git log --oneline -10`, `git pull --ff-only` (or read the
   latest `main`) and compare with the §2 status board.
3. Check CI on the latest commit; if it is red, fixing it comes first.
4. Restate the task, the task ID and any assumption in your first reply.

**At the end of a session** (before your final message)

1. Run the quality gates (§12, §14) and report what ran and what could not.
2. Commit with task-ID messages; push only as agreed with the owner (§14).
3. **Update this file:** the §2 status board (date, commit, test counts, what
   is done, next steps, risks), §17 (new owner decisions), and any section your
   change made stale (§5–§15). Keep it comprehensive — the next agent may be a
   different product with no memory of this session.
4. Update the other docs listed in §13 and the claude.ai Project copies if you
   can.
5. Tell the owner in Chinese what changed, what is verified, and what remains.

---

## 17. Decision log

Owner decisions and scope changes, newest first. Scientific definitions live in
the ADRs; this is the short history agents need.

| Date | Decision |
| --- | --- |
| 30 Sep 2026 | Owner asked to audit what the project does, complete any missing code, then optimize the code. Result: only the RL learning curves were missing (added `rl_curves.py`); the optimization pass removed duplication and fixed one bug, without changing contracts, metrics or experiments. |
| 29 Sep 2026 | Owner switches between Claude and Codex; AGENTS.md must be kept comprehensive and updated every session (§16). `CLAUDE.md` imports this file. |
| 29 Sep 2026 | Course-sized scope: runs B0, B1 (optional), B2, R1, R2 with one seed; removed λ grid, R3 fairness sampler, P1 PPO, A1/A2 ablations, held-out simulator, the 1.7B/1.5B smoke models, the LLM citation judge, reward shaping, and the pre-pipeline tools (server planner, completion parser, offline SFT export). Human audit 50 → 20 dialogues. |
| 29 Sep 2026 | Complete the whole pipeline so it can be cloned and run on the cluster with Slurm; never download datasets or models on the laptop. ADR-001 accepted; ADR-002 to ADR-005 proposed for M1. The deferred ADR-006/007 points were settled by ADR-001 to ADR-005. |
| 29 Sep 2026 | Hardware: one A100 80 GB for the simulator, one H200 for everything else. |
| 27 Sep 2026 | Implement and CPU-test code before cluster access; independent infrastructure first. |
| 27 Sep 2026 | Top-level design (README, README.zh-CN, AGENTS) pushed; proposal due 28 Sep. |

---

## 18. When unsure

- State assumptions in the commit body or a short ADR draft rather than guessing
  silently.
- Ask the owner before you change a shared contract, add a dependency, touch
  evaluation or splits, delete data or runs, add experiments, or widen a task.
- The owner communicates in Simplified Chinese; reply in Chinese, write code and
  docs in English. Do not put the owner's or teammates' personal details
  (student numbers, emails) in the repo.
- Prefer the simplest thing that meets the definition of done.

**Definition of done (any task)**

- [ ] Code implements the task's contract and nothing outside it.
- [ ] Tests cover the new logic and pass on CPU; gates are green.
- [ ] Config entries exist; no hard-coded values.
- [ ] Docs updated per §13, including this file's §2 and §17.
- [ ] A one-line reproduction command is in the commit body.
- [ ] If the task produced numbers: an experiment-log row with the run ID.

---

## 19. Glossary

| Term | Meaning |
| --- | --- |
| B0 / B1 / B2 | Prompted 4B model / prompted 30B reference / SFT warm start |
| R1 / R2 | GRPO with task reward / + Responsible-AI terms (one fixed λ setting) |
| SMOKE / TEACH / SPIKE | Stack check / teacher rollouts for SFT data / 3-step RL timing run |
| Should-transfer label (y) | Whether `transfer_to_human_agents` is the correct action in an episode |
| Kinds | `upstream` (official tasks), `explicit_request`, `out_of_scope` (y = 1), `hard_negative` (y = 0, a difficult user whose issue is solvable) |
| Style card | Text injected into `user_scenario.persona` that sets a communication style, never task facts |
| FP / FN | Over-escalation (transfer when y = 0) / under-escalation (no transfer when y = 1) |
| pass^k | Per task C(c, k) / C(n, k) with c successes in n trials, averaged over tasks |
| Style gap | Max − min success across the six styles; also worst-style success |
| Parity test | Proof that the eval and train drivers compute identical results |
| Test ledger | Scratch file that enforces one test evaluation per checkpoint and simulator |
| Complete run | An evaluation with no lost episodes (or an explicitly accepted share) |
| Finishing point | A milestone (M1, M2) whose result stands on its own if later stages fail |
| Task IDs | T0.x–T4.x in the README execution plan; commit prefixes |
