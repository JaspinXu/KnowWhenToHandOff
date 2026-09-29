# AGENTS.md

Working agreement for AI coding agents (Codex, Claude Code, Copilot, Gemini, …) and humans contributing to **KnowWhenToHandOff**.

The design lives in [README.md](README.md). This file tells you how to work inside that design. If something here conflicts with a direct instruction from the repository owner, follow the owner and flag the conflict.

---

## 1. The project in one paragraph

We train a ~4B open model (Qwen3 family) with multi-turn reinforcement learning on **τ²-bench** (airline and retail) to act as a customer-service agent. It must:

- complete the task (τ²-bench database-state reward),
- follow the domain policy (no violations),
- call `transfer_to_human_agents` only when the user asks for a human or the policy and tools cannot solve the issue,
- serve users with different communication styles equally well,
- cite the policy behind refusals and hand-offs.

**Pipeline:** baseline audit (B0, B1) → imitation warm start (B2) → GRPO with and without Responsible-AI rewards (R1, R2). It is a course project: keep code and experiments small (no sweeps or ablations).

**Status (29 Sep 2026):** the pipeline is code-complete and CPU-verified; experiments remain pending and run on the cluster through `slurm/submit.sh` ([cluster runbook](docs/cluster-runbook.md)). Owner decisions: implement and CPU-test code before cluster access (27 Sep); complete the whole pipeline without downloading data or models on the laptop (29 Sep). ADR-001 is accepted; ADR-002 to ADR-005 are proposed for M1. Scientific definition changes still require approval.

---

## 2. Hard rules

These are non-negotiable. If a task seems to require breaking one, stop and ask.

1. **The test split is sacred.**
    - Never train on, build SFT data from, tune hyper-parameters on, or select checkpoints with test tasks. This covers both the official test split and the hand-off test set.
    - Use the dev split (carved from train) for every selection decision.
2. **Do not edit upstream τ²-bench.** Wrap it with adapters in `src/handoff/`. The τ²-bench version is pinned (commit `fc0055d`, v1.0.1: `TAU2_COMMIT` in `slurm/cluster.env.example`, recorded in the split manifest and ADR-005), and upgrading it needs an ADR.
3. **Training rewards are deterministic code.** Rewards come from the τ²-bench evaluator, rule checks and labels. No LLM judge is used, in training or in evaluation.
4. **The evaluation protocol is frozen after milestone M1 (11 Oct 2026).** Changing a metric definition, split, trial count or simulator for reported numbers requires a new ADR in `docs/adr/`.
5. **No secrets in the repo.**
    - API keys and tokens live in `.env`, which is gitignored.
    - Never print keys in logs or command output.
    - Never commit `.env`, `wandb/` or credentials.
6. **No large artefacts in git.** Checkpoints, trajectories, Hugging Face caches and `data/simulations/` stay on cluster scratch. Git holds code, configs, small manifests and final report assets only.
7. **No result without provenance.** Every number in the README, the report or the slides must trace back to a run ID with its config, git SHA and seeds.
8. **Stay inside the task.**
    - Implement only the task ID you were asked for.
    - If you must change a shared contract (the `Trajectory` schema, `RewardBreakdown`, the `TaskRegistry` API), propose it first in an ADR draft or in the commit body, and update every caller in the same change.
9. **No real user data.** All users are simulated. Persona style cards describe communication style only. They must not encode demographic stereotypes.

---

## 3. Repo map

The package name is `handoff` (src layout). Directories are created by the tasks noted in brackets.

| Path | Owns | Created by |
| --- | --- | --- |
| `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/` | Tooling, pinned deps, CI | T0.1 |
| `requirements/` | Cluster GPU lock (`cluster-gpu.txt`) and its inputs; regenerate with `scripts/lock_cluster_env.sh` | T0.3, ADR-001 |
| `configs/` | Hydra configs: `eval/`, `reward/`, `rl/`, `model/`, `persona/`, `handoff_tasks/`, `paths/` + one entry config per script | T0.1+ |
| `src/handoff/envs/` | Task registry, split manifests, persona injection, hand-off task generator, should-transfer labels | T1.1–T1.4 |
| `src/handoff/ops/` | Run provenance and configuration integrity utilities | T0.3 supporting infrastructure |
| `src/handoff/sim/` | User-simulator clients, serving helpers | T0.3, T0.5 |
| `src/handoff/agents/` | Prompted baseline agent and policy client registered with the τ²-bench runner | T0.5, T1.6 |
| `src/handoff/rollout/` | `Trajectory` schema, eval driver, verl `Tau2AgentLoop` | T1.5, T3.1 |
| `src/handoff/rewards/` | Task reward adapter, violation rules, hand-off term, format term, `RewardComposer` | T3.2 |
| `src/handoff/train/` | SFT data builder and trainer, RL launcher, checkpoint export, monitors | T2.1–T3.5 |
| `src/handoff/eval/` | Evaluation runner, metrics, bootstrap CIs, report tables | T1.5 |
| `src/handoff/demo/` | Small chat UI showing hand-offs and policy citations | T4.3 |
| `scripts/` | Thin CLI entry points only; no business logic | as needed |
| `slurm/` | `submit.sh`, `cluster.env.example`, one job per step in `jobs/` | T0.3 |
| `data/manifests/` | Frozen split and hand-off manifests with hashes | T1.1, T1.3 |
| `docs/adr/` | Architecture decision records (`ADR-00N-title.md`) | T0.4+ |
| `docs/metrics.md`, `docs/labels.md`, `docs/handoff-tasks.md` | Frozen metric, label, rule and hand-off task definitions | T1.2, T1.3, T1.5 |
| `docs/cluster-runbook.md` | The experiment sequence on the cluster, command by command | T0.3+ |
| `docs/experiment-log.md` | One line per run: ID, purpose, result, decision | T1.6+ |
| `tests/` | CPU-only unit, integration and parity tests | every task |
| `reports/` | Generated tables and figures (final versions committed) | T4.1 |

---

## 4. Commands

> Laptop commands run on CPU with no network in tests. Everything that needs GPUs, models or τ²-bench data runs on the cluster through `slurm/submit.sh`; the full sequence is in [docs/cluster-runbook.md](docs/cluster-runbook.md).

```bash
# setup (Python 3.12, uv) — laptop / CI
uv sync --locked
cp .env.example .env                      # fill in keys locally; never commit

# quality gates — run before every commit
uv run ruff check . && uv run ruff format --check .
uv run pytest -q                          # CPU only, sockets disabled; tau2/vLLM/verl tests skip
uv build                                 # build sdist and wheel
for f in slurm/*.sh slurm/lib/common.sh slurm/jobs/*.sbatch; do bash -n "$f" || exit 1; done
shellcheck -x -P slurm slurm/*.sh slurm/lib/common.sh slurm/jobs/*.sbatch   # if installed

# cluster: one-time
cp slurm/cluster.env.example slurm/cluster.env
bash slurm/submit.sh setup                # venv, pinned stack, tau2 v1.0.1, full test suite
bash slurm/submit.sh download MODELS="qwen3_4b qwen3_30b_a3b"
bash slurm/submit.sh tasks                # labels, splits, hand-off tasks (then commit data/)
bash slurm/submit.sh check-tokens MODEL=qwen3_4b

# cluster: evaluation (the simulator server must be running)
bash slurm/submit.sh serve                # ROLE=usersim MODEL=qwen3_30b_a3b
bash slurm/submit.sh eval EVAL=smoke EXPERIMENT=SMOKE POLICY_MODEL=qwen3_4b
bash slurm/submit.sh eval EVAL=dev EXPERIMENT=B0 POLICY_MODEL=qwen3_4b
bash slurm/submit.sh eval EVAL=test_main EXPERIMENT=B0 POLICY_MODEL=qwen3_4b   # once per checkpoint
bash slurm/submit.sh eval RESUME=<run id> POLICY_MODEL=qwen3_4b    # after exit 3 (lost episodes)

# cluster: training
bash slurm/submit.sh teacher POLICY_ENDPOINT=$HANDOFF_SCRATCH/state/endpoints/usersim.json POLICY_LABEL=qwen3_30b_a3b
bash slurm/submit.sh sft-data TEACHER_RUNS=<TEACH run id> DATASET=b2_v1
bash slurm/submit.sh sft DATASET=b2_v1
bash slurm/submit.sh rl REWARD=task_only EXPERIMENT=R1 INIT=<B2 checkpoint>
bash slurm/submit.sh rl REWARD=r2 EXPERIMENT=R2 INIT=<B2 checkpoint>
bash slurm/submit.sh rl RESUME=<RL run id>                             # time limit, simulator outage
bash slurm/run.sh python scripts/monitor_rl.py --run <RL run id> --baseline <B2 dev run id>   # exit 3: stop the run

# results (login node, inside the environment)
bash slurm/run.sh python scripts/append_experiment_log.py <run id> --purpose "..." --decision "..."
bash slurm/run.sh python scripts/make_report.py --name main --baseline B2 \
  --runs B0=<test_main id>+<test_styles id>+<test_handoff id> B2=<...> R2=<...>
```

PowerShell setup uses `Copy-Item .env.example .env`. Dependency installation may use the network; tests must not.

Always run the **smoke** configuration before launching a full run, and a real run only from a clean, committed checkout (provenance refuses dirty code, configs or data).

---

## 5. Coding conventions

- **Language:** Python 3.12 with full type hints. Use `dataclass` or `pydantic` models for schemas (`Trajectory`, `RewardBreakdown`, `TaskRef`, `Persona`).
- **Pure functions** for labels, rules, rewards and metrics: input in, value out, no I/O, no global state.
- **Configuration through Hydra.** No hard-coded paths, hostnames, ports, model names or λ values in library code.
- **Seeds everywhere.** Pass seeds explicitly to samplers, simulators and trainers, and record them.
- **Logging:** use `logging` in library code (`print` only in `scripts/`). Log every reward term separately, never only the total.
- **Async rollouts.** The verl agent loop is async; use async HTTP clients and never block the event loop.
- **Token handling in training.**
    - Keep the token ids returned by generation and accumulate them turn by turn. Do not rebuild them by re-tokenising the final messages.
    - Set `response_mask = 1` for policy tokens and `0` for user and tool tokens.
- **Readable code.** English identifiers, comments and docstrings; small functions; docstrings on every public function; no dead code or commented-out blocks.
- **Dependencies.** Add a new heavy dependency (a new trainer, a new serving stack) only through an ADR.

---

## 6. Testing

Tests run on CPU with no GPU and no network. LLM calls are replaced by recorded fixtures or fakes.

Three tiers, one suite: pure unit tests run everywhere; tests that need tau2 (`test_tau2_integration.py`, `test_cli_pipeline.py`) run a toy tau2 domain with scripted models; tests that need vLLM, verl or PyTorch (`test_gpu_stack.py`, `test_hf_cpu.py`) run on the cluster environment on CPU. Each tier skips itself when its packages are missing; the setup job runs all of them.

**Required coverage:**

| Area | What the test must pin down |
| --- | --- |
| Splits | Train, dev and test (official and hand-off) are disjoint; manifest hashes match |
| Labels | Should-transfer rules on hand-written examples, including the upstream tasks with gold transfers |
| Violation rules | Every rule has a positive and a negative example trajectory |
| Reward composer | Each term in isolation; λ = 0 reduces to the task reward; FP and FN terms trigger correctly |
| Metrics | pass^k estimator against brute force; bootstrap CI shape; style gap; hand-off F1 edge cases (no positives) |
| Persona injection | Only `user_scenario.persona` changes; task facts and evaluation criteria are untouched |
| Parity | The eval driver and the train driver give the same reward on the same recorded trajectory |

When you fix a bug in rewards or metrics, add the trajectory that exposed it as a fixture.

---

## 7. Experiments

- **Run naming** follows the README experiment matrix: `B0`, `B1`, `B2`, `R1`, `R2` (plus `SMOKE`, `TEACH`, `SPIKE`), optional tags, `-s<seed>` and a date, e.g. `R2-s1-20261024`.
- **Run records.** Every run writes its resolved config, git SHA, τ²-bench commit, model revisions, seeds and hardware to `runs/<run_id>/` on cluster scratch (never in git).
- **Experiment log.** After every run that produced numbers, append one line to `docs/experiment-log.md`: run ID, purpose, headline metrics with CIs, decision.
- **Model selection** uses dev only. Evaluate test once per frozen checkpoint and record it as such.
- **Reward-hacking watch-list.** Monitor these on every RL run and stop the run if one drifts more than 2× from B2:
    - transfer rate on should-not-transfer episodes,
    - mean episode length,
    - format-error rate,
    - share of episodes that end on a user "stop" without a completed task.

---

## 8. Cluster etiquette

- Never run GPU work on a login node; use `sbatch` or `srun`.
- Request only the GPUs and time you need, and set `--time`.
- Checkpoints, caches and trajectories go to scratch. Point `HF_HOME` there.
- Kill idle model servers when a campaign finishes.

---

## 9. Git workflow

- **Small commits.** One task step per commit, with the task ID as prefix: `T1.3: generate should-transfer variants for airline`.
- **Never** force-push `main` or rewrite published history.
- **Keep the docs in step with the code.**
    - Tick the task's checkbox in the README when its definition of done is met.
    - Update the Commands section of this file when commands change.
    - Mirror README changes in `README.zh-CN.md` in the same commit. If that is not possible, add a `TODO(zh-CN)` note at the top of the Chinese file.
- **Pre-commit gates** (ruff and pytest) must pass before pushing.

---

## 10. Definition of done (any task)

A task is done when all of these hold:

- [ ] Code implements the task's contract, and nothing outside it.
- [ ] Tests cover the new logic and pass on CPU.
- [ ] Config entries exist; there are no hard-coded values.
- [ ] Docs are updated (README checkbox, this file's Commands section, ADR if a decision was made).
- [ ] A one-line reproduction command is in the commit body.
- [ ] If the task produced numbers: an experiment-log entry with the run ID.

---

## 11. When unsure

- **State your assumptions** in the commit body or in a short ADR draft rather than guessing silently.
- **Ask before you:**
    - change a shared schema,
    - add a dependency,
    - touch evaluation or splits,
    - delete data or runs,
    - widen the scope of a task.
- **Prefer the simplest thing** that meets the definition of done. This is a one-person project with a hard deadline (experiment freeze on 1 Nov 2026, report on 15 Nov 2026).

---

## Glossary

| Term | Meaning |
| --- | --- |
| B0 / B1 / B2 | Prompted 4B model / prompted larger model / SFT warm start |
| R1 / R2 | GRPO with task reward / + Responsible-AI terms (one fixed λ setting) |
| Should-transfer label | Ground truth for whether `transfer_to_human_agents` is the correct action in an episode |
| Hard negative | A difficult user (angry, impatient) whose issue *is* solvable; transferring is wrong |
| Style card | Text injected into `user_scenario.persona` that sets a communication style, never task facts |
| Parity test | Proof that the eval driver and the train driver compute identical rewards |
| Finishing point | A milestone (M1, M2) whose result is complete on its own if later stages fail |
