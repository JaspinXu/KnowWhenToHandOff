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

**Pipeline:** baseline audit (B0, B1) → imitation warm start (B2) → GRPO or PPO with Responsible-AI rewards (R1, R2, R3, P1).

**Status:** foundation implementation started; experiments remain pending. Follow task-ID order for implementation. Owner clarification (27 Sep 2026): implement and CPU-test code before cluster access; GPU-dependent acceptance remains pending and does not block independent code work. Scientific definition changes still require approval.

---

## 2. Hard rules

These are non-negotiable. If a task seems to require breaking one, stop and ask.

1. **The test split is sacred.**
    - Never train on, build SFT data from, tune hyper-parameters on, or select checkpoints with test tasks. This covers both the official test split and the hand-off test set.
    - Use the dev split (carved from train) for every selection decision.
2. **Do not edit upstream τ²-bench.** Wrap it with adapters in `src/handoff/`. The τ²-bench version is pinned in `pyproject.toml`, and upgrading it needs an ADR.
3. **Training rewards are deterministic code.** Rewards come from the τ²-bench evaluator, rule checks and labels. LLM judges may appear only in evaluation metrics, never in a training reward.
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
| `configs/` | Hydra configs: `env/`, `model/`, `reward/`, `train/`, `eval/`, `slurm/` | T0.1+ |
| `src/handoff/envs/` | Task registry, split manifests, persona injection, hand-off task generator, should-transfer labels | T1.1–T1.4 |
| `src/handoff/ops/` | Run provenance and configuration integrity utilities | T0.3 supporting infrastructure |
| `src/handoff/sim/` | User-simulator clients, serving helpers | T0.3, T0.5 |
| `src/handoff/agents/` | Prompted baseline agent and policy client registered with the τ²-bench runner | T0.5, T1.6 |
| `src/handoff/rollout/` | `Trajectory` schema, eval driver, verl `Tau2AgentLoop` | T1.5, T3.1 |
| `src/handoff/rewards/` | Task reward adapter, violation rules, hand-off term, format term, `RewardComposer` | T3.2 |
| `src/handoff/train/` | SFT data builder, RL launchers, fairness sampler | T2.1–T3.7 |
| `src/handoff/eval/` | Evaluation runner, metrics, bootstrap CIs, report tables | T1.5 |
| `src/handoff/demo/` | Small chat UI showing hand-offs and policy citations | T4.3 |
| `scripts/` | Thin CLI entry points only; no business logic | as needed |
| `slurm/` | `sbatch` templates for the policy server, user-simulator server, training and evaluation | T0.3 |
| `data/manifests/` | Frozen split and hand-off manifests with hashes | T1.1, T1.3 |
| `docs/adr/` | Architecture decision records (`ADR-00N-title.md`) | T0.4+ |
| `docs/metrics.md`, `docs/labels.md` | Frozen metric and label definitions | T1.2, T1.5 |
| `docs/experiment-log.md` | One line per run: ID, purpose, result, decision | T1.6+ |
| `tests/` | CPU-only unit, integration and parity tests | every task |
| `reports/` | Generated tables and figures (final versions committed) | T4.1 |

---

## 4. Commands

> Setup, CPU quality and offline planning commands below work locally. Serving, evaluation and training remain planned unless explicitly marked implemented. Remote CI and GPU experiments require their own evidence.

```bash
# setup (Python 3.12, uv)
uv sync --locked
cp .env.example .env                      # fill in keys locally; never commit

# quality gates — run before every commit
uv run ruff check . && uv run ruff format --check .
uv run pytest -q                          # CPU only, sockets disabled
uv build                                 # build sdist and wheel

# offline server planning                 (implemented; no job submission)
uv run python scripts/plan_server.py --help
# See slurm/README.md for required configuration and output bundles.

# persona injection invariance            (implemented; synthetic tasks only)
uv run --locked pytest -q tests/test_personas.py

# model-response parsing                  (implemented; no model requests)
uv run --locked pytest -q tests/test_completions.py

# assistant-only SFT batch preparation     (implemented; no optimization)
uv run --locked pytest -q tests/test_sft_batch.py tests/test_tokens.py

# serving on the cluster                  (planned, T0.3)
sbatch slurm/serve_policy.sbatch   model=qwen3_4b
sbatch slurm/serve_usersim.sbatch  model=qwen3_30b_a3b

# evaluation                              (planned, T1.5–T1.6)
uv run python scripts/run_eval.py eval=smoke model=qwen3_1p7b               # 5 tasks, 1 trial
uv run python scripts/run_eval.py eval=test  model=b0 personas=all trials=4

# training                                (planned, T2.2 / T3.3+)
sbatch slurm/train.sbatch train=sft_b2
sbatch slurm/train.sbatch train=grpo_r1
sbatch slurm/train.sbatch train=grpo_r2 reward.lambda_fp=0.5 reward.lambda_fn=1.0
```

PowerShell setup uses `Copy-Item .env.example .env`. Dependency installation may use the network; tests must not.

Always run the **smoke** configuration before launching a full run.

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

- **Run naming** follows the README experiment matrix: `B0`, `B1`, `B2`, `R1`, `R2-<λ tag>`, `R3`, `P1`, `A1`, `A2`, plus `-s<seed>` and a date, e.g. `R2-fp05-s1-20261024`.
- **Run records.** Every run writes its resolved config, git SHA, τ²-bench commit, model revisions, seeds and hardware to `runs/<run_id>/` (gitignored) and to the tracker (W&B or SwanLab).
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
- Kill idle model servers when a sweep finishes.

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
| R1 / R2 / R3 | GRPO with task reward / + Responsible-AI terms / + fairness sampler |
| P1, A1, A2 | PPO comparison; ablation with naive per-turn shaping; ablation without the over-escalation (FP) term |
| Should-transfer label | Ground truth for whether `transfer_to_human_agents` is the correct action in an episode |
| Hard negative | A difficult user (angry, impatient) whose issue *is* solvable; transferring is wrong |
| Style card | Text injected into `user_scenario.persona` that sets a communication style, never task facts |
| Parity test | Proof that the eval driver and the train driver compute identical rewards |
| Finishing point | A milestone (M1, M2) whose result is complete on its own if later stages fail |
