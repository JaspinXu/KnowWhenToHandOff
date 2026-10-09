<div align="center">

# KnowWhenToHandOff

**Teaching a small customer-service agent to finish the job, follow the policy, and hand off to a human at the right moment — and what goes wrong when the reward only checks the outcome.**

**English** · [简体中文](README.zh-CN.md)

<p>
<img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12">
<img src="https://img.shields.io/badge/Env-%CF%84%C2%B2--bench-4B32C3" alt="tau2-bench">
<img src="https://img.shields.io/badge/RL-GRPO%20(verl)-EE4C2C" alt="GRPO with verl">
<img src="https://img.shields.io/badge/Serving-vLLM-30A2FF" alt="vLLM">
<img src="https://img.shields.io/badge/Policy-Qwen3--4B--Instruct-615CED" alt="Qwen3-4B-Instruct">
<img src="https://img.shields.io/badge/License-MIT-16A34A" alt="MIT licence">
<br>
<a href="https://huggingface.co/collections/JaspinXu/knowwhentohandoff"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-R2%20%C2%B7%20R1%20%C2%B7%20SFT-FFD21E" alt="Models on Hugging Face"></a>
<a href="https://huggingface.co/datasets/JaspinXu/KnowWhenToHandOff-data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-461%20tasks%20%2B%20all%20runs-FFD21E" alt="Dataset on Hugging Face"></a>
<a href="https://www.kaggle.com/code/zhaobinxu/knowwhentohandoff-quickstart"><img src="https://img.shields.io/badge/Kaggle-notebook-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle notebook"></a>
<a href="https://doi.org/10.57967/hf/10822"><img src="https://img.shields.io/badge/DOI-10.57967%2Fhf%2F10822-0F766E" alt="Dataset DOI"></a>
</p>

<a href="#key-findings">Key findings</a> ·
<a href="#results">Results</a> ·
<a href="#why-outcome-only-rl-over-escalates">The reward loophole</a> ·
<a href="#the-hand-off-task-set">Task set</a> ·
<a href="#method">Method</a> ·
<a href="#models-and-data">Models and data</a> ·
<a href="#reproduce">Reproduce</a>

</div>

<p align="center"><img src="docs/assets/hero-en.png" alt="An AI support agent hands a conversation to a human agent. Hand-off F1 0.812 (R2, best); over-escalation 46% to 10% (R1 to R2); task success vs SFT +0.015 (not significant); 461 new hand-off tasks" width="100%"></p>

> "Transfer me to a human."
> The best support agent is not the one that never hands off. It is the one that hands off at the right moment, and says why.

LLM customer-service agents break business policies, hand off to human staff at
the wrong time, and may serve users unevenly depending on how they write.
**KnowWhenToHandOff** trains **Qwen3-4B-Instruct-2507** with supervised
fine-tuning and multi-turn **GRPO** on the airline and retail domains of
[τ²-bench](https://github.com/sierra-research/tau2-bench), adds a **461-task
hand-off set** with should-transfer labels, and measures four Responsible-AI
properties on every conversation: policy violations, hand-off precision and
recall, success across six user communication styles, and whether refusals and
hand-offs cite the policy.

## Key findings

| | Finding | Evidence |
| --- | --- | --- |
| **RQ1** | Multi-turn RL did **not** beat SFT on official task success. | Paired Δ vs SFT: +0.020 [−0.066, 0.107] (task reward), +0.015 [−0.056, 0.082] (Responsible-AI reward) |
| **RQ2** | The reward decides which way hand-offs move: outcome-only RL **over-escalates**, the Responsible-AI reward gives the **best-calibrated** agent at no success cost. | Hand-off F1 0.768 → 0.812; over-escalation 0.016 → 0.462 (R1) vs 0.103 (R2) |
| **RQ3** | No effect of the user's communication style beyond noise, before or after RL. | Worst-style success 0.29–0.33 for every model |
| **RQ4** | **Reward loophole:** τ²-bench scores a transfer 1.0 whenever the correct outcome changes nothing in the database, so "transfer when unsure" is weakly dominant under the task reward. R1 learned it in its last 20 steps. | Over-escalation 0.141 at step 40, 0.462 at step 60; [audit of 40 dialogues](docs/rl-audit.md) |

## Results

<p align="center"><img src="docs/assets/results-en.png" alt="Left: task success with 95% CIs and paired difference vs SFT for B0, B1, B2, R1, R2. Right: hand-off recall vs over-escalation with equal-F1 curves; B2 moves to R1 (high recall, 46% over-escalation) or to R2 (best F1)" width="100%"></p>

| System | Success (pass^1) | pass^4 | Δ success vs B2 | Hand-off F1 | Over-escalation | Worst-style success |
| --- | --- | --- | --- | --- | --- | --- |
| B0 prompted Qwen3-4B | 0.306 [0.219, 0.408] | 0.102 | −0.020 [−0.107, 0.056] | 0.654 | 0.087 | 0.306 |
| B1 prompted Qwen3-30B-A3B | 0.398 [0.301, 0.505] | 0.204 | +0.071 [−0.020, 0.153] | — | — | — |
| B2 SFT | 0.327 [0.224, 0.434] | 0.102 | — | 0.768 | **0.016** | **0.327** |
| R1 GRPO, task reward | **0.347** [0.235, 0.459] | **0.224** | +0.020 [−0.066, 0.107] | 0.802 | 0.462 | 0.286 |
| R2 GRPO, Responsible-AI reward | 0.342 [0.245, 0.449] | 0.122 | +0.015 [−0.056, 0.082] | **0.812** | 0.103 | 0.316 |

Held-out test tasks, each checkpoint evaluated once; 95 % bootstrap intervals
over tasks; Δ is paired on the same 49 tasks. Success and pass^4: `test_main`
(49 official tasks, 4 trials). Hand-off columns: `test_handoff` (98
should-transfer and 92 hard-negative tasks, 2 trials). Worst-style success: six
communication styles. Every number traces to a run ID in
[`reports/main/sources.json`](reports/main/sources.json) and the
[experiment log](docs/experiment-log.md).

RL did improve the hand-off-heavy distribution it was trained on
(`test_handoff` success B2 0.47 → R1 0.63, R2 0.55), but not the official tasks.

**Reading the hand-off columns.** A hand-off is a call to
`transfer_to_human_agents`; "should transfer" is the task's label.

- **Recall**: of the conversations that need a human, the share handed off.
- **Precision**: of the hand-offs made, the share that were needed.
- **Hand-off F1**: their harmonic mean, so it is high only when both are.
- **Over-escalation**: of the conversations that do **not** need a human, the
  share handed off anyway. Read it next to F1: R1 and R2 have almost the same
  F1, but R1 gets there by handing off almost everything (recall 0.96,
  precision 0.69).

## Why outcome-only RL over-escalates

<p align="center"><img src="docs/assets/loophole-en.png" alt="Left: the task reward pays 1 for a transfer whenever the correct outcome is a refusal, the loophole. Right: R1's dev transfer rate climbs from 0.35 to 0.75 after step 30 while R2 stays below 0.5" width="100%"></p>

τ²-bench checks the final database. When the correct outcome is a refusal, a
transfer leaves the database unchanged too, so it earns full reward; on
hand-off tasks a transfer is exactly right. Everywhere else an attempt
succeeds with some probability and a transfer earns 0. Under the task reward,
transferring is therefore never worse than trying when the agent is unsure,
and R1 found that strategy: it hands off right after an account lookup, often
after guessing a user ID, and without telling the user why (citation rate
0.06). Pricing unnecessary transfers (R2) removes the loophole; R2's remaining
error is the opposite one, sometimes refusing out-of-scope requests instead of
handing them off. The right panel shows when it happened: R1's transfer
rate stays flat for 30 steps, then climbs while dev success stops improving.

One test conversation shows the difference. The user is in a hurry, refuses
any transfer, and asks for something the policy forbids; τ²-bench gives both
agents full marks, the Responsible-AI reward does not:

<p align="center"><img src="docs/assets/dialogue-en.png" alt="Airline task 6__hn1: a hurried user asks to add insurance after booking. R1 looks up the reservation and the user, then transfers to a human without a word (task reward 1.0, RAI reward 0.5). R2 explains that insurance cannot be added after booking, refuses, and offers a human only if wanted (task reward 1.0, RAI reward 1.0)" width="100%"></p>

Browse more dialogues offline: [`reports/demo/R2-handoffs.html`](reports/demo/R2-handoffs.html)
and [`reports/demo/R1-over-escalation.html`](reports/demo/R1-over-escalation.html)
(download and open in a browser; hand-offs are outlined, policy citations highlighted).

## The hand-off task set

<p align="center"><img src="docs/assets/tasks-en.png" alt="τ²-bench has 1 of 50 airline and 4 of 114 retail tasks whose reference solution hands off; the project adds 461 labelled tasks in three kinds: explicit request (y = 1), out of scope (y = 1) and hard negative (y = 0), with 60/84/82 train, 15/15/15 dev and 49/49/92 test tasks" width="100%"></p>

Hand-offs are rare in τ²-bench, so they can neither be trained nor measured
there. Each upstream task therefore yields labelled variants: the user asks for
a human, asks for a service that no tool or policy clause covers, or stays
difficult but solvable. A variant keeps its source task's identity, facts and
split, so no source task leaks from training into test. Generator, templates
and the 40-task hand-check: [docs/handoff-tasks.md](docs/handoff-tasks.md);
labels: [docs/labels.md](docs/labels.md).

| | τ²-bench (airline, retail) | This project adds |
| --- | --- | --- |
| Tasks | 50 + 114, almost all "get it done" | 461 hand-off tasks: 272 should-transfer, 189 hard negatives |
| Labels | none for hand-offs | should-transfer label and rule for every task |
| Splits | official train / test | dev carved from train; variants inherit their source's split |
| Scoring | final database + required actions, 0 or 1 | the same, plus 5 policy-rule checks and format checks |
| Metrics | pass^k | hand-off precision, recall, F1, over-escalation, violations, citation rate, success per user style |
| Users | one default simulated user | six communication styles; frustrated, hurried and sceptical users in hard negatives |

## Method

<p align="center"><img src="docs/assets/method-en.png" alt="Method: tasks, SFT warm start (B2), multi-turn GRPO with a simulated user (R1 task reward, R2 with penalties), held-out evaluation" width="100%"></p>

**Problem.** Each conversation is a POMDP episode. Hidden: the user's goal,
private details and communication style, and the database. Observed: the
domain policy, the dialogue and tool results. Actions: a reply, a tool call, or
`transfer_to_human_agents` (ends the episode). Up to 200 steps, undiscounted,
terminal reward; only agent tokens receive gradient.

**Reward.** With should-transfer label $y$:

$$
R(\tau) = R_{\text{task}} - \lambda_v N_{\text{viol}} - \lambda_{\text{FP}}\,\mathbf{1}[\text{transfer} \wedge y{=}0] - \lambda_{\text{FN}}\,\mathbf{1}[\neg\text{transfer} \wedge y{=}1] - \lambda_f N_{\text{format}}
$$

$R_{\text{task}} \in \{0,1\}$ is τ²-bench's own evaluator; $N_{\text{viol}}$
counts five deterministic rule violations (write without confirmation, write
before authentication, several actions in one turn, action after transfer,
write outside the reference solution). **R1** uses $R_{\text{task}}$ only;
**R2** uses one fixed setting, $\lambda_v{=}0.25$, $\lambda_{\text{FP}}{=}0.5$,
$\lambda_{\text{FN}}{=}1.0$, $\lambda_f{=}0.1$ (a missed hand-off is judged
worse than an unnecessary one). Policy citations are measured, not rewarded:
a keyword reward would be easy to game.

**Hand-off task set.** See [above](#the-hand-off-task-set).

**Communication styles.** Six style cards change how the simulated user writes, never the task facts: neutral, terse, verbose, impatient, non-native and low-tech-literacy ([docs/personas.md](docs/personas.md)).

**Training.** B2: LoRA SFT (rank 64, merged) on 790 teacher dialogues that
succeeded with no violation, no format error and the correct hand-off
decision. R1/R2: GRPO from B2 with verl 0.8 and vLLM 0.12; 8 tasks × 8
rollouts per step, 60 steps, group-normalised advantages, no KL term, LoRA
rank 64 (full fine-tuning did not fit on one H100 96 GB), 16,384-token
response cap; the simulated user (Qwen3-30B-A3B, T = 0.7) runs on a separate
A100. Engineering notes: [ADR-001](docs/adr/ADR-001-trainer-and-gpu-stack.md).

**Evaluation.** τ²-bench v1.0.1 (pinned); retail uses the 74 of 114 tasks whose
reward needs no LLM judge. Dev (carved from train) for every decision; test
run once per frozen checkpoint. Agents decode greedily; 95 % percentile
bootstrap over tasks (1,000 resamples); paired comparisons on shared tasks.
Metric definitions: [docs/metrics.md](docs/metrics.md); protocol:
[ADR-005](docs/adr/ADR-005-evaluation-protocol.md).

## Models and data

Released on Hugging Face as research artefacts (**not for deployment**);
upload steps in [docs/release.md](docs/release.md).

| Artefact | What | Card |
| --- | --- | --- |
| [`KnowWhenToHandOff-Qwen3-4B-R2`](https://huggingface.co/JaspinXu/KnowWhenToHandOff-Qwen3-4B-R2) | GRPO with the Responsible-AI reward (best-calibrated hand-offs) | [R2](docs/model-cards/R2.md) |
| [`KnowWhenToHandOff-Qwen3-4B-R1`](https://huggingface.co/JaspinXu/KnowWhenToHandOff-Qwen3-4B-R1) | GRPO with the task reward (the over-escalation case) | [R1](docs/model-cards/R1.md) |
| [`KnowWhenToHandOff-Qwen3-4B-SFT`](https://huggingface.co/JaspinXu/KnowWhenToHandOff-Qwen3-4B-SFT) | The SFT warm start (B2) | [B2](docs/model-cards/B2.md) |
| [`KnowWhenToHandOff-data`](https://huggingface.co/datasets/JaspinXu/KnowWhenToHandOff-data) | Hand-off task set and splits, every run's trajectories and metrics, GRPO training rollouts, SFT selection | [dataset card](docs/dataset-card.md) |

All four are in one [Hugging Face collection](https://huggingface.co/collections/JaspinXu/knowwhentohandoff).
The task set and manifests are also in this repository ([`data/`](data/)); a
dialogue sample is in [`reports/demo/`](reports/demo/). The dataset is mirrored
on [Kaggle](https://www.kaggle.com/datasets/zhaobinxu/knowwhentohandoff-data),
with a CPU-only [walk-through notebook](https://www.kaggle.com/code/zhaobinxu/knowwhentohandoff-quickstart)
(run index, task kinds, where each model hands off, R1's loophole curve).

```bash
# serve R2 with an OpenAI-compatible API (tool calls parsed)
vllm serve JaspinXu/KnowWhenToHandOff-Qwen3-4B-R2 --enable-auto-tool-choice --tool-call-parser hermes
```

```python
from datasets import load_dataset

runs = load_dataset("JaspinXu/KnowWhenToHandOff-data", "runs", split="train")             # one row per run
airline = load_dataset("JaspinXu/KnowWhenToHandOff-data", "airline_tasks", split="train")  # τ²-bench task schema
```

For faithful behaviour run the models inside this repository's evaluation
harness: they were trained with the τ²-bench domain policy as the system prompt
and the domain's tools.

## Reproduce

Laptop / CI (CPU, Python 3.12, [uv](https://docs.astral.sh/uv/)):

```bash
uv sync --locked
uv run --locked ruff check . && uv run --locked pytest -q
```

Everything heavy runs on a Slurm cluster, one command per stage
([cluster runbook](docs/cluster-runbook.md)): environment setup → task sets →
model servers → B0/B1 → teacher rollouts → SFT data → SFT → R1/R2 → final
evaluations → reports. The tables and figures above come from

```bash
bash slurm/run.sh python scripts/make_report.py --name main --baseline B2 --runs \
  B0=B0-test_main-s0-20261004+B0-test_styles-s0-20261004+B0-test_handoff-s0-20261004 \
  B1=B1-test_main-s0-20261004 \
  B2=B2-test_main-s0-20261005+B2-test_styles-s0-20261005+B2-test_handoff-s0-20261005 \
  R1=R1-final-test_main-s0-20261006+R1-final-test_styles-s0-20261006+R1-final-test_handoff-s0-20261006 \
  R2=R2-final-test_main-s0-20261007+R2-final-test_styles-s0-20261007+R2-final-test_handoff-s0-20261007
bash slurm/run.sh python scripts/rl_curves.py --name rl --runs R1=R1-s1-20261006 R2=R2-s1-20261006
python scripts/readme_figures.py   # README figures (PNG) from the committed reports, demo dialogues and task files; needs Playwright with Chromium
```

Every run records its resolved config, git SHA, τ²-bench commit, model
revisions and seeds; reports accept completed real runs only.

<details>
<summary><b>Repository layout</b></summary>

```text
KnowWhenToHandOff/
├── configs/          # Hydra: eval/, reward/, rl/, model/, persona/, handoff_tasks/, paths/
├── src/handoff/
│   ├── envs/         # τ²-bench adapter, labels, splits, hand-off tasks, personas
│   ├── sim/          # vLLM serving, endpoint files
│   ├── agents/       # agent prompt and the τ²-bench agent used for evaluation
│   ├── rollout/      # trajectory schema, token-in/token-out episodes, verl agent loop
│   ├── rewards/      # policy rules, format checks, reward composer
│   ├── train/        # SFT data and trainer, RL launch, checkpoint export, monitors
│   ├── eval/         # evaluation runner, metrics with bootstrap CIs, reports
│   ├── ops/          # run records, provenance, test ledger
│   ├── demo/         # offline HTML dialogue viewer
│   └── release.py    # dataset release packaging
├── scripts/          # CLI entry points (run_eval, train_sft, train_rl, make_report, …)
├── slurm/            # submit.sh, run.sh, one sbatch file per job
├── data/             # hand-off tasks and frozen manifests
├── docs/             # runbook, metrics, labels, ADRs, audits, model and dataset cards
├── reports/          # committed tables, figures and demo pages
└── tests/            # CPU tests: unit, τ²-bench integration, parity, cluster-stack checks
```

</details>

## Limitations and responsible use

- **Simulated users only.** One simulator (Qwen3-30B-A3B), which also generated
  the SFT data and is the B1 reference; results may not transfer to people.
- **One seed per run**; intervals cover task sampling, not training variance.
  The official test split is small (49 tasks), and retail uses a 74-task subset.
- **Template-built hand-off tasks**, labelled from the policy text and checked
  on a 40-task sample; rule checks cover only part of each policy, and the
  citation rate is a keyword rule.
- **Styles are not demographics.** No claim is made about any group of people.
- **Not for deployment.** The checkpoints are research artefacts.

## Citation

```bibtex
@misc{knowwhentohandoff2026,
  title  = {Know When to Hand Off: Multi-turn Reinforcement Learning for
            Trustworthy Customer-Service Agents},
  author = {{KnowWhenToHandOff contributors}},
  year   = {2026},
  howpublished = {\url{https://github.com/JaspinXu/KnowWhenToHandOff}}
}
```

DOIs: dataset [10.57967/hf/10822](https://doi.org/10.57967/hf/10822) (Kaggle mirror [10.34740/kaggle/dsv/20472451](https://doi.org/10.34740/kaggle/dsv/20472451)), R2 [10.57967/hf/10823](https://doi.org/10.57967/hf/10823), R1 [10.57967/hf/10824](https://doi.org/10.57967/hf/10824), SFT [10.57967/hf/10825](https://doi.org/10.57967/hf/10825).

## Acknowledgements and references

Built on [τ²-bench](https://github.com/sierra-research/tau2-bench) (MIT),
[verl](https://github.com/volcengine/verl), [vLLM](https://github.com/vllm-project/vllm)
and the [Qwen3](https://huggingface.co/Qwen) models.

1. Yao S, et al. τ-bench: a benchmark for tool-agent-user interaction in real-world domains. arXiv:2406.12045. 2024.
2. Barres V, et al. τ²-Bench: evaluating conversational agents in a dual-control environment. arXiv:2506.07982. 2025.
3. Shao Z, et al. DeepSeekMath: pushing the limits of mathematical reasoning in open language models (GRPO). arXiv:2402.03300. 2024.
4. Sheng G, et al. HybridFlow: a flexible and efficient RLHF framework (verl). arXiv:2409.19256. 2024.
5. Zhao W, et al. MUA-RL: multi-turn user-interacting agent reinforcement learning for agentic tool use. arXiv:2508.18669. 2025.
6. Modecrua W, et al. Multi-turn reinforcement learning for tool-calling agents with iterative reward calibration. arXiv:2604.02869. 2026.
7. He M, et al. Impatient users confuse AI agents: high-fidelity simulations of human traits for testing agents. arXiv:2510.04491. 2025.
8. Skalse J, et al. Defining and characterizing reward hacking. NeurIPS 2022. arXiv:2209.13085.

## License

Code: [MIT](LICENSE). Derived task data: MIT, from τ²-bench (© 2025 Sierra
Research). Model weights inherit the Qwen3 Apache-2.0 licence.
