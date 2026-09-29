# ADR-003: User simulators, teacher and the B1 reference

- Status: **Proposed** — confirm at M1 (11 Oct 2026)
- Related tasks: T0.3, T0.5, T1.6, T2.1, T3.6

## Decision

| Role | Model | Settings |
| --- | --- | --- |
| User simulator (evaluation and training) | Qwen3-30B-A3B-Instruct-2507, bf16, vLLM (`configs/model/qwen3_30b_a3b.yaml`) | tau2's `UserSimulator` prompt; temperature 0.7; max 1,024 tokens per turn; per-trial seed |
| Teacher for SFT data (T2.1) | Qwen3-30B-A3B-Instruct-2507 as the agent | Temperature 0.6, train split only |
| B1 reference policy | Qwen3-30B-A3B-Instruct-2507 as the agent | Same protocol as B0 |

- The simulator is τ²-bench's own user simulator with its own guidelines; the
  only change is the style card in `user_scenario.persona` (T1.4). Stop and
  transfer tokens (`###STOP###`, `###TRANSFER###`) keep their tau2 meaning.
- Every trial passes a seed to the simulator (tau2 derives per-trial seeds from
  the run seed), so episodes are reproducible for a given server and trials
  differ from each other; this is what makes pass^k meaningful.
- Models are downloaded on the cluster only; their exact Hugging Face commits
  are recorded in the models lock and in every run's provenance.

## Rationale and alternatives

- A local open model costs nothing per call and gives stable versions; paid APIs
  (the README fallback) change silently and cost money at RL scale.
- Qwen3-30B-A3B is a mixture-of-experts model (3B active), fast enough to answer
  many concurrent episodes from one GPU; bf16 weights (~61 GB) fit an 80 GB A100.
- Temperature 0.7 rather than tau2's default 0: with a greedy simulator the
  four trials of a task would be nearly identical.
- Using the same 30B model as teacher, B1 and simulator means the teacher has
  "met" this simulator. A second, held-out simulator would measure that
  overfitting, but it needs another model and a second test campaign; it is out
  of scope for the course project and stated as a limitation.

## Consequences

- Reported numbers are conditional on this one simulator; the manual dialogue
  review checks the worst cases.
- If the A100 has 40 GB, the simulator does not fit in bf16; options are to
  serve it on the H200 between jobs or to use an FP8 variant (needs a new model
  config and an ADR update).
