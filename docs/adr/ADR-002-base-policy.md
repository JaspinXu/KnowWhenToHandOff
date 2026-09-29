# ADR-002: Base policy

- Status: **Proposed** — confirm at M1 (11 Oct 2026) with the B0 dev results
- Related tasks: T1.6, T2.2, T3.1

## Decision

The policy under study is **Qwen3-4B-Instruct-2507** (`configs/model/qwen3_4b.yaml`,
served as `qwen3-4b` with vLLM's `hermes` tool parser).

| Use | Model | Why |
| --- | --- | --- |
| Policy (B0, B2 init, R1, R2) and smoke tests | Qwen3-4B-Instruct-2507 | Strong tool calling at 4B; non-thinking chat template |

No second, smaller model is used: the smoke evaluation and the RL spike (T0.4)
run the same 4B model, so there is one model config and one tokenizer check.

Two properties matter for training and are checked before any RL job
(`scripts/check_tokenization.py`, Slurm job `check-tokens`):

1. **Prefix-stable chat template.** The RL driver appends each observation's
   tokens after the sampled tokens; this equals a full re-rendering only if the
   template renders earlier turns identically in longer conversations. Qwen3's
   hybrid-thinking templates drop earlier reasoning and fail this check; the
   2507 Instruct template has no thinking block.
2. **Prompt budget.** The domain policy, tool schemas and greeting must leave
   room for the first user message within `episode.max_prompt_tokens`.

## Decision rule at M1

Keep Qwen3-4B-Instruct-2507 unless, on the dev split, B0's
`format_error_rate` exceeds 5 % (README gate) or `check-tokens` fails. The
alternative is Qwen3.5-4B, if an instruct checkpoint with a prefix-stable
template is available on Hugging Face; it would need its own model config and a
rerun of the checks. Record the dev numbers (run IDs) here when deciding.

## Consequences

- SFT, RL and every checkpoint evaluation use the Qwen3-4B tokenizer; B2 and RL
  checkpoints keep the base model's tokenizer files.
- The README scope (policy <= 8B parameters) is met; larger models are used only
  as simulators, teacher and the B1 reference (ADR-003).
