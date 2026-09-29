#!/usr/bin/env bash
# T3.1 prerequisite: exact incremental tokenization for the policy tokenizer.
#   MODEL=qwen3_4b (models-lock entry; also gives the chat-template arguments)
#   TOKENIZER=/path/to/checkpoint  (check a trained checkpoint's tokenizer instead)
#   TRAJECTORIES=<trajectories.jsonl>  MAX_PROMPT_TOKENS=8192 (episode.max_prompt_tokens)
source "${HANDOFF_REPO}/slurm/lib/common.sh"
job_banner
activate_venv
args=(--model "${MODEL:-qwen3_4b}" --max-prompt-tokens "${MAX_PROMPT_TOKENS:-8192}")
if [[ -n "${TOKENIZER:-}" ]]; then args+=(--tokenizer "${TOKENIZER}"); fi
if [[ -n "${TRAJECTORIES:-}" ]]; then args+=(--trajectories "${TRAJECTORIES}"); fi
python scripts/check_tokenization.py "${args[@]}"
