#!/usr/bin/env bash
# T2.2: assistant-only SFT (run B2). DATASET=<name from build_sft_data.py>
# NGPUS>1 uses torchrun with FSDP (full fine-tuning only).  EXTRA="sft.num_train_epochs=3"
source "${HANDOFF_REPO}/slurm/lib/common.sh"
job_banner
activate_venv
parse_extra
: "${DATASET:?set DATASET=<dataset_name>}"
if [[ ! -d "${HANDOFF_SCRATCH}/datasets/${DATASET}" ]]; then
  echo "no dataset ${DATASET} in ${HANDOFF_SCRATCH}/datasets (run sft-data first)" >&2
  exit 2
fi
if [[ "${NGPUS:-1}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NGPUS}" scripts/train_sft.py \
    "dataset_name=${DATASET}" 'sft.fsdp=full_shard auto_wrap' "${EXTRA_ARGS[@]}"
else
  python scripts/train_sft.py "dataset_name=${DATASET}" "${EXTRA_ARGS[@]}"
fi
