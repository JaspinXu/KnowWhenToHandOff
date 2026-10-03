#!/usr/bin/env bash
# Submit a project job with the resources configured in slurm/cluster.env.
#
#   bash slurm/submit.sh <job> [KEY=VALUE ...] [-- extra sbatch flags]
#
# Jobs: setup, download, tasks, check-tokens, serve, eval, teacher, sft-data,
#       sft, rl, cpu.
# KEY=VALUE pairs are exported to the job (see docs/cluster-runbook.md), e.g.
#   bash slurm/submit.sh eval EVAL=test_main EXPERIMENT=B0 POLICY_MODEL=qwen3_4b
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${here}/.." && pwd)"
env_file="${HANDOFF_CLUSTER_ENV:-${here}/cluster.env}"
if [[ ! -f "${env_file}" ]]; then
  echo "Create ${env_file} from slurm/cluster.env.example first." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "${env_file}"
export HANDOFF_REPO="${repo}" HANDOFF_CLUSTER_ENV="${env_file}"

job="${1:-}"
if [[ -z "${job}" ]]; then
  sed -n '2,10p' "$0"
  exit 2
fi
shift
extra_sbatch=()
while (($#)); do
  case "$1" in
    --) shift; extra_sbatch+=("$@"); break ;;
    *=*)
      if [[ ! "${1%%=*}" =~ ^[A-Z_][A-Z0-9_]*$ ]]; then
        echo "invalid variable name in '$1' (use UPPER_CASE=value)" >&2; exit 2
      fi
      # shellcheck disable=SC2163  # exports the KEY=VALUE pair itself
      export "$1" ;;
    *) echo "unexpected argument: $1 (use KEY=VALUE)" >&2; exit 2 ;;
  esac
  shift
done

gpu() {  # gpu <type-or-empty> <count>
  if [[ -n "$1" ]]; then echo "--gres=gpu:$1:$2"; else echo "--gpus=$2"; fi
}
config_exists() {  # config_exists <group> <name>: catch typos before queueing
  if [[ "$2" == _* || ! -f "${repo}/configs/$1/$2.yaml" ]]; then
    echo "unknown $1 config '$2' (see configs/$1/)" >&2; exit 2
  fi
}
case "${job}" in
  eval|teacher)
    if [[ -z "${RESUME:-}" && "${job}" == eval ]]; then config_exists eval "${EVAL:-smoke}"; fi ;;
  rl)
    if [[ -z "${RESUME:-}" ]]; then
      : "${INIT:?set INIT=<policy checkpoint dir>}"
      config_exists rl "${RL:-grpo}"
      config_exists reward "${REWARD:-task_only}"
    fi ;;
esac
# Logs go next to the checkout (home is visible from every node, even if
# scratch turns out not to be); override with HANDOFF_LOGS.
logs="${HANDOFF_LOGS:-${repo}/logs}"
mkdir -p "${logs}"
common=(--export=ALL --output="${logs}/%x-%j.out" --chdir="${repo}")
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then common+=(--account="${SLURM_ACCOUNT}"); fi
gpu_job=(--partition="${PARTITION_GPU}" --cpus-per-task="${CPUS_GPU_JOB}" --mem="${MEM_GPU_JOB}")
cpu_job=(--partition="${PARTITION_CPU}" --cpus-per-task=8 --mem=32G --time="${TIME_CPU}")

case "${job}" in
  setup)        flags=(--partition="${PARTITION_CPU}" --cpus-per-task=8 --mem=32G --time="${TIME_SETUP}"); script=setup_env.sbatch ;;
  download)     flags=("${cpu_job[@]}"); script=download_models.sbatch ;;
  tasks)        flags=("${cpu_job[@]}"); script=build_task_sets.sbatch ;;
  check-tokens) flags=("${cpu_job[@]}"); script=check_tokenization.sbatch ;;
  sft-data)     flags=("${cpu_job[@]}"); script=build_sft_data.sbatch ;;
  cpu)          flags=("${cpu_job[@]}"); script=cpu.sbatch ;;
  serve)        flags=("${gpu_job[@]}" "$(gpu "${GPU_TYPE_SIM}" "${NGPUS:-1}")" --time="${TIME_SERVE}"); script=serve.sbatch ;;
  eval|teacher)
    if [[ -n "${POLICY_ENDPOINT:-}" ]]; then  # policy already served elsewhere (B1)
      flags=(--partition="${PARTITION_CPU}" --cpus-per-task=8 --mem=32G --time="${TIME_EVAL}")
    else
      flags=("${gpu_job[@]}" "$(gpu "${GPU_TYPE_BIG}" 1)" --time="${TIME_EVAL}")
    fi
    script=eval.sbatch ;;
  sft)          flags=("${gpu_job[@]}" "$(gpu "${GPU_TYPE_BIG}" "${NGPUS:-1}")" --time="${TIME_SFT}"); script=sft.sbatch ;;
  rl)           flags=("${gpu_job[@]}" "$(gpu "${GPU_TYPE_BIG}" "${NGPUS:-1}")" --time="${TIME_RL}"); script=rl.sbatch ;;
  *) echo "unknown job '${job}'" >&2; sed -n '2,10p' "$0"; exit 2 ;;
esac
if [[ "${job}" == teacher ]]; then export EVAL="${EVAL:-teacher_train}" EXPERIMENT="${EXPERIMENT:-TEACH}"; fi
sbatch "${common[@]}" "${flags[@]}" --job-name="handoff-${job}" "${extra_sbatch[@]}" "${here}/jobs/${script}"
