# Offline server preparation (T0.3 code only)

The implemented command generates a plan and never calls `sbatch`, downloads
weights, installs vLLM or runs a model. Real server responses, tool calling and
throughput still require cluster acceptance. Training launchers await the trainer
decision in T0.4; no placeholder trainer is provided.

The owner has one A100 and one H200. Their memory sizes, placement and scheduler
settings are unconfirmed. Plan each server as a separate single-GPU job; do not
assume the GPUs form a homogeneous two-GPU training allocation.

## Inspect settings locally

```bash
uv run python scripts/plan_server.py --help
uv run python scripts/plan_server.py
```

The second command intentionally exits with the missing fields. Fill
`model.revision` with the exact Hugging Face commit, and
`server.vllm_version` with the version of an approved cluster environment.
Provide absolute POSIX scratch, repository and Python executable paths. These
values are not guessed by the planner. Paths currently must not contain spaces.

One-line template (replace every angle-bracket placeholder before use):

```bash
uv run python scripts/plan_server.py model=qwen3_4b model.revision=<40-character-SHA> server.vllm_version=<version> cluster.scratch=<scratch-path> cluster.workdir=<repo-path> cluster.python=<venv-python-path> output_dir=scratch/plans/policy
```

For a user simulator, select `model=custom`, set `model.repository`,
`model.served_name` and `model.revision`, and choose a distinct
`cluster.job_name` and `server.port`. This supplies no new simulator choice or
evaluation protocol. `server.host=127.0.0.1` is the conservative default; a
cross-node client requires an approved reachable binding and cluster networking.

Optional scheduler settings are `cluster.partition`, `cluster.account`, and
`cluster.constraint`. Tensor parallel size must match GPU count; defaults are
one. The default resource limits are planning estimates, not measured requirements.

The new output directory contains:

- `serve.sbatch`: LF-terminated Slurm script, with explicit GPU and time limits.
- `resolved_config.json`: canonical deployment settings without secrets.
- `plan.json`: hashes, source Git SHA, dirty state, and `status: not_executed`.

Existing output directories are never overwritten. Do not put tokens into Hydra
overrides; credentials remain in the cluster environment.

## Later cluster acceptance

1. Verify the chosen vLLM/CUDA/Python/model combination on the compute environment.
   The generated script checks vLLM's exact installed version; it does not install it.
2. Ensure scratch and repository directories exist before submission: Slurm opens
   the output log before the script can create directories. Copy the complete
   planning bundle to scratch and inspect the resolved config and job script.
3. From the cluster, submit `sbatch /path/to/plan/serve.sbatch`. No local planner
   command submits a job. Running the script directly outside a Slurm GPU
   allocation fails immediately.
4. The job stores its submitted script, resolved config, actual Git SHA/dirty
   status, runtime versions and GPU inventory under
   `<scratch>/handoff-jobs/<SLURM_JOB_ID>/`. HF cache and Slurm output stay on scratch.
5. Check the OpenAI-compatible endpoint and real tool-call response from a compute
   node, then measure throughput with approved tasks. A successful script render,
   shell parse, or health endpoint is not proof of tool-call correctness.
6. Stop idle servers using the corresponding job IDs when their work finishes.

References verified during implementation:
[vLLM server CLI](https://docs.vllm.ai/en/latest/cli/serve/),
[vLLM tool calling](https://docs.vllm.ai/en/latest/features/tool_calling/),
[Slurm sbatch](https://slurm.schedmd.com/sbatch.html).
