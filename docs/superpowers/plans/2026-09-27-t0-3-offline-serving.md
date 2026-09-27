# T0.3 Offline Serving Preparation

> Execute natively with superpowers:executing-plans and test-driven-development.

**Goal:** Prepare inspectable vLLM Slurm jobs on CPU without connecting to a
cluster, installing vLLM, downloading weights, or changing research definitions.

**Architecture:** Pure validation and rendering in `handoff.sim`; a thin Hydra
CLI writes an immutable planning bundle. Separate model, serving and scheduler
configs supply every operational value. The bundle is explicitly an offline
plan, never an experiment result.

**Spec:** README T0.3 and the owner's 27 September direction to implement
independent code before cluster access. ADR-006 remains deferred.

**Dependencies:** Existing Hydra and Python standard library only. The cluster's
future vLLM environment is an external prerequisite, not installed by this task.

## File ownership and tests

- `src/handoff/sim/server_plan.py`: validate resolved config, construct tokenized
  vLLM argv, and render a single-node Slurm script.
- `src/handoff/sim/plan_io.py`: write new plan directories exclusively, containing
  `serve.sbatch`, resolved config, and hashed planning metadata; refuse overwrite.
- `scripts/plan_server.py`: compose Hydra config, render/write the plan; no submit.
- `configs/serve.yaml`, `configs/model/`, `configs/server/`,
  `configs/slurm/`: templates with mandatory deployment paths and revisions.
- `slurm/README.md`: inspect, stage and submit generated scripts later.
- `tests/test_server_plan.py`, `tests/test_plan_io.py`: CPU contract tests.

## Requirements

- Missing scratch/workdir/interpreter/revisions fail before output is written.
- Model revisions are immutable 40-character lowercase Git SHAs.
- Reject unknown keys, secret-bearing config, malformed numbers, invalid
  resources and control characters; never print rejected raw values.
- No shell interpolation of user config: argv uses shell quoting; scheduler
  tokens and absolute POSIX paths have strict allowlists.
- Require positive GPU/CPU/memory/time resources, valid port and GPU utilization,
  single-node TP equal to the GPU count, and an explicit nonnegative seed.
- Require a numeric vLLM version; generated script checks that installed version
  before launching. Actual package/CUDA/model compatibility remains unverified.
- Enforce Slurm allocation at runtime; HF cache and logs go to scratch.
- Log script/config identities and hardware/version metadata, not credentials.
- Do not auto-submit, install a serving stack, or access any model/API.
- Preserve existing target paths; failed bundle creation cleans up only files
  created by that invocation. Two writers must not overwrite each other.

## Steps

- [x] Write invalid-config, quoting/resource, pinning and output-integrity tests.
- [x] Run them and observe missing implementation failures.
- [x] Implement pure validation, command construction and script rendering.
- [x] Implement exclusive bundle writes and a thin Hydra CLI.
- [x] Run unit tests, CLI dry runs, shell syntax checks if bash is available,
  lint/format, hooks and diff checks.
- [x] Review the complete branch and fix important findings.
- [x] Commit with T0.3 prefix and reproduction command; retain real cluster DoD
  as pending in both READMEs and implementation status.

## Evidence

Initial tests failed because the serving modules did not exist. The implemented
suite passed 49 tests, then review-driven secret-output tests reproduced three
failures. The fix passes all 53 CPU tests. Ruff, hooks, build and Bash syntax
checks pass; the allocation guard exits 2 outside Slurm. Real cluster acceptance
and the training template remain pending; no T0.3 DoD completion is claimed.
