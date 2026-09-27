# T0.1 Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task after owner review. Native execution in this
> chat is recommended for this single tightly scoped deliverable; no parallel
> agent work is requested by this plan.

**Goal:** Make the existing design repository installable and verifiable on CPU
with Python 3.12, while keeping all later task IDs unimplemented.

**Architecture:** A minimal `src/handoff` package with project-root Hydra config
resources, uv packaging, and offline tests. T0.1 has no benchmark adapter or
training behavior. The CI workflow runs the same gates as local development.

**Tech stack:** CPython 3.12, uv, hatchling, hydra-core, pytest, pytest-socket,
ruff, pre-commit, GitHub Actions. Exact compatible versions will be resolved once
and recorded in `uv.lock`; no floating Git branch dependencies.

**Spec:** [README T0.1](../../../README.md#phase-0--foundations-28-sep--4-oct),
[AGENTS.md](../../../AGENTS.md), and the
[repository audit](../../repository-audit-2026-09-27.md).

**Status:** Implemented and locally verified under the owner's direction to proceed with independent code. Remote CI remains pending.

## Global constraints

- Python 3.12 with full type hints; package name `handoff`, src layout.
- Implement only T0.1. Follow T0.x → T4.x ordering.
- Tests run on CPU with no GPU and no network.
- Configuration through Hydra; no hard-coded paths, hostnames, ports, model names
  or lambda values in library code.
- No secrets, checkpoints, trajectories, model caches, or personal proposal data
  in git. A blank `.env.example` is a template, not a credential file.
- README changes must be mirrored in `README.zh-CN.md`.
- A real CI run is required before marking the complete T0.1 DoD satisfied.

## Dependency approval requested

| Group | Packages | Purpose |
| --- | --- | --- |
| Build | hatchling | Build the small src-layout package |
| Runtime | hydra-core | Compose YAML configuration, as required by the design |
| Development | pytest, pytest-socket, ruff, pre-commit | CPU tests with network disabled, lint/format, commit hooks |

`pytest-socket` is proposed so “no network” is enforced, not merely a convention.
No tau2, torch, vLLM, verl, tracker, or web UI package is installed in this task.

## Review focus

- A package import must work from a different working directory after install.
- Offline testing must fail a deliberate socket attempt instead of permitting it.
- Hydra config composition must not depend on the caller's current directory.
- Ignore rules must block both top-level and nested secrets/artifacts while
  allowing `.env.example` and future small manifests.
- Docs must distinguish a locally passing CI command from a green remote CI run.

## File map

| File | Responsibility |
| --- | --- |
| `pyproject.toml`, `uv.lock`, `.python-version` | Build metadata, exact resolution, Python selection, quality-tool config |
| `src/handoff/__init__.py` | Importable package and initial version `0.1.0` |
| `configs/config.yaml`, `configs/README.md` | Minimal root config (`seed: 0`) and conventions; domain/model groups arrive in their own tasks |
| `tests/test_foundations.py` | Packaging, config and enforced-network-boundary checks |
| `.pre-commit-config.yaml` | Pinned lint/format hooks, whitespace and merge-conflict checks |
| `.github/workflows/ci.yml` | Python 3.12 Ubuntu/Windows CPU matrix with locked dependencies |
| `.gitignore`, `.env.example`, `LICENSE` | Artifact/secret boundaries, blank configuration template, MIT license |
| `README.md`, `README.zh-CN.md`, `AGENTS.md` | Accurate setup commands and execution status |

## Task: Installable, offline-verified foundations

**Consumes:** The README task definition; no existing Python APIs.

**Produces:** `import handoff`, distribution metadata for `handoff==0.1.0`, and
Hydra root config with `seed: 0`. No public functions or shared schemas are
introduced. All future subsystem APIs remain owned by their assigned tasks.

- [x] **1. Verify a clean baseline and choose the implementation checkout.**
  Preserve the audit/ADR/plan; inspect attached worktrees before creating one.
  Use a `codex/` branch for implementation. Do not overwrite concurrent changes.

- [x] **2. Define packaging and write the contract tests.**
  Set `requires-python = ">=3.12,<3.13"`, hatchling src layout, dependency groups,
  pytest `testpaths = ["tests"]` and `--disable-socket`, ruff target `py312`,
  line length 88, and lint rules `E`, `F`, `I`, `UP`, `B`.
  Tests must assert installed metadata is `0.1.0`; an import in an isolated child
  Python process from a temporary directory succeeds; composing the root config
  by absolute config directory gives integer seed 0; a socket attempt raises
  `pytest_socket.SocketBlockedError`. Use absolute repository paths in tests.

- [x] **3. Run the initial tests and retain genuine failure output.**
  The absent package/config must fail their tests before those files are added.
  Do not claim a test-first cycle without capturing its actual result.

- [x] **4. Add the minimal package and root config, resolve the lock, then test.**
  `src/handoff/__init__.py` contains a package docstring and version only.
  Create the root config and explain future config-group ownership. Run
  `uv sync --locked` after lock creation; run `uv run pytest -q`.
  Do not add placeholder training/evaluation commands that return fake success.

- [x] **5. Establish secret/artifact boundaries and tooling.**
  Ignore `.env` and variants (allow `.env.example`), virtual environments,
  caches, `runs/`, `wandb/`, `data/simulations/`, downloaded upstream data,
  checkpoints, model weight formats, and local scratch outputs.
  Use `git check-ignore --no-index` on representative nested and top-level
  paths; verify manifest JSON and `.env.example` remain trackable.
  Add MIT license and an empty-key environment template without personal data.
  Pin hook and Actions revisions to verified immutable versions/SHAs.

- [x] **6. Configure CPU CI and update documentation together.**
  CI: check out, set up uv/Python 3.12, sync from lock, run ruff check,
  ruff format check and pytest on Ubuntu and Windows. Give only read permissions.
  Document PowerShell `Copy-Item .env.example .env` and POSIX equivalent,
  reproducible checks, and the still-planned status of serving/eval/train.
  Leave the T0.1 checkbox unchecked until a real remote CI run passes.

- [x] **7. Verify the complete local deliverable.**
  Run each command and inspect its result:
  `uv sync --locked`; `uv run ruff check .`;
  `uv run ruff format --check .`; `uv run pytest -q`;
  `uv run pre-commit run --all-files`; `uv build`; `git diff --check`.
  Packaging/network/config tests must pass on CPU. Inspect staged filenames
  before any commit. Do not use a GPU or make model calls.

- [ ] **8. Commit the bounded task, then verify remote CI when publishing is authorized.**
  Prefix the commit `T0.1:` and include reproduction command
  `uv sync --locked && uv run pytest -q` in its body.
  Report local verification separately while CI is pending. Only after the
  actual remote job succeeds, update both README checkboxes and record the run.

## After T0.1

Next is T0.2 after the benchmark pin is approved. Five real train-only episodes
require a configured model service. The owner subsequently authorized independent code work before GPU access. Missing
services keep T0.2 acceptance open, while independent infrastructure may advance.
ADR-006 remains deferred, so no benchmark pin is installed.
