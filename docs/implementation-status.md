# Implementation status

Updated: 27 September 2026. This file separates code verification from research
results. No benchmark score, trained checkpoint, or measured cluster throughput
has been produced.

## Owner decisions

- Implement code and run CPU tests locally before cluster access is supplied.
- One A100 and one H200 are available according to the owner. VRAM, scheduler
  topology, account, partition and scratch path are not yet specified.
- ADR-006 is deferred. Do not adopt its benchmark pin, split or reward changes.
- Independent infrastructure may advance while real GPU acceptance remains open.

## Progress

| Task | Code / local verification | Outstanding acceptance |
| --- | --- | --- |
| T0.1 | Installable package, Hydra root config, locked tooling, CPU tests, CI workflow, license and secret/artifact ignore rules | Actual remote CI run |
| T0.2 | Not implemented; proposed upstream pin is deferred | Accepted pin, real five-task smoke and parser |
| T0.3 | Validated offline server planner, Hydra templates, exclusive output bundles and CLI | Confirmed cluster configuration, server responses and measured throughput |
| T0.4–T4.5 | Not implemented | Their README definitions of done |

## Foundation evidence

The initial contract tests produced three expected failures (missing package,
missing distribution metadata and missing config) and one passing network guard.
After implementation, all four tests passed. Ruff checks and sdist/wheel build
also passed. These are local engineering checks, not model experiments.

Reproduce: `uv sync --locked && uv run pytest -q` (run as separate commands in
Windows PowerShell 5). Full gates are listed in AGENTS.md.

## Offline serving evidence

The CPU suite currently has 53 passing tests, including script/argv rendering,
invalid configuration, immutable revision validation, non-overwrite behavior,
partial-write cleanup and real CLI calls from another working directory. Generated Bash passed syntax validation and refused execution outside a Slurm
allocation. An independent review found CLI argument-value leakage; regression
tests reproduced it, the handler was fixed, and the complete suite passed. No
Slurm job was submitted; no model weights or GPU runtime were installed.

Reproduce: `uv run pytest -q`. Usage: [offline server preparation](../slurm/README.md).

## Operational boundaries

The upstream scientific definitions in both READMEs remain unchanged. Planning
files and CPU fake responses must never be cited as real model results. Do not
tick a README task until its full definition of done is supported by evidence.
