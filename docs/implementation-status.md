# Implementation status

Updated: 28 September 2026. This file separates code verification from research
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
| T0.3 | Offline server planner, Hydra templates, exclusive plan bundles, run provenance and CLI | Confirmed cluster configuration, server responses and measured throughput |
| T1.4 | Six configured style cards and serialized-task persona injector | Accepted upstream Task adapter and ten reviewed simulated dialogues per style |
| T3.1 prerequisite | Sampled-token accumulator and policy/context masks | Live verl adapter, generation log probabilities, reward parity and GPU validation |
| Other T0.4–T4.5 work (except prerequisites listed above) | Not implemented | Their README definitions of done |

## Foundation evidence

The initial contract tests produced three expected failures (missing package,
missing distribution metadata and missing config) and one passing network guard.
After implementation, all four tests passed. Ruff checks and sdist/wheel build
also passed. These are local engineering checks, not model experiments.

Reproduce: `uv sync --locked && uv run pytest -q` (run as separate commands in
Windows PowerShell 5). Full gates are listed in AGENTS.md.

## Offline serving evidence

The serving implementation passed 53 CPU tests, including script/argv rendering,
invalid configuration, immutable revision validation, non-overwrite behavior,
partial-write cleanup and real CLI calls from another working directory. Generated Bash passed syntax validation and refused execution outside a Slurm
allocation. An independent review found CLI argument-value leakage; regression
tests reproduced it, the handler was fixed, and the complete suite passed. No
Slurm job was submitted; no model weights or GPU runtime were installed.

Reproduce: `uv run pytest -q`. Usage: [offline server preparation](../slurm/README.md).

## Runtime utility evidence

The runtime utility slice brought the CPU suite to 103 passing tests. Its tests
cover immutable run records, nested credential redaction, unresolved config
rejection, config hashes,
exclusive completion markers, input validation and exact token/mask preservation.
Independent review exposed cleanup failures during file close/flush and URL
validation bypasses with leading whitespace. Five regression tests reproduced
these failures and passed after fixes. Ruff and wheel/sdist builds also passed.
No benchmark/reward/evaluation definition was changed. API usage and limitations
are in [runtime utilities](runtime-utilities.md).

The G: checkout is on `codex/handoff-local`. Initial audit drafts were preserved
in a Git stash before checking out the implemented code. The implementation
worktree and earlier commits are retained. No remote push was performed.

## Persona injection evidence

The initial 44 persona tests failed because the module did not exist, then passed
after implementation. They exercise the six styles on synthetic task dictionaries
and prove that injection preserves every field except `user_scenario.persona`,
does not mutate source tasks, and keeps nested variants independent. Neutral is
a no-op control; populated personas cannot be overwritten with a style card.
The library composes through Hydra and provides content hashes for run records.
Independent review found no actionable issues. A subsequent boundary regression
exposed shared scenario references changing unrelated fields; detaching the styled
branch fixed it. The full suite passes 148 CPU tests; Ruff, format and package
build checks pass.
See [persona usage and review requirements](personas.md). Live upstream Task
integration and simulator behavior have not been validated.

## Operational boundaries

The upstream scientific definitions in both READMEs remain unchanged. Planning
files and CPU fake responses must never be cited as real model results. Do not
tick a README task until its full definition of done is supported by evidence.
