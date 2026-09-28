# Persona Injection Implementation Plan

> Execute inline with superpowers:executing-plans and test-driven-development.

**Goal:** Implement the README's six communication styles without altering task
facts, labels, evaluation criteria, splits, rewards or benchmark dependencies.

**Architecture:** Hydra supplies versioned style text. A frozen `Persona` carries
the ID and final prompt plus a content hash. `PersonaInjector.apply` takes a
serialized task dictionary, deep-copies it and changes only
`user_scenario.persona`. Reconstructing an upstream Task belongs to the eventual
approved benchmark adapter, outside this dependency-free boundary.

**Tech stack:** Python 3.12 standard library, existing Hydra and pytest.

**Spec:** README Communication styles and T1.4; AGENTS sections 2, 5 and 6.
Owner has authorized independent code while ADR-006 and GPU details are deferred.

## Constraints and decisions

- Exactly `neutral`, `terse`, `verbose`, `impatient`, `non-native`,
  `low-tech-literacy`; no new style or demographic claim.
- Neutral is a deep-copy no-op, preserving the unmodified control prompt.
- Other styles require an absent, null or empty original persona. Reject an
  existing nonempty persona rather than silently replacing task information or
  combining two styles. Callers must start each variant from the original task.
- Configured common instructions limit changes to communication style, preserving
  facts, goals, tool rules and disclosure requirements. No nationality, age,
  education, disability or demographic traits are assigned.
- No upstream Task dependency, label generation, evaluation runner or live model.
  T1.4 remains open until ten real dialogues per style are reviewed.

## Review focus

1. Nested task values must never alias the input after injection.
2. Neutral must not add instructions or populate an absent persona field.
3. Existing persona content must not be discarded or combined silently.
4. Missing/unknown/blank styles and unresolved config must fail explicitly.
5. Style prompts must not change required disclosures or demand escalation.

### Task 1: T1.4 style library and serialized-task injector

**Files:** `src/handoff/envs/{__init__,personas}.py`,
`configs/personas.yaml`, `configs/persona/styles_v1.yaml`,
`tests/test_personas.py`, `docs/personas.md`, both READMEs, AGENTS and status.

**Interfaces:** frozen `Persona(id: str, text: str)`;
`Persona.sha256: str`; `personas_from_config(config: Mapping[str, object])
-> dict[str, Persona]`; `PersonaInjector.apply(task: dict[str, object],
persona: Persona) -> dict[str, object]`.

- [x] Write failing tests: neutral equality with independent nested copies,
  airline/retail synthetic task invariance for each style, source nonmutation,
  sequential variants isolated, rejection of populated personas/invalid task
  structures, config composition from another cwd, invalid cards and hashes.
- [x] Run `uv run --locked pytest -q tests/test_personas.py`; expect failures
  because the persona module/config does not exist yet.
- [x] Implement pure configuration validation, frozen cards, exact content hashes,
  guarded deep-copy injection and the six configured styles.
- [x] Run focused tests, then full CPU suite, Ruff, format, build and diff checks.
- [x] Obtain one independent review; reproduce and fix important findings.
- [x] Update bilingual usage/status without ticking the live dialogue audit.
- [x] Commit with T1.4 prefix and the reproduction command in the body.

Evidence: 44 initial missing-module failures; 44 focused passes after implementation.
Independent review found no actionable issues. An additional shared-reference
regression failed before detaching the modified scenario branch; full suite then
passed 148 tests. No upstream data, model calls or experimental results were used.
