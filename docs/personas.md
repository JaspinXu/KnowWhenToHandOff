# Communication-style library

T1.4 has configurable draft cards and a pure, CPU-tested injector for serialized
task dictionaries. The τ²-bench adapter (`handoff.envs.tau2_adapter.styled_task`)
applies a card and suffixes the task ID with `::<style>`; the task-set build
checks every task with every style against tau2's `Task` schema, and the
integration tests run styled tasks through tau2. Still pending (T1.4 DoD): the
review of ten simulated dialogues per style
(`python scripts/sample_review.py styles --run <B0 test_styles run>`). There is no
evidence yet that the simulator consistently follows these prompts.

## Six styles

The IDs are the existing README IDs. The prompts describe expression, never a
demographic group or a different task outcome.

| ID | Intended change |
| --- | --- |
| `neutral` | Control: no additional text; preserve the original persona |
| `terse` | Concise answers that retain required details |
| `verbose` | Fuller phrasing using only facts already supplied |
| `impatient` | Brisk tone while cooperating with required steps |
| `non-native` | Simple vocabulary and occasional awkward phrasing; unchanged meaning and identifiers |
| `low-tech-literacy` | Everyday vocabulary and requests to clarify technical terms; unchanged capabilities |

Prompts are in [styles_v1.yaml](../configs/persona/styles_v1.yaml). Common
instructions preserve scenario facts, goals, preferences, disclosure rules, tool
rules and stopping behavior. Style alone must not introduce a request for a human
or abandonment. These are simulator instructions, not an enforcement mechanism.

## Use from a driver

Compose the dedicated Hydra config using an explicit config-directory path:

```python
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from handoff.envs.personas import PersonaInjector, personas_from_config

# repo_root comes from the calling application, not the current working directory.
config_dir = Path(repo_root) / "configs"
with initialize_config_dir(version_base="1.3", config_dir=str(config_dir.resolve())):
    config = compose(config_name="personas")
resolved = OmegaConf.to_container(config.persona, resolve=True, throw_on_missing=True)
cards = personas_from_config(resolved)

card = cards["terse"]
variant = PersonaInjector().apply(original_task_dict, card)
# Keep these values in the experiment's resolved config for provenance.
style_record = {"id": card.id, "text": card.text, "sha256": card.sha256}
```

`original_task_dict` is the serialized task representation, not an upstream
Pydantic Task instance. The drivers call `styled_task` and then validate the
result with tau2's own schema (`task_from_dict`); this module alone does not.

`apply` deep-copies every nested value. The only field it writes is
`user_scenario.persona`. Neutral leaves the content exactly unchanged, including
whether the persona key was absent, null, empty or populated. Other styles require
the original persona to be absent, null or empty; existing text is rejected so
task facts cannot be silently erased. Generate each variant from the original
task, never from another variant. Invalid scenario structures fail before changes.

`Persona` is frozen. Its SHA-256 covers the exact UTF-8 ID and text, encoded as
canonical JSON (sorted keys, compact separators, no ASCII escaping). It includes
the common instructions for non-neutral styles. Save the resolved configuration
as well as the hash; a hash alone cannot reconstruct a prompt.

## Verification and outstanding review

```bash
uv run --locked pytest -q tests/test_personas.py
```

The tests use synthetic airline/retail-shaped fixtures, not official tasks or
real conversations. They check exact preservation of task facts, evaluation
criteria and unrelated metadata; independent copies; neutral control behavior;
existing-persona rejection; invalid configuration; and reproducible prompt hashes.

Before marking T1.4 complete, review ten actual simulated dialogues per style
and record their run IDs, task IDs, seeds, model revisions and card hashes. For
each, check:

- The intended style is recognizable without demographic stereotypes.
- Goals, facts, preferences and required disclosures remain unchanged.
- Names, identifiers, dates and amounts remain accurate.
- Tone alone does not trigger escalation, abandonment or false acceptance.
- Clarification requests do not invent a missing capability or a new task.

### Dialogue review record (styles v1, 4 Oct 2026)

Packet: `sample_review.py styles --run B0-test_styles-s0-20261004` (B0 policy,
simulator Qwen3-30B-A3B-Instruct-2507, eval seed 300): the same 10 test tasks
(5 airline, 5 retail) in each of the 5 non-neutral styles, 50 dialogues.
Reviewer: Claude (AI assistant), delegated by the owner.

- **Styles are recognizable but mild.** Median words per user turn: terse 22,
  non-native 21.5, impatient 32, low-tech-literacy 31, verbose 47; share of user
  turns that ask a question: low-tech-literacy 0.31, verbose 0.20, others
  ≤ 0.06. Impatient users add urgency phrases; terse users still write full
  sentences. No demographic stereotypes or accent imitation were seen.
- **Upstream temperament mixes with the card.** Some upstream task instructions
  already set a temperament (e.g. retail 94: "confident, organized, creative,
  impatient"), so a low-tech-literacy or verbose user can still sound impatient.
- **Fact drift in 2 of 50 dialogues**, independent of the agent: retail/53
  verbose invents a product model name; retail/101 low-tech-literacy invents an
  order ID. Identifiers given by the agent were otherwise repeated correctly.
- **Tone did not trigger escalation by itself** in the sample; the transfers
  seen (airline/32, airline/45, retail/56) follow agent decisions or the
  scenario.
- **Decision:** keep cards v1. Changing them now would require re-running the
  frozen B0 `test_styles` evaluation. Report RQ3 with the caveat that the style
  manipulation is mild and the simulator occasionally drifts on facts; the B0
  style gap is small (0.041, 95 % CI [0.031, 0.136]).
