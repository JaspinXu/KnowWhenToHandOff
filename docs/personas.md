# Communication-style library

T1.4 now has configurable draft cards and a pure, CPU-tested injector for
serialized task dictionaries. No upstream benchmark version or task split is
selected. The live Task adapter and ten-dialogue-per-style review remain pending.
There is no evidence yet that a simulator consistently follows these prompts.

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
Pydantic Task instance. The future benchmark adapter must serialize and validate
its Task with the accepted version's API. This module does not validate the whole
benchmark schema or claim compatibility with a currently unaccepted pin.

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

No dialogue audit has been performed. Do not infer fairness or task success from
the CPU invariance tests, or change a frozen protocol based on these drafts.
