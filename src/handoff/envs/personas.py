"""Configure communication styles and inject them into serialized task copies."""

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

_STYLES = (
    "neutral",
    "terse",
    "verbose",
    "impatient",
    "non-native",
    "low-tech-literacy",
)


def _resolved_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value.strip() != "???"
        and "${" not in value
    )


@dataclass(frozen=True)
class Persona:
    """An immutable style ID and exact simulator prompt; neutral adds no text."""

    id: str
    text: str

    def __post_init__(self) -> None:
        if self.id not in _STYLES:
            raise ValueError("persona ID must be one of the six README styles")
        if self.id == "neutral":
            if self.text != "":
                raise ValueError("neutral persona must have empty text")
        elif not _resolved_text(self.text):
            raise ValueError("persona text must be nonempty and fully resolved")

    @property
    def sha256(self) -> str:
        """Hash the exact ID and text for recording in a resolved run config."""
        payload = json.dumps(
            {"id": self.id, "text": self.text},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def personas_from_config(config: Mapping[str, object]) -> dict[str, Persona]:
    """Validate a resolved v1 Hydra style library and build its immutable cards.

    This pure function accepts a plain mapping. Resolve Hydra interpolations and
    missing values before calling it; it neither reads files nor selects styles.
    """
    if not isinstance(config, Mapping) or set(config) != {
        "version",
        "guardrails",
        "cards",
    }:
        raise ValueError("persona config has missing or unknown fields")
    if type(config["version"]) is not int or config["version"] != 1:
        raise ValueError("persona library version must be 1")
    guardrails = config["guardrails"]
    if not _resolved_text(guardrails):
        raise ValueError("persona guardrails must be nonempty and fully resolved")
    cards = config["cards"]
    if not isinstance(cards, Mapping) or set(cards) != set(_STYLES):
        raise ValueError("persona cards must contain exactly the six README styles")
    result: dict[str, Persona] = {}
    for name in _STYLES:
        # Validate the individual card before adding common text; otherwise an
        # empty style could be hidden by the nonempty common instructions.
        card = Persona(name, cards[name])
        result[name] = (
            card if name == "neutral" else Persona(name, f"{guardrails}\n\n{card.text}")
        )
    return result


class PersonaInjector:
    """Change only user_scenario.persona on a deep copy of a serialized task."""

    def apply(self, task: dict[str, object], persona: Persona) -> dict[str, object]:
        """Apply one style to an original task, preserving all other fields.

        Neutral returns an unchanged deep copy. Other styles reject existing
        persona text to avoid erasing facts or silently mixing styles. A pinned
        benchmark adapter must reconstruct and validate its own upstream Task.
        """
        if not isinstance(persona, Persona):
            raise ValueError("persona must be a validated Persona")
        if not isinstance(task, dict):
            raise ValueError("task must be a serialized task dictionary")
        scenario = task.get("user_scenario")
        if not isinstance(scenario, dict):
            raise ValueError("task must contain a user_scenario dictionary")
        existing = scenario.get("persona")
        if existing is not None and not isinstance(existing, str):
            raise ValueError("existing persona must be text or null")
        if persona.id != "neutral" and existing not in (None, ""):
            raise ValueError(
                "cannot overwrite an existing persona; use the original task"
            )
        result = deepcopy(task)
        if persona.id != "neutral":
            # deepcopy preserves shared references. Detach this branch so task
            # builders reusing it elsewhere cannot change unrelated fields.
            styled_scenario = deepcopy(scenario)
            styled_scenario["persona"] = persona.text
            result["user_scenario"] = styled_scenario
        return result
