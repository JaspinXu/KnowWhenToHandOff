"""Pin style-only task changes using synthetic tasks, without upstream data."""

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
STYLES = ("neutral", "terse", "verbose", "impatient", "non-native", "low-tech-literacy")


def style_config() -> dict:
    return {
        "version": 1,
        "guardrails": "Change expression only; preserve task facts and requirements.",
        "cards": {
            style: "" if style == "neutral" else f"Style: {style}." for style in STYLES
        },
    }


def synthetic_task(domain: str) -> dict:
    return {
        "id": f"synthetic-{domain}",
        "user_scenario": {
            "persona": None,
            "instructions": {
                "task_instructions": "Follow the synthetic task.",
                "facts": [1, 2],
            },
        },
        "initial_state": {
            "initialization_actions": [{"name": "fixture", "arguments": {"a": 1}}]
        },
        "evaluation_criteria": {
            "actions": [{"name": "synthetic_action", "arguments": {"id": "x"}}]
        },
        "metadata": {"domain": domain, "should_transfer": False},
    }


@pytest.mark.parametrize("domain", ["airline", "retail"])
@pytest.mark.parametrize("style", STYLES)
def test_only_persona_changes_and_nested_input_is_preserved(
    domain: str, style: str
) -> None:
    from handoff.envs.personas import PersonaInjector, personas_from_config

    task = synthetic_task(domain)
    original = copy.deepcopy(task)
    card = personas_from_config(style_config())[style]
    result = PersonaInjector().apply(task, card)
    assert task == original
    if style == "neutral":
        assert result == original
    else:
        assert result["user_scenario"]["persona"] == card.text
        restored = copy.deepcopy(result)
        restored["user_scenario"]["persona"] = None
        assert restored == original
    result["user_scenario"]["instructions"]["facts"].append(3)
    result["evaluation_criteria"]["actions"][0]["arguments"]["id"] = "changed"
    assert task == original


@pytest.mark.parametrize(
    "existing", [None, "", "The original persona includes task facts."]
)
def test_neutral_preserves_existing_persona(existing: str | None) -> None:
    from handoff.envs.personas import Persona, PersonaInjector

    task = synthetic_task("airline")
    task["user_scenario"]["persona"] = existing
    assert PersonaInjector().apply(task, Persona("neutral", "")) == task


def test_absent_persona_and_independent_variants() -> None:
    from handoff.envs.personas import PersonaInjector, personas_from_config

    task = synthetic_task("retail")
    del task["user_scenario"]["persona"]
    cards = personas_from_config(style_config())
    injector = PersonaInjector()
    assert injector.apply(task, cards["neutral"]) == task
    terse = injector.apply(task, cards["terse"])
    verbose = injector.apply(task, cards["verbose"])
    assert "persona" not in task["user_scenario"]
    assert terse["user_scenario"]["persona"] == cards["terse"].text
    assert verbose["user_scenario"]["persona"] == cards["verbose"].text
    with pytest.raises(ValueError, match="existing persona"):
        injector.apply(terse, cards["verbose"])


def test_shared_scenario_reference_does_not_change_unrelated_task_fields() -> None:
    from handoff.envs.personas import Persona, PersonaInjector

    task = synthetic_task("retail")
    # Task builders can reuse a dictionary even though JSON serialization would
    # represent the values as separate objects. Injection must remain path-local.
    task["metadata"]["scenario_snapshot"] = task["user_scenario"]
    original = copy.deepcopy(task)
    result = PersonaInjector().apply(task, Persona("terse", "Brief phrasing."))
    assert result["metadata"] == original["metadata"]
    assert result["user_scenario"]["persona"] == "Brief phrasing."
    assert task == original


@pytest.mark.parametrize("existing", ["Contains facts.", "  ", 12, {}, []])
def test_rejects_replacing_existing_persona(existing: object) -> None:
    from handoff.envs.personas import Persona, PersonaInjector

    task = synthetic_task("airline")
    task["user_scenario"]["persona"] = existing
    original = copy.deepcopy(task)
    with pytest.raises(ValueError, match="persona"):
        PersonaInjector().apply(task, Persona("terse", "Brief phrasing."))
    assert task == original


@pytest.mark.parametrize(
    "task", [None, [], {}, {"user_scenario": None}, {"user_scenario": []}]
)
def test_rejects_invalid_task_structure(task: object) -> None:
    from handoff.envs.personas import Persona, PersonaInjector

    with pytest.raises(ValueError, match="task|user_scenario"):
        PersonaInjector().apply(task, Persona("neutral", ""))


@pytest.mark.parametrize(
    "style,text",
    [
        ("unknown", "Text"),
        ("neutral", "Added instructions"),
        ("terse", " "),
        ("verbose", None),
        ("impatient", "${unresolved}"),
        ("terse", "???"),
    ],
)
def test_rejects_invalid_persona(style: str, text: object) -> None:
    from handoff.envs.personas import Persona

    with pytest.raises(ValueError):
        Persona(style, text)


def test_card_is_frozen_and_hash_tracks_exact_content() -> None:
    from handoff.envs.personas import Persona

    card = Persona("terse", "Brief phrasing.")
    expected = json.dumps(
        {"id": card.id, "text": card.text},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert card.sha256 == hashlib.sha256(expected.encode("utf-8")).hexdigest()
    assert Persona("terse", card.text).sha256 == card.sha256
    assert Persona("verbose", card.text).sha256 != card.sha256
    assert Persona("terse", card.text + " More.").sha256 != card.sha256
    with pytest.raises(dataclasses.FrozenInstanceError):
        card.text = "changed"


@pytest.mark.parametrize(
    "change",
    [
        "missing_style",
        "unknown_style",
        "blank_style",
        "neutral_text",
        "version",
        "bool_version",
        "unknown_key",
        "blank_guardrails",
        "unresolved",
        "non_mapping",
    ],
)
def test_rejects_invalid_library(change: str) -> None:
    from handoff.envs.personas import personas_from_config

    config = style_config()
    if change == "missing_style":
        del config["cards"]["terse"]
    elif change == "unknown_style":
        config["cards"]["new-style"] = "Text"
    elif change == "blank_style":
        config["cards"]["verbose"] = " "
    elif change == "neutral_text":
        config["cards"]["neutral"] = "Override the control"
    elif change == "version":
        config["version"] = 2
    elif change == "bool_version":
        config["version"] = True
    elif change == "unknown_key":
        config["facts"] = "extra"
    elif change == "blank_guardrails":
        config["guardrails"] = " "
    elif change == "unresolved":
        config["guardrails"] = "${unresolved}"
    elif change == "non_mapping":
        config["cards"] = []
    with pytest.raises(ValueError):
        personas_from_config(config)


def test_hydra_library_loads_outside_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from handoff.envs.personas import personas_from_config

    monkeypatch.chdir(tmp_path)
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="personas")
    raw = OmegaConf.to_container(cfg.persona, resolve=True, throw_on_missing=True)
    cards = personas_from_config(raw)
    assert set(cards) == set(STYLES)
    assert cards["neutral"].text == ""
    assert len({card.sha256 for card in cards.values()}) == 6
    for name in STYLES[1:]:
        assert cards[name].text.startswith(raw["guardrails"])
        assert raw["cards"][name] in cards[name].text
