"""Labels (T1.2), frozen splits (T1.1) and the hand-off task set (T1.3)."""

import copy

import pytest
from fakes import COMMIT, build_all, synthetic_task

from handoff.envs.handoff_tasks import (
    EXPLICIT_REQUEST_TEMPLATES,
    HARD_NEGATIVE_TEMPLATES,
    OUT_OF_SCOPE_TEMPLATES,
    derived_task_id,
    transfer_criteria,
)
from handoff.envs.labels import (
    build_label_file,
    kind_label,
    upstream_label,
    validate_label_file,
)
from handoff.envs.registry import (
    TaskRef,
    TaskRegistry,
    build_split_manifest,
    carve_dev,
    validate_handoff_manifest,
    validate_split_manifest,
)


@pytest.fixture
def domain_data():
    tasks = [synthetic_task(str(i), transfer=(i in (3, 13))) for i in range(20)]
    splits = {
        "train": [str(i) for i in range(12)],
        "test": [str(i) for i in range(12, 20)],
        "base": [str(i) for i in range(20)],
    }
    return tasks, splits


def test_upstream_and_kind_labels():
    assert upstream_label(synthetic_task("1")) == (False, "no_transfer")
    assert upstream_label(synthetic_task("1", transfer=True)) == (
        True,
        "gold_transfer_action",
    )
    assert kind_label("out_of_scope") == (True, "out_of_scope")
    assert kind_label("hard_negative") == (False, "hard_negative")
    with pytest.raises(ValueError):
        kind_label("upstream")


def test_label_file_hash_and_overrides(domain_data):
    tasks, _ = domain_data
    label_file = build_label_file(
        {"airline": tasks},
        tau2_commit=COMMIT,
        overrides={"airline": {"5": {"should_transfer": True, "reason": "audited"}}},
    )
    validate_label_file(label_file)
    assert label_file["domains"]["airline"]["labels"]["5"]["rule"] == "audited_override"
    assert label_file["domains"]["airline"]["n_positive"] == 3
    tampered = copy.deepcopy(label_file)
    tampered["domains"]["airline"]["labels"]["1"]["should_transfer"] = True
    with pytest.raises(ValueError, match="hash"):
        validate_label_file(tampered)
    with pytest.raises(ValueError, match="reason"):
        build_label_file(
            {"airline": tasks},
            tau2_commit=COMMIT,
            overrides={"airline": {"5": {"should_transfer": True, "reason": " "}}},
        )


def test_carve_dev_is_deterministic_stratified_and_disjoint():
    train_ids = [str(i) for i in range(30)]
    labels = {"3": True, "7": True, "9": True, "11": True}
    train, dev = carve_dev(train_ids, fraction=0.2, seed=5, labels=labels)
    assert (train, dev) == carve_dev(train_ids, fraction=0.2, seed=5, labels=labels)
    assert set(train).isdisjoint(dev) and set(train) | set(dev) == set(train_ids)
    assert len(dev) == round(0.2 * 26) + round(0.2 * 4)
    assert sum(labels.get(t, False) for t in dev) == 1
    assert carve_dev(train_ids, fraction=0.2, seed=6, labels=labels)[1] != dev
    with pytest.raises(ValueError):
        carve_dev(["1", "1"], fraction=0.2, seed=1)


def test_split_manifest_is_disjoint_and_hash_checked(domain_data):
    _, _, manifest, _, _ = build_all(domain_data)
    validate_split_manifest(manifest)
    splits = manifest["domains"]["airline"]["splits"]
    assert all(
        set(splits[a]).isdisjoint(splits[b]) for a in splits for b in splits if a != b
    )
    assert manifest["domains"]["airline"]["counts"]["test"] == 8
    broken = copy.deepcopy(manifest)
    broken["domains"]["airline"]["splits"]["dev"].append(splits["test"][0])
    with pytest.raises(ValueError):
        validate_split_manifest(broken)


def test_official_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        build_split_manifest(
            {"airline": {"official_splits": {"train": ["1"], "test": ["1"]}}},
            dev_fraction=0.2,
            seed=1,
            tau2_commit=COMMIT,
            labels_sha256="x",
        )


def test_handoff_variants_inherit_split_and_labels(domain_data):
    _, labels, manifest, pairs, handoff = build_all(domain_data)
    validate_handoff_manifest(handoff, manifest)
    split_of = {
        tid: name
        for name, ids in manifest["domains"]["airline"]["splits"].items()
        for tid in ids
    }
    for task, ref in pairs:
        assert ref.split == split_of[ref.source_task_id]
        assert task["id"] == ref.task_id
        if ref.kind == "hard_negative":
            assert not ref.should_transfer
            assert not labels[("airline", ref.source_task_id)]
            assert task["evaluation_criteria"]["reward_basis"] == ["DB", "COMMUNICATE"]
            assert (
                "never ask to speak with a human"
                in task["user_scenario"]["instructions"]["task_instructions"]
            )
        else:
            assert ref.should_transfer
            assert task["evaluation_criteria"] == transfer_criteria()
        assert (
            task["initial_state"] == synthetic_task(ref.source_task_id)["initial_state"]
        )
        assert task["user_scenario"]["persona"] is None
    ids = [task["id"] for task, _ in pairs]
    assert len(ids) == len(set(ids))
    registry = TaskRegistry(manifest, labels, handoff)
    registry.assert_lineage_disjoint()


def test_generation_counts_templates_and_determinism(domain_data):
    first = build_all(domain_data)[3]
    second = build_all(domain_data)[3]
    assert [t for t, _ in first] == [t for t, _ in second]
    test_refs = [r for _, r in first if r.split == "test"]
    sources = {r.source_task_id for r in test_refs}
    assert sum(r.kind == "explicit_request" for r in test_refs) == len(sources)
    assert sum(r.kind == "hard_negative" for r in test_refs) == len(sources) - 1
    used = {r.template_id for _, r in first}
    assert {t.template_id for t in HARD_NEGATIVE_TEMPLATES} <= used
    assert {t.template_id for t in OUT_OF_SCOPE_TEMPLATES["airline"]} <= used
    assert {t.template_id for t in EXPLICIT_REQUEST_TEMPLATES} <= used
    per_source = {}
    for _, ref in first:
        if ref.kind == "out_of_scope" and ref.split == "train":
            per_source.setdefault(ref.source_task_id, set()).add(ref.template_id)
    assert all(len(v) == 2 for v in per_source.values())


def test_positive_variant_keeps_identity_and_drops_original_goal(domain_data):
    pairs = build_all(domain_data)[3]
    task, ref = next((t, r) for t, r in pairs if r.kind == "out_of_scope")
    instructions = task["user_scenario"]["instructions"]
    assert instructions["known_info"] == "You are Ada Lovelace, user id ada_1."
    assert instructions["unknown_info"] is None
    assert "Cancel reservation" not in instructions["reason_for_call"]
    explicit = next(t for t, r in pairs if r.kind == "explicit_request")
    assert (
        "Cancel reservation"
        in explicit["user_scenario"]["instructions"]["reason_for_call"]
    )
    assert derived_task_id("12", "out_of_scope", 1) == "12__oos1"


def test_registry_load_filters_by_split_and_kind(domain_data):
    _, labels, manifest, _, handoff = build_all(domain_data)
    registry = TaskRegistry(manifest, labels, handoff)
    upstream_test = registry.load("airline", "test")
    assert {r.kind for r in upstream_test} == {"upstream"}
    everything = registry.load("airline", "test", include_handoff_set=True)
    assert len(everything) > len(upstream_test)
    only_oos = registry.load("airline", "test", True, kinds=["out_of_scope"])
    assert {r.kind for r in only_oos} == {"out_of_scope"}
    assert registry.get("airline", "13").should_transfer
    with pytest.raises(KeyError):
        registry.get("airline", "999")
    with pytest.raises(ValueError):
        registry.load("airline", "validation")


def test_registry_rejects_split_mismatch_and_missing_labels(domain_data):
    _, labels, manifest, _, handoff = build_all(domain_data)
    broken = copy.deepcopy(handoff)
    task = broken["domains"]["airline"]["tasks"][0]
    task["split"] = "train" if task["split"] != "train" else "test"
    with pytest.raises(ValueError):
        TaskRegistry(manifest, labels, broken)
    partial = dict(labels)
    partial.pop(("airline", "0"))
    with pytest.raises(ValueError, match="label"):
        TaskRegistry(manifest, partial)


def test_task_ref_validation():
    with pytest.raises(ValueError):
        TaskRef("airline", "1", "validation", "1")
    with pytest.raises(ValueError):
        TaskRef("airline", "1__xr0", "test", "2")
    assert TaskRef("airline", "1__xr0", "test", "1", kind="explicit_request").key == (
        "airline",
        "1__xr0",
    )
