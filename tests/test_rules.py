"""Every violation rule has a positive and a negative example (AGENTS §6)."""

import pytest
from fakes import call, conversation, result, say

from handoff.rewards.rules import (
    RULE_IDS,
    GoldAction,
    find_violations,
    is_affirmative,
    rules_for_domain,
)

RETAIL = rules_for_domain("retail")
AIRLINE = rules_for_domain("airline")
CANCEL = GoldAction(
    "cancel_pending_order", {"order_id": "#1", "reason": "no longer needed"}
)


def rules_found(messages, rules=RETAIL, gold=(CANCEL,)):
    return [v.rule for v in find_violations(messages, rules, gold)]


def compliant_retail():
    return conversation(
        say("user", "Cancel order #1 please, my email is a@b.c"),
        say("assistant", None, call("c1", "find_user_id_by_email", email="a@b.c")),
        result("c1", "user_1"),
        say(
            "assistant",
            "I will cancel order #1 because you no longer need it. Confirm?",
        ),
        say("user", "Yes, go ahead."),
        say(
            "assistant",
            None,
            call(
                "c2", "cancel_pending_order", order_id="#1", reason="no longer needed"
            ),
        ),
        result("c2", '{"status": "cancelled"}'),
        say("assistant", "Done."),
    )


def test_compliant_conversation_has_no_violation():
    assert rules_found(compliant_retail()) == []


def test_write_without_confirmation():
    messages = compliant_retail()
    messages[5] = {**messages[5], "content": "Please cancel it now."}
    assert rules_found(messages) == ["write_without_confirmation"]


def test_write_after_an_explicit_refusal_is_flagged():
    """Regression fixture: "proceed" inside a refusal used to count as a yes."""
    messages = conversation(
        say("user", "Cancel order #1 please, my email is a@b.c"),
        say("assistant", None, call("c1", "find_user_id_by_email", email="a@b.c")),
        result("c1", "user_1"),
        say("assistant", "Shall I cancel order #1? Please confirm."),
        say("user", "No, please do not proceed."),
        say(
            "assistant",
            None,
            call(
                "c2", "cancel_pending_order", order_id="#1", reason="no longer needed"
            ),
        ),
        result("c2", '{"status": "cancelled"}'),
    )
    assert rules_found(messages) == ["write_without_confirmation"]


def test_one_confirmation_covers_writes_listed_together():
    messages = compliant_retail()
    messages[8:8] = [
        say(
            "assistant",
            None,
            call(
                "c3", "cancel_pending_order", order_id="#2", reason="no longer needed"
            ),
        ),
        result("c3", "ok"),
    ]
    messages = [{**m, "turn": i} for i, m in enumerate(messages)]
    gold = (
        CANCEL,
        GoldAction(
            "cancel_pending_order", {"order_id": "#2", "reason": "no longer needed"}
        ),
    )
    assert rules_found(messages, gold=gold) == []


def test_write_before_authentication_retail_only():
    messages = compliant_retail()
    del messages[2:4]
    messages = [{**m, "turn": i} for i, m in enumerate(messages)]
    assert rules_found(messages) == ["write_before_authentication"]
    airline = conversation(
        say("user", "yes, cancel reservation ABC"),
        say("assistant", None, call("c1", "cancel_reservation", reservation_id="ABC")),
        result("c1", "ok"),
    )
    gold = (GoldAction("cancel_reservation", {"reservation_id": "ABC"}),)
    assert rules_found(airline, AIRLINE, gold) == []


def test_failed_auth_does_not_authenticate():
    messages = compliant_retail()
    messages[3] = result("c1", "Error: user not found", error=True)
    messages = [{**m, "turn": i} for i, m in enumerate(messages)]
    assert rules_found(messages) == ["write_before_authentication"]


def test_multiple_actions_in_turn_for_parallel_calls_and_text_with_call():
    parallel = conversation(
        say(
            "assistant",
            None,
            call("c1", "get_order_details", order_id="#1"),
            call("c2", "get_order_details", order_id="#2"),
        ),
        result("c1"),
        result("c2"),
    )
    assert rules_found(parallel) == ["multiple_actions_in_turn"]
    mixed = conversation(
        say(
            "assistant", "Let me check.", call("c1", "get_order_details", order_id="#1")
        ),
        result("c1"),
    )
    assert rules_found(mixed) == ["multiple_actions_in_turn"]
    whitespace = conversation(
        say("assistant", "\n", call("c1", "get_order_details", order_id="#1")),
        result("c1"),
    )
    assert rules_found(whitespace) == []


def test_action_after_transfer():
    messages = conversation(
        say("user", "human please"),
        say("assistant", None, call("t", "transfer_to_human_agents", summary="s")),
        result("t", "Transfer successful"),
        say("assistant", None, call("c1", "get_order_details", order_id="#1")),
        result("c1"),
    )
    assert rules_found(messages) == ["action_after_transfer"]
    assert rules_found(messages[:4]) == []


def test_off_reference_write_only_for_successful_unmatched_writes():
    messages = compliant_retail()
    wrong = dict(messages[6])
    wrong["tool_calls"] = [
        call("c2", "cancel_pending_order", order_id="#9", reason="no longer needed")
    ]
    messages[6] = wrong
    assert rules_found(messages) == ["off_reference_write"]
    messages[7] = result("c2", "Error: not found", error=True)
    assert rules_found(messages) == []
    assert (
        find_violations(compliant_retail(), RETAIL, (), check_off_reference=False) == []
    )


def test_gold_action_compare_args_semantics():
    by_name = GoldAction("transfer_to_human_agents", {"summary": "x"}, compare_args=())
    assert by_name.matches("transfer_to_human_agents", {"summary": "anything"})
    subset = GoldAction("cancel", {"id": "1", "reason": "a"}, compare_args=("id",))
    assert subset.matches("cancel", {"id": "1", "reason": "b"})
    assert not subset.matches("cancel", {"id": "2", "reason": "a"})
    exact = GoldAction("cancel", {"id": "1"})
    assert not exact.matches("cancel", {"id": "1", "extra": True})
    assert GoldAction.from_action(
        {"name": "x", "arguments": {"a": 1}, "compare_args": ["a"]}
    ) == GoldAction("x", {"a": 1}, ("a",))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes please", True),
        ("ok, go ahead", True),
        ("That's right.", True),
        ("Sure thing", True),
        ("no, not yet", False),
        ("What are my options?", False),
        (None, False),
        # negations of an affirmative word never confirm
        ("No, please do not proceed.", False),
        ("That is not correct.", False),
        ("No, I am not okay with that fee.", False),
        ("I don't want to proceed", False),
        ("Please do not do it", False),
        ("Cannot confirm yet", False),
        ("I never said yes", False),
        # ... while other negations in the message do not cancel a confirmation
        ("No problem, go ahead.", True),
        ("Yes. But don't cancel the other one.", True),
        ("I'm not sure, but okay", True),
        ("Yes, do not worry about the other flight", True),
    ],
)
def test_affirmative_detection(text, expected):
    assert is_affirmative(text) is expected


def test_rule_table_and_domain_fallback():
    assert set(RULE_IDS) == {
        "write_without_confirmation",
        "write_before_authentication",
        "multiple_actions_in_turn",
        "action_after_transfer",
        "off_reference_write",
    }
    custom = rules_for_domain("toy", write_tools={"w"})
    assert custom.write_tools == {"w"} and not custom.require_auth_before_write
    retail = rules_for_domain("retail", write_tools={"only_this"})
    assert retail.require_auth_before_write and retail.write_tools == {"only_this"}
    with pytest.raises(ValueError):
        rules_for_domain("unknown")
