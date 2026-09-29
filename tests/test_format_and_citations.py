"""Format-error detection and rule-based citation events."""

from fakes import call, conversation, result, say

from handoff.eval.citations import citation_events
from handoff.rewards.format import find_format_errors


def kinds(messages, **kwargs):
    return [e.kind for e in find_format_errors(messages, **kwargs)]


def test_clean_conversation_has_no_format_error():
    messages = conversation(
        say("user", "hi"),
        say("assistant", None, call("c1", "get_user_details", user_id="u")),
        result("c1"),
        say("assistant", "Hello"),
    )
    assert kinds(messages, known_tools={"get_user_details"}) == []


def test_unparsed_tool_call_and_empty_message():
    messages = conversation(
        say("user", "hi"),
        say("assistant", '<tool_call>{"name": "x", "arguments": {}'),
        say("assistant", ""),
    )
    assert kinds(messages) == ["unparsed_tool_call", "empty_message"]


def test_unknown_tool_counted_once():
    messages = conversation(
        say("assistant", None, call("c1", "fly_me", to="moon")),
        result("c1", "Error: Tool 'fly_me' not found.", error=True),
    )
    assert kinds(messages, known_tools={"get_user_details"}) == ["unknown_tool"]
    assert kinds(messages) == ["unknown_tool"]


def test_argument_errors_but_not_business_errors():
    messages = conversation(
        say("assistant", None, call("c1", "get_user_details", uid="u")),
        result(
            "c1",
            "Error: get_user_details() got an unexpected keyword argument 'uid'",
            error=True,
        ),
        say("assistant", None, call("c2", "get_user_details", user_id="nope")),
        result("c2", "Error: User not found", error=True),
    )
    assert kinds(messages) == ["invalid_arguments"]


def test_protocol_error_from_termination_can_be_suppressed():
    messages = conversation(say("user", "hi"))
    assert kinds(messages, termination_reason="agent_error") == ["protocol_error"]
    assert (
        kinds(messages, termination_reason="agent_error", count_protocol_error=False)
        == []
    )


def test_handoff_citation_window():
    messages = conversation(
        say("user", "I need wheelchair assistance."),
        say(
            "assistant", "Per our policy I cannot arrange that, so I will transfer you."
        ),
        say("assistant", None, call("t", "transfer_to_human_agents", summary="s")),
        result("t", "Transfer successful"),
        say("assistant", "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."),
    )
    events = citation_events(messages)
    assert [(e.kind, e.cited) for e in events] == [("handoff", True)]


def test_refusals_with_and_without_policy_reference():
    messages = conversation(
        say("user", "Refund my basic economy ticket."),
        say("assistant", "Unfortunately I cannot refund it."),
        say("user", "Why?"),
        say("assistant", "I'm afraid our policy does not allow refunds for this fare."),
        say("assistant", "Anything else I can help with?"),
    )
    assert [(e.kind, e.cited) for e in citation_events(messages)] == [
        ("refusal", False),
        ("refusal", True),
    ]
