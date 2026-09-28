"""Validate real parser behavior on synthetic chat-completion wire payloads."""

import copy
import json

import pytest


def completion_payload() -> dict:
    return {
        "model": "synthetic-policy",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "How can I help?"},
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 6, "total_tokens": 14},
    }


def tool_payload() -> dict:
    payload = completion_payload()
    payload["choices"][0].update(
        {
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-a",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{ "items": ["synthetic"], "n": 1 }',
                        },
                    },
                    {
                        "id": "call-b",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"items": ["second"]}',
                        },
                    },
                ],
            },
        }
    )
    return payload


def parse(payload: object):
    from handoff.agents.completions import parse_completion

    return parse_completion(
        json.dumps(payload).encode(), allowed_tools={"lookup"}, max_tool_calls=2
    )


def test_text_completion_usage_and_roundtrip() -> None:
    result = parse(completion_payload())
    assert result.model == "synthetic-policy"
    assert result.content == "How can I help?"
    assert result.tool_calls == ()
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (
        8,
        6,
        14,
    )
    assert result.to_message() == {"role": "assistant", "content": "How can I help?"}


@pytest.mark.parametrize("finish_reason", ["stop", "tool_calls"])
def test_tools_preserve_arguments_ids_and_return_independent_messages(
    finish_reason: str,
) -> None:
    payload = tool_payload()
    payload["choices"][0]["finish_reason"] = finish_reason
    result = parse(payload)
    assert [call.id for call in result.tool_calls] == ["call-a", "call-b"]
    assert result.tool_calls[0].arguments == '{ "items": ["synthetic"], "n": 1 }'
    args = result.tool_calls[0].parsed_arguments()
    assert args == {"items": ["synthetic"], "n": 1}
    args["items"].append("modified")
    assert result.tool_calls[0].parsed_arguments()["items"] == ["synthetic"]
    expected = copy.deepcopy(payload["choices"][0]["message"])
    exported = result.to_message()
    assert exported == expected
    exported["tool_calls"][0]["id"] = "mutated"
    assert result.to_message() == expected


def test_usage_absent_is_unknown_not_zero() -> None:
    payload = completion_payload()
    del payload["usage"]
    result = parse(payload)
    assert (
        result.prompt_tokens is result.completion_tokens is result.total_tokens is None
    )


@pytest.mark.parametrize(
    "mutation,code",
    [
        ("unknown_tool", "UNKNOWN_TOOL"),
        ("duplicate_id", "INVALID_TOOL_CALL"),
        ("missing_id", "INVALID_TOOL_CALL"),
        ("wrong_type", "INVALID_TOOL_CALL"),
        ("missing_function", "INVALID_TOOL_CALL"),
        ("too_many_calls", "TOO_MANY_TOOL_CALLS"),
        ("bad_role", "INVALID_MESSAGE"),
        ("empty_turn", "EMPTY_ASSISTANT"),
        ("content_object", "INVALID_MESSAGE"),
        ("multiple_choices", "INVALID_COMPLETION"),
        ("missing_choice", "INVALID_COMPLETION"),
        ("wrong_index", "INVALID_COMPLETION"),
        ("truncated", "TRUNCATED_COMPLETION"),
        ("filtered", "FILTERED_COMPLETION"),
        ("unknown_finish", "INVALID_COMPLETION"),
        ("tool_finish_without_calls", "INVALID_TOOL_CALL"),
        ("usage_bool", "INVALID_USAGE"),
        ("usage_negative", "INVALID_USAGE"),
        ("usage_inconsistent", "INVALID_USAGE"),
        ("usage_missing_field", "INVALID_USAGE"),
        ("missing_model", "INVALID_COMPLETION"),
        ("legacy_call", "INVALID_MESSAGE"),
    ],
)
def test_malformed_replies_have_bounded_safe_errors(mutation: str, code: str) -> None:
    from handoff.agents.completions import CompletionError

    payload = (
        tool_payload()
        if mutation
        in {
            "unknown_tool",
            "duplicate_id",
            "missing_id",
            "wrong_type",
            "missing_function",
            "too_many_calls",
        }
        else completion_payload()
    )
    choice = payload["choices"][0]
    message = choice["message"]
    if mutation == "unknown_tool":
        message["tool_calls"][0]["function"]["name"] = "sensitive-untrusted-tool"
    elif mutation == "duplicate_id":
        message["tool_calls"][1]["id"] = "call-a"
    elif mutation == "missing_id":
        del message["tool_calls"][0]["id"]
    elif mutation == "wrong_type":
        message["tool_calls"][0]["type"] = "code"
    elif mutation == "missing_function":
        del message["tool_calls"][0]["function"]
    elif mutation == "too_many_calls":
        message["tool_calls"].append(copy.deepcopy(message["tool_calls"][0]))
    elif mutation == "bad_role":
        message["role"] = "user"
    elif mutation == "empty_turn":
        message["content"] = " "
    elif mutation == "content_object":
        message["content"] = {"text": "sensitive-untrusted-content"}
    elif mutation == "multiple_choices":
        payload["choices"].append(copy.deepcopy(choice))
    elif mutation == "missing_choice":
        payload["choices"] = []
    elif mutation == "wrong_index":
        choice["index"] = True
    elif mutation == "truncated":
        choice["finish_reason"] = "length"
    elif mutation == "filtered":
        choice["finish_reason"] = "content_filter"
    elif mutation == "unknown_finish":
        choice["finish_reason"] = "sensitive-untrusted-finish"
    elif mutation == "tool_finish_without_calls":
        choice["finish_reason"] = "tool_calls"
    elif mutation == "usage_bool":
        payload["usage"]["prompt_tokens"] = True
    elif mutation == "usage_negative":
        payload["usage"]["completion_tokens"] = -1
    elif mutation == "usage_inconsistent":
        payload["usage"]["total_tokens"] = 999
    elif mutation == "usage_missing_field":
        del payload["usage"]["total_tokens"]
    elif mutation == "missing_model":
        del payload["model"]
    elif mutation == "legacy_call":
        message["function_call"] = {"name": "lookup", "arguments": "{}"}
    with pytest.raises(CompletionError) as error:
        parse(payload)
    assert error.value.code == code
    assert str(error.value) == code
    assert "sensitive" not in repr(error.value)


@pytest.mark.parametrize(
    "arguments",
    [
        '{"a": 1, "a": 2}',
        '{"x":{"a":1,"a":2}}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":1e999}',
        "[]",
        "null",
        '"text"',
        '{"unfinished":',
        "{} trailing",
    ],
)
def test_rejects_ambiguous_or_non_object_arguments(arguments: str) -> None:
    from handoff.agents.completions import CompletionError

    payload = tool_payload()
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        arguments
    )
    with pytest.raises(CompletionError, match="INVALID_ARGUMENTS"):
        parse(payload)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"choices": [], "choices": []}',
        b"NaN",
        b"[]",
        b"null",
        b"\xff",
        b"{malformed sensitive-content}",
    ],
)
def test_rejects_invalid_wire_json_without_echoing_content(raw: bytes) -> None:
    from handoff.agents.completions import CompletionError, parse_completion

    with pytest.raises(CompletionError, match="INVALID_JSON"):
        parse_completion(raw, allowed_tools={"lookup"}, max_tool_calls=2)


def test_limits_and_allowed_tools_configuration_is_validated() -> None:
    from handoff.agents.completions import parse_completion

    for tools, limit in [
        ("lookup", 2),
        ([None], 2),
        ({""}, 2),
        ({"lookup"}, True),
        ({"lookup"}, 0),
    ]:
        with pytest.raises(ValueError):
            parse_completion(
                json.dumps(completion_payload()).encode(),
                allowed_tools=tools,
                max_tool_calls=limit,
            )


@pytest.mark.parametrize(
    "location",
    [
        "content",
        "model",
        "call_id",
        "object_key",
        "argument_direct",
        "argument_key",
        "argument_value",
    ],
)
def test_rejects_lone_surrogates_after_json_unescaping(location: str) -> None:
    from handoff.agents.completions import CompletionError

    payload = tool_payload()
    message = payload["choices"][0]["message"]
    expected = "INVALID_JSON"
    if location == "content":
        message["content"] = "\ud800"
    elif location == "model":
        payload["model"] = "\udfff"
    elif location == "call_id":
        message["tool_calls"][0]["id"] = "\ud800"
    elif location == "object_key":
        payload["\ud800"] = "extension"
    elif location == "argument_direct":
        message["tool_calls"][0]["function"]["arguments"] = '{"x":"\ud800"}'
    elif location == "argument_key":
        message["tool_calls"][0]["function"]["arguments"] = '{"nested":{"\\ud800":1}}'
        expected = "INVALID_ARGUMENTS"
    else:
        message["tool_calls"][0]["function"]["arguments"] = '{"x":["\\udfff"]}'
        expected = "INVALID_ARGUMENTS"
    with pytest.raises(CompletionError, match=expected):
        parse(payload)


@pytest.mark.parametrize("escape", [False, True])
def test_valid_surrogate_pairs_and_unicode_remain_utf8_serializable(
    escape: bool,
) -> None:
    payload = tool_payload()
    message = payload["choices"][0]["message"]
    message["content"] = "查詢 😀"
    arguments = json.dumps({"nested": {"😀": ["𐀀", "中文"]}}, ensure_ascii=escape)
    message["tool_calls"][0]["function"]["arguments"] = arguments
    result = parse(payload)
    assert result.tool_calls[0].arguments == arguments
    assert result.tool_calls[0].parsed_arguments() == {"nested": {"😀": ["𐀀", "中文"]}}
    encoded = json.dumps(result.to_message(), ensure_ascii=False).encode("utf-8")
    assert json.loads(encoded) == message
