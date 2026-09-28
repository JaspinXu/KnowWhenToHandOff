# Model response parser

`handoff.agents.completions` implements the dependency-free response boundary
needed by T0.5. It does not send model requests, execute tools, register a tau2
agent, calculate a reward or perform an experiment. HTTPX and the async transport
remain deferred by the owner in ADR-007.

## Supported input

The input is the UTF-8 bytes of one nonstreaming chat-completion response, with
one indexed choice, an assistant message and a nonempty model name. Text replies
and multiple function calls are supported. Streaming deltas, legacy
`function_call`, multimodal content, and provider-specific reasoning fields are
not supported by the exported message interface.

This boundary follows vLLM's
[chat tool-calling protocol](https://docs.vllm.ai/en/latest/features/tool_calling/).
It accepts `stop` with function calls as well as `tool_calls`; a `tool_calls`
finish reason without any call is rejected. No vLLM version is selected here.

```python
from handoff.agents.completions import CompletionError, parse_completion

# raw_bytes comes from a future bounded transport or a recorded response file.
# allowed_names and max_calls come from the environment/configuration.
try:
    turn = parse_completion(
        raw_bytes, allowed_tools=allowed_names, max_tool_calls=max_calls
    )
except CompletionError as error:
    # Record only the bounded code, never the raw model response or exception locals.
    failure_code = error.code
else:
    next_history_message = turn.to_message()
    for call in turn.tool_calls:
        arguments = call.parsed_arguments()
        # The environment must validate its tool schema and policy before execution.
```

Call IDs and JSON argument strings are preserved exactly. Parsed arguments and
exported messages are independent copies. Records are frozen. Usage fields are
nonnegative integer counts with a consistent total; omitted/null usage is unknown
(`None`), not zero. The `model` value is the server's claim; this parser does not
verify a model revision or assert it matches the requested server.

## Failures

`CompletionError` exposes only one of these parser-generated codes:

| Code | Meaning |
| --- | --- |
| `INVALID_JSON` | Invalid UTF-8/object JSON, lone Unicode surrogates, duplicate keys or nonfinite numbers |
| `INVALID_COMPLETION` | Invalid envelope, choice/index/model or unsupported finish reason |
| `TRUNCATED_COMPLETION` | Generation ended on a length limit |
| `FILTERED_COMPLETION` | Provider returned a content-filter finish reason |
| `INVALID_MESSAGE` | Wrong role/content type or unsupported legacy call |
| `EMPTY_ASSISTANT` | Neither usable text nor a tool call |
| `INVALID_TOOL_CALL` | Missing, repeated or malformed call IDs/structure |
| `TOO_MANY_TOOL_CALLS` | More calls than the caller's configured bound |
| `UNKNOWN_TOOL` | Function name is absent from the environment's allowlist |
| `INVALID_ARGUMENTS` | Arguments are not unambiguous finite JSON objects |
| `INVALID_USAGE` | Incomplete, negative, noninteger or inconsistent token counts |

Invalid parser configuration raises `ValueError` before parsing. The parser does
not repair/truncate arguments, choose a different tool or partially execute a
valid prefix of a malformed call list. Duplicate JSON keys and lone Unicode
surrogates are rejected at every nesting level, including decoded arguments.
Valid non-BMP characters and paired surrogate escapes remain supported.
These errors are technical diagnostics, not definitions of a
format penalty, policy violation, task failure or hand-off label. Diagnostic
messages contain only codes; Python exception contexts/traceback locals may still
reference input data. Record the code rather than serializing exception objects.

The eventual transport must bound response bytes before buffering and parsing.
The tool environment must validate parameter types/required fields against its
own schema and apply its policy rules. A recognized tool name and JSON object
alone are not authorization to execute the call. Chat text must never be
retokenized to stand in for the sampled IDs needed by RL.

## Verification

```bash
uv run --locked pytest -q tests/test_completions.py
```

Tests use synthetic protocol payloads and require neither sockets nor model
weights. They verify parsing and round trips, not endpoint availability, tau2
integration or model quality.
