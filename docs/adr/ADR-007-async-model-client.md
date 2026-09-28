# ADR-007: Async model transport for baseline and simulator clients

- Status: Deferred by owner — dependency not accepted
- Date: 2026-09-28
- Related tasks: T0.3, T0.5; no change to ADR-006

## Problem and scope

Owner response on 28 September: do not add dependencies; complete only the pure
parser. HTTPX and the network client/smoke CLI below remain a proposal.

The repository cannot yet send a model request. Baseline and simulator integration
need a reusable, asynchronous client for the local vLLM chat-completion endpoint.
Implementing a new HTTP stack or blocking rollouts on synchronous I/O is unnecessary.

## Proposed dependency decision

Add exactly `httpx==0.28.1` as a runtime dependency and lock its transitive
dependencies with uv. No OpenAI SDK, trainer, benchmark or serving runtime is
added. HTTPX supplies asynchronous requests and in-process test transports.

Sources checked on 28 September:
[release metadata](https://pypi.org/project/httpx/0.28.1/),
[async client](https://www.python-httpx.org/async/),
[mock transports](https://www.python-httpx.org/advanced/transports/).

## Concrete implementation

- Pure response parsing in `src/handoff/agents/completions.py` can be implemented
  before dependency approval using only the standard library. It validates a
  single completed assistant turn, known function names, unique call IDs, JSON
  object arguments and token-usage fields. It returns immutable local client
  records and preserves original argument strings. It never executes a tool.
- `src/handoff/agents/client.py` owns an async context-managed HTTPX client with
  explicit endpoint, served model, timeout, response byte limit, concurrency and
  sampling settings. All values come from a resolved Hydra client config.
- Credentials are read from a named environment variable at the CLI boundary and
  passed in an Authorization header. They are not included in configuration,
  exception messages or persisted responses. Ignore proxy/environment settings
  and reject redirects so requests cannot silently move to another endpoint.
- No automatic generation retries: callers see bounded error codes and decide
  how to handle failed episodes. Never print server bodies or raw exceptions.
- A thin smoke CLI sends a synthetic tool-calling request to an explicitly
  configured endpoint and validates the returned call. It runs only when invoked,
  reports infrastructure checks, and never claims a benchmark score or throughput.
- CPU tests use HTTPX MockTransport; no network/GPU/model downloads. Cover
  malformed responses, HTTP failures, timeouts, cancellation, response size bounds,
  credential handling, concurrency limits and reproducible request parameters.

## Boundaries

This adds client-local return types, not changes to Trajectory, RewardBreakdown,
TaskRegistry, labels, splits, metrics, reward rules or the simulator selection.
It does not pin tau2 or vLLM. Registering the client with tau2 waits for the
benchmark decision. Chat text is not a source of sampled token IDs: the RL loop
still needs its own token-in/token-out interface.

## Acceptance

Owner approval is required only for the new HTTPX dependency under AGENTS section
11. Until approval, implement/test the pure parser and leave the transport pending.
