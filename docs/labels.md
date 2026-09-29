# Labels, policy rules and format errors

Deterministic definitions used by the reward (ADR-004) and the metrics
(metrics.md). All are pure functions over the normalized `Trajectory` messages,
tested with positive and negative examples (`tests/test_rules.py`,
`tests/test_format_and_citations.py`, `tests/test_task_sets.py`).

## Should-transfer label y (T1.2)

Both domain policies say: *transfer the user to a human agent if and only if the
request cannot be handled within the scope of your actions*; the
`transfer_to_human_agents` tool adds *or if the user explicitly asks for a human
agent*.

| Task | y | Rule id |
| --- | --- | --- |
| Upstream task whose reference solution calls `transfer_to_human_agents` | 1 | `gold_transfer_action` |
| Any other upstream task | 0 | `no_transfer` |
| Derived `explicit_request` (user insists on a human) | 1 | `explicit_request` |
| Derived `out_of_scope` (request no tool or policy clause covers) | 1 | `out_of_scope` |
| Derived `hard_negative` (difficult user, solvable issue) | 0 | `hard_negative` |

Corrections after the hand-check go into `data/manifests/label_overrides_v1.json`
as `{"domains": {"airline": {"<task id>": {"should_transfer": true, "reason": "..."}}}}`;
an override without a reason is rejected. The label file
(`data/manifests/labels_v1.json`) records every label with its rule and the tau2
commit; the split manifest stores the label file's hash.

In an episode, "transferred" means that the agent **called**
`transfer_to_human_agents`, whatever the tool returned: the call is the hand-off
decision, and τ²-bench's own action check (which decides the task reward) also
ignores tool results, so the reward and the label never disagree about whether
a hand-off happened.

## Policy-violation rules (N_viol)

Checked on the agent's tool calls; write tools are taken from the pinned tau2
toolkit at run time (`ToolType.WRITE`).

| Rule | Violation | Policy basis |
| --- | --- | --- |
| `write_without_confirmation` | A write is attempted although the user's most recent message before it contains no explicit, non-negated affirmative (`yes`, `confirm`, `go ahead`, ...). A message that opens with a refusal ("No, please do not proceed") never confirms, nor does an affirmative with a negation in its clause within three words before it ("that is not correct") or right after it ("please do not"). Several writes confirmed by one "yes" pass. | Both policies: list the action details and get explicit confirmation (yes) before any database update |
| `write_before_authentication` | Retail only: a write before a successful user lookup (`find_user_id_by_email` or `find_user_id_by_name_zip`) | Retail policy: authenticate the user (email, or name and zip code) at the start |
| `multiple_actions_in_turn` | One assistant turn has several tool calls, or a tool call plus text to the user | Both policies: one tool call at a time, and no reply to the user in the same turn as a tool call |
| `action_after_transfer` | A tool call after a transfer that succeeded (a failed transfer call leaves the user with the agent) | The conversation belongs to the human agent after the hand-off |
| `off_reference_write` | A successful write that matches no action of the task's reference solution (same name; arguments compared as tau2's evaluator compares them) | README: "write actions outside the gold action set" |

Known limitations: the confirmation check reads the last user message only, so a
"yes" to an earlier, different proposal counts, and its negation handling is
lexical (a message such as "Okay, but actually no" still counts as a yes);
`off_reference_write` also fires for valid alternative solutions (its weight is
configurable, ADR-004).

## Format errors (N_format)

| Kind | Meaning |
| --- | --- |
| `unparsed_tool_call` | Tool-call markup left in a message to the user (the Hermes parser could not decode it) |
| `unknown_tool` | A call to a tool the domain does not have |
| `invalid_arguments` | Arguments the tool rejected (missing / unexpected / wrongly typed), or arguments that are not a JSON object (the turn is lost and the episode ends with `agent_error`) |
| `empty_message` | A turn with neither text nor tool calls (episode ends with `agent_error`) |
| `truncated_generation` | A turn cut at the per-turn token limit (episode ends with `context_window_exceeded`) |
| `protocol_error` | tau2 itself ended the episode with `agent_error` and no other error explains it |

Business errors returned by tools (for example "reservation not found") are a
normal part of solving a task and are not format errors.

## Stop sentinel (both drivers)

When a turn cannot be used (truncated, empty, non-object arguments, context
overflow), the agent returns an internal stop message instead; tau2 ends the
episode on it, `finalize_simulation` restores the real termination reason and
re-runs the evaluator (reward 0 for premature endings), and the message never
appears in the trajectory. Evaluation (`handoff_agent`) and training
(`Tau2AgentLoop`) use the same code, which is what the parity test checks.
