# Responsible-AI audit of the baselines (T1.6, M1)

What the prompted baselines do that a deployed customer-service agent must not,
measured before any training. Runs: `B0-{dev,test_main,test_styles,test_handoff}-s0-20261004`
(Qwen3-4B-Instruct-2507, prompted) and `B1-test_main-s0-20261004`
(Qwen3-30B-A3B-Instruct-2507, prompted; the same model as the user simulator).
Protocol: ADR-005 (frozen 4 Oct); metric definitions: `docs/metrics.md`. All
numbers are 95 % bootstrap CIs over tasks from the runs' `summary.md`; rows are
in `docs/experiment-log.md`. B2 (SFT) is shown only to preview M2.

## 1. Policy compliance

| test_main | B0 | B1 | B2 (preview) |
| --- | --- | --- | --- |
| Success (pass^1) | 0.306 [0.214, 0.408] | 0.398 [0.291, 0.510] | 0.327 [0.224, 0.439] |
| pass^4 | 0.102 [0.020, 0.184] | 0.204 [0.102, 0.327] | 0.102 [0.020, 0.204] |
| Rule violations (rules only) | 0.179 [0.128, 0.230] | 0.163 | 0.031 [0.005, 0.061] |
| Writes outside the reference solution (`off_reference_write`) | 0.408 [0.326, 0.495] | — | 0.459 [0.357, 0.571] |
| Format errors | 0.026 [0.005, 0.056] | 0.000 | 0.000 |

- **No write before authentication** in any baseline episode (0.000), and few
  writes without explicit confirmation (B0 0.051 on test_main).
- **The common rule breach is several actions in one turn** (B0 0.143): more
  than one tool call, or a tool call plus a reply, in the same turn, which both
  policies forbid.
- **The largest risk is a wrong write, not a forbidden one**: 41 % of B0
  test_main episodes contain a successful write that matches no action of the
  reference solution (wrong item, amount, cabin or payment method in the
  dialogues reviewed; the check also flags valid alternative solutions, so it
  is an upper bound, `docs/labels.md`). Task success and pass^4 show that
  errors are not rare one-offs: B0 solves a task in all four trials only 10 %
  of the time.
- The larger prompted model (B1) solves more tasks but breaks rules about as
  often as B0; scale alone does not buy compliance.

## 2. Hand-off calibration (test_handoff, B0)

| Metric | B0 |
| --- | --- |
| Hand-off precision | 0.866 [0.788, 0.933] |
| Hand-off recall | 0.526 [0.435, 0.618] |
| Over-escalation (transfer when not needed) | 0.087 [0.043, 0.134] |
| Under-escalation (no transfer when needed) | 0.474 [0.382, 0.565] |

- **B0 under-escalates far more than it over-escalates.** Almost half of the
  users who ask for a human or need something the agent cannot do are kept in
  the conversation. Under-escalation is the more harmful error here: the user
  is refused, stalled or given a false promise instead of a person who can
  help.
- Over-escalation (8.7 %) has a real cost too (human time, a worse experience
  for routine requests) and is the failure an escalation reward could
  over-correct into; RQ2 therefore reports both rates, not F1 alone.
- B1 was evaluated on test_main only, where few tasks require a transfer; its
  hand-off F1 there (0.40, CI [0.00, 0.70]) is not interpretable.
- The B2 error analysis (`docs/error-analysis.md`) shows the mechanism behind
  under-escalation that also exists in prompted models: announcing a transfer
  in text without calling the tool, and pretending to support out-of-scope
  requests.

## 3. Robustness across communication styles (RQ3, B0)

Success by style (test_styles, neutral from test_main): impatient 0.306,
terse 0.316, low-tech-literacy 0.337, non-native 0.347, verbose 0.347, neutral
0.306. Style gap 0.041 [0.031, 0.136]; worst-style success 0.306
[0.217, 0.379].

- **No style is served clearly worse by B0**; all CIs overlap. This is a
  measurement under a mild manipulation (style review in `docs/personas.md`:
  styles recognisable but mild; 2 of 50 dialogues drift on facts), so it is not
  evidence that real users with these styles are served equally.
- Styles describe expression only, never demographic groups; no stereotypes
  were seen in the 50 reviewed dialogues.
- Worst-style success is the primary RQ3 number: the max − min gap can grow
  when one style improves (B2: impatient 0.439, gap 0.112, worst style 0.327).

## 4. Transparency

Citation rate (a refusal or hand-off that names the policy reason): B0 0.30 on
test_handoff, 0.60 on test_main. Unexplained refusals are common; R2's reward
does not target this, so it is reported, not optimised.

## 5. Limitations of this audit

- **Simulated users.** Every user is Qwen3-30B-A3B following a scenario; it
  sometimes keeps talking after a transfer, repeats itself, or invents an
  identifier. B1 is the same model as the simulator, which may favour it.
- **Coverage.** Retail results use 74 of 114 tasks (tasks whose reward needs
  tau2's LLM judge are excluded, ADR-005); hand-off tasks are generated from
  templates and reviewed by Claude (40/40, owner spot-check), not by domain
  staff.
- **Rule checks are automatic** (`docs/metrics.md`); a 20-dialogue manual audit
  per final checkpoint (T3.5/T3.6) checks them.
- **One seed per run**; CIs cover task sampling, not training variance.

## 6. Consequences for training

- R2 penalises each weighted violation (λ_v = 0.25) and weights missed transfers
  (λ_FN = 1.0) above unnecessary ones (λ_FP = 0.5), matching the harm ranking
  above (ADR-004).
- Watch-list for the RL audits (RQ4): transfer rate drifting up (escalating to
  avoid hard tasks), text-only transfers, actions after transfer, very long
  episodes, and over-escalation on hard negatives.
