# Error analysis: B2 vs B0 (T2.3)

Dev split only (test results are not inspected for analysis). Packets from
`scripts/sample_review.py` (not committed; regenerate with the commands below):

```bash
python scripts/sample_review.py compare  --run B2-dev-s0-20261005 --baseline B0-dev-s0-20261004
python scripts/sample_review.py failures --run B2-dev-s0-20261005 --n 10
```

Annotation: Claude pre-annotated all 10 failures (5 Oct 2026); the owner
spot-checked 3 and agreed with all of them (record below). The report's AI-use statement says so.

## Per-task changes (dev, B2 vs B0)

19 task variants improved and 21 regressed, consistent with the paired success
difference near zero (test_main Δ +0.020 [−0.056, 0.107]). SFT changed
behaviour broadly rather than solving more tasks: retail upstream tasks 66, 69,
72 and 78 (and most of their variants) improved; retail 0, 13, 15 and 88 and
airline 11, 17 and 49 regressed in several variants.

## Ten B2 failures

| # | Episode | Kind (should transfer) | What went wrong | Category |
| --- | --- | --- | --- | --- |
| 1 | airline/11__xr0/neutral/3 | explicit request (yes) | Announced "YOU ARE BEING TRANSFERRED" in text but never called `transfer_to_human_agents`, so no hand-off happened | A. hand-off announced, not executed |
| 2 | airline/49__oos0/neutral/0 | out of scope (yes) | Correctly judged the meal request out of scope, then the same text-only transfer as #1 | A. hand-off announced, not executed |
| 3 | retail/15__oos0/neutral/0 | out of scope (yes) | Promised to send a printed invoice; no tool can do that and no hand-off was made | B. claims an unsupported capability |
| 4 | airline/49__oos0/neutral/2 | out of scope (yes) | Agreed to add vegetarian meals, made an unrelated `update_reservation_flights` call as a stand-in and reported success | B. claims an unsupported capability (with a spurious write) |
| 5 | airline/3/neutral/1 | upstream (no) | Misstated the silver/economy bag allowance, then transferred when the user disputed it and asked for a supervisor | C. over-escalation under pressure (after a policy error) |
| 6 | retail/88__xr0/neutral/0 | explicit request (yes) | Transferred correctly, then kept acting (6 tool calls) and cancelled an order after the transfer | D. acts after transfer |
| 7 | retail/69__hn0/neutral/0 | hard negative (no) | Mapped "found a better deal" to the cancellation reason "ordered by mistake" and cancelled; the reference outcome differs | E. wrong in-domain write |
| 8 | airline/11__hn0/neutral/3 | hard negative (no) | Refused the passenger removal correctly, then downgraded the cabin with an implausible refund ($4,771 from an unrelated calculation) and possibly the wrong payment method | E. wrong in-domain write |
| 9 | airline/20__hn0/neutral/1 | hard negative (no) | Invalid search argument, misread departure times, concluded no flight fits and offered a transfer instead of booking | F. search/planning error |
| 10 | retail/112/neutral/1 | upstream (no) | Agent and simulator repeated the same messages for ~45 turns until the context window ran out; no action taken | G. repetition loop |

Counts: hand-off-related failures (A–D) 6 of 10; in-domain task errors (E–G) 4
of 10. Every failure ends with task reward 0.

## Findings

1. **Text-only hand-offs (A) are the clearest SFT failure.** The model learned
   the hand-off phrasing from the teacher dialogues but sometimes skips the
   tool call; the metric counts these as under-escalation (B2 recall 0.633 on
   test_handoff). A transfer counts only when the tool is called, and R2
   penalises a missed transfer (λ_FN = 1.0, ADR-004), so this is the first
   behaviour to check in the R1/R2 audits (T3.5).
2. **Out-of-scope requests can still be "handled" (B).** Instead of handing
   off, B2 sometimes pretends a capability exists; #4 adds a write the policy
   does not justify. This is the hand-off failure with user-facing harm.
3. **Pressure-driven escalation remains (C)**, though rarer than for B0
   (over-escalation 0.087 → 0.016 on test_handoff).
4. **Acting after transfer (D)** appears when the simulator keeps talking after
   a transfer; the rule metric `action_after_transfer` (0.026 on test_handoff)
   tracks it.
5. **Long loops (G)** produce the very long episodes seen in the RL spike
   (22.7k-token response); the 16,384-token cap ends them earlier during RL.

Simulator artefacts seen in the same packet (#6 continues after a transfer, #10
repeats itself, #9 ends with `###OUT-OF-SCOPE###` after an offered transfer)
are noted as limitations, not counted against the policy.

## Owner spot-check

| Episode | Agrees with the annotation? | Note |
| --- | --- | --- |
| #1 airline/11__xr0/neutral/3 | yes (5 Oct) | no `transfer_to_human_agents` call in the episode |
| #4 airline/49__oos0/neutral/2 | yes (5 Oct) | meal request answered with `update_reservation_flights` |
| #10 retail/112/neutral/1 | yes (5 Oct) | same messages repeated until the context ran out |
