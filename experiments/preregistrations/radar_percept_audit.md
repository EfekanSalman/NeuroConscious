# HOT-2, fourth part: a label source that is not a cheat (paid audits)

Written before the runs. `radar_percept_learned.md`: the learned monitor found the threshold with 'free' labels (593, a
cheat that labels every offered report with the truth) and failed with labels from checked reports only (851, worse than
trusting everything, 736). The open question was whether an unbiased label source that is not a cheat closes the gap.

## The audit
`RADAR_LABELS = 'audit'`: the learned monitor keeps the labels from admitted reports it checks with its own view, and in
addition remembers the (strength, truth) of up to five recently *rejected* reports. Each step, with probability
`RADAR_AUDIT_P` and if one is pending, it spends the step (the chosen action is replaced by a stationary one, the same
cost model as an orienting step in the attention problems) and learns whether one random pending rejected report was
real. This gives labels on the side of the threshold that checked-only labels never see. `RADAR_AUDIT_FREE` makes the
audit cost no step (an upper bound on the value of this label source, separating information from step cost).
Everything else as in `radar_percept_learned.md` (same ghost radar, strength N(2,1)/N(0,1), regimes of 100 steps, no remap
or slip, needs-only workspace, total critical steps, mean over agents).

## Design
Selection seeds 4400-4429: ghost weight in {12, 30} x probe in {0, 0.02} x audit probability in {0.02, 0.05, 0.1}
(12 runs, paid audits); the lowest total is used. Test seeds 4300-4399 (new, 100 seeds): none, never, percept_oracle,
percept_threshold (tau 2.5), learned-free (w 30, probe 0, a cheat), learned-verified (w 3, probe 0.02, as selected before),
learned-audit (selected setting), learned-audit-free (same setting, audits cost no step).

## Disclosed look
A 4-seed run (9600-9603) to check that audits happen: about 47 audits per agent in 1000 steps, 17% of reports admitted.
I saw the critical-step means of those four seeds and did not use them. An earlier version of the code had a bug that
turned verified labels off in audit mode (zero audits, everything admitted); it was fixed before this registration.

## Predictions (my probabilities)
- **A1 (60%):** learned-audit beats learned-verified by at least 100.
- **A2 (45%):** learned-audit beats `none` by at least 30 (a monitor with realistic labels, net of what audits cost).
- **A3 (50%):** learned-audit ends within 80 of the hand threshold.
- **A4 (55%):** learned-audit-free beats learned-audit by at least 20 (the step cost matters).
- Also reported: audits per run, share of reports admitted, hits and ghosts verified.

## Reading in advance
A1 holding with A2 failing: the label source is what was missing, but at this price of audits it does not pay for itself
over trusting everything. A1 and A2 holding: an outcome-trained monitor with realistic labels is worth having; HOT-2 in its
per-percept learned form is then met at an honest cost. A1 failing: labels from the rejected side are not the limit, and
the previous failure has another cause. A4 failing: the information, not the lost steps, decides.

## Limits stated in advance
One environment designed by me; the audit reveals the truth of a report for one step of cost (a designer-given oracle query,
not something the agent builds); the ghost weight is a number I set and take from a small grid; four predictions, no
multiple-comparison correction.
