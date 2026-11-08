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

## Result (`experiments/results/radar_percept_audit/`, commit 5dc613e, dirty=false)
Selection (seeds 4400-4429, ghost weight_probe_audit probability): 12_0_0.02 **842**, 30_0_0.02 858, 30_0.02_0.02 896,
12_0.02_0.02 898, 12_0_0.05 954, 30_0_0.05 965, 12_0.02_0.05 976, 30_0.02_0.05 986, 12_0_0.1 1047, 30_0_0.1 1059,
12_0.02_0.1 1093, 30_0.02_0.1 1095. Used w=12, probe 0, audit probability 0.02 (the smallest in the grid: each added audit
cost more than it taught). Test (seeds 4300-4399), total critical steps (hits / ghosts verified; share admitted; audits):
none 723 (49 / 170), never 967, oracle 589 (42 / 7), hand threshold 620 (36 / 7), learned-free 599 (39 / 8; 29%),
learned-verified 846 (29 / 21; 23%), **learned-audit 883** (23 / 14; 17%; 17.6 audits), learned-audit-free 842 (23 / 14; 17%).
- **A1 failed:** learned-audit - learned-verified = +37 [+5, +70] (predicted at least 100 better). **A2 failed:**
  learned-audit - none = +160 [+125, +195]. **A3 failed:** learned-audit - threshold = +263 [+227, +299].
  **A4 held:** audit-free - audit = -41 [-73, -9]: the audit steps themselves cost about 40.
- The learned-free control reproduces (599 vs 593 in the previous test, 124 better than none), as do none (723 vs 736) and
  verified (846 vs 851).
- **Post-hoc diagnostic (not pre-registered; 20 seeds 9500-9519, always trusting 813, oracle 605; w=12, probe 0):** the
  audit with free steps at audit probability 1.0 (584 labels of rejected reports per run) gives 843, at 0.2 (172 labels)
  832; paid at 0.2 1231 and at 1.0 1748. So the amount of audit labels is not what limits the monitor; even 584 free
  labels per run leave it at the level of trusting everything. My reading, untested: the labelled set is truncated on
  both sides. Audits label only rejected reports (all below the threshold), and admitted reports get a label only if the
  agent later checks them with its own view; admitted ghosts it never walks to are never counted against the threshold,
  so the fitted threshold cannot rise to where the ghosts are. The free-label control (599), which labels every offered
  report including the admitted ones, escapes this. The audit I built samples the wrong side.
- Reading: a paid audit of rejected reports does not make the learned monitor worth having here. HOT-2 in its learned
  per-percept form remains met only with a label source that covers what the agent admits (the free-label control), which is
  a cheat as built. Not tested: auditing a random sample of all offered reports (admitted ones included), which the
  diagnostic points to; learning the ghost weight. Four predictions, no multiple-comparison correction, one environment.
