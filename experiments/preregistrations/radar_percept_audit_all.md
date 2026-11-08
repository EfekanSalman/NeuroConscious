# HOT-2, fifth part: auditing a sample of all reports, admitted ones included

Written before any run with the new audit scope. `radar_percept_audit.md`: paid audits of *rejected* reports left the
learned monitor worse than trusting everything (883 vs 723), and a post-hoc diagnostic (not pre-registered) showed that
even 584 free audit labels per run did not help (843 vs 813, 20 seeds). My reading was that the labelled set is truncated:
audits label only reports below the threshold, and admitted ghosts the agent never walks to are never counted against it.
This test is the direct check of that reading. If it is right, auditing a sample of *all* recent reports should fix it.

## Change
`RADAR_AUDIT_SCOPE = 'all'`: the pool of reports an audit can pick from (the five most recent offered) includes admitted
reports as well as rejected ones. Everything else as in `radar_percept_audit.md`: ghost radar, strength N(2,1)/N(0,1), an audit
replaces the action with a stationary step (`RADAR_AUDIT_FREE` makes it free), labels from checked admitted reports are kept,
needs-only workspace, total critical steps, mean over agents, no remap or slip.

## Design
Selection seeds 4600-4629: ghost weight in {12, 30} x audit probability in {0.02, 0.05, 0.1, 0.2}, probe 0 (8 paid runs);
the lowest is used. Test seeds 4500-4599 (new, 100 seeds): none, never, percept_oracle, percept_threshold (tau 2.5),
learned-free (w 30, probe 0; a cheat), learned-audit-rejected (the previous setting: w 12, probe 0, audit probability
0.02, scope 'rejected'), learned-audit-all (selected, paid), learned-audit-all-free (same setting, audits cost no step).

## Predictions (my probabilities)
- **B1 (50%):** learned-audit-all (paid) beats learned-audit-rejected by at least 100.
- **B2 (35%):** learned-audit-all (paid) beats `none` by at least 30.
- **B3 (40%):** learned-audit-all (paid) ends within 80 of the hand threshold.
- **B4 (60%):** learned-audit-all-free beats the paid one by at least 20.
- **B5 (55%):** learned-audit-all-free beats `none` by at least 50 (if information is what was missing, then with free
  audits it should show).
- Also reported: audits per run, share admitted, hits and ghosts verified.

## Reading in advance
B1 and B5 holding with B2 failing: the truncation reading is right (the information was missing) but at this price the audits
cost more than they teach, so the learned form is met only if checking is cheap. B1, B2 and B3 holding: a learned per-percept
monitor with an honest, paid label source reaches the hand rule; HOT-2 in this form is met. B1 and B5 failing: truncation is
not the cause and I do not know what is; this closes the learned branch. The result is then about this learner and this
world, not about HOT-2.

## Limits stated in advance
The audit is a designer-given oracle query at the cost of a step, and the ghost weight and the pool size (five) are numbers I
set; one environment; five predictions, no multiple-comparison correction; this is the fifth experiment on the same radar,
so the world and the learner have been shaped by what I saw earlier (all looks disclosed in the preceding files).
