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

## Result (`experiments/results/radar_percept_audit_all/`, commit 1b0f1b6, dirty=false)
Selection (seeds 4600-4629, ghost weight_audit probability, scope 'all', paid): 12_0.02 **874**, 30_0.02 885, 12_0.05 931,
30_0.05 952, 12_0.1 1020, 30_0.1 1044, 12_0.2 1212, 30_0.2 1224 (again the smallest audit probability). Test (seeds
4500-4599), total critical steps (hits / ghosts verified; share admitted; audits per run):
none 748 (48 / 169), never 939, oracle 612 (42 / 7), hand threshold 614 (35 / 7), learned-free 635 (39 / 9; 30%),
learned-audit-rejected 867 (24 / 14; 17%; 18 audits), **learned-audit-all 892** (24 / 13; 16%; 19 audits),
learned-audit-all-free 844 (22 / 13; 16%; 19 audits).
- **B1 failed:** audit-all - audit-rejected = +25 [-10, +59] (predicted at least 100 better). **B2 failed:** audit-all - none =
  +144 [+108, +180]. **B3 failed:** audit-all - threshold = +278 [+242, +314]. **B4 held:** free - paid = -48 [-82, -14].
  **B5 failed:** audit-all-free - none = +96 [+60, +132] (predicted at least 50 better).
- The earlier test reproduces: learned-free 635 (vs 593, 599), audit-rejected 867 (vs 883), none 748 (vs 723, 736).
- **The truncation reading is not supported.** Covering the admitted reports in the audit pool did not help. But B5 is
  confounded with label count: the selected audit probability (0.02) gives only about 19 labels per run, so it does not test
  coverage at a large label budget. A post-hoc diagnostic (not pre-registered; 20 seeds 9500-9519, always trusting 813,
  free-label learner 647 with the same weight and probe) with scope 'all' and free audits: audit probability 1.0 (781 labels
  per run) 851, 0.2 (189 labels) 810, 0.05 (49 labels) 798. So even with free audits of every kind of report and hundreds of
  labels, the audit learner stays at or near the level of trusting everything, while the free-label learner, which gets
  similar numbers of labels, reaches 647. I do not know why. A candidate I did not test: in audit mode the labels from
  reports the agent checked itself are mixed in with the audit labels, and they are biased; in free mode they are not used.
- Reading: the learned per-percept monitor works in this world only with the free-label control (a cheat), and neither
  checked-only labels, paid audits of rejected reports, nor paid or free audits of all recent reports reproduce it. My two
  explanations (label bias, truncation) were not confirmed. The learned branch of HOT-2 is closed here, with one unexplained
  discrepancy left as is. Five predictions, no multiple-comparison correction, one environment; this is the fifth experiment
  on the same radar, shaped by what I saw earlier.

**Correction (later):** results here that use `RADAR_LABELS='free'` are inflated by a leak (the truth of the report being judged was used to judge it); see `radar_percept_leak.md`.
