# HOT-2, sixth part: are the agent's own checked labels what spoils the audit learner?

Written before any run with the new label mode. `radar_percept_audit_all.md` left one unexplained fact: with free audits of
all reports and hundreds of labels the audit learner stays near always-trusting (851, 810, 798 vs 813; 20 seeds, post hoc),
while the free-label learner (labels = truth of every offered report) reaches 647. Untested candidate: in audit mode the
labels from reports the agent walked to and checked itself are mixed in; they are biased (only admitted, nearby reports, and
the agent sees a ghost only if it goes there). In free mode they are not used. This is a cause-finding diagnostic, not a new
indicator test.

## Change
`RADAR_LABELS = 'audit_only'`: identical to 'audit' except `radar_verified` no longer adds a label (hit/ghost counts and the
regime estimate still update). Settings as in the diagnostic: ghost weight 30, probe 0, scope 'all', free audits.

## Design
Test seeds 5000-5099 (new, 100 seeds), needs-only workspace, ghost radar as before, total critical steps, mean over agents:
(a) none; (b) learned-free; (c) audit (mixed labels), audit probability 1.0, free; (d) audit_only, probability 1.0, free;
(e) audit_only, probability 0.2, free; (f) audit_only, probability 0.02, paid (the previously selected cost level).
No parameter selection.

## Predictions (my probabilities)
- **C1 (45%):** (d) beats (c) by at least 100 (the mixed labels are what spoils it).
- **C2 (35%):** (d) ends within 60 of (b).
- **C3 (40%):** (d) beats (a) by at least 50.
- **C4 (30%):** (f) beats (a) by at least 30 (a cheap honest label source works once the cause is removed).
- Also reported: label counts, share admitted, hits and ghosts verified.

## Reading in advance
C1 and C2 holding: the cause is the biased self-checked labels; the learned form works with unbiased labels from any source,
and HOT-2's learned branch is reopened with that as a design rule. C1 failing: the mixed labels are not the cause, and the
remaining difference between audits and free labels is unknown (label timing, pool of five, or the admission loop). Either
way the result is about this learner and this world.

## Limits stated in advance
Sixth experiment on the same radar; world and learner shaped by earlier looks; four predictions, no correction; free audits
are an upper bound; one environment.

## Result (`experiments/results/radar_percept_audit_only/`, commit 1e74f24)
Seeds 5000-5099, total critical steps (hits / ghosts verified; audits per run): none 726 (49 / 170), learned-free 617 (39 / 9),
audit mixed labels, probability 1.0, free 772 (33 / 18; 785 audits), **audit_only probability 1.0, free 790** (35 / 22; 787),
audit_only 0.2, free 804 (36 / 23; 193), audit_only 0.02, paid 997 (39 / 56; 20).
- **C1 failed:** audit_only - mixed = +17 [-6, +41] (predicted at least 100 better). **C2 failed:** audit_only - learned-free =
  +173 [+142, +203]. **C3 failed:** audit_only - none = +64 [+40, +88] (worse than no monitor). **C4 failed:** paid audit_only
  0.02 - none = +271 [+245, +297] (the pool-of-five audit at that setting is much worse than the earlier selected one, 867 vs 748).
- **The candidate cause is refuted.** Removing the agent's own checked labels changed nothing; with ~790 free, unbiased audit
  labels the learner is still 64 worse than no monitor and 173 worse than the free-label learner. The cause stays unknown.
  Untested differences between the two label sources: free labels cover every offered report at once, the audit labels a
  random one of the five most recent per step (older ones are discarded) and arrive later; I did not count the free labels.
- Not tested further; the learned branch of HOT-2 stays closed.
