# HOT-2, seventh part: label count and the pool of five

Written before any run with the new setting. `radar_percept_audit_only.md` refuted the biased-label candidate: ~790 free,
unbiased audit labels still gave 790 vs 726 (no monitor) and 617 (free-label learner). Remaining untested differences: the
free-label learner labels every offered report at once, while an audit labels one random report among the five most recent
per step (older ones are discarded). I also never counted the free-label learner's labels. Cause-finding diagnostic.

## Change
`RADAR_AUDIT_POOL` (default 5, the former hard-coded number) sets how many recent offered reports an audit can pick from; the
number of labels the learner receives is now counted (`radar_labels`, emitted only with the radar on).

## Design
Seeds 5100-5199 (new, 100), needs-only workspace, ghost radar as before, total critical steps, ghost weight 30, probe 0,
scope 'all', free audits with probability 1.0, labels 'audit_only'; pools 5, 50, 500. Plus none and learned-free (label counts).

## Predictions (my probabilities)
- **D1 (50%):** the free-label learner receives at least twice as many labels per run as audit pool 5.
- **D2 (35%):** pool 500 beats pool 5 by at least 100.
- **D3 (25%):** pool 500 ends within 60 of learned-free.

## Reading in advance
D1 holding with D2 failing: label count is not the cause either. D2 and D3 holding: discarding reports (pool) is the cause.
D1 failing and D2 failing: both candidates are out and the difference stays unexplained; I stop this line.

## Limits stated in advance
Seventh experiment on the same radar; three predictions, no correction; free audits are an upper bound; one environment.

## Result (`experiments/results/radar_percept_audit_pool/`, commit 7e7711c, dirty=false)
Seeds 5100-5199, total critical steps (ghosts verified; labels per run): none 733 (170; 219 labels from own checks),
learned-free 603 (9; 1017), audit_only pool 5 **807** (21; 778), pool 50 778 (20; 942), pool 500 802 (19; 952).
- **D1 failed:** free-label learner gets 1017 labels vs 778 (ratio 1.3, predicted at least 2). **D2 failed:** pool 500 - pool 5 =
  -6 [-31, +19]. **D3 failed:** pool 500 - learned-free = +198 [+175, +222].
- Both candidates are out: with 952 labels per run (937 more than 5-pool, vs 1017 for free labels) and almost every offered
  report eligible, the audit learner is still ~200 worse than the free-label learner and 70 worse than no monitor. What
  remains different is when and in what order a label arrives (free: at the offer, in order; audit: later, random pick), or an
  error in my audit path that tests did not catch. I did not look further; this line stops here.
