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
