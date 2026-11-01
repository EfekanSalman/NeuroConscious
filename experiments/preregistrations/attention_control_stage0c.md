# AST-1, stage 0c: is the headroom reachable?

Written before any run. Stage 0b left a 164-step gap between the capped hand-written policies and `wired_cap1`, but those policies
are no better than no attention (`none_cap1`). Before building a learned controller, check that a good capped policy exists at all.

## Policy
`BEACON_POLICY` 'oracle' cheats: it bids (at `BEACON_FORCE`) exactly when the remembered cue differs from the true active half
(or there is none), so it refreshes the trace the first moment the slot is winnable after a redraw. It is an upper bound for any
attention policy under the same cap; no learner can know the redraw times.

## Conditions
Scenario `beacon_irregular`, frontier exploration, lean workspace, capacity 1, trace TTL 300, seeds 2500-2599, 1000 steps, total critical
steps: `wired_cap1`, `none_cap1`, `oracle_c` (bid capped at 1.0), `oracle_free` (bid 10.0, no cap).

## Predictions (my probabilities)
- **O1 (60%):** `oracle_c` is within 60 steps of `wired_cap1`.
- **O2 (80%):** `oracle_free` is within 30 steps of `wired_cap1`.
- **O3 (70%):** `oracle_c` beats `none_cap1` by at least 60.
- Decision rule. O1 holds: a good capped policy exists and stage 1 (learned controllers) can be informative; its headroom is the gap between
  the hand-written policies and `oracle_c`. O1 fails but O3 holds: reachable headroom is limited to `none - oracle_c`; stage 1 only if that
  is at least 100 steps. O3 fails: nothing reachable under the cap, the AST line is closed as a negative result (the control problem
  I built has no accessible gain for a schema).
