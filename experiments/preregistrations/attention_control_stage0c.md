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

## Result (`experiments/results/ast_stage0c/`, commit b056a07, dirty=false; 100 seeds)
Total critical steps (refreshes per 1000; usable-cue steps / of which correct): wired_cap1 1451, none_cap1 1602, oracle_c 1556 (2.3; 570 / 443),
oracle_free 1482 (4.2; 834 / 834).
- **O1 failed:** oracle_c - wired = +106 [+86, +127] (predicted within 60).
- **O2 failed narrowly:** oracle_free - wired = +32 [+16, +48] (predicted within 30; the interval contains 30).
- **O3 failed:** none - oracle_c = +46 [+26, +66] (predicted at least 60).
- Decision rule: O3 failed, so **the AST line is closed as a negative result.** Even a policy that knows the redraw times gains only about 46
  steps over no attention under the cap, because the cue can enter the slot only in rare windows (2.3 refreshes per 1000 steps).
  Stage 1 (a learned schema, a model-free controller, a random-weight schema) would have been decided by differences far below that ceiling.
- What this does not show: that an attention schema cannot help. It shows that the control problem I built (capacity 1, a discrete refresh
  of one cue, a cap that makes attention available only in rare calm windows) has no reachable gain for any controller. A different problem
  (attention that is graded rather than all-or-none, a signal that cannot be read from the agent's own state, a cost that is not tied to
  need pressure) might. Whether such a problem exists without being built to order is open.
