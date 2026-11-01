# AST-1, stage 0: is there anything to control?

Written before any run. The attention schema (AST-1) only matters if deciding where attention goes pays off. This stage uses
no learning: it asks whether a hand-written attention policy beats the constant-salience policies in a world where attending
costs something. If it does not, there is nothing for a schema to learn and the AST line stops here.

## Setting (the choice Efekan accepted: capacity 1 plus a cue trace)
Scenario `beacon`, frontier exploration, lean workspace (habit and goal off, threshold 0.3), **capacity 1**, memory `nearest`/salience10,
seeds 2300-2399, 1000 steps, metric total critical-need steps. With capacity 1 a cue that wins the slot is broadcast and the agent
idles for that step (`explore`): attending costs one step. New: a **cue trace** (`CUE_TRACE_TTL`): the last broadcast cue keeps steering
recall and exploration for up to TTL steps after it left the workspace (here 300). The active half is redrawn every 150 steps and
the agent is not told when, so a trace can be wrong for some time after a redraw.
Attention policies (`BEACON_POLICY`): 'constant' = the beacon bids at a fixed salience every step (as in `beacon_signatures`);
'interval' = silent until the cue has been out of the workspace for k steps, then a bid high enough to win the slot (one idle step).

## Conditions
1. `wired_cap1`: the cue is read directly every step (no attention needed, upper reference). 2. `none_cap1`: no cue.
3. `const_s06_notrace`: constant salience 0.6, no trace (the part B condition, new seeds). 4. `const_s06_trace`, 5. `const_s10_trace`: constant with trace.
6-9. `int10`, `int25`, `int50`, `int100`: interval policy with k = 10, 25, 50, 100, trace on.

## Predictions (my probabilities)
- **S1 (85%):** the best interval policy beats the best constant policy (conditions 3-5) by at least 100 critical steps.
- **S2 (70%):** the best interval policy is within 60 steps of `wired_cap1`.
- **S3 (65%):** the best k is 25 or 50 (too frequent wastes steps, too rare leaves a stale or missing trace).
- **S4 (75%):** the trace itself helps: `const_s06_trace` is at least 100 steps better than `const_s06_notrace`.
- Decision rule: if S1 holds there is a control problem worth learning and stage 1 (a learned schema, a model-free controller, a
  random-weight schema and an oracle, all choosing when to attend) is worth building. If S1 fails, attention control has no
  value here and the AST line stops, reported as a negative result.

## Result (`experiments/results/ast_stage0/`, commit 34a72b6, dirty=false; 100 seeds)
Total critical steps (cue refreshes per 1000; usable-cue steps / of which correct): wired_cap1 1438, none_cap1 1581,
const_s06_notrace 1763 (34), const_s06_trace 1783 (33; 849 / 508), const_s10_trace 1807 (41; 872 / 544), int10 1573 (87; 996 / 985),
int25 1492 (37; 996 / 965), int50 1459 (19; 996 / 918), int100 1480 (10; 996 / 825).
- **S1 held:** best constant (no trace, 1763) - best interval (k=50, 1459) = +304 [+291, +317].
- **S2 held:** int50 - wired = +21 [+8, +33].
- **S3 held:** best k is 50; the curve is U-shaped (int10 1573, int25 1492, int50 1459, int100 1480).
- **S4 failed, in the opposite direction:** `const_s06_notrace - const_s06_trace` = -20 [-35, -4], the trace made the constant policy a little worse.
  With a constant bid the cue is refreshed rarely (33 times), so the trace is stale for long: of 849 steps with a usable cue only 508 are right
  (60%), against 918 of 996 at k=50. A remembered cue that is not refreshed at the right time is a liability, not a gain.
- Decision rule says: S1 held, so stage 1 is worth building. **Caveat I add after seeing the data:** the best hand-written policy is only 21
  steps from `wired_cap1`, which pays nothing for attention and cannot be beaten. A learned controller (schema or model-free) can therefore gain
  at most about 21 steps and the stage 1 comparisons (schema vs model-free vs random-weight schema) would be decided by differences of that size.
  This environment leaves almost no room for a schema to matter; it needs a harder control problem first (for instance irregular redraw times
  or a refresh that cannot win the slot while a need is critical) or stage 1 is not informative.
