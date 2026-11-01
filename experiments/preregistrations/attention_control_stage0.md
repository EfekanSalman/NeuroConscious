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
