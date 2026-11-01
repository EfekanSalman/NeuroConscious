# AST-1, stage 0b: a harder attention problem

Written before any run. Stage 0 (`attention_control_stage0`) found that attention policy matters (+304 between the best
constant and the best interval policy) but that a hand-written interval policy ends 21 steps from the free-attention reference
`wired_cap1`, leaving a learned schema almost nothing to gain. This stage makes the control problem harder in two ways, both
chosen because they make a fixed rule weaker:
1. **Irregular redraws:** the active half is redrawn after 50-250 steps (`beacon_irregular`, `BEACON_CYCLE_JITTER` 0.67), so the
   agent cannot count on a period and a stale trace costs more.
2. **Attention that cannot override a critical need:** the forced cue bid is capped at salience 1.0 (`BEACON_FORCE`), below the
   bid of any need above 0.85 (level squared + 0.5 >= 1.22). The cue can enter the single slot only while no need is critical.

## Conditions
Scenario `beacon_irregular`, frontier exploration, lean workspace, capacity 1, memory `nearest`/salience10, trace TTL 300,
seeds 2400-2499, 1000 steps, metric total critical-need steps.
`wired_cap1` (reference, no attention cost), `none_cap1`, `int25_c`, `int50_c`, `int100_c` (interval policy with the capped bid),
and `int50_free` (interval k=50 with the uncapped bid 10, to isolate the effect of the cap).

## Predictions (my probabilities)
- **T1 (70%):** headroom exists: the best capped interval policy is at least 60 steps behind `wired_cap1`.
- **T2 (70%):** the cap costs: `int50_c - int50_free >= 40`.
- **T3 (75%):** attention still beats nothing: the best capped interval policy is at least 30 steps better than `none_cap1`.
- Decision rule: if T1 holds there is room for a learned controller and stage 1 is worth building (a schema-based controller,
  a model-free controller with the same inputs, a random-weight schema and a fixed-interval baseline). If T1 fails, hand-written
  control is already at the ceiling in a harder world and the AST line is closed as a negative result (nothing for a schema to add).
