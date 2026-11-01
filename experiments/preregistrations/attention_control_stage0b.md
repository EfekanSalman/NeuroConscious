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

## Result (`experiments/results/ast_stage0b/`, commit 3220692, dirty=false; 100 seeds)
Total critical steps (refreshes per 1000; usable-cue steps / of which correct): wired_cap1 1449, none_cap1 1594, int25_c 1637 (8; 866 / 544),
int50_c 1624 (6.5; 860 / 540), int100_c 1613 (4.4; 832 / 528), int50_free 1459 (19; 996 / 915).
- **T1 held:** best capped interval policy (k=100) - wired = +164 [+143, +183]: a large gap.
- **T2 held:** int50_c - int50_free = +165 [+141, +188].
- **T3 failed:** none_cap1 - best capped = -19.5 [-41, +3]: the capped hand-written policies are no better than no attention at all
  (int25_c and int50_c are worse than none). Under the cap the cue can win the slot only while no need is critical, and that happens
  rarely (4-8 refreshes per 1000 steps); the traces are right only 63% of the time, and a stale trace is worse than none.
- Irregular redraws themselves cost the uncapped interval policy nothing (int50_free 1459, +10 over wired; the regular world gave 1459, +21).
  All of the headroom comes from the cap.
- Caveat on the decision rule: T1 holding does not show that the headroom is **reachable**. The cap creates a trap (without a cue the agent
  is critical most of the time and cannot attend; with a cue it is calm and can), and a learned controller starts inside it. Stage 1 is
  uninformative unless a good capped policy exists. Stage 0c checks that with a policy that cheats (`attention_control_stage0c`).
