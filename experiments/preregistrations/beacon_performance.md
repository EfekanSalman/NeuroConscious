# Beacon task, part A: does reaching the cue only through the workspace matter for performance?

Written before any run. Part B (signatures: capacity competition, simultaneous effect on consumers) is a separate
pre-registration, written after this result.

## Why this task
Every specialist reads the whole agent, so nothing in the current agent needs to be broadcast. The beacon regime
creates information that is only sensed by one module (`BeaconSpecialist`, the sole reader of `read_regime`):
in the `beacon` scenario only one half of the 20x20 grid holds resources (6 food and 6 water sites, 3 per half
each), the active half is redrawn at random every 150 steps and the other half is emptied. The consumers of the
cue are memory recall (sightings in the dormant half are skipped) and exploration (frontier cells and the
least-visited fallback prefer the active half). Without the cue the agent walks to remembered sites that were
emptied and searches both halves.

## Conditions (`agent.base_agent:BEACON_ACCESS`)
Scenario `beacon`, frontier exploration, seeds 2100-2199, 1000 steps, sighting/workspace memory `nearest`/salience10.
Metric: total critical-need steps (also cue availability, `cue_steps`, `cue_correct_steps`).
Workspace agent = the lean one from earlier runs (needs-only: habit and goal off, `NEED_THRESHOLD` 0.3, capacity 3):
1. `ws_none`: the cue never reaches the consumers (the sensor is not wired anywhere).
2. `ws_wired`: every consumer reads the sensor directly, no workspace involved in the cue. **Minimal-implementation control.**
3. `ws_broadcast`: the beacon bids (salience 0.6, never acts) and the cue is available only on steps when it wins one of the 3 slots.
4. `ws_wrong`: as 3, but the sensor reports the wrong half (`specialists:BEACON_INVERT`). Content-specificity control.
5. `rules_none`, 6. `rules_wired`: rule chain (`CRITICAL_OVERRIDE_ORDER` neediest, level 0.3, sighting memory) with no cue / a wired cue.

## Predictions (my probabilities)
- **A1 (90%):** the cue is worth at least 60 critical steps: `ws_none - ws_wired >= 60`.
- **A2 (65%):** broadcast is as good as wiring: `|ws_broadcast - ws_wired| < 40`.
- **A3 (80%):** a misleading cue hurts: `ws_wrong - ws_none >= 30`.
- **A4 (60%):** `ws_none - ws_broadcast >= 50` (the broadcast route delivers most of the benefit).
- **A5 (70%):** the cue helps the rule chain too: `rules_none - rules_wired >= 60`.
- Reading. A1 and A4 hold: the task needs the information to reach the consumers. A2 holding means performance cannot tell
  the workspace from direct wiring (minimal implementation: a point-to-point link does the same job); that is the
  expected outcome and the reason for part B. If broadcast is more than 40 steps worse than wiring, competition for slots costs
  something measurable. A5 holding means the benefit is a property of the information, not of the workspace.

## Result (`experiments/results/beacon_performance/`, commit 29d13c1, dirty=false; 100 seeds)
Total critical steps (cue availability in steps of 1000): ws_none 1584 (0), ws_wired 1422 (1000), ws_broadcast 1424 (858),
ws_wrong 2006 (763 steps, all wrong), rules_none 1608, rules_wired 1480.
- **A1 held:** ws_none - ws_wired = +162 [+144, +181].
- **A2 held:** ws_broadcast - ws_wired = +3 [-10, +16]. Broadcast through the 3-slot workspace is as good as wiring even
  though the cue is unavailable on 14% of steps (cue absence does not matter for a cue that lasts 150 steps).
- **A3 held, strongly:** ws_wrong - ws_none = +423 [+406, +439]; a misleading cue is far worse than none (meals 1.0 vs 9.1).
- **A4 held:** ws_none - ws_broadcast = +159 [+138, +181].
- **A5 held:** rules_none - rules_wired = +128 [+111, +144]. The cue helps the rule chain as much as the workspace agent.
- Not predicted: rules_wired is still 56-59 steps behind ws_wired / ws_broadcast [+40, +73], the same ~60 steps as in `need_order`.
- Reading: the task makes the information necessary (A1, A3), a point-to-point wire does the job as well as the broadcast
  (A2) and the rule chain benefits equally (A5). Performance does not distinguish the workspace from direct wiring.
