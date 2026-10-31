# Does the workspace lead come from acting on a need earlier than the rule chain does?

Written before any run. The specialist ablation (`specialist_ablation`) showed that three need specialists alone
beat the rule-based agent by 203 critical steps. They bid from level 0.3 (`NEED_THRESHOLD`); the rule chain acts on
a need only above 0.85 (`CRITICAL_OVERRIDE_LEVEL`, the same level that metrics count as critical) and otherwise
follows the DQN suggestion, goals and procedures. So the rule agent only starts seeking a resource once the need is
already critical, and that alone could explain the gap. Not a workspace property.

## Intervention
Scenario `sites`, frontier exploration on, seeds 1100-1199, 1000 steps, metric total critical-need steps.
Rule-based agent with `decision_maker:CRITICAL_OVERRIDE_LEVEL` at 0.85 (default), 0.6 and 0.3.
Needs-only workspace (habit and goal disabled, `nearest` salience 10 as before) with `specialists:NEED_THRESHOLD`
at 0.3 (default), 0.6 and 0.85. With 0.85 nothing bids below it, so the agent idles; that is part of the test.

## Predictions (my probabilities)
- **B1 (70%):** rules at 0.3 beat rules at 0.85 by at least 100 critical steps.
- **B2 (55%):** rules at 0.3 end within 60 steps of needs-only at 0.3 (the gap closes). If B1 holds but B2 fails,
  a second difference remains (the lead is not only the threshold).
- **B3 (65%):** needs-only at 0.85 is worse than needs-only at 0.3 by at least 100.
- Reading: B1 and B2 hold -> the lead is acting earlier, nothing specific to the workspace. B1 holds, B2 fails ->
  earlier action explains part. B1 fails -> the threshold is not the cause and something else in the rule chain is.

## Result (added after the run; results in `experiments/results/act_threshold/`, commit f0f96a9, dirty=false)
Total critical steps (mean over 100 seeds): rules 0.85/0.6/0.3 = 1262/1215/1195; needs-only 0.85/0.6/0.3 = 1412/1117/942.
- **B1 failed on size:** rules 0.3 - rules 0.85 = -67 [-89, -44] (predicted at least 100; direction right).
- **B2 failed:** rules 0.3 - needs 0.3 = +252 [+223, +283]; the gap is not closed (it falls from 319 to 252).
- **B3 held:** needs 0.85 - needs 0.3 = +469 [+449, +489].
- Acting earlier explains about a fifth of the gap. Drinks: rules 9.8-10.9 at every threshold vs needs-only 18.5-20.2,
  even needs-only at 0.85 (18.5), so the drinking gap does not depend on the threshold at all.

## Correction (found after the first run)
My runner gave the rule-based agents no sighting memory (they used the FIFO buffer), while the needs-only
workspace used `nearest`/salience10 memory; `specialist_ablation` had matched them (`SIGHTING_MEMORY_*`). The
first results are therefore confounded for every rules-vs-needs comparison (B2) and are kept as
`rules_fifo_*.json`. B3 and the needs-only rows are unaffected. The rule conditions are re-run with
`agent.base_agent:SIGHTING_MEMORY_POLICY='salience'`, `SIGHTING_MEMORY_RECALL='nearest'`, `SIGHTING_MEMORY_SLOTS=10`
as `rules_0.85/0.6/0.3.json`, same seeds. B1 and B2 are judged on the matched re-run; no new predictions added.
