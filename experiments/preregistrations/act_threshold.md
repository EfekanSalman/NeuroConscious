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
