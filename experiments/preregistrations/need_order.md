# Is the rule chain's fixed need order what the need arbiter beats?

Written before any run. Follows `act_threshold` and a diagnostic run (not a result, 20 seeds, rule agent vs needs-only
workspace). Findings that motivate this test:
- The rule chain checks hunger, then thirst, then fatigue, in that fixed order. In 56% of its critical-thirst steps
  hunger is critical too and it seeks food.
- With the override level at 0.3 the rule agent seeks food 895 times and drinks 75 times per 1000 steps (needs-only
  workspace: 537 and 430). So the lower threshold in `act_threshold` also starved thirst, and B1 was not a clean test
  of "acting earlier".
- The need arbiter serves the need with the highest level (salience = level squared).

## Intervention
New switch `decision_maker:CRITICAL_OVERRIDE_ORDER`: 'fixed' (default, behaviour unchanged, verified against
`baseline.json`) or 'neediest' (among needs above the override level, the highest level is served first; ties keep
the fixed order). Scenario `sites`, frontier on, sighting memory `nearest`/salience10, seeds 1100-1199, 1000 steps,
metric total critical-need steps. New runs: rules with 'neediest' at override level 0.85, 0.6, 0.3.
Reference runs reused from `experiments/results/act_threshold/` (same seeds and settings): rules 'fixed' 0.85/0.6/0.3
(1135/1064/1077) and needs-only workspace 0.3 (942). Reuse is valid because the default is unchanged.
Not changed: the focused-state override (a food focus turns 'drink_water' into 'seek_food', about 6% of critical-thirst
steps) and the DQN's rests; both stay in the rule chain.

## Predictions (my probabilities)
- **C1 (65%):** 'neediest' at 0.85 beats 'fixed' at 0.85 by at least 50 critical steps.
- **C2 (55%):** 'neediest' at 0.3 ends within 60 steps of needs-only at 0.3 (942), i.e. the 193-step gap mostly closes.
- **C3 (60%):** with 'neediest', 0.3 is better than 0.85 by at least 100 (the threshold works once thirst is no longer starved).
- Reading: C1 and C2 hold -> the lead is balancing needs by level, not the workspace. C1 holds, C2 fails -> balancing
  explains part and the focus lock, DQN rests or something else explains the rest. C1 fails -> order is not the cause.

## Result (`experiments/results/need_order/`, commit 80d19ad, dirty=false; reference runs from `act_threshold`)
Total critical steps (mean, 100 seeds): fixed 0.85/0.6/0.3 = 1135/1064/1077; neediest 0.85/0.6/0.3 = 1135/1058/1003;
needs-only workspace 0.3 = 942.
- **C1 failed, trivially:** neediest 0.85 - fixed 0.85 = 0.0 [0.0, 0.0] in every seed. Cause found afterwards:
  `FocusedState.think` rewrites the decision after the DecisionMaker (a food focus turns 'drink_water' into
  'seek_food'), so at 0.85 the order never changes what is executed. The prediction ignored that the order switch sits
  upstream of the focus lock, although I had named the lock myself as not changed.
- **C2 failed narrowly:** neediest 0.3 - needs-only 0.3 = +61 [+31, +89] (predicted within 60). The gap to the need
  arbiter fell from 193 (fixed 0.85) to 61; of the 132 steps gained, threshold and order contribute about 58 and 74.
- **C3 held:** neediest 0.3 - neediest 0.85 = -132 [-158, -107].
- Caveat on earlier tables: `action_counts` record the decision made in `_think_default` (`_last_performed_action`),
  before the focused-state rewrite, so they are decision labels, not executed actions. The diagnostic run used the
  final action and is unaffected. The 'seek_food 895 / drink_water 75' figures at override 0.3 therefore understate food.
