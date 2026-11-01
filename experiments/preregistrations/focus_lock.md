# Does the focused-state rewrite explain the remaining gap to the need arbiter?

Written before any run. Follows `need_order`: rules with 'neediest' order at override 0.3 end at 1003 critical steps,
needs-only workspace at 942 (+61 [+31, +89]). Two untested differences were named there: the focus lock and the DQN.

## What the code shows (read, not run)
- In the rule agent `attention_focus` is set by `hunger > 0.7` first ('food'), then thirst > 0.7 ('water'), then fatigue.
  Mean hunger is about 0.75, so the food focus is on much of the time.
- `FocusedState.think` runs after the DecisionMaker. With a food focus and hunger > 0.3 it turns every non-move
  decision, including 'drink_water' and 'rest', into 'seek_food'. The need-order switch sits upstream of it.
- In the neediest-0.3 result the rule agent decided 'drink_water' 334 times but drank 13 times, the needs-only
  workspace 20 times. The decisions are labels from before the rewrite (see `need_order`), so this is suggestive only.
- The DQN only acts when no need is above the override level; at 0.3 that is rare (hunger mean 0.75).
  I treat it as a small suspect and do not test it here.
- The workspace agent has other differences I cannot remove from the rule agent: bid persistence (0.15 salience bonus
  for the previous winner), `level^2` salience, perceptual attention steered by the broadcast, and no `Asleep`/`Awake`
  state split in the decision. I test persistence and feedback by switching them off in the workspace agent.

## Intervention
New switch `core.consciousness.focused_state:FOCUS_REWRITES_DECISION` (default True = behaviour unchanged, verified
against `baseline.json`; the focus still boosts perception, only the rewrite of the decision is off).
Scenario `sites`, frontier on, sighting memory `nearest`/salience10, seeds 1100-1199, 1000 steps, total critical steps.
New runs (all at override level 0.3, order 'neediest' for rules):
- R1: rules, rewrite off.
- R2: rules, rewrite off, order 'fixed' (does the order still matter once the lock is gone?).
- W1: needs-only workspace, `WORKSPACE_PERSISTENCE` 0.
- W2: needs-only workspace, `WORKSPACE_FEEDBACK` False.
Reused (same seeds/settings, default unchanged): rules neediest 0.3 = 1003, rules fixed 0.3 = 1077, needs-only 942.

## Predictions (my probabilities)
- **D1 (65%):** R1 beats rules neediest 0.3 (1003) by at least 30 critical steps.
- **D2 (50%):** R1 ends within 30 steps of needs-only (942), i.e. rules with no focus lock match the arbiter.
- **D3 (55%):** R2 is worse than R1 by at least 40 (order still matters without the lock).
- **D4 (40%):** W1 (no persistence) is worse than needs-only 942 by at least 30.
- **D5 (30%):** W2 (no feedback) is worse than 942 by at least 30.
- Reading: D1 and D2 hold -> the workspace lead was the rule agent's own focus lock plus threshold and order; a rule
  agent with the same need policy does as well. D1 holds, D2 fails -> the lock explains part; the remainder is persistence,
  attention feedback, DQN or something I have not named. D1 fails -> the lock is not the cause and the 61 steps remain unexplained.

## Limits stated in advance
Same environment, same seeds as the earlier tests (reused, so the choices of threshold and order are tuned on them);
no multiple-comparison correction across five predictions; a null for D2 would not say the workspace adds something,
only that these named differences do not account for the gap. Report the results either way, with commit and dirty=false.
