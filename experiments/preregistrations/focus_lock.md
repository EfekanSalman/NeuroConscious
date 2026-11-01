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

## Result (`experiments/results/focus_lock/`, commit 2333223, dirty=false; references from `act_threshold` and `need_order`)
Total critical steps (mean over agents, 100 seeds): rules neediest 0.3 with lock 1003 -> without lock 979; rules fixed 0.3
without lock 1290 (with lock 1077); needs-only 942; needs-only without persistence 932, without feedback 941.
(A first launch of these runs failed on unquoted string parameters before producing any output; the command line was
corrected and the same four runs repeated. Nothing else changed.)
- **D1 failed, narrowly:** no lock - lock = -24.7 [-46.2, -3.1] (predicted at least 30). The lock costs a little, not much.
- **D2 failed:** no lock - needs-only = +36.3 [+11.6, +60.9] (predicted within 30). About 25 of the 61 steps are the lock;
  about 36 remain and the interval excludes zero.
- **D3 held, in the opposite way than I framed it:** fixed order without the lock is worse than neediest without the
  lock by +311 [+289, +333], and worse than fixed *with* the lock (1290 vs 1077). Drinks fall to 7.3.
  Reading from the code, not tested: with a water focus (thirst > 0.7, hunger <= 0.7) the lock turns 'seek_food' into
  'drink_water', so in the fixed chain it acts as a crude thirst rescue. The lock is both a cost (food focus
  overwriting drinking) and a hidden priority mechanism; removing it only helps once the order is already by need level.
- **D4 failed:** no persistence - needs-only = -10.8 [-34.2, +12.6] (no effect).
- **D5 failed:** no feedback - needs-only = -1.5 [-6.1, +3.1] (no effect).
- Summary: of the named differences only the lock matters, and by about 25 of 61 steps. Persistence and attention
  feedback do nothing here. The remaining ~36 steps are not explained. Untested suspects: DQN or procedural memory when
  all needs are at or below 0.3, the Asleep/Awake split, capacity 3, how 'seek_food'/'drink_water' are executed in the
  rule agent versus the workspace. Five predictions, no multiple-comparison correction; the references were tuned on these seeds.
