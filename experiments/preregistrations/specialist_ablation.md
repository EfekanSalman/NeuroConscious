# Which specialist gives the workspace agent its lead over the rule-based agent?

Written before any run. Two hypotheses (fixed need order, stormy-weather rest) were refuted; the workspace
agent is still 96-134 steps ahead (`frontier_interaction`).

## What the earlier data already showed (frontier exploration, salience10, `nearest`)
Winners per run: need:hunger ~440, need:thirst ~355, habit ~140, goal 19, need:fatigue 10. Actions: seek_food
460 and drink_water 373 (833 searching) against 715 for the rule-based agent, which rests 211 times against
123. The workspace agent's rests come almost entirely from the habit specialist (a DQN that wins whenever
every need is under about 0.55). So the lead might be time: more steps searching, fewer resting.

## Intervention
`core.workspace.workspace:DISABLED_SPECIALISTS` leaves specialists out; when nothing bids the agent idles
(`explore`, a stationary no-op). Scenario `sites`, frontier exploration on, workspace agent with `nearest`
salience 10 (parameters of `frontier_interaction/nearest_salience10`), seeds 1000-1099, 1000 steps, 5 conditions:
full, no habit, no goal, no habit and no goal (needs only), and the rule-based agent as reference.
Metric: total critical-need steps. Also reported: search and rest steps.

## Predictions (my probabilities)
- **A1 (60%):** without the habit the total is lower than the full workspace by at least 30 (interval excluding zero).
- **A2 (80%):** without the goal specialist the total differs from the full workspace by less than 30.
- **A3 (55%):** needs-only differs from no-habit by less than 30.
- **A4 (60%):** needs-only still beats the rule-based agent by at least 50.
- If A1 and A4 hold, the lead is need-driven action selection without the rule chain's rests and pushes, and the
  habit is a cost; "global workspace" would not be what gives the lead. If A4 fails, some part of the lead
  needs the habit or goal.
