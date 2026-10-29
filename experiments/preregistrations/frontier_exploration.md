# Is exploration the bottleneck? Frontier exploration against the random walk

Written before any run. `path_planning.md` ended with a failed prediction: agents spend about 870 of 1000
steps searching and eat about 15 times, so the limit looks like finding resources, not reaching seen ones.
Today, with nothing remembered, the agent steps to its least-visited neighbouring cell (a random walk).

## Intervention
`USE_FRONTIER_EXPLORATION`: with nothing remembered, walk (breadth-first, around known obstacles) to the
nearest cell that has not been in view for 100 steps, or never. 100 is fixed now (regrow time is 50).
Everything else is unchanged; path planning stays off.

## Pilot (seeds 900-903, 600 steps, not part of the test)
Meals 9.25 with frontier exploration, 8.75 with the random walk: little difference. Distinct cells
walked on: 247 vs 316.

## Phase 1
Scenario `sites`, rule-based agent, `nearest` recall, seeds 700-799, 1000 steps, 8 conditions:
{legacy buffer, fifo 10, salience 10, fifo 100} x {random walk, frontier}. Metric: total critical-need steps.
- **E1 (about 45%):** frontier exploration lowers total critical steps in all four memory conditions
  (95% interval excluding zero in at least three).
- **E2 (about 25%):** the effect with the legacy buffer is at least 100 steps.
- If E1 fails: exploration is not the bottleneck either, and the ceiling is the world's difficulty
  (resources are scarce and regrow slowly), not agent navigation.

## Phase 2 (run only if E1 holds in at least three of four conditions)
With frontier exploration on, seeds 800-899, 1000 steps, scenario `sites`:
- the workspace-vs-rules admission interaction (the 8 `memory_interaction` conditions), and
- controller vs rule-based agent with the legacy buffer and with salience 10 (30 seeds, 3000 steps, seeds 200-229 as
  in `controller_memory`).
Predictions: the workspace agent keeps at least 50 steps of lead over the rule-based one (about 70%); the
rule-based agent with salience 10 is still not worse than the controller (about 60%).
