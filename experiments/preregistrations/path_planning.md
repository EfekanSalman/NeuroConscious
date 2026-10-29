# Does path planning remove the navigation ceiling on memory?

Written before any run. Every memory and controller result so far used greedy navigation: step toward the
remembered target, give it up when every closer step is blocked. With unbounded memory, 5-9 of 40 agent-runs
earlier ended up trapped. With `USE_PATH_PLANNING` the agent keeps a map of the obstacles it has seen and
walks a shortest path around them (breadth-first, unknown cells passable). The planner is the only change.

## Design
Scenario `sites`, rule-based agent, `nearest` recall, seeds 300-399, 1000 steps. 8 conditions:
{legacy 5-item buffer, fifo 10, salience 10, fifo 100} x {greedy, planned}.
Primary metric: total critical-need steps. Secondary: distinct cells visited.

## Predictions (my probabilities)
- **A (about 60%):** planning lowers total critical steps in every memory condition (95% interval excluding
  zero in at least three of four), and its effect is larger the more locations are remembered: the effect
  with fifo 100 is at least as large as with the legacy buffer.
- **B (about 50%):** competitive admission keeps at least half of its benefit under planning
  (salience 10 minus fifo 10, planned, versus greedy).
- If A fails: navigation traps were not what limited memory in this world, and planning is not worth keeping on.
