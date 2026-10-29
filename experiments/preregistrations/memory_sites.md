# Memory in a world with persistent resource locations

Written before any run. Scenario `sites` (see `experiments/scenarios.py`): 20x20, 15 obstacles, 6 fixed
food sites and 6 fixed water sites per world, a consumed resource regrows in place after 50 steps.
The scenario and the parameters below are fixed; they will not be changed after seeing results.

## Why
In the random-regeneration world a remembered location is stale once the resource is eaten, and the
competitive-memory benefit measured earlier did not replicate (10 slots salience 893 vs FIFO 928).
If memory has no value even where locations persist, the memory line is finished for this agent.

## Conditions (30 seeds x 1000 steps each)
Workspace agent (as in `gwt_memory`): `nearest` and `recent` recall x {fifo5, fifo10, fifo100, salience5,
salience10}, plus `gwt_legacymem`. Rule-based agent: `rules_fifo5`, `rules_fifo100`, and with the same
sighting memory `rules_{nearest,recent}_{fifo10,salience10}`. 17 conditions. Metric: total critical-need steps.

## Predictions
- **A (memory has value here):** unbounded memory (fifo100) beats fifo5 by at least 100 steps with a 95%
  interval excluding zero, for both recall rules. If this fails, memory is worthless in this
  architecture even with persistent locations (greedy navigation, no planning) and the memory line stops.
- **B (admission policy):** salience10 beats fifo10 by at least 50 steps, interval excluding zero.
  I give this about one chance in three: the earlier effect vanished, and may have come from clustered
  resources. A holds and B fails would mean capacity matters and admission does not.
- **C (workspace adds little over the same memory in a rule-based agent):** the workspace agent is
  still ahead of the rule-based agent at equal memory, by 50-200 steps, through its opportunistic
  consumption and not through broadcast.

## Known limits
The memory drops a sighting when it sees the cell empty, which discards a regrowing site; this caps the
benefit and is left as is. One world design, one regrow time.
