# Does the benefit of competitive memory admission depend on the global workspace?

Written before any run. Follow-up to `memory_sites.md`, where the benefit of salience-based admission
(salience10 vs fifo10) was clear for the workspace agent and not significant for the rule-based agent with
the same memory. An exploratory difference-of-differences over six earlier comparisons found no significant
interaction, with intervals of +-80 to +-220 (underpowered).

## Design
Scenario `sites` (unchanged), seeds 100-199 (disjoint from the earlier 0-29), 1000 steps, 8 conditions:
{workspace agent, rule-based agent with the same sighting memory} x {salience10, fifo10} x
{nearest, recent} recall. Parameters are those of the earlier `gwt_memory_uniform` / `gwt_memory_control_uniform`
conditions.

## Primary measure
Interaction = (effect in the workspace agent) - (effect in the rule-based agent), where effect is
total critical-need steps for salience10 minus fifo10. Averaged over the two recall rules, so that no
recall rule is chosen because it showed the larger sign. Interval: normal approximation, 95%.

## Interpretation rule (fixed now)
- Interval excludes zero: the workspace agent benefits from admission differently; open the cause as a new question.
- Interval inside +-50: no interaction beyond 50 steps; the benefit belongs to the memory.
- Otherwise: undetermined.

## Prediction
Mean interaction slightly negative (about -40), interval not inside +-50, so most likely "undetermined"
(about even odds). Expected standard error of the average interaction: about 23.
