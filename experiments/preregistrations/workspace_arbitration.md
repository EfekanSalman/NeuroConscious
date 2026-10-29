# Why the workspace agent gains more from competitive memory admission than the rule-based agent

Written before any run. Follow-up to `memory_interaction.md`, whose pre-registered rule said to open the
cause as a new question. Interaction found: -56 +-16 steps (workspace benefits about twice as much).

## What the earlier data already showed (seeds 100-199, `sites`, 10 slots, `nearest`)
Hunger-critical steps are equal for the two agents (about 550). Thirst-critical steps differ: workspace
517-566, rule-based 658-694; drinks 14-15 vs 9-11. Opportunistic consumption is used 4-5 times per agent,
too rarely to matter. Both agents eat and drink at a median level of 0.99, i.e. in this world needs sit at
the ceiling. `DecisionMaker` checks `hunger > 0.85` before `thirst > 0.85`, so with both critical it always
serves hunger first and starves thirst; the workspace ranks critical needs by their level.

## Hypothesis
The workspace advantage comes from arbitrating between simultaneously critical needs, and the larger
benefit of admission follows from it (serving both needs makes both kinds of remembered location useful).

## Intervention
`core.workspace.specialists:FIXED_NEED_PRIORITY=True` makes critical needs outrank each other in the rule
agent's fixed order (hunger, thirst, fatigue). Non-critical behaviour is unchanged. New conditions: workspace
agent with it on x {salience10, fifo10} x {nearest, recent} recall, 100 seeds (100-199), 1000 steps, scenario
`sites`. Compared with the 8 existing `memory_interaction` conditions (same seeds, same parameters; the
runs are not paired).

## Predictions
- **P1:** with fixed priority the workspace agent's thirst-critical steps rise to the rule-based level
  (within +-60 of 660) and its total critical steps end within +-40 of the rule-based agent's at equal memory.
- **P2:** the average interaction (fixed-priority workspace minus rule-based) falls to within +-30 of zero.
- If P1 holds and P2 fails: arbitration explains the level but not the larger benefit of admission.
- If P1 fails: the thirst gap is not the fixed order; the cause is elsewhere (navigation, timing of action).
