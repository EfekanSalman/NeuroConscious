# Is the workspace advantage over the rule-based agent a stormy-weather effect?

Written before any run. Follow-up to `workspace_arbitration.md`, whose P1 failed: forcing the rule agent's
fixed critical-need order on the workspace did not move its total (1248 vs 1244 with fifo10, `nearest`) and
left it 82-114 steps ahead of the rule-based agent.

## Observation that prompted it
The rule-based agent rests 149-166 steps per run, the workspace agent 87-101, although fatigue-critical steps
are equal (78-84). `DecisionMaker` (lines 367-369) forces `rest` in stormy weather whatever else was chosen;
the weather changes every 20 steps among sunny, rainy and stormy. The workspace agent ignores the weather.

## Hypothesis
Most of the gap between the two agents, and of the admission-policy interaction, comes from time the
rule-based agent loses to weather-forced rest.

## Intervention
Remove storms from the world: `environment.world:WEATHER_TYPES=['sunny','rainy']`, for all conditions.
8 conditions, the same as `memory_interaction` (workspace and rule-based agent x {salience10, fifo10} x
{nearest, recent}), seeds 100-199, 1000 steps, scenario `sites`. Compared with the existing runs (same seeds
and parameters; not paired).

## Predictions
- **S1:** the rule-based agent's rest steps fall to the workspace agent's level (at most 110 per run).
- **S2:** total critical-need steps, rule-based minus workspace at equal memory, fall into +-40 in all four
  recall x admission combinations (they were 78 to 143).
- **S3:** the average admission x workspace interaction ends within +-30 of zero (it was -56 +-16).
- If S1 holds and S2 fails: the storm costs the rule agent time but is not what makes the workspace better.
- If S1 fails: the extra rest comes from somewhere else.
