# Indicator AE-2: a model of the agent's own output-input contingencies

Written before the 100-seed runs. Butlin et al. (2023) agency-and-embodiment indicator AE-2: the system models
output-input contingencies and uses the model in control. This test does not claim the agent is conscious; it asks
whether such a model causally matters in a world where it has to track a changing body, and what the simplest
implementation is worth.

## Problem
`environment.world:ACTION_REMAP_PERIOD`: every P steps the four move commands are reassigned to a random permutation
of the four directions (a new "body"); `agent.base_agent:ACTION_SLIP`: each executed move goes in a random direction with
that probability. The agent's planner, frontier exploration and resource search all name the direction they want. A
motor layer turns the wanted direction into a command (`MOTOR_MODEL`):
- `none`: issue the wanted direction as the command (no model).
- `oracle`: read the true mapping (cheats).
- `last`: remember, per command, the direction it last produced (from the displacement it caused); to go somewhere, use
  the command believed to produce it (the most recently checked one), or, if none is believed to, the command checked
  longest ago. Blocked moves carry no information.
- `vote`: the same, with the majority of each command's last 5 outcomes.
- `wrong`: a fixed random table that never updates (a model that does not track the world; the capacity-matched control).
The model is a 4-entry table. That is deliberate (minimal-implementation check): it is the simplest thing that counts as
modelling output-input contingencies.

## Design
Needs-only workspace (capacity 3, habit and goal off, threshold 0.3, workspace memory 10 slots nearest/salience),
scenario `sites`, frontier on, 1000 steps, total critical steps (mean over agents), seeds 3500-3599 (new), 100 seeds.
- Main (P = 150, slip 0.15): none, wrong, last, vote, oracle.
- Volatility: P = 30 and P = 600 (slip 0.15) for none, vote, oracle.
- Noise: slip 0 and 0.3 (P = 150) for none, last, vote.
Informed by a 10-seed smoke run on other seeds (9300-9309), which I looked at while debugging: it showed the first
version of `last` doing nothing (no command was tried when none was believed to produce the wanted direction; fixed
before this registration) and a rough ordering none 1780, oracle ~1006, last ~1250, vote ~1090 at P=150, slip 0.15.
So H1-H4 are not blind; H5-H6 are the open ones.

## Predictions (my probabilities)
- **H1 (95%):** oracle beats none by at least 500 critical steps (a changing body is a real problem, and knowing it is worth a lot).
- **H2 (90%):** vote beats none by at least 400 (P = 150, slip 0.15).
- **H3 (75%):** wrong is not better than none by more than 100 (a model that does not track the world is no help).
- **H4 (60%):** vote ends within 150 of oracle.
- **H5 (70%):** vote's gain over none is smaller at P = 30 than at P = 600 by at least 200 (the model needs time to relearn).
- **H6a (70%):** at slip 0.3, vote beats last by at least 50. **H6b (65%):** at slip 0, last beats vote by at least 50
  (voting costs speed when there is no noise).
- Also reported: the share of moves that went where the planner wanted (`motor_as_intended` / `motor_moves`).

## Reading in advance
H1-H3 hold -> the contingency model is what separates a working body from a broken one here, and a dummy model with the
same size gives nothing: causal dependence in the sense of the framework. That is also the weakest possible form of the
indicator: a 4-entry table passes it, so passing says little about the indicator as a marker of consciousness and a lot
about how cheaply it is met. H1 failing would mean the agent does not depend on its own motor mapping in this world
(unlikely). H4 failing means the learning cost, not the idea, is the limit.

## Limits stated in advance
One environment designed by me; the table and its fallback rule are my own engineering, so a null for `last`/`vote` would
be about my implementation; seven predictions, no multiple-comparison correction; no learned-network control (a DQN that
relearns after remaps) was built, so "model-free would do worse" is not tested here.
