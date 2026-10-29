# Homeostatic controller combined with the competitive sighting memory

Written before any run. The controller (learned, drive-reduction reward) chooses which need to serve; the
sighting memory says where. They cannot be combined with the workspace, so the memory is attached directly
(`SIGHTING_MEMORY_*`), as in the earlier control experiments.

## Scale, from a pilot on seeds 900-903 (not part of the test)
In scenario `sites` the mean drive is near its ceiling: about 1.19 for the rule-based agent and 1.17 for the
controller (possible maximum 1.73), standard deviation across agent-runs about 0.03. There is little room.

## Design
Scenario `sites`, seeds 200-229, 3000 steps (the controller needs time to learn). 7 conditions:
rule-based and controller x {legacy 5-item buffer, `fifo` 10 slots, `salience` 10 slots}, all `nearest`
recall; plus a random-option control (controller flag on, `CONTROLLER_EPSILON_DECAY=1.0`) with `salience` 10.
Primary metric: mean drive. Secondary: total critical-need steps.

## Predictions (stated with my probabilities)
- **H1 (about 40%):** the controller beats the rule-based agent in mean drive by at least 0.02 with the
  legacy buffer (95% interval excluding zero). The pilot suggests about 0.015.
- **H2 (about 40%):** competitive admission lowers mean drive by at least 0.02 against `fifo` 10, in both
  the rule-based agent and the controller (intervals excluding zero).
- **H3 (random control, about 75%):** the controller with salience 10 beats the random-option control
  by at least 0.03.
- Complementarity (controller x admission interaction) is reported with its interval, with no prediction.
If H1 and H2 both fail the result is "no usable room in this world", not evidence against either component.
