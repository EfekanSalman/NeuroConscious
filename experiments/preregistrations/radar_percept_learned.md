# HOT-2, third part: why the outcome-trained monitor failed, and whether a better-specified one can find the threshold

Written before the runs. In `radar_percept_monitor.md` a hand-set threshold reached the oracle (605 vs 600) and my
outcome-trained monitor did not (923, no better than no radar). I wrote that the likely causes were biased, sparse labels
and the 1:1 weighting of a hit against a ghost, and that this was a reading. Before this pre-registration I took a
diagnostic look (20 seeds 9500-9519, the same seeds as the first look, disclosed) at the learner's label source, ghost
weight and probe rate. It changes what I should say about the first result.

## What the diagnostic look showed (not pre-registered)
Mean total critical steps on 9500-9519 (always trusting 813, oracle 605, hand threshold 2.5 648):
- 'free' labels (every offered report labelled with the truth at once: a cheat that removes the label-source problem),
  probe 0.1: ghost weight 1 953, 3 937, 12 919. With probe 0: weight 1 932, 3 837, 12 **647**.
- 'verified' labels (only reports the agent admitted and checked itself), weight 12: probe 0.1 924, 0.02 869, 0 860.
So the first failure came mostly from two choices of mine: a 10% probe admits a tenth of an overwhelming flow of ghosts
(distinct cells, each filling a memory slot), and a ghost weight of 1 undervalues how much more a ghost costs than a hit is
worth. With both fixed and free labels the learner reaches the hand rule. The label source (verified only) still holds it
back (about 860).

## The learners
Same radar, world, agent and metric as `radar_percept_monitor.md` (ghost radar range 8, strength N(2,1) real / N(0,1) ghost,
regimes 100 steps, no remap or slip). `percept_learned` as before (threshold maximising hits - w * ghosts over the last 60
labelled reports, everything admitted until 10 labels, probe chance `RADAR_PROBE` of admitting anyway) with two knobs:
- `RADAR_LABELS = 'verified'` (labels from reports the agent admitted and then confirmed or contradicted with its own view)
  or `'free'` (every offered report labelled with the truth: a cheating control for the label source);
- `RADAR_GHOST_WEIGHT` w.
Selection on seeds 4100-4129: for 'verified', w in {3, 12, 30} x probe in {0, 0.02, 0.05} (9 runs); for 'free', w in
{3, 12, 30} with probe 0 (3 runs). The lowest total in each family is used on the test seeds.
Test seeds 4200-4299 (100 seeds, new): none, never, percept_oracle, percept_threshold (tau 2.5, chosen earlier on 3900-3929),
learned-free and learned-verified with their selected settings.

## Predictions (my probabilities)
- **L1 (70%):** learned-free beats `none` by at least 50.
- **L2 (60%):** learned-free ends within 60 of the hand threshold.
- **L3 (20%):** learned-verified beats `none` by at least 50.
- **L4 (65%):** learned-free beats learned-verified by at least 100.
- **L5 (55%):** learned-verified beats `never` by at least 50 (it at least keeps part of the radar's value).
- Also reported: share of reports admitted, hits and ghosts verified, the selected settings.

## Reading in advance
L1, L2 and L4 holding with L3 failing: given plentiful, unbiased labels and a ghost weight that reflects what a ghost costs,
an outcome-trained monitor does find the threshold, and what a realistic agent lacks is the label source: it learns about a
report only if it admitted it and checked it. The weight is still a number I set, so "learned" here means "the threshold is
fitted" and not "the agent works out what a ghost costs". L3 holding would mean checked-only labels are enough. L1 failing:
the 20-seed look was noise.

## Limits stated in advance
One environment designed by me; settings found after disclosed looks on the seeds above; the free-label learner cheats by
construction and is a control, not a candidate; the ghost weight is chosen from three values on 30 seeds; five predictions,
no multiple-comparison correction.
