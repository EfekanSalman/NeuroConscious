# PP-1: infer the regime from prediction errors instead of being told (beacon world)

Written before any selection or test run. Butlin PP-1 asks for input processing by a predictive model whose error signals
drive updating. Test-bed form: in the `beacon` world (only one half of the grid holds resources, redrawn every 150 steps) the
agent gets **no cue** (`BEACON_ACCESS='inferred'`) and must infer the active half from what it sees. The headroom is the known
value of the cue: no cue 1584 vs wired cue 1422 (`beacon_performance.md`, -162), and a wrong cue is far worse than none.

## The model (`core/navigation/regime_filter.py`)
A generative model with two hypotheses (left/right half active). A resource seen in a half is decisive evidence for that half
(the dormant half is emptied). A cell seen empty is weak evidence against its own half: in the active half a cell holds a
resource with probability r, so each empty cell shifts the log-odds by -log(1-r). Belief decays toward 0 (hazard), r is learned
online by a delta rule on the prediction error (seen - r) on cells of the half believed active. The cue is the believed
half, or none while |log-odds| < theta. The cue enters through the existing consumers (recall, frontier exploration).
Modes: `filter` (above), `filter_sightings` (the same, without evidence from empty cells: the sighting-driven part only),
`last` (the half of the latest resource sighting, held: the cheapest reference; no model, no error signal).

## Looks before this file (disclosed)
Seeds 7900-7909, 10 seeds, defaults (theta 1, cap 2, hazard 1/150, r0 0.1, lr 0.02): none 1609, wired 1437, last 1587,
filter 1596, filter_sightings 1514. The inferred cue was correct on only about 60% of the steps where it existed (last: 532 of
892; filter: 369 of 575). I expect a modest gain at best and that the empty-cell evidence does not help at these settings.

## Design
Scenario `beacon`, needs-only workspace (capacity 3, habit/goal off, threshold 0.3, memory 10 slots salience/nearest),
frontier exploration, 1000 steps, total critical steps, mean over agents. Selection seeds 7000-7029: for `filter` and for
`filter_sightings` separately, the lowest of cap {2, 4} x hazard {1/150, 1/50} x theta {0.5, 1.0}. Test seeds 7100-7199
(new, 100): none, wired, last, filter (selected), filter_sightings (selected), filter_inverted (the selected filter acting on
the opposite half: content control). Also reported: cue steps and correct cue steps.

## Predictions (my probabilities)
- **P1 (80%):** the selected filter beats `none` by at least 60 (the inferred cue is worth something).
- **P2 (45%):** the selected filter beats `last` by at least 40 (the learned prediction model beats the cheap reference).
- **P3 (25%):** the selected filter ends within 60 of `wired`.
- **P4 (85%):** `filter_inverted` is worse than `none` by at least 100 (the content matters; wrong beliefs are costly).
- **P5 (30%):** the selected `filter` beats the selected `filter_sightings` by at least 20 (the empty-cell prediction errors add
  something beyond sightings).

## Reading in advance
P1 and P4 holding: the inference works and is causally used. P2 holding: a predictive model earns more than the cheapest
reference. P2 failing with P1 holding: PP-1 is met in the weak sense that a belief updated from sensory evidence helps, and
`last` does the same job (the repeated pattern here: the cheap reference sits near the model). P5 failing: the error-driven part
(learned resource rate) adds nothing at this world's resource density (about 3% of cells), and that is a statement about this
world, not about predictive processing. Also expected: even the best inferred cue stays far from `wired`, because a 3x3 view
sees too little of a 20x20 world to know the regime fast.

## Limits stated in advance
One world I designed; the belief model's structure (two hypotheses, decisive sightings) is mine and nearly identical to the
`last` reference; five predictions, no multiple-comparison correction; defaults looked at on 10 other seeds; a 3x3 view
and a 150-step regime.
