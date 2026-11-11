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

## Result (`experiments/results/predictive_regime/`, commit 3acc3de, dirty=false; seeds 7100-7199, total critical steps)
none 1586, wired 1414, last 1604 (cue on 939 steps, 534 correct), **filter 1512** (746; 558; selected cap 4, hazard 1/150, theta 0.5),
filter_sightings 1464 (751; 611; selected cap 2, hazard 1/50, theta 0.5), filter_inverted 2021 (993; 211).
- **P1 held:** filter - none = -75 [-92, -58]. **P2 held:** filter - last = -92 [-121, -64]. **P3 failed:** filter - wired =
  +98 [+81, +116]. **P4 held:** inverted - none = +434 [+419, +449]. **P5 failed, in the opposite direction:** filter -
  filter_sightings = +48 [+33, +63]: the empty-cell prediction errors made the belief worse.
- Not predicted: `last` is no better than no cue (+18 [-11, +46]); filter_sightings is the best inferred form (-123 vs none,
  +50 from wired). What separates it from `last` is that its belief decays and the cue lapses when unsure, so it stops
  steering toward a half it has not seen resources in for a while; a wrong cue is expensive. The cue is correct on
  only 60-80% of the steps it exists, against 100% for the sensor.
- So the working ingredient looks like abstention, not the prediction-error learning of the resource rate. This is the
  explanation I will test next, below.

## Addendum (written before running it): is a cue that lapses enough?
Post hoc, motivated by the result above. A cheap reference with the abstention but no belief model and no error signal:
`last_ttl` (the half of the latest sighting, held for REGIME_TTL steps, then no cue). Selection seeds 7000-7029, TTL in
{20, 40, 80, 160}; test seeds 7200-7299 (new, 100): none, wired, filter_sightings (the selection above), last_ttl.
- **Q1 (60%):** last_ttl beats none by at least 60.
- **Q2 (50%):** last_ttl is within 40 of filter_sightings.
- Reading: Q1 and Q2 holding: the lapse is what helps, and a timer does what the filter does (predictive processing is not
  needed in this weak sense). Q2 failing with Q1 holding: the graded belief beats a timer. Q1 failing: the lapse is not enough.
