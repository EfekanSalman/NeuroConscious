# Beacon task, part B: what limited capacity does to the cue

Written before any run, after `beacon_performance` (part A: all five predictions held; broadcast = wiring in performance).

## What this part can and cannot show
Part A could not separate the workspace from a direct wire. The one place where they differ by design is capacity:
a wire delivers the cue every step, the workspace only on steps when the cue wins one of its slots (GWT-2). So
this part measures the cost and the shape of that limit. **It is not independent evidence for the theory:** the limit
is something I put into the implementation, so a lower cue availability under a smaller capacity is a check that the
code obeys its specification (the first of the criteria), not a discovery. What is not by construction is the size of
the behavioural cost and whether it depends on need pressure. A claim that the cue reaches all consumers at once is
also true by construction here (one variable, `agent.regime_cue`), so it is not tested.

## Change made for this part
When the cue takes every admitted slot (capacity 1) it is broadcast and the agent idles (`explore`, stationary) for that
step; before, the cue was silently dropped in favour of the best action bid. For capacity 3 nothing changes (the cue
can never take all slots), so part A's results stand. The tracker now also reports cue availability by need pressure:
steps with the highest need below 0.6 (`calm`) and above 0.85 (`urgent`).

## Conditions
Scenario `beacon`, frontier exploration, lean workspace agent as in part A (habit and goal off, threshold 0.3),
seeds 2200-2299, 1000 steps, metric total critical-need steps.
Controls (no beacon bid, the cue is wired): `wired_cap3`, `wired_cap1` (capacity 1 changes the agent by itself:
no opportunistic action, one content per step; this control isolates the competition for the cue).
Broadcast (`BEACON_ACCESS` 'broadcast'): `bc_cap3` (salience 0.6), `bc_cap2` (0.6), `bc_cap1_s03`, `bc_cap1_s06`, `bc_cap1_s10`
(capacity 1 and beacon salience 0.3 / 0.6 / 1.0; a need bids level squared, plus 0.5 above 0.85).

## Predictions (my probabilities)
- **B1 (85%):** cue availability (steps with a cue) falls with capacity at salience 0.6: `bc_cap3 > bc_cap2 > bc_cap1_s06`, each step at least 50 of 1000 steps.
- **B2 (80%):** at `bc_cap1_s06` the cue is available in calm steps and rarely in urgent ones: share of calm steps with a cue minus share of urgent steps with a cue is at least 0.4.
- **B3 (75%):** the limit costs performance: `bc_cap1_s06 - wired_cap1 >= 30` critical steps.
- **B4 (70%):** a cue that is too salient displaces action: `bc_cap1_s10 - bc_cap1_s06 >= 100`.
- **B5 (60%):** part A's result replicates: `|bc_cap3 - wired_cap3| < 40`.
- Reading: B1 and B2 holding show the cue's availability is governed by competition with need pressure, as specified
  (expected, by construction). B3 holding gives the cost of the limit that a wire does not pay; if B3 fails the limit is
  free in this world. Neither tests the theory; the finding is the size of the cost.
