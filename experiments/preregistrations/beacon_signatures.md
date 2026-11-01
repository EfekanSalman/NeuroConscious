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

## Result (`experiments/results/beacon_signatures/`, commit 9b82cc1, dirty=false; 100 seeds)
Total critical steps (cue steps of 1000; share of calm / urgent steps with a cue): wired_cap3 1415 (1000; 1.00/1.00),
wired_cap1 1449, bc_cap3 1424 (855; 0.93/0.83), bc_cap2 1544 (290; 0.92/0.16), bc_cap1_s03 1659 (15; 0.38/0.00),
bc_cap1_s06 1771 (35; 0.47/0.00), bc_cap1_s10 1804 (43; 0.47/0.01).
- **B1 held:** cue steps 855 > 290 > 35 for capacity 3 > 2 > 1.
- **B2 held, on a thin base:** share with a cue is 0.47 in calm steps and 0.00 in urgent ones (difference 0.47). But the agent is
  calm in only 18 of 1000 steps and urgent in 930 at capacity 1 (the beacon world is harsh), so the calm share rests on few steps.
- **B3 held, strongly:** bc_cap1_s06 - wired_cap1 = +322 [+306, +337].
- **B4 failed on size:** bc_cap1_s10 - bc_cap1_s06 = +33 [+20, +46] (predicted at least 100; direction right).
- **B5 held:** bc_cap3 - wired_cap3 = +9 [-4, +22].
- Not predicted: capacity 2 already loses the cue in urgent steps (0.16) and costs +129 over wired_cap3 [no interval computed;
  different capacities, same seeds]. With two needs critical at once they outrank the cue. And wired_cap1 - wired_cap3 = +34 [+21, +47]:
  capacity 1 costs something by itself (no opportunistic action).
- Reading: the cue is available exactly when needs are calm and lost when they press, as specified (by construction); the cost of
  that limit is large (up to 322 steps) when capacity is smaller than the number of concurrent contenders and zero at capacity 3.
  In this world a wire is never worse than the workspace and a workspace of capacity below 3 is clearly worse.
