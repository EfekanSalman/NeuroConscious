# RPT-1: evidence accumulated across processing cycles (ghost radar, weak signal)

Written before any selection or test run. Butlin RPT-1: input modules using algorithmic recurrence. This is the **weakest,
purely algorithmic sense** only: state carried from one processing cycle (step) to the next inside an input module. It
says nothing about spatial feedback in a perceptual hierarchy (the 3x3 view and the radar have none), and I do not claim that
it tests recurrent processing theory beyond this. RPT-2 (organised, integrated perceptual representations) is not tested.

## Task and headroom
The ghost radar of HOT-2 with a weak reliability signal: strength N(1,1) for a real site, N(0,1) for a ghost (separation 1.0).
Real sites are reported again and again at the same cell; ghosts fall on random cells and rarely repeat. Looked at before this
file (disclosed; seeds 7900-7919, 20 seeds, needs-only workspace, total critical steps): no monitor 709, never trust 968,
oracle 634, single-report threshold at tau 0.5/1.0/1.5/2.0: 970/948/864/812. So the oracle is 75 better than no monitor,
and no single-report threshold beats no monitor: there is room for a mechanism that uses more than one report.
One-seed decision logs (seed 5100, before this file): recurrent (tau 2.5, decay 0.9) admitted 53% of real and 0.4% of ghost
reports and scored 610 against oracle 502 and threshold (2.0) 772; window (sum over 20 steps, tau 2.5) 738; repeat count 2 in 20
steps 873, count 3 719. I expect the recurrent form to work; the open questions are the controls.

## Forms (new `RADAR_MONITOR` values; constants `RADAR_RECUR_*`)
- `percept_recurrent`: leaky integrator per (kind, cell): `s = s * decay**dt + strength`, admit if `s >= tau`. State carried across steps.
- `percept_window`: sum of the strengths reported for the cell in the last W steps, from a raw log; no integrator state. Same history, no recurrence.
- `percept_repeat`: admit once the cell was reported n times in the last W steps; ignores strength. Cheapest reference.
- `RADAR_RECUR_POOLED`: one state per kind for all cells (content control: evidence not tied to a cell).
- Single-report threshold (`percept_threshold`) is the feedforward control; oracle and none as before.

## Design
Scenario `sites`, needs-only workspace, ghost radar (range 8, rate high 0.5), separation 1.0, no remap or slip, 1000 steps, total
critical steps, mean over agents. Selection seeds 8000-8029, each form's own grid: threshold tau {1.5, 2.0, 2.5, 3.0};
recurrent decay {0.8, 0.9, 0.95} x tau {2.0, 2.5, 3.0, 3.5}; window W {10, 20, 40} x tau {2.0, 2.5, 3.0, 3.5}; repeat n {2, 3, 4} x W {10, 20, 40}.
The lowest of each is used. Test seeds 8100-8199 (new, 100): none, oracle, threshold, recurrent, window, repeat, recurrent-pooled
(the selected recurrent setting with one pooled state). Also reported: admitted share of real and ghost reports is not logged
by the metrics; ghosts and hits verified are.

## Predictions (my probabilities)
- **R1 (80%):** recurrent beats `none` by at least 30.
- **R2 (80%):** recurrent beats the single-report threshold by at least 60.
- **R3 (40%):** recurrent ends within 30 of the oracle.
- **R4 (70%):** window is within 30 of recurrent (history access is what matters, not recurrence as such).
- **R5 (50%):** repeat is within 30 of recurrent (the cheapest reference is as good).
- **R6 (80%):** pooled recurrent does not beat `none` by 30 or more (the evidence must belong to the cell).

## Reading in advance
R1, R2 and R6 holding: evidence accumulated across cycles helps and is causally tied to the cell, so RPT-1 is met in this weak
sense. R4 holding: recurrence is not what does it; a window of the same history does (the minimal-implementation reading).
R5 holding: even the strength is unnecessary here; repetition alone separates real from ghost, which says the task is solved by
a property of the world (real sites repeat) rather than by anything perceptual. R2 failing: accumulation did not beat the
single report and the headroom was not reachable. All of this concerns this simulator, not recurrent processing theory.

## Limits stated in advance
One world I designed; the repetition of real sites and the randomness of ghosts are my construction and make accumulation
useful by design; six predictions, no multiple-comparison correction; parameters picked on separate seeds with grids I chose;
no spatial recurrence or feedback hierarchy was tried.
