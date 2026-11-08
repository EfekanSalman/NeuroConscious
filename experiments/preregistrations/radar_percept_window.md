# HOT-2, eighth part: is the learned threshold too noisy (label window)?

Written before any run of this test. `radar_percept_leak.md`: with the leak removed, the learner that gets correct labels for
every offered report (`free_after`) is no better than no monitor (823 vs 813), while the hand threshold 2.5 reaches 648.
Decision log of one seed (post hoc, read before this prereg): the learner admitted 52% of real and 3% of ghost reports, the
hand threshold admits about 31% and 0.6%; the learned threshold varies a lot (quantiles 10%/50%/90%: 1.34 / 1.99 / 2.77,
minimum -0.31). A few admitted ghosts are costly, so my reading is that the threshold, fitted on a window of 60 labels, is too
noisy. Direct test: a longer window.

## Design
Mode `free_after`, ghost weight 30, probe 0, window `RADAR_LABEL_WINDOW` in {60, 200, 1000} (existing constant). Seeds
6000-6049 (new, 50). Also none and the hand threshold (2.5). Needs-only workspace, ghost radar as before, total critical steps.
Also reported: reports admitted / offered, ghosts verified.

## Predictions (my probabilities)
- **E1 (60%):** window 1000 beats window 60 by at least 100.
- **E2 (45%):** window 1000 ends within 60 of the hand threshold.
- **E3 (55%):** window 200 lies between 60 and 1000 (monotone).

## Reading in advance
E1 and E2 holding: a learned per-percept monitor works given correct labels and enough of them; the failure was noise in the
threshold. E1 failing: noise in the threshold is not the cause (admission quality or the ghost-cost trade-off is), and I stop.

## Limits stated in advance
Eighth test on one radar, shaped by what I saw; labels are free and correct (an upper bound); one environment; three
predictions, no correction.

## Result (`experiments/results/radar_percept_window/`, commit 37763ba, dirty=false; seeds 6000-6049, total critical steps)
none 712 (170 ghosts verified), hand threshold 611 (7), learned `free_after` window 60 **785** (19; 314 of 1046 offered reports
admitted), window 200 722 (13; 217/1032), window 1000 **719** (12; 203/1036).
- **E1 failed:** window 1000 - window 60 = -66 [-102, -29] (right direction, predicted at least 100 better). **E2 failed:**
  window 1000 - hand threshold = +108 [+72, +145]. **E3 held:** 785 > 722 > 719.
- Window 1000 - none = +7 [-37, +52]: even with a stable threshold the learner is no better than trusting everything.
- So noise in the threshold costs about 65 steps (window 60 is worse than no monitor), but removing it does not make the
  learned monitor useful. The learned threshold admits ~20% of reports and lets ~12 ghosts through, the hand threshold lets 7
  through; I do not know why that gap costs ~100 steps, and I stop here (eighth test on this radar).
