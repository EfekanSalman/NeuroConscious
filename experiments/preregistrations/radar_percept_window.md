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
