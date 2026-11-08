# AE-2, second part: is a model of the commands needed, or does a model-free learner cope as well?

Written before any run of the new condition. Follows `motor_model.md`, whose open point was: "a model works and a
non-tracking model hurts" was shown, but not that a model is *needed*.

## The control
`MOTOR_MODEL = 'modelfree'`: no table of what each command does. A 4x4 table `Q[wanted][command]` holds the running
average (step `MOTOR_LR`) of whether issuing that command, when that direction was wanted, produced the wanted
displacement; the agent issues the best-valued command, or a random one with probability `MOTOR_EPSILON`. Blocked moves
teach nothing, as for the model. It starts favouring the identity command, like the other models. It is tabular, not a
neural DQN: the task has no state beyond the wanted direction, so a network would only add parameters (not tested).
The structural difference to `vote`: an observation updates one cell here (one wanted direction), whereas the model's
observation fixes what a command does for every direction at once. 16 cells to relearn after a remap against 4.

## Design
Same as `motor_model.md` (needs-only workspace, capacity 3, scenario `sites`, frontier, 1000 steps, total critical
steps, 100 seeds) on new test seeds 3700-3799, re-running none, vote and oracle too (same commit, no reuse).
- **Hyperparameter selection** for `modelfree` only, on separate seeds 3600-3629 at P=150, slip 0.15:
  LR in {0.2, 0.5, 0.8} x EPSILON in {0.05, 0.1, 0.2}; the lowest total critical steps is used everywhere. This gives
  the model-free control a tuning advantage that `vote` (window 5, fixed in advance) did not have: it biases against my
  own claim that a model matters.
- Test: P = 30, 150, 600 (slip 0.15) for none, vote, oracle, modelfree.
- Notation: A(P) = modelfree - vote (positive: the model is better).

## Predictions (my probabilities)
- **F1 (90%):** at P=150 modelfree beats none by at least 300.
- **F2 (65%):** at P=150 vote beats modelfree by at least 50 (A >= 50).
- **F3 (60%):** the model's advantage grows with volatility: A(30) - A(600) >= 100.
- **F4 (50%):** at P=600 modelfree is within 100 of vote (|A(600)| <= 100).
- **F5 (55%):** at P=150 modelfree's share of moves that went where wanted is below vote's.
- Reported but not predicted: modelfree - oracle at each P.

## Reading in advance
F2 and F3 holding: the model's generalisation across directions is worth something, and "model needed" is supported
against this competitor (the weakest form: a tabular learner on my task). F2 failing with |A| <= 50 at P=150: here a
model-free learner copes as well, and the AE-2 result of `motor_model.md` says only that *some* adaptation is needed.
F3 failing while F2 holds: the model is better but not because of faster relearning.

## Limits stated in advance
One environment designed by me; the competitor is the simplest model-free learner, so a loss for it would not exclude
a cleverer one; five predictions, no multiple-comparison correction; the explanation "16 cells vs 4" is a reading, not
tested by a separate manipulation.
