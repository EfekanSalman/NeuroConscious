# Indicator HOT-2: monitoring how reliable one's own perception is

Written before the 100-seed runs. Butlin et al. (2023) higher-order-theory indicator HOT-2: metacognitive monitoring that
distinguishes reliable perceptual representations from noise. This test does not claim consciousness. Like the three
AST problems, it first asks whether a monitor has anything to do in this simulator (headroom), and only then compares
monitors.

## Problem
A **ghost radar** (`GHOST_RADAR_RANGE = 8`): each step, per channel (food, water), the nearest real site within range is
reported with probability 0.5, and in addition a ghost (a site that is not there, at a random other cell in range) with
probability 0.01 (low regime) or 0.5 (high regime). The regime alternates every 100 steps with a random hidden phase.
Reports enter the workspace memory as `radar` sightings; one confirmed by the agent's own 3x3 view is a hit, one the view
contradicts is a ghost and is removed. `RADAR_MONITOR` decides whether recall may use radar sightings on a given step:
- `none`: always (no monitor). `never`: radar sightings are not stored (no radar). `oracle`: only in the low regime (cheats).
- `wrong`: a coin flipped each 100 steps, unrelated to the regime (capacity-matched control: same size of decision,
  uncorrelated with reality).
- `learned`: a running average (rate 0.2, start 0.5) of hits versus ghosts over the agent's own verifications; the radar
  is used while it is at least 0.6, or with probability 0.1 anyway (so that distrusted sightings still get checked).
Agent and world as in `motor_model.md` (needs-only workspace, capacity 3, memory 10 slots nearest/salience, scenario
`sites`, frontier, 1000 steps, total critical steps, mean over agents), no remap or slip. Seeds 3800-3899 (new), 100 seeds.

## Disclosed design iteration
Parameters were set after a 10-seed look (seeds 9400-9409, not reused). With high-regime ghost rates 0.15, 0.5 and 1.0
(detect 0.5) always trusting the radar (`none`: 564, 668, 744) beat the oracle (735, 786, 815) and `never` was 949, 946,
943; `learned` (rate 0.15 setting) was 970, with the trust agreeing with the oracle's on 55% of steps. So I already expect
the headroom gate to fail; 0.5 is the middle setting. I did not search for a setting where it passes.

## Predictions (my probabilities)
- **G0a, headroom (15%):** oracle beats `none` by at least 80. **G0b, leverage (90%):** oracle beats `never` by at least 100.
  The monitor has a job only if both hold.
- **H1 (80%):** `none` ends no more than 50 worse than the oracle (ghosts are cheap next to what the real detections earn).
- **H2 (70%):** `learned` does not beat `none` by 50 or more (nothing for a monitor to add).
- **H3 (65%):** `learned`'s trust agrees with the oracle's on at most 65% of steps (the estimate barely tracks the regime;
  verification events are sparse).
- Also reported: `wrong`, hits and ghosts verified per run.

## Reading in advance
G0a failing: in this world, learning when to distrust the radar has no value, so any monitor is idle and HOT-2 cannot be
tested here in this form; the result is about the world, not the theory. A per-detection reliability signal (each report
carries a noisy strength that predicts whether it is real) is a different problem and is not tested. G0a and G0b holding:
compare `learned` against `none`, `wrong` and the oracle; as before, `learned` must beat `wrong` (rate-matched noise), or
the claim is only that using the radar less helps.

## Limits stated in advance
One environment designed by me and tuned in its ghost cost only by a disclosed 10-seed look; the monitor is a running
average, the simplest possible, and a tracking failure of `learned` would be about this implementation; four predictions,
no multiple-comparison correction.
