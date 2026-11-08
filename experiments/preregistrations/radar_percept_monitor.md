# HOT-2, second form: a reliability signal on every percept

Written before the 100-seed runs. `radar_monitor.md` closed the regime-level form (always trusting the radar beat the
oracle monitor). The untested variant named there: each report carries its own noisy reliability signal, and a monitor
has to decide per report. This is the closer reading of Butlin et al.'s HOT-2 (telling reliable representations from
noise). It does not claim consciousness.

## Problem
Same ghost radar as `radar_monitor.md` (range 8, real nearest site reported with probability 0.5 per channel and step,
ghosts at 0.01 / 0.5 per channel and step in alternating hidden regimes of 100 steps), with `GHOST_STRENGTH_SEP = 2`: every
report carries a strength drawn from N(2, 1) if it is real and N(0, 1) if it is a ghost; the agent sees the strength, not
the truth. A radar sighting is confirmed by the 3x3 view (hit) or removed when the view contradicts it (ghost), as before.
Monitors (`RADAR_MONITOR`), all acting per report on admission to memory:
- `none`: admit all. `never`: no radar.
- `percept_oracle`: admit exactly the real reports (cheats; the upper bound).
- `percept_threshold`: admit a report whose strength is at least tau; the best tau is chosen on selection seeds 3900-3929
  from {1, 1.5, 2, 2.5, 3, 3.5} and used on the test seeds. This is a rule set by the designer who knows the signal's scale.
- `percept_random`: admit each report with probability 0.3, strength ignored (the capacity-matched control).
- `percept_learned`: keeps the last 60 verified (strength, hit?) pairs, admits a report whose strength is at least the
  threshold that maximises hits minus ghosts among them (everything until 10 pairs exist), or with probability 0.1
  anyway so that weak reports still get verified. It learns from outcomes only; it is not told the scale.
Agent, world and metric as in `radar_monitor.md` (needs-only workspace, capacity 3, memory 10 slots nearest/salience,
scenario `sites`, frontier, 1000 steps, total critical steps, mean over agents, no remap or slip), test seeds 4000-4099.

## Disclosed looks
Not blind. On seeds 9500-9519 (20 seeds, not reused here) I looked at: always trusting 813, the oracle 605, fixed tau
1.5 / 2 / 2.5 / 3 / 3.5 = 834 / 690 / 648 / 716 / 809, tau 1 = 952, random-0.3 1017, learned 958, no radar about 960. I also
looked at a variant where ghosts are only removed once the agent stands on them (oracle gain similar) and dropped it.
The learned monitor was written once and not changed after seeing its 958; I expect it to fail and wrote the predictions so.

## Predictions (my probabilities)
- **G0, headroom (85%):** `percept_oracle` beats `none` by at least 100.
- **G1, the signal is usable (70%):** the best `percept_threshold` beats `none` by at least 50.
- **H1 (10%):** `percept_learned` beats `none` by at least 50.
- **H2 (75%):** the best `percept_threshold` beats `percept_learned` by at least 100.
- **H3 (55%):** `percept_learned` beats `percept_random` by at least 40.
- Also reported: share of reports admitted, hits and ghosts verified, the chosen tau.

## Reading in advance
G0 and G1 holding with H1 failing: a per-percept reliability signal gives a monitor something to do (causal dependence),
a tuned rule uses it well, and my outcome-trained monitor does not find the rule: HOT-2 is met only in the weak form where
the designer sets the threshold, and the learned form is a failure of this learner, not a test of the idea. H1 holding:
learning the reliability from outcomes is enough here. G0 failing: the headroom seen in the 20-seed look was noise.
`percept_learned` not beating `percept_random` (H3 failing): its signal use is no better than chance admission.

## Limits stated in advance
One environment designed by me and set after disclosed looks on other seeds; the signal is synthetic (Gaussian,
d' = 2) and the learned monitor is the simplest one I could write, so a failure is about this implementation; five
predictions, no multiple-comparison correction.
