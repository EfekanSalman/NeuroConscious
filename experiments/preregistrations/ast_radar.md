# AST problem 3: attention gates a long-range sensor

Written before any run of this problem. Follows `attention_gate_headroom` (problem 2), which failed because perception
was a weak lever in this simulator (3x3 view; the agent finds resources by walking onto cells and from memory) and an
orienting step cost more than the better perception earned. Here attention gates a sensor that matters.

## Problem
`ATTENTION_GATE` as before (hidden `attention_actual`, flipped to the other channel by capture, moved to the intended
channel only by an orienting step that replaces the action with a stationary one) plus a **radar**
(`ATTENTION_RADAR_RANGE`): each step, for each channel, the nearest site of that kind within Manhattan range 8 is
detected with probability 0.9 if attention is on that channel, else 0.0. A detection enters memory like a sighting. Radar
and the 3x3 view are both gated. Agent: needs-only workspace (capacity 3, habit and goal off, threshold 0.3, workspace
memory 10 slots nearest/salience), scenario `sites` (6 food + 6 water sites, regrow 50), frontier exploration,
1000 steps, total critical steps (mean over agents). Capture rate 0.05 per step.
**All parameters above are fixed now. If a gate fails I will not retune them to rescue the problem.**

## Policies
none (never orient), oracle (cheats: orients exactly when wrong, costs the step), oracle_free (same, no step),
and hand policies using only what the agent can observe: switch (orient when the intended channel changes),
interval k in {10, 20, 40} (switch plus every k steps), silence m in {4, 8, 16} (switch plus after m steps without
seeing the intended resource). The best hand policy is chosen on selection seeds 3000-3029 and evaluated on test
seeds 3100-3199 (never used before). none, oracle and oracle_free are run on the test seeds.

## Gates (a failed gate closes the problem; no learned schema is built)
- **G0, leverage (55%):** oracle_free beats none by at least 150 critical steps. (If correct attention is not worth
  that much even for free, there is nothing to control.)
- **G1, cost (45%):** oracle (paying the orienting step) beats none by at least 100.
- **G2, headroom for a schema (35%):** the chosen hand policy is at least 60 worse than oracle, i.e. simple
  observable-only rules leave something for a model of attention to add.
If all three hold, Stage 1 (written then, with its own pre-registration): a learned attention schema (predicts the hidden
attention state from the agent's own recent history and efference copy, trained on a self-supervised target) against a
model-free controller with the same history window, a random-weight schema and the best hand policy. The decision rule
there will say that a schema must beat the model-free controller with the same inputs, otherwise the claim is only
"history helps".

## Reading in advance
G0 failing again would say perception, even long-range, is not a lever in this world. G0 and G1 holding while G2 fails
would say a hand rule suffices and a schema adds nothing here. The most likely failure, in my judgement, is G2: a
silence rule sees most captures because a missing detection is itself the evidence.

## Limits stated in advance
One environment and one set of parameters, designed by me; no multiple-comparison correction across three gates; a
success of Stage 1 would show a controller benefit in this task, not that anything is conscious or that this is how
attention schema theory's mechanism works.

## Result (`experiments/results/ast_radar/`, commit 7a7f33e, dirty=false)
Selection (30 seeds, hand policies): switch 781, interval 10/20/40 = 929/790/748, silence 4/8/16 = 1293/946/820; chosen interval 40.
Test (100 seeds, total critical steps; share of steps with attention on the intended channel; orients per 1000 steps):
none 683 (44%, 0), oracle 971 (95%, 89), oracle_free 772 (95%, 88), chosen hand policy (interval 40) 703 (69%, 60).
- **G0 failed, in the opposite direction:** oracle_free - none = +89 [+19, +160] (predicted at most -150).
- **G1 failed, opposite direction:** oracle - none = +288 [+226, +351] (predicted at most -100).
- **G2 not meaningful:** chosen - oracle = -268 [-324, -211]; the "hand policy worse than oracle" condition is met
  trivially the wrong way round (the hand policy is better than the oracle), so there is no headroom to read.
- By the pre-registered rule the problem is closed; no learned schema was built. Chosen vs none: +20 [-6, +47].
- The radar is a strong lever (683 here against 923-ish without it), so problem 2's failure cause is gone. What
  failed is my definition of good attention. Reading, not tested: with random captures the agent's attention sits on
  each channel about half the time, so the radar prospects for both food and water and memory fills up for both needs;
  an agent that always looks at the channel it needs now sees only that channel's sites and learns the other need's
  sites late. The oracle optimises "attend to what I intend now", which is not the best use of a limited sensor.
  Whether another oracle (alternating, or prospecting the channel not currently needed) would have beaten none was
  not tested: choosing it now would be redefining the target after seeing the result.
- Tally of my AST attempts: three problems, three closures without a learned schema (cue broadcast with trace;
  gated attention; gated radar). In each the cheating or near-optimal reference failed to beat a trivial one by a
  reachable margin, so the comparison the theory needs (a schema against a model-free controller with the same inputs) was never run.
