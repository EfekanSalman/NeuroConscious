# What the experiments found (and did not)

A summary of the pre-registered experiments in `experiments/preregistrations/` (one file per experiment, each with its
own Result section) and the data in `experiments/results/` (every file stamped with the commit and `dirty=false`).
Numbers below are copied from those files; read them there for intervals, seeds and caveats.

## Frame
NeuroConscious is used here as a **computational test bed for consciousness theories**, not as a claim that the agent is
conscious. Each theory-derived property is turned into a structural specification, tested for causal dependence
(removing it must change behaviour), compared against capacity-matched controls and the simplest implementation that
meets it, and negative results are reported. IIT is out of scope. Computational functionalism is a working hypothesis
only. Metric throughout: total critical steps (hunger + thirst + fatigue above the critical level; lower is better),
mean over the agents of a seed, 100 seeds unless stated.

## Results at a glance

| Question | Result | Status |
|---|---|---|
| Does a learned homeostatic controller beat the rules? | Yes: mean drive 0.495 vs 0.674 (rules) vs 0.765 (random choice) | solid |
| Does bounded, priority-based memory admission help? | Yes when resource sites persist: salience10 vs fifo10 about -45..-100 critical steps; 10 slots equal unbounded memory | solid |
| Is exploration or planning the bigger lever? | Frontier exploration: -147..-187; path planning: no effect | solid |
| Is the global workspace better than the rule chain? | Yes by 96-134 steps, but the lead decomposes into ordinary mechanisms (need arbiter at threshold 0.3 with level order, no focus lock, one-cell opportunism). Rules reach 923 vs the needs-only workspace 942 | explained, not a theory result |
| Does broadcast beat direct wiring when information must pass a bottleneck (beacon task)? | No: broadcast = wiring (+3 [-10,+16]); information is necessary (wrong cue is far worse than no cue); capacity limit costs +322 at capacity 1, nothing at 3 | information-need shown, broadcast-specific advantage not |
| Attention schema (AST), three problems | No reachable gain over trivial references in any; learned schema never built | closed negative |
| AE-2: a model of one's own output-input contingencies | A 4-entry command table recovers ~90% of the oracle's gain; a non-tracking table is worse than none (+135); a tabular model-free learner is 230 steps worse than the model | model needed in the weak sense; cheap to meet |
| HOT-2: monitoring how reliable one's perception is (regime level) | Trusting the radar on every step (742) beats the oracle monitor (849) and a learned monitor (1010); no job for a monitor | closed negative |
| HOT-2, per-percept form (each radar report carries a noisy reliability signal) | An oracle gains 122 over trusting everything; a hand-set threshold reaches it (605 vs 600); an outcome-trained monitor does not (923, no better than no radar) | met when the designer sets the threshold; learned form: yes with free (cheating) labels (593), no with labels from checked reports only (851) or with paid audits of rejected reports (883); both worse than no monitor (723) |

## Findings that held up
- The homeostatic controller, bounded memory admission and frontier exploration effects above replicate across the
  conditions tried (and the controller and the memory policy substitute for each other rather than adding).
- AE-2 (`motor_model`, `motor_model_free`): when the four move commands are reassigned every 150 steps and 15% of moves
  slip, no model 1758 / wrong fixed table 1893 / last-outcome table 1204 / majority-of-5 table 1099 / oracle 1028; tabular
  model-free control 1316 against 1086 for the model on fresh seeds. The best update rule depends on noise (the
  majority table loses 175 to last-outcome without slip and wins 200 with 30% slip).
- Information bottleneck (`beacon_*`): the cue's value is real (wrong cue is worse than none), so the information barrier works as a test.

## Why four lines closed negative (my reading)
In the three attention problems and the monitoring problem the cheapest reference already sat at or above the cheating
oracle: orienting attention for free was almost as good as the oracle, an oracle that attends to the intended channel
was worse than random attention, and trusting the radar always was better than trusting it only when it is clean. (The per-percept form of the monitoring
problem is the exception: there the oracle and a tuned rule beat trusting everything.)
Two features of this simulator seem to cause it: perception is a weak lever (a 3x3 view; the agent mostly finds
resources by walking onto cells and from memory, and urgent needs fill about 93% of steps), and a mistaken percept is
cheap to correct because the agent's own view removes it on sight. A different world or a per-percept reliability signal
could behave differently; neither was tested. This is a statement about this test bed, not about the theories.

## Claims retracted or corrected along the way
- "Competitive memory removes distractors" did not replicate (-35, ns); the first result came from a world bug (regrown
  resources piled into the top rows), fixed in `792b26f`.
- "Persistence reduces movement" was wrong; it cost ~130 critical steps, ~60% of it explained by a threshold artefact of my own choices.
- Two explanations for the workspace's lead failed (fixed-order arbitration, forced rest in storms); path planning had no effect.
- My early "learned prediction" claim lacked a random control; "70% explained" for the workspace lead was overstated.
- AE-2 H5 failed because I framed it against the no-model baseline; the quantity that grows with volatility is the gap to the oracle.

## Method, and its limits
- Hypotheses and probabilities were committed before runs; runs came from a clean `git worktree` at that commit; negative
  and failed predictions are recorded in each Result section. Default behaviour was checked against `results/baseline.json`
  and the unit tests after each feature.
- Limits: every environment was designed by me; seeds were reused in some earlier comparisons; no multiple-comparison
  correction anywhere; several controls (the rule-based opportunism, the motor table, the monitor) are my own
  implementations, so a null can be about the implementation; ghost-radar and motor-model parameters were set after
  disclosed small looks at other seeds; the model-free learner is tabular and its exploration rate sat at the tuning grid's edge.

## Open
- HOT-2: auditing a random sample of all offered reports (admitted ones included; a post-hoc diagnostic points there because audits of rejected reports only did not help), learning the ghost weight, other signal strengths.
- A neural model-free learner for AE-2, and a manipulation that isolates why the model beats the tabular learner.
- Indicators not touched: recurrent processing (RPT), predictive processing (PP).
- Known weak spots in the code: `explore` is a stationary no-op (left as is: every measurement here was taken with it);
  emotions barely influence behaviour. The unused `AgentInitializer` cluster (with `ActionExecutor`, `PerceptionManager`,
  `ThoughtProcessor` and their base classes) and the empty `tests/test_motivation.py` were removed; they stay in git history.
