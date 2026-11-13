# Six consciousness-theory indicators in a toy world: the cheap reference usually wins

*A short account of the pre-registered experiments in this repository. Numbers are copied from
[`FINDINGS.md`](FINDINGS.md); each experiment has its own file in `preregistrations/` and its data in `results/`.*

## What this is, and is not
NeuroConscious is a small simulated world with one or two agents that must keep hunger, thirst and fatigue under control.
I use it as a **test bed for consciousness theories**, not as a claim that the agent is conscious. For each theory I take one
property that the literature lists as an *indicator* (Butlin et al., 2023), write it as a structural specification, and ask four
things: does removing it change behaviour (causal dependence), does it beat a control of the same capacity, does the simplest
implementation that meets the specification do as well, and what happens if it fails. IIT is out of scope. The metric is
"critical steps": steps where a need is above its critical level, summed over the three needs (lower is better), averaged over
100 seeds.

## How the experiments were run
Hypotheses and my probabilities for them were committed to git *before* any run. Runs came from a clean checkout of that
commit; every result file carries the commit and `dirty=false`. Failed predictions are written into each file's Result
section. Defaults were checked against a stored baseline and 177 unit tests after every feature.

## What I found

| Property tested | Result |
|---|---|
| Learned homeostatic control, bounded memory, exploration | Real effects that replicate (e.g. frontier exploration −147 to −187 steps; 10 memory slots equal unbounded memory when resource sites persist). |
| Global workspace vs. a rule chain | The workspace leads by 96–134 steps, but the lead decomposes into ordinary mechanisms (a need arbiter at the right threshold, no focus lock, one-cell opportunism). The rules reach 923 vs 942 for a needs-only workspace. |
| Broadcast vs. direct wiring when information must pass a bottleneck | No difference (+3 [−10, +16]). The information itself is necessary: a wrong cue is far worse than none. |
| Attention schema (three problems) | No gain over trivial references in any; a learned schema was never built. |
| AE-2: a model of one's own output–input contingencies | A 4-entry command table recovers about 90% of the oracle's gain and beats a tabular model-free learner by about 230 steps. Met in the weak sense, cheaply. |
| HOT-2: monitoring how reliable one's perception is | At regime level, trusting always (742) beats the oracle monitor (849). Per percept, an oracle gains 122 and a designer-set threshold reaches it, but **no learned monitor ever worked**. |
| PP-1: inferring the active half of the world from prediction errors | A belief filter beats having no cue by 75–123 steps and an inverted belief is 434 worse, so the content is used. But a timer that holds the last sighting for 80 steps matches it (+12 [−2, +26]). |
| RPT-1 (weakest form): evidence accumulated across steps | The integrator beats no monitor by 85 and a single report by 110. But a raw window of the same history matches it, and a repeat counter that ignores signal strength beats it (653 vs 684). |

## The pattern
In most lines the cheapest reference already sat at or above the cheating oracle, or matched the theory-shaped mechanism:
orienting attention for free was almost as good as the oracle; trusting the radar always beat trusting it selectively; a held
timer did what a belief filter did; a repeat counter beat an integrator. Two features of this simulator seem to cause it:
perception is a weak lever (a 3×3 view; the agent mostly finds resources by walking onto cells and from memory), and a mistaken
percept is cheap to correct because the agent's own view removes it on sight. This is a statement about the test bed. It is not
evidence against any theory.

## Mistakes I made, and corrected
- A "competitive memory removes distractors" result did not replicate; it came from a world bug (fixed).
- "Persistence reduces movement" was wrong; it cost about 130 steps, mostly from a threshold artefact of my own choices.
- I told a story about why audits of a learned monitor failed, then a seventh test chasing it. The real cause was a leak in my
  code: the "free label" learner was shown the truth of the very report it was judging. With the leak removed it is no better
  than no monitor (823 vs 813). Results in five preregistration files and several of my explanations had to be corrected. Reading the agent's decision log
  next to the aggregate numbers would have found it in one test; it is now a step in my pre-registrations.
- Several of my predictions failed, in both directions; they are recorded as such.

## Limits
Every environment was designed by me, and some of the mechanisms I expected to help were helpful by construction (for example,
real resource sites repeat while radar ghosts do not). Several controls are my own implementations, so a null result can be about
the implementation. There is no multiple-comparison correction. Seeds were reused in some early comparisons, and some parameters
were set after small disclosed looks at other seeds. RPT was only tested in its weakest, purely algorithmic sense (state carried
across steps); spatial feedback in a perceptual hierarchy and RPT-2 were not tried. No result here says anything about whether
any system is conscious.

## Reproduce
```bash
pip install "numpy~=2.3.1" "matplotlib>=3.7" "torch~=2.7.1"
python -m unittest discover tests
python -m experiments.run --seeds 0-9 --steps 1000      # default world, compare with results/baseline.json
```
Experiment-specific parameters are in each preregistration and in the `params` field of its result files.
