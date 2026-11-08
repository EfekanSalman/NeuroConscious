# HOT-2: the "free label" learner was leaking (post-hoc finding, NOT pre-registered)

Found while looking for a flaw in my audit path, as offered after the seventh test. I logged what each learner admitted
(one seed, then 20 seeds 9500-9519, which were also used in earlier post-hoc looks). In 'free' mode the offered report is
labelled with its truth *before* the monitor judges that same report, so the threshold is refitted with the report's own
label (ghost weight 30 pushes the threshold above a ghost's strength). One seed: ghosts offered ~1010, admitted 0% in free
mode, 4% in audit mode; the label streams themselves (share of ghosts, mean strengths) were the same.

## Test
New mode `RADAR_LABELS = 'free_after'`: same labels, applied after the judgement of that report. Ghost weight 30, probe 0.

## Result (`experiments/results/radar_percept_leak/`, commit 680efac, dirty=false; seeds 9500-9519, total critical steps)
none 813, hand threshold (2.5) 648, learned 'free' (leaking) 627, **learned 'free_after' 823**.
With the leak removed the learner with free labels for every offered report is no better than no monitor.

## What this changes
- The "learned monitor works with free labels (593-635)" result in `radar_percept_learned.md`, `radar_percept_audit.md` and
  `radar_percept_audit_all.md` was an artefact of the leak. The label source never mattered; no learner here has been
  shown to work.
- The "unexplained discrepancy" between free and audit labels is explained: it was not the audit path; the free path was
  the cheat. The sixth and seventh tests were chasing a difference that did not exist.
- The oracle and the hand-set threshold results stand (they use no labels).
- What the learner actually fails at (why a labelled threshold fit does not beat always trusting) is still unknown. A
  caution: this is one more error of mine found after the fact; the earlier "free-label" labels were called "a cheat" in my
  own text, but the strength of the cheat was larger than I described.
