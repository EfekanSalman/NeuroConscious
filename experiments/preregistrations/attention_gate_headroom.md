# Gated attention: is there reachable headroom for an attention schema? (NOT pre-registered)

Honest status: this is a diagnostic. I wrote the mechanism, saw a first 20-seed smoke run, changed one setting after
seeing it (setting b), and then re-ran both settings at a clean commit. The thresholds below were not fixed in advance.
Stated plan, had it gone on: gate 1 (a cheating oracle that knows the hidden attention state beats no orienting by >= 150),
gate 2 (the best hand policy leaves >= 60 to the oracle), only then a learned schema against a model-free controller,
a random-weight schema and a fixed interval. Gate 1 failed, so none of the later stages was built.

## Problem
`ATTENTION_GATE`: food and water cells are perceived at ATTENTION_HIGH (0.95) only when a hidden `attention_actual`
is on that channel, else at ATTENTION_LOW. The agent cannot read `attention_actual`; it can spend a step orienting
it to the channel it intends (a stationary step), and captures flip it at random. Workspace agent: needs-only, capacity 3,
threshold 0.3, scenario `sites`, 20 seeds (9000-9019, not used elsewhere), 1000 steps, total critical steps.
Setting a: capture 0.03/step, LOW 0.15. Setting b: capture 0.10/step, LOW 0.0.
Policies: none (never orient), oracle (orients exactly when wrong; cheats), oracle_free (same, orienting costs no step),
switch (orients when the intended channel changes; uses nothing hidden).

## Result (`experiments/results/attention_gate_headroom/`, commit 7cc272b, dirty=false)
| policy | a: critical steps (correct attention, orients) | b |
|---|---|---|
| none | 1087 (46%, 0) | 1170 (48%, 0) |
| oracle | 1217 (97%, 61) | 1337 (90%, 117) |
| oracle_free | 1060 (97%, 60) | 1076 (90%, 121) |
| switch | 1187 (65%, 34) | 1231 (56%, 31) |
Differences from none: oracle +130 [+92,+169] / +167 [+121,+213] (worse); oracle_free -27 [-70,+16] / -94 [-145,-43];
switch +100 [+56,+144] / +61 [+18,+104] (worse).
- Gate 1 failed by a wide margin: even a cheating oracle does worse than never orienting, because an orienting step
  costs more than the better perception earns. Setting b: oracle - oracle_free = +261 over about 117 orients, about 2.2
  critical steps per orient, against a gain of 94 over about 120 orients, about 0.8 per orient.
- Correct attention is worth something only when orienting is free (setting b: -94), and then no schema is needed:
  orienting to the intended channel every step reaches the oracle without knowing the hidden state.
- Why perception is a weak lever here (reading, not tested): the view is 3x3 and the agent finds resources by
  walking onto cells and from memory; the same agent lost nothing when workspace attention feedback was off (`focus_lock`, D5).
- Structural point: a schema can pay only if checking or correcting attention has a price between "free" (trivial
  policy) and "more than it earns" (no policy). I did not search for such a price; that would be tuning the world until
  the schema looks necessary, and I would have to state it as designed by construction.

## Limits
Two settings, 20 seeds, no multiple-comparison correction, one environment I designed, post-hoc changes disclosed above.
This closes this problem, not the idea. It does not show that attention schemas cannot help.
