# Is the workspace's opportunistic consumption what the rule agent still lacks?

Written before any run. Follows `focus_lock`: rules (neediest order, override 0.3, focus rewrite off) end at 979, the
needs-only workspace at 942 (+36 [+12, +61]). Persistence and attention feedback were ruled out there.

## What the data and code show (not a test)
- The two agents are close on everything but thirst: meals 20.7 vs 21.1, moves 895 vs 895, hunger-critical 436 vs 457
  (rules are better), thirst-critical 471 vs 410 and drinks 17.1 vs 20.2 (workspace is better).
- `GlobalWorkspace._select_action`: if the winner cannot be satisfied on the agent's own cell but another admitted need
  can (it stands on food or water), the agent takes that one instead. With capacity 3 and three need specialists every
  pressing need is admitted. The workspace counts this as `opportunistic_actions`: 5.5 per agent per 1000 steps in the
  needs-only runs. The rule chain has nothing like it: standing on water while hungrier, it walks on.
- This is also the one place where several simultaneously broadcast contents change the action, which is the
  textbook point of a workspace, so it is worth separating from the arbiter.

## Intervention
Two switches, both verified to leave default behaviour unchanged (baseline.json equal, 117 tests):
- `core.workspace.workspace:OPPORTUNISM` (default True): off = the winner always acts.
- `core.decision.decision_maker:CRITICAL_OPPORTUNISM` (default False): on = among the needs above the override level, if
  the first one's resource is not on the agent's cell and a later one's is, serve the later one.
Scenario `sites`, frontier on, sighting memory/workspace memory `nearest`/salience10, seeds 1100-1199, 1000 steps, total
critical steps (mean over agents), override/threshold 0.3. New runs:
- N1: needs-only workspace (capacity 3, habit and goal off), OPPORTUNISM off.
- R3: rules, 'neediest', focus rewrite off, CRITICAL_OPPORTUNISM on.
- R4: rules, 'neediest', focus rewrite ON (the original agent), CRITICAL_OPPORTUNISM on.
Reused (same seeds/settings): needs-only 942, rules neediest no-lock 979, rules neediest with lock 1003.

## Predictions (my probabilities)
- **E1 (50%):** N1 is worse than 942 by at least 20 critical steps.
- **E2 (50%):** R3 beats rules no-lock (979) by at least 20.
- **E3 (40%):** R3 ends within 25 steps of needs-only (942).
- **E4 (55%):** R3 has at least 2 more drinks than rules no-lock (17.1) and fewer thirst-critical steps (by 25 or more).
- R4 is descriptive, no prediction (the lock may undo the opportunism).
- Reading: E1 and E2 hold -> a one-cell, multi-content shortcut accounts for part of the 36; E3 too -> the rule agent
  with the same shortcut matches the workspace, so on this task neither the arbiter nor the broadcast adds anything beyond
  hand-written rules. E1 holds, E2 fails -> the shortcut matters only with the workspace's other machinery. E1 fails -> the
  remaining 36 steps lie elsewhere (DQN or procedural memory when all needs are low, Asleep/Awake, execution details).

## Limits stated in advance
Same environment and reused seeds (tuned on); four predictions, no multiple-comparison correction; the rule version of the
shortcut is my own implementation, so a null there could be my implementation, not the idea; with only ~5 events per
agent the effect is small by construction and may not clear the stated margins. Report either way, commit and dirty=false.
