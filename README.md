# 🧠 NeuroConscious

**NeuroConscious** is a small agent-based simulation, written in Python, used as a **computational test bed for consciousness
theories**. Agents live in a grid world and must keep hunger, thirst and fatigue under control. The project asks what
theory-derived properties (global workspace, attention schema, higher-order monitoring, predictive and recurrent processing,
and others) actually do for such an agent, using pre-registered experiments, ablations and controls.

It is **not** a claim that the agent is conscious. Negative results are reported, and the repository records my own corrections
and retractions along the way.

---

### 🔬 What the experiments found

Six consciousness-theory properties were tested, plus the agent's controller, memory and exploration. In short:

- The homeostatic controller, bounded memory admission and frontier exploration have real, replicating effects.
- The global workspace's lead over a rule chain breaks down into ordinary mechanisms, and broadcast does no better than direct wiring.
- Attention schema (three problems): no gain over trivial references.
- A model of the agent's own output-input contingencies (AE-2) is needed in a weak sense, and a 4-entry table does it cheaply.
- Higher-order monitoring (HOT-2): a designer-set threshold works; no learned monitor did, and an early "it works" result was a leak in my code, since corrected.
- Predictive processing (PP-1) and recurrent evidence accumulation (RPT-1, weakest form): the mechanisms help, but a timer, a window or a repeat counter does as well or better.

In this simulator the cheapest reference usually matches or beats the theory-shaped mechanism. That is a statement about this test
bed, not about the theories.

- Short readable account: [`experiments/WRITEUP.md`](experiments/WRITEUP.md)
- Full table with numbers, retractions and limits: [`experiments/FINDINGS.md`](experiments/FINDINGS.md)
- One pre-registration per experiment (hypotheses and probabilities written before the runs, with a Result section added after):
  [`experiments/preregistrations/`](experiments/preregistrations/); data (every file stamped with its commit and `dirty=false`):
  [`experiments/results/`](experiments/results/)

---

### 🧬 The agent and its world

- Internal state (hunger, thirst, fatigue, mood, emotions) with strategy-based mood calculation
- Several decision machines that can be switched: a rule chain, a learned homeostatic controller, and an optional global
  workspace of competing specialists with its own bounded memory
- Memory systems (working, episodic, semantic, procedural, sighting memory), frontier exploration, optional path planning
- Optional perception and action manipulations used by the experiments: a ghost radar with unreliable reports, a beacon regime
  that moves resources between halves of the world, remapped and slipping move commands
- A grid world with food and water sites, obstacles and weather; scenarios in `experiments/scenarios.py`

---

### 🚀 Getting started

```bash
pip install -r requirements.txt          # numpy, matplotlib, torch
python -m unittest discover tests        # unit tests

python main.py                           # 1: full simulation with plots, 2: interactive CLI

# headless, seeded experiment runs (one process per seed; reproducible)
python -m experiments.run --seeds 0-9 --steps 1000 --out my_run.json
python -m experiments.run --scenario sites --seeds 0-99 --param "agent.base_agent:USE_FRONTIER_EXPLORATION=True" --out run.json
python -m experiments.compare experiments/results/baseline.json my_run.json
```

Experiment parameters are module-level constants (`config/constants.py`, `environment/world.py`) set with `--param module:NAME=value`.
Defaults are checked against `experiments/results/baseline.json` after every feature; new behaviour is off by default.

---

### 🗂️ Layout

| Path | What is in it |
|---|---|
| `agent/` | the agent and its sense-think-act loop |
| `core/` | state, mood, emotion, homeostasis, memory, learning, decision, workspace, navigation |
| `environment/` | the grid world |
| `config/` | constants and switches for every experimental feature |
| `experiments/` | runner, comparison tool, scenarios, metrics, pre-registrations, results, findings |
| `tests/` | unit tests |
| `visualization/` | plots used by `main.py` |

---

### 🧑‍💻 Author

Developed by [Efekan Salman](https://github.com/EfekanSalman), a developer interested in artificial life, computational
neuroscience, and conscious systems.

---

### 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
