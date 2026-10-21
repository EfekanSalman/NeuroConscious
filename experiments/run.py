"""
Headless, seeded runner: simulates the default two-agent world and records metrics.

Usage (from the repository root):
    python -m experiments.run --seeds 0-9 --steps 1000 --out experiments/results/baseline.json

Every seed runs in its own process, so global state (torch, random, module-level
counters) never leaks between runs, and the same seed reproduces the same result.
Nothing is written unless --out is given; the simulation's own console output is discarded.
"""

import argparse
import contextlib
import json
import multiprocessing
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from agent.base_agent import Agent
from core.mood.basic_mood import BasicMoodStrategy
from environment.world import GRID_SIZE, World
from experiments.metrics import AgentTracker, EventCounter, aggregate

SCHEMA_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parent.parent


def run_once(seed: int, steps: int) -> Dict[str, Any]:
    """Runs one seeded simulation and returns per-agent metrics."""
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    events = EventCounter()
    with contextlib.redirect_stdout(events):
        world = World()
        mood_strategy = BasicMoodStrategy()
        agents = [Agent(name="SimBot-1", mood_strategy=mood_strategy),
                  Agent(name="SimBot-2", mood_strategy=mood_strategy)]
        world.add_agent(agents[0], start_pos=(0, 0))
        world.add_agent(agents[1], start_pos=(GRID_SIZE - 1, GRID_SIZE - 1))
        trackers = [AgentTracker(agent) for agent in agents]

        for step in range(steps):
            world.step()
            for tracker in trackers:
                tracker.observe(step)

    return {
        "seed": seed,
        "agents": {t.agent.name: t.result(events.counts[t.agent.name], GRID_SIZE) for t in trackers},
    }


def parse_seeds(spec: str) -> List[int]:
    """Parses '7', '0,3,5' or '0-9' (inclusive), in any comma-separated combination."""
    seeds: List[int] = []
    for part in spec.split(","):
        if "-" in part:
            first, last = part.split("-")
            seeds.extend(range(int(first), int(last) + 1))
        else:
            seeds.append(int(part))
    return seeds


def _git_state() -> Dict[str, Any]:
    """Commit and whether tracked files differ from it, so results stay attributable."""
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _print_summary(summary: Dict[str, Any]) -> None:
    print(f"\nSummary over {summary['agent_runs']} agent-runs (mean ± std [min, max]):")
    for key, stats in summary.items():
        if isinstance(stats, dict) and "mean" in stats and "std" in stats:
            print(f"  {key:<24} {stats['mean']:8.2f} ± {stats['std']:6.2f}  [{stats['min']:g}, {stats['max']:g}]")
    reach = summary["reach_center"]
    mean_step = "n/a" if reach["mean_step"] is None else f"{reach['mean_step']:.0f}"
    print(f"  {'reach_center':<24} completed {reach['completed_rate']:.0%}, mean step {mean_step}")
    print("  mean action counts:     " + ", ".join(f"{k}={v:.0f}" for k, v in summary["action_counts"].items()))
    print("  mean consciousness:     " + ", ".join(f"{k}={v:.0f}" for k, v in summary["consciousness_steps"].items()))


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="0-9", help="e.g. 7, 0,3,5 or 0-9 (default: 0-9)")
    parser.add_argument("--steps", type=int, default=1000, help="simulation steps per run (default: 1000)")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1), help="parallel processes")
    parser.add_argument("--out", type=Path, help="write the full results as JSON to this path")
    args = parser.parse_args(argv)

    seeds = parse_seeds(args.seeds)
    print(f"Running {len(seeds)} seed(s) x {args.steps} steps on {args.jobs} process(es)...", flush=True)

    # 'spawn' gives each worker a clean interpreter, so seeding is the only source of randomness.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
        runs = list(pool.map(run_once, seeds, [args.steps] * len(seeds)))

    summary = aggregate(runs)
    _print_summary(summary)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema": SCHEMA_VERSION, "steps": args.steps, "seeds": seeds, "git": _git_state(),
                   "summary": summary, "runs": runs}
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
