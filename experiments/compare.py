"""
Compares experiment result files against a reference.

Usage (from the repository root):
    python -m experiments.compare REFERENCE.json OTHER.json [OTHER.json ...]

The two agents of a seed share one world, so they are not independent samples. Every
metric is therefore averaged over the agents of a seed first, and statistics are taken
over seeds. A difference is marked '*' when its 95% interval (normal approximation of the
difference of means) excludes zero. Seeds are not paired: the same seed gives a different
world once the decision machinery changes, so runs are compared as independent samples.
"""

import argparse
import math
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional

from experiments.resultfile import load_results

CRITICAL_KEYS = ("hunger_critical_steps", "thirst_critical_steps", "fatigue_critical_steps")
DEFAULT_METRICS = ["drive_mean", "hunger_at_meal_median", "thirst_at_drink_median", "meals", "drinks", "total_critical_steps", "hunger_critical_steps", "thirst_critical_steps",
                   "fatigue_critical_steps", "moves", "failed_pushes", "reach_center_step", "opportunistic_actions"]


def seed_values(results: dict, metric: str) -> List[float]:
    """One value per seed: the metric averaged over that seed's agents (None values skipped)."""
    values = []
    for run in results["runs"]:
        agents = list(run["agents"].values())
        if metric == "total_critical_steps":
            per_agent = [sum(a[k] for k in CRITICAL_KEYS) for a in agents]
        else:
            per_agent = [a[metric] for a in agents if a.get(metric) is not None]
        if per_agent:
            values.append(mean(per_agent))
    return values


def describe(values: List[float]) -> Optional[Dict[str, float]]:
    if len(values) < 2:
        return None
    return {"mean": mean(values), "se": stdev(values) / math.sqrt(len(values)), "n": len(values)}


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("others", type=Path, nargs="+")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="comma-separated metric names")
    args = parser.parse_args(argv)

    metrics = args.metrics.split(",")
    reference = load_results(args.reference)
    print(f"Reference: {args.reference.stem}  ({len(reference['seeds'])} seeds x {reference['steps']} steps, "
          f"params {reference.get('params') or {}})")

    for path in args.others:
        other = load_results(path)
        print(f"\n{path.stem}  ({len(other['seeds'])} seeds, params {other.get('params') or {}})")
        print(f"  {'metric':<24} {'value':>16} {'reference':>16} {'difference [95% interval]':>34}")
        for metric in metrics:
            a, b = describe(seed_values(other, metric)), describe(seed_values(reference, metric))
            if a is None or b is None:
                continue
            diff = a["mean"] - b["mean"]
            half = 1.96 * math.hypot(a["se"], b["se"])
            mark = "*" if abs(diff) > half else " "
            print(f"  {metric:<24} {a['mean']:>9.1f} ± {a['se']:<4.1f} {b['mean']:>9.1f} ± {b['se']:<4.1f} "
                  f"{diff:>+10.1f} [{diff - half:+.1f}, {diff + half:+.1f}] {mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
