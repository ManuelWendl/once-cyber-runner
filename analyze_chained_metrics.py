"""Faithful SOOPER episode metrics from the per-step dump, with chained-episode
aggregation.

Why this exists
---------------
The agent-side metrics file (`metrics_*.jsonl`) aggregates episode outcomes via
`epstats` (elements.Agg), which AVERAGES per-episode results within each
wall-clock log window. So `epstats/log/hole_terminated/sum` is a per-window
*fraction*, not a count — summing it across windows only equals the true count
when there is exactly one episode per window. With ~2 episodes/window that
undercounts everything ~2x. The per-step PolicySwitcher dump
(`sooper_steps_*.jsonl`, written when `sooper.dump_steps=True`) captures every
env-step with contiguous `step_in_ep`, so it is the ground truth for episode
structure.

Chained aggregation
-------------------
With `chain_only_on_timeout=True`, a physical episode that ends in a timeout is
followed by a chained episode that RESTORES the timed-out physical state (the
marble keeps going from where it was). Those physical episodes are really one
continuous exploration trajectory, broken into 2000-step segments by the
episode_length cap. This script merges them: a LOGICAL episode accumulates
physical episodes linked by timeout and closes on a hole (is_terminal) or at
run end. Aggregated length = sum of the segment lengths.

Termination inference from the dump
-----------------------------------
The dump has no termination-reason string, only flags. In the current SOOPER
setup goal=0 and prior_hold no longer terminates (release-with-cost-reset), so:
  - is_terminal=True            -> hole
  - is_last=True & ~is_terminal -> timeout

Usage
-----
  python analyze_chained_metrics.py --dump metrics/sooper_steps_mode2_v7.jsonl
  python analyze_chained_metrics.py --dump metrics/sooper_steps_mode2_v7.jsonl \
      --compare metrics/sooper_steps_mode2_v6.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


@dataclass
class PhysEpisode:
    env: int
    episode: int
    length: int            # number of env-steps (rows) in this physical episode
    is_hole: bool          # ended via hole-fall (is_terminal)
    is_timeout: bool       # ended via episode_length cap (is_last & ~is_terminal)
    completed: bool        # is_last seen (False = still running at dump end)
    triggers: int          # count of triggered=True steps
    prior_steps: int       # count of prior_active=True steps


def extract_physical_episodes(dump: list[dict[str, Any]]) -> list[PhysEpisode]:
    by_key: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in dump:
        by_key[(int(r["env"]), int(r["episode"]))].append(r)
    eps: list[PhysEpisode] = []
    for (env, episode), rows in by_key.items():
        rows.sort(key=lambda r: int(r["step_in_ep"]))
        is_hole = any(bool(r["is_terminal"]) for r in rows)
        completed = any(bool(r["is_last"]) for r in rows)
        is_timeout = completed and not is_hole
        eps.append(PhysEpisode(
            env=env, episode=episode, length=len(rows),
            is_hole=is_hole, is_timeout=is_timeout, completed=completed,
            triggers=sum(int(r["triggered"]) for r in rows),
            prior_steps=sum(int(r["prior_active"]) for r in rows),
        ))
    return eps


def raw_summary(eps: list[PhysEpisode]) -> dict[str, float]:
    completed = [e for e in eps if e.completed]
    holes = sum(e.is_hole for e in completed)
    timeouts = sum(e.is_timeout for e in completed)
    total_steps = sum(e.length for e in eps)
    return {
        "physical_episodes_completed": len(completed),
        "physical_episodes_started": len(eps),
        "holes": holes,
        "timeouts": timeouts,
        "hole_rate_per_episode": holes / max(len(completed), 1),
        "total_env_steps": total_steps,
        "holes_per_1k_steps": holes / max(total_steps, 1) * 1000,
        "mean_phys_ep_len": total_steps / max(len(eps), 1),
        "total_triggers": sum(e.triggers for e in eps),
        "triggers_per_episode": sum(e.triggers for e in eps) / max(len(completed), 1),
        "prior_active_frac": sum(e.prior_steps for e in eps) / max(total_steps, 1),
    }


def chained_summary(eps: list[PhysEpisode]) -> dict[str, float]:
    """Merge timeout-linked physical episodes per env into logical trajectories."""
    by_env: dict[int, list[PhysEpisode]] = defaultdict(list)
    for e in eps:
        by_env[e.env].append(e)
    logical_lengths: list[int] = []
    logical_holes = 0
    logical_timeouts_at_end = 0  # chains still open / ended by run termination
    n_logical = 0
    for env, env_eps in by_env.items():
        env_eps.sort(key=lambda e: e.episode)
        acc_len = 0
        for e in env_eps:
            acc_len += e.length
            if e.is_hole:
                # hole closes the logical trajectory
                logical_lengths.append(acc_len)
                logical_holes += 1
                n_logical += 1
                acc_len = 0
            elif e.is_timeout:
                # timeout -> chained into the next episode; keep accumulating
                continue
            else:
                # still-running episode at dump end -> close as open trajectory
                if acc_len > 0:
                    logical_lengths.append(acc_len)
                    logical_timeouts_at_end += 1
                    n_logical += 1
                    acc_len = 0
        if acc_len > 0:  # trailing accumulation (chain open at run end)
            logical_lengths.append(acc_len)
            logical_timeouts_at_end += 1
            n_logical += 1
    import numpy as np
    ll = np.array(logical_lengths) if logical_lengths else np.array([0])
    return {
        "logical_episodes": n_logical,
        "logical_holes": logical_holes,
        "logical_open_at_end": logical_timeouts_at_end,
        "hole_rate_per_logical_ep": logical_holes / max(n_logical, 1),
        "mean_logical_ep_len": float(ll.mean()),
        "median_logical_ep_len": float(np.median(ll)),
        "max_logical_ep_len": int(ll.max()),
    }


def print_summary(tag: str, eps: list[PhysEpisode]) -> None:
    raw = raw_summary(eps)
    ch = chained_summary(eps)
    print(f"\n===== {tag} =====")
    print("  -- RAW (each physical episode counted separately) --")
    for k, v in raw.items():
        print(f"    {k:32s} {v:10.3f}")
    print("  -- CHAINED (timeout-linked episodes merged into logical trajectories) --")
    for k, v in ch.items():
        print(f"    {k:32s} {v:10.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", type=Path, required=True,
                    help="Per-step PolicySwitcher dump (sooper_steps_*.jsonl).")
    ap.add_argument("--compare", type=Path, default=None,
                    help="Optional second dump to compare against.")
    args = ap.parse_args()

    eps = extract_physical_episodes(load_jsonl(args.dump))
    print_summary(args.dump.stem, eps)
    if args.compare:
        eps2 = extract_physical_episodes(load_jsonl(args.compare))
        print_summary(args.compare.stem, eps2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
