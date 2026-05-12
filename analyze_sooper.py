"""SOOPER metrics analysis.

Two subcommands matching the diagnostic passes used during Phase-3 tuning:

    # Macro: compare run-level stats across one or more metrics.jsonl files
    #   (window-aggregated dreamer logs from `dreamerv3_opax.sbatch`).
    python analyze_sooper.py macro metrics/metrics.jsonl metrics/metrics_sooper_v3.jsonl

    # Per-step: pick apart a single sooper_steps.jsonl
    #   (per-env per-control-step dump from PolicySwitcher when
    #    sooper.dump_steps=true).
    python analyze_sooper.py per-step metrics/sooper_steps_v3.jsonl

The macro pass prints:
  - per-run summary (hole rate, coverage, gate trigger/release counts, V_prior
    range, etc.)
  - hole-rate-by-chunk and coverage-by-chunk tables across all files

The per-step pass prints:
  - episode termination breakdown (hole/timeout/incomplete)
  - prior-active interval shape (count, length distribution, soft/hard/fell)
  - state at trigger moment (V_prior, risk_critic percentiles)
  - re-trigger gap distribution (cooldown sanity check)
  - distance-to-fall risk-signal histograms (for picking tau_high)

Pure stdlib + numpy. No JAX, no matplotlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def col(rows: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows if key in r], dtype=float)


def steps_col(rows: List[Dict[str, Any]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    s, v = [], []
    for r in rows:
        if key in r and "step" in r:
            s.append(r["step"]); v.append(r[key])
    return np.array(s), np.array(v, dtype=float)


# --------------------------------------------------------------------------
# Macro analysis (one row per metrics flush window)
# --------------------------------------------------------------------------


def report_macro(label: str, rows: List[Dict[str, Any]]) -> None:
    hole = col(rows, "epstats/log/hole_terminated/sum")
    goal = col(rows, "epstats/log/goal_terminated/sum")
    timeout = col(rows, "epstats/log/timeout_terminated/sum")
    total = hole + goal + timeout
    cov = col(rows, "exploration/coverage")
    prog = col(rows, "exploration/mean_max_path_progress")

    print(f"\n=== {label} ===")
    print(
        f"  episodes: tot={int(total.sum())} hole={int(hole.sum())} "
        f"goal={int(goal.sum())} timeout={int(timeout.sum())}"
    )
    if total.sum() > 0:
        print(f"  hole rate:    {hole.sum() / total.sum():.3f}")
        print(f"  timeout rate: {timeout.sum() / total.sum():.3f}")
    if len(cov):
        print(f"  final coverage (last 5):       {cov[-5:].mean():.3f}")
        print(f"  final max_path_progress (l5):  {prog[-5:].mean():.3f}")

    if any("epstats/log/gate/prior_active/avg" in r for r in rows):
        active = col(rows, "epstats/log/gate/prior_active/avg")
        trig = col(rows, "epstats/log/gate/triggered/sum")
        rel = col(rows, "epstats/log/gate/released/sum")
        hard = col(rows, "epstats/log/gate/hard_released/sum")
        hold = col(rows, "epstats/log/gate/hold_count/max")
        v_avg = col(rows, "epstats/log/gate/V_prior/avg")
        rc_avg = col(rows, "epstats/log/gate/risk_critic/avg")
        rcm_avg = col(rows, "epstats/log/gate/risk_cont_max/avg")
        print(f"  prior_active avg: {active.mean():.4f}  max {active.max():.4f}")
        print(
            f"  triggers: {int(trig.sum())}  releases: {int(rel.sum())}  "
            f"hard_rel: {int(hard.sum())}"
        )
        if trig.sum() > 0:
            print(f"    releases/triggers: {rel.sum() / trig.sum():.3f}")
        if rel.sum() > 0:
            print(f"    hard / total releases: {hard.sum() / rel.sum():.3f}")
        if hold.size:
            print(f"  hold/max p50/p95/max: {np.percentile(hold, [50, 95, 100])}")
        if v_avg.size:
            print(
                f"  V_prior /avg range:     "
                f"{v_avg.min():.1f} -> {v_avg.mean():.1f} -> {v_avg.max():.1f}"
            )
            print(
                f"  risk_critic /avg range: "
                f"{rc_avg.min():.3f} -> {rc_avg.mean():.3f} -> {rc_avg.max():.3f}"
            )
            print(
                f"  risk_cont_max /avg:     "
                f"{rcm_avg.min():.3f} -> {rcm_avg.mean():.3f} -> {rcm_avg.max():.3f}"
            )


def chunk_table(
    title: str,
    runs: List[Tuple[str, List[Dict[str, Any]]]],
    compute: callable,
    fmt: str = ".3f",
    n_chunks: int = 5,
    label_width: int = 30,
) -> None:
    """Print a per-chunk table where `compute(rows, chunk_indices)` returns
    a scalar per chunk."""
    print(f"\n=== {title} ===")
    for label, rows in runs:
        chunks = compute(rows, n_chunks)
        if chunks is None:
            continue
        print(f"  {label:{label_width}s}: " + "  ".join(format(x, fmt) for x in chunks))


def hole_rate_chunks(rows: List[Dict[str, Any]], n: int):
    hole = col(rows, "epstats/log/hole_terminated/sum")
    goal = col(rows, "epstats/log/goal_terminated/sum")
    timeout = col(rows, "epstats/log/timeout_terminated/sum")
    tot = hole + goal + timeout
    if len(hole) == 0:
        return None
    chunks = np.array_split(np.arange(len(hole)), n)
    return [hole[c].sum() / max(tot[c].sum(), 1) for c in chunks]


def coverage_chunks(rows: List[Dict[str, Any]], n: int):
    s, v = steps_col(rows, "exploration/coverage")
    if len(s) == 0:
        return None
    order = np.argsort(s)
    chunks = np.array_split(order, n)
    return [v[c].mean() for c in chunks]


def prior_active_chunks(rows: List[Dict[str, Any]], n: int):
    s, v = steps_col(rows, "epstats/log/gate/prior_active/avg")
    if len(s) == 0:
        return None
    order = np.argsort(s)
    chunks = np.array_split(order, n)
    return [v[c].mean() for c in chunks]


def progress_chunks(rows: List[Dict[str, Any]], n: int):
    s, v = steps_col(rows, "exploration/mean_max_path_progress")
    if len(s) == 0:
        return None
    order = np.argsort(s)
    chunks = np.array_split(order, n)
    return [v[c].mean() for c in chunks]


def cmd_macro(args: argparse.Namespace) -> None:
    runs = []
    for path in args.files:
        p = Path(path)
        if not p.is_file():
            print(f"WARNING: {p} not found, skipping", file=sys.stderr)
            continue
        runs.append((p.stem, load_jsonl(p)))
    if not runs:
        raise SystemExit("No metrics files loaded.")

    for label, rows in runs:
        report_macro(label, rows)

    chunk_table("Hole rate by chunk", runs, hole_rate_chunks)
    chunk_table("Coverage by chunk", runs, coverage_chunks)
    chunk_table("max_path_progress by chunk", runs, progress_chunks)
    # Only meaningful for SOOPER runs; safe to print zeros for plain.
    chunk_table(
        "prior_active fraction by chunk", runs, prior_active_chunks, fmt=".4f"
    )


# --------------------------------------------------------------------------
# Per-step analysis (one row per env per control step)
# --------------------------------------------------------------------------


def extract_intervals(eps: Dict[Tuple[int, int], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """A `prior-active interval` is a contiguous stretch of steps in one
    episode where the gate fires (`triggered`) and is later released (or
    the episode terminates before release)."""
    intervals = []
    for (env, episode), traj in eps.items():
        in_iv = False
        start = V_trig = risk_trig = None
        for i, r in enumerate(traj):
            if r["triggered"] and not in_iv:
                in_iv = True
                start = i
                V_trig = r["V_prior"]
                risk_trig = r["risk_critic"]
            if in_iv and r["released"]:
                in_iv = False
                intervals.append({
                    "env": env, "episode": episode,
                    "start": start, "end": i, "length": i - start,
                    "outcome": "hard_released" if r["hard_released"] else "soft_released",
                    "V_trigger": V_trig, "risk_trigger": risk_trig,
                })
                start = V_trig = risk_trig = None
            elif in_iv and r["is_terminal"]:
                intervals.append({
                    "env": env, "episode": episode,
                    "start": start, "end": i, "length": i - start,
                    "outcome": "fell_mid_hold",
                    "V_trigger": V_trig, "risk_trigger": risk_trig,
                })
                in_iv = False
                start = V_trig = risk_trig = None
        if in_iv:
            intervals.append({
                "env": env, "episode": episode,
                "start": start, "end": len(traj) - 1, "length": len(traj) - 1 - start,
                "outcome": "truncated",
                "V_trigger": V_trig, "risk_trigger": risk_trig,
            })
    return intervals


DISTANCE_BUCKETS = [
    "safe (60+ before fall)",
    "30-60 before fall",
    "10-30 before fall",
    "0-10 before fall (doom)",
]


def _bucket_of(d: int) -> str:
    if d > 60:
        return DISTANCE_BUCKETS[0]
    if d > 30:
        return DISTANCE_BUCKETS[1]
    if d > 10:
        return DISTANCE_BUCKETS[2]
    return DISTANCE_BUCKETS[3]


def distance_to_fall_buckets(eps, signals: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """For each terminated episode, bin steps by distance to terminal and
    return per-bucket arrays for each signal. Steps where prior is driving
    are EXCLUDED — we want the WM's natural risk reading, not perturbed by
    gate intervention.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {b: [] for b in DISTANCE_BUCKETS}
    for key, traj in eps.items():
        if not traj[-1]["is_terminal"]:
            continue
        T = len(traj)
        for r in traj:
            if r["prior_active"]:
                continue
            d = T - 1 - r["step_in_ep"]
            buckets[_bucket_of(d)].append(r)
    out: Dict[str, Dict[str, np.ndarray]] = {label: {} for label in buckets}
    for label, rs in buckets.items():
        for sig in signals:
            out[label][sig] = np.array([r[sig] for r in rs]) if rs else np.array([])
        out[label]["_n"] = len(rs)
    return out


def fp_detection_table(buckets: Dict[str, Dict[str, np.ndarray]], sig: str,
                        taus: Iterable[float]) -> None:
    safe = buckets["safe (60+ before fall)"][sig]
    near = buckets["10-30 before fall"][sig]
    doom = buckets["0-10 before fall (doom)"][sig]
    if min(safe.size, near.size, doom.size) == 0:
        print("  (insufficient data for FP/detection table)")
        return
    print(f"  {'tau':>6} {'fp(safe)':>10} {'det(10-30)':>12} {'det(0-10)':>12}")
    for tau in taus:
        fp = (safe > tau).mean()
        det_near = (near > tau).mean()
        det_doom = (doom > tau).mean()
        print(f"  {tau:>6.2f} {fp:>10.3f} {det_near:>12.3f} {det_doom:>12.3f}")


def cmd_per_step(args: argparse.Namespace) -> None:
    path = Path(args.file)
    if not path.is_file():
        raise SystemExit(f"{path} not found.")
    print(f"Loading {path}")
    rows = load_jsonl(path)
    print(f"  {len(rows):,} rows")

    eps: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        eps[(r["env"], r["episode"])].append(r)
    for k in eps:
        eps[k].sort(key=lambda r: r["step_in_ep"])
    print(f"  {len(eps)} (env, episode) pairs")

    # --- 1. Termination breakdown ---
    n_term = sum(1 for k, v in eps.items() if v[-1]["is_terminal"])
    n_to = sum(1 for k, v in eps.items()
               if v[-1].get("is_last") and not v[-1]["is_terminal"])
    n_inc = len(eps) - n_term - n_to
    print("\nEpisode termination breakdown:")
    print(f"  terminated (hole/goal):       {n_term}")
    print(f"  timeout:                       {n_to}")
    print(f"  incomplete at end of file:     {n_inc}")

    # --- 2. Prior-active intervals ---
    intervals = extract_intervals(eps)
    print(f"\nPrior-active intervals: {len(intervals)}")
    by_outcome = defaultdict(int)
    for x in intervals:
        by_outcome[x["outcome"]] += 1
    for k in ("soft_released", "hard_released", "fell_mid_hold", "truncated"):
        print(f"  {k:18s}  {by_outcome[k]}")

    if intervals:
        lengths = np.array([x["length"] for x in intervals])
        print("\nInterval length (control steps):")
        print(
            f"  p25/p50/p75/p95/max: "
            f"{np.percentile(lengths, [25, 50, 75, 95, 100])}"
        )
        print(
            f"  mean: {lengths.mean():.1f}   "
            f"total prior-driven steps: {int(lengths.sum())}"
        )
        for outcome in ("soft_released", "hard_released", "fell_mid_hold"):
            L = np.array([x["length"] for x in intervals if x["outcome"] == outcome])
            if L.size:
                print(
                    f"  {outcome:18s} n={L.size:4d}  "
                    f"p50/p95/max = {np.percentile(L, [50, 95, 100])}"
                )

        V = np.array([x["V_trigger"] for x in intervals])
        risk = np.array([x["risk_trigger"] for x in intervals])
        print("\nState at trigger moment:")
        print(f"  V_prior     p25/p50/p75: {np.percentile(V, [25, 50, 75])}")
        print(f"  risk_critic p25/p50/p75: {np.percentile(risk, [25, 50, 75])}")

    # --- 3. Re-trigger gaps (cooldown sanity) ---
    by_ep = defaultdict(list)
    for x in intervals:
        by_ep[(x["env"], x["episode"])].append(x)
    gaps = []
    for k, ivs in by_ep.items():
        ivs.sort(key=lambda x: x["start"])
        for a, b in zip(ivs, ivs[1:]):
            gaps.append(b["start"] - a["end"])
    if gaps:
        gaps = np.array(gaps)
        print(f"\nRe-trigger gaps (control steps between release and next trigger, same episode):")
        print(f"  n={len(gaps)}  p25/p50/p75/p95: {np.percentile(gaps, [25, 50, 75, 95])}")
        print(f"  fraction <= cooldown_typical(60): {(gaps <= 60).mean():.3f}")
        print(f"  fraction <= 100 (rapid re-trigger): {(gaps <= 100).mean():.3f}")

    # --- 4. Distance-to-fall histograms ---
    print("\n=== Risk signal distributions by distance-to-fall ===")
    signals = ["risk_cont_max", "risk_cont_product", "risk_critic", "V_prior"]
    buckets = distance_to_fall_buckets(eps, signals)
    for sig in signals:
        print(f"\n  {sig}:")
        print(
            f"    {'bucket':28s}  {'p25':>7s} {'p50':>7s} {'p75':>7s} "
            f"{'p90':>7s} {'p95':>7s} {'p99':>7s}  {'n':>6s}"
        )
        for label in DISTANCE_BUCKETS:
            arr = buckets[label][sig]
            n = buckets[label]["_n"]
            if arr.size == 0:
                continue
            ps = np.percentile(arr, [25, 50, 75, 90, 95, 99])
            print(
                f"    {label:28s}  " + " ".join(f"{p:7.3f}" for p in ps)
                + f"  {n:>6d}"
            )

    # --- 5. FP/detection trade-off for risk_critic (the v3 gate signal) ---
    print("\n=== FP/detection trade-off (risk_critic) ===")
    fp_detection_table(buckets, "risk_critic",
                        [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90])


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pm = sub.add_parser(
        "macro",
        help="Compare run-level stats across metrics.jsonl files.",
    )
    pm.add_argument(
        "files", nargs="+",
        help="One or more dreamer metrics.jsonl files. Labels in output are taken from filenames.",
    )
    pm.set_defaults(func=cmd_macro)

    pp = sub.add_parser(
        "per-step",
        help="Analyze a single sooper_steps.jsonl dump.",
    )
    pp.add_argument("file", help="Path to a sooper_steps.jsonl file.")
    pp.set_defaults(func=cmd_per_step)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
