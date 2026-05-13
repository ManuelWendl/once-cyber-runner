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
  - per-run summary including BOTH per-episode and per-step normalisations
    (hole/goal counts per 100k env-steps + mean steps between fall/goal).
    Per-step is the right unit for fixed-compute comparisons; per-episode
    is the right unit for "given the marble is alive, will it fall?"
  - hole-rate-by-chunk, holes-per-100k-by-chunk, goals-per-100k-by-chunk,
    coverage-by-chunk tables across all files
  - gate trigger/release counts, V_prior range, etc. (SOOPER runs only)

The per-step pass prints:
  - episode termination breakdown (hole/timeout/incomplete)
  - prior-active interval shape (count, length distribution, soft/hard/fell)
  - state at trigger moment (V_prior, risk_critic percentiles)
  - re-trigger gap distribution (cooldown sanity check)
  - distance-to-fall risk-signal histograms (for picking tau_high)

Pure stdlib + numpy. Add `--plot` to also write PNG dashboards (needs
matplotlib).
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


def total_env_steps(rows: List[Dict[str, Any]]) -> int:
    """Return the env-step horizon covered by this metrics file."""
    steps = [r["step"] for r in rows if "step" in r]
    return int(max(steps)) if steps else 0


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
    env_steps = total_env_steps(rows)

    h_sum, g_sum, t_sum = float(hole.sum()), float(goal.sum()), float(timeout.sum())
    e_sum = float(total.sum())

    # Macro counts can come out fractional because event totals get averaged
    # across the eval-window log flushes — print one decimal so a single goal
    # logged as 0.33 doesn't round to 0 and disappear.
    def _fmt(x: float) -> str:
        return f"{x:.0f}" if abs(x - round(x)) < 1e-6 else f"{x:.2f}"

    print(f"\n=== {label} ===")
    print(f"  env-steps: {env_steps:,}")
    print(
        f"  episodes: tot={_fmt(e_sum)} hole={_fmt(h_sum)} "
        f"goal={_fmt(g_sum)} timeout={_fmt(t_sum)}"
    )
    if e_sum > 0:
        print(f"  hole rate (per-episode):    {h_sum / e_sum:.3f}")
        print(f"  goal rate (per-episode):    {g_sum / e_sum:.3f}")
        print(f"  timeout rate (per-episode): {t_sum / e_sum:.3f}")
        avg_len = env_steps / e_sum if e_sum > 0 else 0
        print(f"  mean episode length:        {avg_len:.0f} env-steps")

    # Per-step normalisation — fair comparison across runs with different
    # episode-length distributions. Both metrics are reported because they
    # answer different questions:
    #   - hole_rate (per-episode): "given an episode, P(it falls)?"
    #   - holes_per_100k: "fixed compute budget, # of fall events?"
    # The latter is the relevant unit for Phase-4 world-model training and
    # for the actual safety question ("how often does the marble die?").
    if env_steps > 0:
        per_100k = 100_000 / env_steps
        h_100k = h_sum * per_100k
        g_100k = g_sum * per_100k
        steps_per_hole = env_steps / h_sum if h_sum > 0 else float("inf")
        steps_per_goal = env_steps / g_sum if g_sum > 0 else float("inf")
        print(f"  holes per 100k env-steps:   {h_100k:.1f}")
        print(f"  goals per 100k env-steps:   {g_100k:.2f}")
        print(f"  mean steps between falls:   {steps_per_hole:,.0f}")
        if g_sum > 0:
            print(f"  mean steps between goals:   {steps_per_goal:,.0f}")

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


def _events_per_100k_chunks(rows: List[Dict[str, Any]], key: str, n: int):
    """Normalise an episode-event count by the env-step span of each chunk.

    Returns the rate at which `key` events fire per 100k env-steps within
    each equal slice of the run. Robust to runs where window-flush cadence
    varies — divides by actual step delta inside the chunk.
    """
    s, v = steps_col(rows, key)
    if len(s) == 0:
        return None
    order = np.argsort(s)
    s_sorted = s[order]
    v_sorted = v[order]
    chunks = np.array_split(np.arange(len(s_sorted)), n)
    out = []
    for c in chunks:
        if len(c) == 0:
            out.append(0.0)
            continue
        # span = last step - first step in chunk; fall back to chunk-share
        # of the global span if the chunk has only one window.
        span = s_sorted[c[-1]] - s_sorted[c[0]] if len(c) > 1 else (
            s_sorted[-1] / n
        )
        span = max(span, 1.0)
        out.append(float(v_sorted[c].sum()) * 100_000.0 / span)
    return out


def holes_per_100k_chunks(rows: List[Dict[str, Any]], n: int):
    return _events_per_100k_chunks(rows, "epstats/log/hole_terminated/sum", n)


def goals_per_100k_chunks(rows: List[Dict[str, Any]], n: int):
    return _events_per_100k_chunks(rows, "epstats/log/goal_terminated/sum", n)


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

    chunk_table("Hole rate by chunk (per-episode)", runs, hole_rate_chunks)
    chunk_table("Holes per 100k env-steps by chunk", runs,
                holes_per_100k_chunks, fmt=".1f")
    chunk_table("Goals per 100k env-steps by chunk", runs,
                goals_per_100k_chunks, fmt=".2f")
    chunk_table("Coverage by chunk", runs, coverage_chunks)
    chunk_table("max_path_progress by chunk", runs, progress_chunks)
    # Only meaningful for SOOPER runs; safe to print zeros for plain.
    chunk_table(
        "prior_active fraction by chunk", runs, prior_active_chunks, fmt=".4f"
    )

    if args.plot:
        outdir = Path(args.plot_dir)
        plot_macro(runs, outdir)
        print(f"\nWrote plot: {outdir / 'macro_comparison.png'}")


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
    "safe (timeout episode)",
    "safe (60+ before fall)",
    "30-60 before fall",
    "10-30 before fall",
    "0-10 before fall (doom)",
]


def _bucket_of(d: int) -> str:
    if d > 60:
        return DISTANCE_BUCKETS[1]
    if d > 30:
        return DISTANCE_BUCKETS[2]
    if d > 10:
        return DISTANCE_BUCKETS[3]
    return DISTANCE_BUCKETS[4]


def distance_to_fall_buckets(eps, signals: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    """Bin OPAX-driven steps by distance-to-fall.

    Population is split into two cohorts of "safe" plus three pre-fall
    distance buckets:
      - "safe (timeout episode)"     — episodes that survived to timeout.
                                       The cleanest no-fall sample.
      - "safe (60+ before fall)"    — early/middle steps of episodes that
                                       eventually fell. Often elevated
                                       risk_critic vs. timeout episodes
                                       because these episodes were near
                                       the boundary throughout.
      - 30-60 / 10-30 / 0-10 before fall — the doom approach in terminated
                                            episodes.

    Splitting both safe populations explicitly avoids the bias of treating
    "safe-looking steps in doomed eps" as representative of all safe steps.
    Steps where the prior is driving are excluded from every bucket — we
    want the WM's natural risk reading, not the gate's perturbation.
    """
    buckets: Dict[str, List[Dict[str, Any]]] = {b: [] for b in DISTANCE_BUCKETS}
    for key, traj in eps.items():
        last = traj[-1]
        if last["is_terminal"]:
            T = len(traj)
            for r in traj:
                if r["prior_active"]:
                    continue
                d = T - 1 - r["step_in_ep"]
                buckets[_bucket_of(d)].append(r)
        elif last.get("is_last"):
            for r in traj:
                if r["prior_active"]:
                    continue
                buckets["safe (timeout episode)"].append(r)
        # Incomplete episodes (file ended mid-episode) are skipped.
    out: Dict[str, Dict[str, np.ndarray]] = {label: {} for label in buckets}
    for label, rs in buckets.items():
        for sig in signals:
            out[label][sig] = np.array([r[sig] for r in rs]) if rs else np.array([])
        out[label]["_n"] = len(rs)
    return out


def fp_detection_table(buckets: Dict[str, Dict[str, np.ndarray]], sig: str,
                        taus: Iterable[float]) -> None:
    # Pool both safe cohorts for an unbiased FP estimate. Previously only
    # the 'safe (60+ before fall)' bucket was used, which excluded all
    # timeout episodes — biasing the FP rate downward by ~25%.
    safe_timeout = buckets["safe (timeout episode)"][sig]
    safe_far = buckets["safe (60+ before fall)"][sig]
    safe = np.concatenate([safe_timeout, safe_far]) if (
        safe_timeout.size + safe_far.size > 0
    ) else np.array([])
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


def cmd_report(args: argparse.Namespace) -> None:
    runs = []
    for path in args.files:
        p = Path(path)
        if not p.is_file():
            print(f"WARNING: {p} not found, skipping", file=sys.stderr)
            continue
        runs.append((p.stem, load_jsonl(p)))
    if not runs:
        raise SystemExit("No metrics files loaded.")
    outdir = Path(args.plot_dir)
    p1 = plot_report(runs, outdir)
    p2 = plot_report_final_summary(runs, outdir)
    print(f"Wrote: {p1}")
    print(f"Wrote: {p2}")


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

    if args.plot:
        outdir = Path(args.plot_dir)
        plot_per_step(path.stem, intervals, buckets, outdir)
        print(f"\nWrote plots: {outdir}/{{calibration,intervals}}_{path.stem}.png")


# --------------------------------------------------------------------------
# Plots (lazy-import matplotlib)
# --------------------------------------------------------------------------


def _ema(v: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    out = np.empty_like(v)
    if len(v) == 0:
        return out
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
    return out


def plot_macro(runs: List[Tuple[str, List[Dict[str, Any]]]], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Coverage
    ax = axes[0, 0]
    for label, rows in runs:
        s, v = steps_col(rows, "exploration/coverage")
        if len(s) == 0:
            continue
        order = np.argsort(s)
        ax.plot(s[order], _ema(v[order]), lw=2, label=label)
        ax.plot(s[order], v[order], alpha=0.2)
    ax.set_title("exploration/coverage")
    ax.set_xlabel("step"); ax.set_ylabel("coverage")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Holes per 100k env-steps (rate per fixed compute, NOT per-episode rate).
    # Per-episode rate hides SOOPER's win because SOOPER lengthens episodes,
    # so fewer-but-longer episodes can have the same "hole rate" with half
    # the absolute fall count.
    ax = axes[0, 1]
    for label, rows in runs:
        s, hole_v = steps_col(rows, "epstats/log/hole_terminated/sum")
        if len(s) == 0:
            continue
        order = np.argsort(s)
        s_o, h_o = s[order], hole_v[order]
        # Per-window rate: holes in window / step delta to next window.
        deltas = np.diff(s_o, prepend=0.0)
        deltas = np.where(deltas <= 0, 1.0, deltas)
        per_100k = h_o * 100_000.0 / deltas
        ax.plot(s_o, _ema(per_100k), lw=2, label=label)
    ax.set_title("holes per 100k env-steps")
    ax.set_xlabel("env-step"); ax.set_ylabel("holes per 100k steps")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # max_path_progress
    ax = axes[1, 0]
    for label, rows in runs:
        s, v = steps_col(rows, "exploration/mean_max_path_progress")
        if len(s) == 0:
            continue
        order = np.argsort(s)
        ax.plot(s[order], _ema(v[order]), lw=2, label=label)
        ax.plot(s[order], v[order], alpha=0.2)
    ax.set_title("exploration/mean_max_path_progress")
    ax.set_xlabel("step"); ax.set_ylabel("progress")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # prior_active fraction
    ax = axes[1, 1]
    for label, rows in runs:
        s, v = steps_col(rows, "epstats/log/gate/prior_active/avg")
        if len(s) == 0:
            continue
        order = np.argsort(s)
        ax.plot(s[order], _ema(v[order]), lw=2, label=label)
    ax.set_title("prior_active fraction (SOOPER runs)")
    ax.set_xlabel("step"); ax.set_ylabel("fraction of steps under prior")
    ax.set_ylim(0, 1.0); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outdir / "macro_comparison.png", dpi=110)
    plt.close(fig)


def _cumulative_episodes(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (steps, cum_episode_count) per window flush, sorted by step."""
    s_h, h = steps_col(rows, "epstats/log/hole_terminated/sum")
    s_g, g = steps_col(rows, "epstats/log/goal_terminated/sum")
    s_t, t = steps_col(rows, "epstats/log/timeout_terminated/sum")
    # All three should be aligned with the same step grid; trust hole's grid.
    if len(s_h) == 0:
        return np.array([]), np.array([])
    order = np.argsort(s_h)
    s = s_h[order]
    per_window = h[order] + (
        g[order] if g.size == h.size else 0
    ) + (t[order] if t.size == h.size else 0)
    return s, np.cumsum(per_window)


def _cumulative_holes(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    s, h = steps_col(rows, "epstats/log/hole_terminated/sum")
    if len(s) == 0:
        return np.array([]), np.array([])
    order = np.argsort(s)
    return s[order], np.cumsum(h[order])


def _cumulative_goals(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    s, g = steps_col(rows, "epstats/log/goal_terminated/sum")
    if len(s) == 0:
        return np.array([]), np.array([])
    order = np.argsort(s)
    return s[order], np.cumsum(g[order])


def _label_for(stem: str) -> str:
    """Pretty-print run filenames for plot legends."""
    mapping = {
        "metrics": "plain OPAX (baseline)",
        "metrics_sooper_v3": "SOOPER v3 (τ=0.65, H_min=300)",
        "metrics_sooper_v4": "SOOPER v4 (τ=0.65, H_min=100)",
        "metrics_sooper_v5": "SOOPER v5 (τ=0.50, H_max=300)",
        "metrics_sooper_v6": "SOOPER v6 (τ=0.70, cd=150)",
        "metrics_sooper_v7": "SOOPER v7 (τ=0.70, cd=60)",
        "metrics_sooper_v8": "SOOPER v8 (τ=0.65, τ_low=0.25)",
    }
    return mapping.get(stem, stem)


def _running_max(v: np.ndarray) -> np.ndarray:
    out = np.empty_like(v)
    cur = -np.inf
    for i, x in enumerate(v):
        cur = max(cur, x)
        out[i] = cur
    return out


def _final_stats(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    hole = col(rows, "epstats/log/hole_terminated/sum")
    goal = col(rows, "epstats/log/goal_terminated/sum")
    timeout = col(rows, "epstats/log/timeout_terminated/sum")
    cov = col(rows, "exploration/coverage")
    prog = col(rows, "exploration/mean_max_path_progress")
    eps = float((hole + goal + timeout).sum())
    return {
        "episodes": eps,
        "holes": float(hole.sum()),
        "goals": float(goal.sum()),
        "coverage": float(cov[-5:].mean()) if len(cov) else 0.0,
        "progress": float(prog[-5:].mean()) if len(prog) else 0.0,
        "env_steps": float(total_env_steps(rows)),
    }


def plot_report(
    runs: List[Tuple[str, List[Dict[str, Any]]]],
    outdir: Path,
) -> Path:
    """Episode-efficiency dashboard: same coverage with fewer episodes.

    Four panels:
      1. Coverage (running max of EMA) vs cumulative episodes — monotone
         curves so 'further left wins' reads at a glance.
      2. Max-path-progress (running max of EMA) vs cumulative episodes.
      3. Cumulative holes vs cumulative episodes — slope = holes/episode.
      4. Cumulative episodes vs env-steps — flatter = longer episodes per
         fixed compute.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    pretty = {label: _label_for(label) for label, _ in runs}
    handles = {label: c for (label, _), c in zip(runs, colors)}

    # --- Panel 1: coverage vs cumulative episodes (running max) ----------
    ax = axes[0, 0]
    max_eps_seen = 0
    for label, rows in runs:
        s_steps, cum_eps = _cumulative_episodes(rows)
        s_cov, cov = steps_col(rows, "exploration/coverage")
        if len(s_steps) == 0 or len(s_cov) == 0:
            continue
        order = np.argsort(s_cov)
        s_cov_o, cov_o = s_cov[order], cov[order]
        eps_at = np.interp(s_cov_o, s_steps, cum_eps)
        cov_smooth = _running_max(_ema(cov_o))
        ax.plot(eps_at, cov_smooth, lw=2.5, color=handles[label],
                label=pretty[label])
        # mark the final point with a dot + episode-count annotation
        ax.scatter(eps_at[-1], cov_smooth[-1], s=60, color=handles[label],
                   zorder=5, edgecolor="white", linewidth=1.5)
        ax.annotate(f" {int(eps_at[-1])} eps",
                    (eps_at[-1], cov_smooth[-1]),
                    fontsize=9, color=handles[label])
        max_eps_seen = max(max_eps_seen, eps_at[-1])
    ax.set_xlim(0, max_eps_seen * 1.10)
    ax.set_title("Coverage vs episodes consumed\n"
                 "(curves further LEFT reach the same coverage with FEWER episodes)",
                 fontsize=12)
    ax.set_xlabel("cumulative episodes", fontsize=11)
    ax.set_ylabel("coverage (running max of EMA)", fontsize=11)
    ax.set_ylim(0, 1.0); ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.3)

    # --- Panel 2: max progress vs cumulative episodes (running max) ------
    ax = axes[0, 1]
    max_eps_seen = 0
    for label, rows in runs:
        s_steps, cum_eps = _cumulative_episodes(rows)
        s_p, prog = steps_col(rows, "exploration/mean_max_path_progress")
        if len(s_steps) == 0 or len(s_p) == 0:
            continue
        order = np.argsort(s_p)
        s_p_o, prog_o = s_p[order], prog[order]
        eps_at = np.interp(s_p_o, s_steps, cum_eps)
        prog_smooth = _running_max(_ema(prog_o))
        ax.plot(eps_at, prog_smooth, lw=2.5, color=handles[label],
                label=pretty[label])
        ax.scatter(eps_at[-1], prog_smooth[-1], s=60, color=handles[label],
                   zorder=5, edgecolor="white", linewidth=1.5)
        ax.annotate(f" {int(eps_at[-1])} eps",
                    (eps_at[-1], prog_smooth[-1]),
                    fontsize=9, color=handles[label])
        max_eps_seen = max(max_eps_seen, eps_at[-1])
    ax.set_xlim(0, max_eps_seen * 1.10)
    ax.set_title("Max-path-progress vs episodes consumed\n"
                 "(curves further LEFT make the same progress with FEWER episodes)",
                 fontsize=12)
    ax.set_xlabel("cumulative episodes", fontsize=11)
    ax.set_ylabel("max_path_progress (running max of EMA)", fontsize=11)
    ax.legend(fontsize=10, loc="lower right"); ax.grid(alpha=0.3)

    # --- Panel 3: cumulative holes vs cumulative episodes ----------------
    ax = axes[1, 0]
    max_eps_seen = 0
    for label, rows in runs:
        s_steps, cum_eps = _cumulative_episodes(rows)
        s_h, cum_h = _cumulative_holes(rows)
        if len(cum_eps) == 0 or len(cum_h) == 0:
            continue
        eps_at_h = np.interp(s_h, s_steps, cum_eps)
        ax.plot(eps_at_h, cum_h, lw=2.5, color=handles[label],
                label=pretty[label])
        ax.scatter(eps_at_h[-1], cum_h[-1], s=60, color=handles[label],
                   zorder=5, edgecolor="white", linewidth=1.5)
        max_eps_seen = max(max_eps_seen, eps_at_h[-1])
    upper = max_eps_seen * 1.05
    ax.plot([0, upper], [0, upper], color="black", linestyle="--", alpha=0.4,
            lw=1.2, label="hole/episode = 1 (every episode falls)")
    ax.set_xlim(0, upper)
    ax.set_title("Cumulative holes vs episodes\n"
                 "(steeper slope = more fall-prone)",
                 fontsize=12)
    ax.set_xlabel("cumulative episodes", fontsize=11)
    ax.set_ylabel("cumulative fall events", fontsize=11)
    ax.legend(fontsize=10, loc="upper left"); ax.grid(alpha=0.3)

    # --- Panel 4: episodes vs env-steps ----------------------------------
    ax = axes[1, 1]
    for label, rows in runs:
        s_steps, cum_eps = _cumulative_episodes(rows)
        if len(s_steps) == 0:
            continue
        ax.plot(s_steps, cum_eps, lw=2.5, color=handles[label],
                label=pretty[label])
        ax.scatter(s_steps[-1], cum_eps[-1], s=60, color=handles[label],
                   zorder=5, edgecolor="white", linewidth=1.5)
    ax.set_title("Episodes consumed vs env-steps (compute budget)\n"
                 "(flatter line = LONGER episodes per compute unit)",
                 fontsize=12)
    ax.set_xlabel("env-steps", fontsize=11)
    ax.set_ylabel("cumulative episodes", fontsize=11)
    ax.legend(fontsize=10, loc="upper left"); ax.grid(alpha=0.3)

    plt.suptitle(
        "SOOPER episode-efficiency: reaching the same coverage with fewer episodes",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = outdir / "report_episode_efficiency.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_report_final_summary(
    runs: List[Tuple[str, List[Dict[str, Any]]]],
    outdir: Path,
) -> Path:
    """Compact summary: coverage and progress achieved per N episodes.

    One bar per run, grouped by metric. Above each bar the episode count
    is annotated; runs that reach similar y values with smaller annotated
    counts are the headline SOOPER win.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    stats = [(label, _final_stats(rows)) for label, rows in runs]
    labels = [_label_for(s[0]) for s in stats]
    cov = [s[1]["coverage"] for s in stats]
    prog = [s[1]["progress"] for s in stats]
    eps = [int(s[1]["episodes"]) for s in stats]
    holes = [int(round(s[1]["holes"])) for s in stats]
    goals = [s[1]["goals"] for s in stats]
    colors = plt.cm.tab10(np.linspace(0, 1, len(stats)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel A: final coverage with #episodes annotation
    ax = axes[0]
    bars = ax.bar(range(len(stats)), cov, color=colors)
    for i, (b, e) in enumerate(zip(bars, eps)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{e} eps", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("final coverage (last 5 windows)")
    ax.set_title("Coverage achieved (lower #eps with same height = win)",
                 fontsize=11)
    ax.set_ylim(0, max(cov) * 1.15 if cov else 1.0)
    ax.grid(alpha=0.3, axis="y")

    # Panel B: final max progress with #episodes annotation
    ax = axes[1]
    bars = ax.bar(range(len(stats)), prog, color=colors)
    for i, (b, e) in enumerate(zip(bars, eps)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1,
                f"{e} eps", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("final max_path_progress (last 5 windows)")
    ax.set_title("Max progress achieved (lower #eps with same height = win)",
                 fontsize=11)
    ax.set_ylim(0, max(prog) * 1.15 if prog else 1.0)
    ax.grid(alpha=0.3, axis="y")

    # Panel C: holes + goals
    ax = axes[2]
    x = np.arange(len(stats))
    w = 0.4
    ax.bar(x - w / 2, holes, w, color=colors, label="falls")
    ax.bar(x + w / 2, goals, w, color="gold", edgecolor="black",
           label="goals")
    for i, (h, g) in enumerate(zip(holes, goals)):
        ax.text(i - w / 2, h + 0.5, str(h), ha="center", fontsize=9)
        if g > 0.01:
            ax.text(i + w / 2, g + 0.5, f"{g:.2f}", ha="center",
                    fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("event count (per 100k env-steps)")
    ax.set_title("Falls vs goals per fixed compute budget", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    plt.suptitle("SOOPER final-state summary", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = outdir / "report_final_summary.png"
    plt.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_per_step(
    stem: str,
    intervals: List[Dict[str, Any]],
    buckets: Dict[str, Dict[str, np.ndarray]],
    outdir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    # ---- Calibration histograms (risk signals vs distance-to-fall) ----
    bucket_colors = ["tab:green", "tab:blue", "tab:cyan", "tab:orange", "tab:red"]
    signals = [
        ("risk_cont_max", (0, 1), None),
        ("risk_cont_product", (0, 1), None),
        ("risk_critic", (0, 1), 0.65),
        ("V_prior", (0, 130), None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (sig, xlim, tau) in zip(axes.flat, signals):
        for color, label in zip(bucket_colors, DISTANCE_BUCKETS):
            arr = buckets[label][sig]
            if arr.size == 0:
                continue
            ax.hist(
                arr, bins=60, range=xlim, alpha=0.55, color=color,
                label=f"{label} (n={arr.size})", density=True,
            )
        if tau is not None:
            ax.axvline(tau, color="black", linestyle="--", lw=2,
                       label=f"tau_high={tau}")
        ax.set_title(sig)
        ax.set_xlabel(sig); ax.set_ylabel("density")
        ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(*xlim)
    plt.suptitle(f"Risk-signal calibration — {stem}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outdir / f"calibration_{stem}.png", dpi=110)
    plt.close(fig)

    # ---- Interval-length distribution, coloured by outcome ----
    fig, ax = plt.subplots(figsize=(10, 6))
    outcome_colors = {
        "soft_released": "tab:green",
        "hard_released": "tab:purple",
        "fell_mid_hold": "tab:red",
        "truncated":     "tab:gray",
    }
    max_len = max((x["length"] for x in intervals), default=600) + 1
    for outcome, color in outcome_colors.items():
        L = [x["length"] for x in intervals if x["outcome"] == outcome]
        if not L:
            continue
        ax.hist(L, bins=40, range=(0, max_len), alpha=0.6, color=color,
                label=f"{outcome} (n={len(L)})")
    ax.set_title(f"Prior-active interval length — {stem}")
    ax.set_xlabel("interval length (control steps)")
    ax.set_ylabel("count"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"intervals_{stem}.png", dpi=110)
    plt.close(fig)


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
    pm.add_argument("--plot", action="store_true",
                    help="Also write a multi-run PNG dashboard (needs matplotlib).")
    pm.add_argument("--plot-dir", default="metrics/plots",
                    help="Output directory for plots. Default: metrics/plots.")
    pm.set_defaults(func=cmd_macro)

    pr = sub.add_parser(
        "report",
        help="Generate publication-style plots for the SOOPER report.",
    )
    pr.add_argument(
        "files", nargs="+",
        help="One or more dreamer metrics.jsonl files.",
    )
    pr.add_argument("--plot-dir", default="metrics/plots",
                    help="Output directory for plots. Default: metrics/plots.")
    pr.set_defaults(func=cmd_report)

    pp = sub.add_parser(
        "per-step",
        help="Analyze a single sooper_steps.jsonl dump.",
    )
    pp.add_argument("file", help="Path to a sooper_steps.jsonl file.")
    pp.add_argument("--plot", action="store_true",
                    help="Also write calibration + interval-length PNGs (needs matplotlib).")
    pp.add_argument("--plot-dir", default="metrics/plots",
                    help="Output directory for plots. Default: metrics/plots.")
    pp.set_defaults(func=cmd_per_step)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
