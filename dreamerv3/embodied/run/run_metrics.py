"""Faithful, budget-normalized training metrics for the cyberrunner
OPAX-vs-SOOPER comparison.

Replaces the window-averaged `epstats` accounting (which reported per-window
fractions, not counts, and undercounted episode outcomes ~2x) and the old
`_CoverageTracker`. Everything here is fed one per-env transition at a time
from `train.py:logfn(tran, worker)` and emitted at log time.

Coordinate conventions
----------------------
`tran['states'][2:4]` are the marble's BOARD-FRAME coordinates, normalized by
(BOARD_WIDTH/2, BOARD_HEIGHT/2). The board frame is centered at the board's
geometric center, so states[2:4] ∈ [-1, 1]. Converting to corner-origin
fraction: `frac = (states[2:4] + 1) / 2 ∈ [0, 1]`. Grid cell:
`cx = int(frac_x * grid_res)`. The maze layout (walls/waypoints) is in
corner-origin physical meters, so it maps to the SAME grid via
`int(x / board_w * grid_res)` — consistent with the ball mapping.
"""
from __future__ import annotations

import collections
import pathlib
import random
import sys
from typing import Any

import numpy as np

BOARD_WIDTH = 0.276
BOARD_HEIGHT = 0.231


# --------------------------------------------------------------------------
# Maze layout
# --------------------------------------------------------------------------

def _load_layout(layout: str):
    """Return (walls_h, walls_v, holes, waypoints) for a maze layout."""
    repo_root = str(pathlib.Path(__file__).resolve().parents[3])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from cyberrunner_env_vision import _LAYOUT_LOADERS
    return _LAYOUT_LOADERS[layout]()


# --------------------------------------------------------------------------
# Exploration
# --------------------------------------------------------------------------

class ExplorationStats:
    """Segment-based coverage (matching the proven `coverage_frac` in
    `.vendor/cyberrunner_ppo/env_mjx.py`), coverage delta, path progress, and
    spatial visitation entropy.

    Coverage = fraction of the maze's polyline segments the ball has been seen
    on. The segments ARE the navigable maze, so this is inherently
    "reachable-normalized" — far more robust than a spatial flood-fill mask
    (which is meaningless at the 30x30 resolution where every cell overlaps a
    corridor). The env's `compute_path_progress` returns
    `progress = arc_length * 10`, so the segment index is recovered from
    `log/path_progress` via the cumulative waypoint distances.
    """

    def __init__(self, layout: str = 'hard', grid_res: int = 30,
                 delta_window: int = 30_000, recent_n: int = 100):
        self.grid_res = grid_res
        self.delta_window = delta_window
        self.recent_n = recent_n
        self._latest_step = 0
        # Path-progress trackers.
        self.max_progress_per_episode: collections.deque = collections.deque(
            maxlen=recent_n)
        self._cur_max: dict[int, float] = {}
        self.run_max_progress = 0.0
        # Segment coverage: cumulative arc-length boundaries for seg recovery.
        try:
            _wh, _wv, _holes, wps = _load_layout(layout)
            wps = np.asarray(wps, dtype=np.float32)
            seg_len = np.linalg.norm(np.diff(wps, axis=0), axis=1)
            self._cum = np.concatenate([[0.0], np.cumsum(seg_len)])
            self.total_segments = len(wps) - 1
        except Exception as e:
            print(f'[run_metrics] layout load failed ({e}); coverage disabled',
                  flush=True)
            self._cum = None
            self.total_segments = 1
        self._seg_first_seen: dict[int, int] = {}
        # Spatial visit grid (entropy only — NOT used for coverage).
        self.visit_counts = np.zeros((grid_res, grid_res), dtype=np.int64)

    def _seg_idx(self, path_progress: float) -> int:
        # progress = arc_length * 10; segment k spans [cum[k], cum[k+1]).
        arc = path_progress / 10.0
        seg = int(np.searchsorted(self._cum, arc, side='right')) - 1
        return min(max(seg, 0), self.total_segments - 1)

    def _cell(self, states) -> tuple[int, int]:
        fx = (float(states[2]) + 1.0) * 0.5
        fy = (float(states[3]) + 1.0) * 0.5
        cx = min(max(int(fx * self.grid_res), 0), self.grid_res - 1)
        cy = min(max(int(fy * self.grid_res), 0), self.grid_res - 1)
        return cy, cx

    def update(self, worker, states, path_progress, is_first, is_last,
               env_step=0):
        self._latest_step = max(self._latest_step, int(env_step))
        pp = float(path_progress)
        self.run_max_progress = max(self.run_max_progress, pp)
        if is_first:
            self._cur_max[worker] = 0.0
        self._cur_max[worker] = max(self._cur_max.get(worker, 0.0), pp)
        if is_last:
            self.max_progress_per_episode.append(self._cur_max.pop(worker, 0.0))
        # Segment coverage (only when path is detected → positive progress).
        if self._cum is not None and pp > 0:
            seg = self._seg_idx(pp)
            if seg not in self._seg_first_seen:
                self._seg_first_seen[seg] = self._latest_step
        # Spatial grid (entropy).
        if states is not None and np.ndim(states) >= 1 and np.shape(states)[-1] >= 4:
            cy, cx = self._cell(states)
            self.visit_counts[cy, cx] += 1

    def emit(self) -> dict[str, float]:
        n_seen = len(self._seg_first_seen)
        coverage = n_seen / max(self.total_segments, 1)
        cutoff = self._latest_step - self.delta_window
        fresh = sum(1 for s in self._seg_first_seen.values() if s > cutoff)
        delta = fresh / max(self.total_segments, 1)
        counts = self.visit_counts.astype(np.float64).ravel()
        total = counts.sum()
        if total > 0:
            p = counts[counts > 0] / total
            entropy = float(-np.sum(p * np.log(p)))
        else:
            entropy = 0.0
        mmpp = (float(np.mean(self.max_progress_per_episode))
                if self.max_progress_per_episode else 0.0)
        return {
            'exploration/coverage': float(coverage),
            'exploration/coverage_delta': float(delta),
            'exploration/segments_visited': float(n_seen),
            'exploration/total_segments': float(self.total_segments),
            'exploration/max_path_progress': float(self.run_max_progress),
            'exploration/mean_max_path_progress': mmpp,
            'exploration/visitation_entropy': entropy,
        }


# --------------------------------------------------------------------------
# Safety — faithful cumulative counters
# --------------------------------------------------------------------------

class SafetyStats:
    """Monotonic episode-outcome counters + budget-normalized rates.

    Classifies each completed episode from the env's mutually-exclusive
    termination flags (log/hole_terminated, log/goal_terminated,
    log/timeout_terminated) rather than the buggy window-averaged epstats.
    """

    def __init__(self):
        self.env_steps = 0
        self.episodes = 0
        self.holes = 0
        self.timeouts = 0
        self.goals = 0

    def update(self, tran: dict[str, Any], is_last: bool):
        self.env_steps += 1
        if not is_last:
            return
        self.episodes += 1
        if float(tran.get('log/hole_terminated', 0.0)) > 0.5:
            self.holes += 1
        elif float(tran.get('log/goal_terminated', 0.0)) > 0.5:
            self.goals += 1
        else:
            self.timeouts += 1

    def emit(self) -> dict[str, float]:
        steps = max(self.env_steps, 1)
        completed = max(self.episodes, 1)
        return {
            'safety/holes_cumulative': float(self.holes),
            'safety/holes_per_1k_steps': self.holes / steps * 1000.0,
            'safety/mean_steps_between_falls': (
                self.env_steps / self.holes if self.holes > 0
                else float(self.env_steps)),
            'safety/fall_free_episode_frac': (
                (self.episodes - self.holes) / completed),
            'safety/episodes_completed': float(self.episodes),
            'safety/env_steps': float(self.env_steps),
        }


# --------------------------------------------------------------------------
# Gate diagnostics — only fed when the gate is active
# --------------------------------------------------------------------------

class GateStats:
    """Trigger rate, prior-active fraction, prior-segment length, episode-level
    save rate, and trigger->fall lead time. Updated only when `tran` carries
    `log/gate/triggered` (i.e. SOOPER runs)."""

    def __init__(self):
        self.env_steps = 0
        self.triggers = 0
        self.prior_steps = 0
        self._seg_lens: list[int] = []
        self._cur_seg: dict[int, int] = collections.defaultdict(int)
        # per-episode bookkeeping
        self._ep_triggered: dict[int, bool] = collections.defaultdict(bool)
        self._ep_first_trig_step: dict[int, int] = {}
        self._ep_step: dict[int, int] = collections.defaultdict(int)
        self.triggered_eps = 0
        self.triggered_and_fell = 0
        self.triggered_and_survived = 0
        self._lead_times: list[int] = []

    @staticmethod
    def present(tran: dict[str, Any]) -> bool:
        return 'log/gate/triggered' in tran

    def update(self, tran: dict[str, Any], worker: int,
               is_first: bool, is_last: bool):
        self.env_steps += 1
        if is_first:
            self._ep_triggered[worker] = False
            self._ep_first_trig_step.pop(worker, None)
            self._ep_step[worker] = 0
        step_in_ep = self._ep_step[worker]

        triggered = float(tran.get('log/gate/triggered', 0.0)) > 0.5
        prior_active = float(tran.get('prior_active',
                                      tran.get('log/gate/prior_active', 0.0))) > 0.5
        if triggered:
            self.triggers += 1
            if not self._ep_triggered[worker]:
                self._ep_triggered[worker] = True
                self._ep_first_trig_step[worker] = step_in_ep
        if prior_active:
            self.prior_steps += 1
            self._cur_seg[worker] += 1
        elif self._cur_seg[worker] > 0:
            self._seg_lens.append(self._cur_seg[worker])
            self._cur_seg[worker] = 0

        if is_last:
            if self._cur_seg[worker] > 0:
                self._seg_lens.append(self._cur_seg[worker])
                self._cur_seg[worker] = 0
            if self._ep_triggered[worker]:
                self.triggered_eps += 1
                fell = float(tran.get('log/hole_terminated', 0.0)) > 0.5
                if fell:
                    self.triggered_and_fell += 1
                    first = self._ep_first_trig_step.get(worker, step_in_ep)
                    self._lead_times.append(step_in_ep - first)
                else:
                    self.triggered_and_survived += 1
        self._ep_step[worker] = step_in_ep + 1

    def emit(self) -> dict[str, float]:
        steps = max(self.env_steps, 1)
        out = {
            'gate/trigger_rate_per_1k': self.triggers / steps * 1000.0,
            'gate/prior_active_frac': self.prior_steps / steps,
            'gate/mean_segment_len': (
                float(np.mean(self._seg_lens)) if self._seg_lens else 0.0),
            'gate/triggers_cumulative': float(self.triggers),
        }
        if self.triggered_eps > 0:
            out['gate/save_rate'] = self.triggered_and_survived / self.triggered_eps
            out['gate/fail_rate'] = self.triggered_and_fell / self.triggered_eps
        if self._lead_times:
            out['gate/lead_time_median'] = float(np.median(self._lead_times))
        return out


# --------------------------------------------------------------------------
# Gate-trigger video clips
# --------------------------------------------------------------------------

class TriggerClips:
    """Per-env ring buffer of recent frames; on a gate trigger, capture a
    window from `pre` steps before to `post` steps after, reservoir-sampled to
    `max_clips` per logging interval."""

    def __init__(self, num_envs: int, pre: int = 60, post: int = 120,
                 max_clips: int = 3):
        self.pre = pre
        self.post = post
        self.max_clips = max_clips
        self._ring: dict[int, collections.deque] = {
            i: collections.deque(maxlen=pre) for i in range(num_envs)}
        self._recording: dict[int, list] = {}
        self._clips: list[np.ndarray] = []
        self._seen = 0  # triggers observed this interval (for reservoir)

    def update(self, worker: int, frame, triggered: bool, is_last: bool):
        if frame is None:
            return
        frame = np.asarray(frame)
        if worker in self._recording:
            rec = self._recording[worker]
            rec.append(frame)
            if len(rec) >= self.pre + self.post or is_last:
                self._finalize(worker)
        else:
            self._ring[worker].append(frame)
            if triggered:
                # seed clip with buffered pre-frames
                self._recording[worker] = list(self._ring[worker])

    def _finalize(self, worker: int):
        rec = self._recording.pop(worker, None)
        if not rec or len(rec) < 5:
            return
        clip = np.stack(rec)
        # reservoir sampling of size max_clips
        self._seen += 1
        if len(self._clips) < self.max_clips:
            self._clips.append(clip)
        else:
            j = random.randint(0, self._seen - 1)
            if j < self.max_clips:
                self._clips[j] = clip

    def drain(self) -> dict[str, np.ndarray]:
        out = {f'gate/trigger_clip_{i}': c for i, c in enumerate(self._clips)}
        self._clips = []
        self._seen = 0
        return out
