"""Parse the W&B output.log and produce publication-ready training curve figures."""
import re
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

LOG_PATH = pathlib.Path(
    "wandb/run-20260421_182904-ej6mr2l9/files/output.log"
)
OUT_DIR = pathlib.Path("visualizations")
OUT_DIR.mkdir(exist_ok=True)

# ── parse spawn-eval lines ────────────────────────────────────────────────────
# [step 50016] spawn_success_mean=0.417 spawn_success_p10=0.000 ...
SPAWN_RE = re.compile(
    r"\[step\s+(\d+)\].*?"
    r"spawn_success_mean=([\d.]+).*?"
    r"spawn_success_p10=([\d.]+).*?"
    r"spawn_success_min=([\d.]+)"
)

# ── parse per-step train lines ────────────────────────────────────────────────
# [step 10000] ep_rew_mean=... success_rate=... final_ball_speed=...
TRAIN_RE = re.compile(
    r"\[step\s+(\d+)\]\s+"
    r"ep_rew_mean=([\d.]+)\s+"
    r"ep_len_mean=([\d.]+)\s+"
    r"success_rate=([\d.]+).*?"
    r"final_ball_speed=([\d.]+).*?"
    r"final_safe_margin=([-\d.]+)"
)

spawn_steps, spawn_mean, spawn_p10, spawn_min = [], [], [], []
train_steps, train_success, train_speed, train_margin = [], [], [], []

with open(LOG_PATH) as f:
    for line in f:
        m = SPAWN_RE.search(line)
        if m:
            spawn_steps.append(int(m.group(1)))
            spawn_mean.append(float(m.group(2)))
            spawn_p10.append(float(m.group(3)))
            spawn_min.append(float(m.group(4)))
            continue
        m = TRAIN_RE.search(line)
        if m:
            train_steps.append(int(m.group(1)))
            train_success.append(float(m.group(4)))
            train_speed.append(float(m.group(5)))
            train_margin.append(float(m.group(6)))

spawn_steps = np.array(spawn_steps) / 1e6
train_steps = np.array(train_steps) / 1e6


def smooth(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < w:
        return x, np.arange(len(x))
    return np.convolve(x, np.ones(w) / w, mode="valid"), np.arange(len(x) - w + 1)


# ── Figure 1: spawn success rate (mean / p10 / min) ──────────────────────────
fig, ax = plt.subplots(figsize=(6, 3.5))

ax.plot(spawn_steps, spawn_mean, label="Mean",    color="#2196F3", lw=1.8)
ax.plot(spawn_steps, spawn_p10,  label="P10",     color="#FF9800", lw=1.4, ls="--")
ax.plot(spawn_steps, spawn_min,  label="Min",     color="#F44336", lw=1.2, ls=":")
ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.4)

ax.set_xlabel("Environment steps ($\\times 10^6$)")
ax.set_ylabel("Success rate")
ax.set_title("Prior stabilization: spawn success rate")
ax.set_ylim(-0.05, 1.05)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "prior_spawn_success.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "prior_spawn_success.png", dpi=150, bbox_inches="tight")
print("Saved prior_spawn_success.pdf/.png")
plt.close(fig)

# ── Figure 2: training episode success rate (smoothed) ───────────────────────
window = 20
s_success, s_idx = smooth(np.array(train_success), window)
s_steps = train_steps[s_idx]

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(train_steps, train_success, color="#2196F3", lw=0.6, alpha=0.3)
ax.plot(s_steps, s_success, color="#2196F3", lw=1.8, label=f"Rolling mean (w={window})")
ax.set_xlabel("Environment steps ($\\times 10^6$)")
ax.set_ylabel("Episode success rate")
ax.set_title("Prior stabilization: training success rate")
ax.set_ylim(-0.05, 1.05)
ax.legend(framealpha=0.9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "prior_train_success.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "prior_train_success.png", dpi=150, bbox_inches="tight")
print("Saved prior_train_success.pdf/.png")
plt.close(fig)

# ── Figure 3: final ball speed over training ──────────────────────────────────
s_speed, s_idx = smooth(np.array(train_speed), window)
s_steps_sp = train_steps[s_idx]

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(train_steps, train_speed, color="#9C27B0", lw=0.6, alpha=0.3)
ax.plot(s_steps_sp, s_speed, color="#9C27B0", lw=1.8)
ax.set_xlabel("Environment steps ($\\times 10^6$)")
ax.set_ylabel("Final ball speed (m/s)")
ax.set_title("Prior stabilization: final ball speed")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "prior_ball_speed.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "prior_ball_speed.png", dpi=150, bbox_inches="tight")
print("Saved prior_ball_speed.pdf/.png")
plt.close(fig)
