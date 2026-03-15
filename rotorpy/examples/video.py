import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# ── Load data ──────────────────────────────────────────────────────────────────
import argparse

parser = argparse.ArgumentParser(description="Visualize UAV swarm trajectories.")
parser.add_argument("npz_file", help="Path to the .npz data file")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--show", action="store_true", help="Display the animation in a window")
group.add_argument("--save", metavar="OUTPUT.mp4", help="Save the animation to an mp4 file")
args = parser.parse_args()

data = np.load(args.npz_file)

positions = data["all_pos"]                          # (1023, 4, 3)
desired   = data["desired_trajectories"]             # (4, 1023, 3, 1)
times     = data["all_time"]                         # (1022,)

n_frames, n_uavs, _ = positions.shape
if desired.ndim == 4:
    desired = desired[:, :, :, 0]                    # (4, 1023, 3)

# ── Style ──────────────────────────────────────────────────────────────────────
_BASE_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
                "#9b59b6", "#1abc9c", "#e67e22", "#00bcd4"]
COLORS = [_BASE_COLORS[i % len(_BASE_COLORS)] for i in range(n_uavs)]
UAV_LABELS  = [f"UAV {i+1}" for i in range(n_uavs)]
TRAIL_LEN   = 80        # frames of trail to show
SKIP        = 3         # render every Nth frame (speed vs quality)
FPS         = 30

frames_to_render = range(0, n_frames, SKIP)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 8), facecolor="#0d1117")
ax  = fig.add_subplot(111, projection="3d", facecolor="#0d1117")

ax.tick_params(colors="white")
for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor("#333333")
for spine in ax.spines.values():
    spine.set_edgecolor("#444444")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.zaxis.label.set_color("white")
ax.set_xlabel("X (m)", labelpad=8)
ax.set_ylabel("Y (m)", labelpad=8)
ax.set_zlabel("Z (m)", labelpad=8)
ax.set_title("UAV Swarm — 3D Trajectory", color="white", fontsize=14, pad=12)

# fixed axis limits
pad = 2
ax.set_xlim(positions[:, :, 0].min() - pad, positions[:, :, 0].max() + pad)
ax.set_ylim(positions[:, :, 1].min() - pad, positions[:, :, 1].max() + pad)
ax.set_zlim(positions[:, :, 2].min() - pad, positions[:, :, 2].max() + pad)

# ── Artists ────────────────────────────────────────────────────────────────────
trails    = [ax.plot([], [], [], "-",  color=COLORS[i], alpha=0.5, lw=1.2)[0] for i in range(n_uavs)]
dots      = [ax.plot([], [], [], "o",  color=COLORS[i], ms=8,  label=UAV_LABELS[i])[0] for i in range(n_uavs)]
des_lines = [ax.plot([], [], [], "--", color=COLORS[i], alpha=0.25, lw=0.8)[0] for i in range(n_uavs)]

time_text = ax.text2D(0.02, 0.96, "", transform=ax.transAxes,
                      color="white", fontsize=10, va="top")

ax.legend(loc="upper right", facecolor="#1a1a2e", edgecolor="#444",
          labelcolor="white", fontsize=9)

# ── Update function ────────────────────────────────────────────────────────────
def update(frame_idx):
    f = frame_idx  # actual frame in positions array

    # clamp trail
    t0 = max(0, f - TRAIL_LEN)

    for i in range(n_uavs):
        # trail
        trails[i].set_data(positions[t0:f+1, i, 0], positions[t0:f+1, i, 1])
        trails[i].set_3d_properties(positions[t0:f+1, i, 2])
        # current marker
        dots[i].set_data([positions[f, i, 0]], [positions[f, i, 1]])
        dots[i].set_3d_properties([positions[f, i, 2]])
        # desired trajectory (full)
        des_lines[i].set_data(desired[i, :, 0], desired[i, :, 1])
        des_lines[i].set_3d_properties(desired[i, :, 2])

    # time label (times array is 1 shorter than positions)
    t_val = times[min(f, len(times)-1)]
    time_text.set_text(f"t = {t_val:.2f} s")

    # slow azimuth rotation for cinematic effect
    ax.view_init(elev=25, azim=f * 0.07)

    return trails + dots + des_lines + [time_text]

# ── Build animation ────────────────────────────────────────────────────────────
frames = list(frames_to_render)
ani = animation.FuncAnimation(fig, update, frames=frames,
                               interval=1000 / FPS, blit=False)

# ── Save or Show ──────────────────────────────────────────────────────────────
plt.tight_layout()
if args.save:
    writer = animation.FFMpegWriter(fps=FPS, bitrate=2000,
                                    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
    print(f"Rendering video to {args.save} ...")
    ani.save(args.save, writer=writer, dpi=120)
    print("Done!")
    plt.close()
else:
    plt.show()