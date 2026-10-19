"""Generate the map-fidelity figures for multi_agent/local.

Writes into multi_agent/local/docs/:

  map_vs_reference.png  reference screenshot (with the reconstructed footprints
                        reprojected onto it) next to the new map's plan view
  arena_elements.png    labelled plan of every arena element
  camera_views.png      84x84 camera frames next to the reference FPV inset

Run from multi_agent/local:   python tools/render_map_figures.py

The photogrammetry constants below are the measurements the reconstruction was
built from; see the comment at the top of scripts/sim/maps/map1.json.
"""

import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Affine2D

HERE = os.path.dirname(os.path.abspath(__file__))
LOCAL = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(LOCAL))
sys.path.insert(0, LOCAL)

from scripts.sim.camera import RayCamera          # noqa: E402
from scripts.sim.world import World, rotation_matrix  # noqa: E402

REF = os.path.join(REPO, "imgs", "trajectory_map1_3drones.png")
DOCS = os.path.join(LOCAL, "docs")

# --- measured image geometry of the reference screenshot --------------------
# vanishing points of the floor plane and the image of the top-left floor
# corner, from the wall-base lines and the near-wall cap line.
V_X = np.array([33379.9, 776.2])     # world +x direction (nearly affine)
V_Y = np.array([796.7, -1262.2])     # world +y direction
O_TL = np.array([217.6, 81.4])       # image of the top-left floor corner
TARGET_PX = np.array([1449.0, 386.0])  # measured target sphere contact point
MEAS_X_L, MEAS_W = -0.5, 40.0        # left wall x and room width used in the fit


def floor_homography():
    """Homography mapping world (x, y) on the floor to screenshot pixels."""
    vx = np.array([V_X[0], V_X[1], 1.0])
    vy = np.array([V_Y[0], V_Y[1], 1.0])
    o = np.array([O_TL[0], O_TL[1], 1.0])
    t = np.array([TARGET_PX[0], TARGET_PX[1], 1.0])
    u_t, v_t = 26.8 - MEAS_X_L, 3.9 + MEAS_W / 2.0
    a, b, _ = np.linalg.solve(np.array([u_t * vx, v_t * vy, -t]).T, -o)
    H = np.array([a * vx, b * vy, o]).T

    def project(x, y):
        q = H @ np.array([x - MEAS_X_L, y + MEAS_W / 2.0, 1.0])
        return q[:2] / q[2]
    return project


def expanded_boxes(world):
    """Base boxes only (skip the cosmetic coping slabs) for plan drawing."""
    return [b for b in world.boxes if not b.name.endswith("_cap")]


def box_corners_2d(b):
    """Plan-view corners of a box, honouring its yaw."""
    hx, hy = b.half[0], b.half[1]
    loc = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    r = b.rot[:2, :2]
    return (r @ loc.T).T + b.center[:2]


MAT_COLORS = None


def plan_axes(ax, world, mapd, labels=False):
    a = world.arena
    ax.add_patch(mp.Rectangle((a["x_min"], a["y_min"]),
                              a["x_max"] - a["x_min"], a["y_max"] - a["y_min"],
                              fill=False, lw=2.5, ec="#222"))
    for b in expanded_boxes(world):
        if b.name.startswith("wall_"):
            continue
        col = np.array(mapd["materials"][b.material]["color"]) / 255.0
        poly = mp.Polygon(box_corners_2d(b), closed=True, fc=col, ec="#333", lw=0.6)
        ax.add_patch(poly)
    t = mapd["objects"]["target"]
    ax.add_patch(mp.Circle((t["pos"][0], t["pos"][1]), t["radius"], fc="#e02020", zorder=5))
    ax.add_patch(mp.Circle((t["pos"][0], t["pos"][1]), 12, fill=False,
                           ec="#e02020", ls="--", lw=1.2, zorder=5))
    sp = mapd["spawn"]
    ax.plot([sp["x"], sp["x"]], sp["y_range"], color="#1a9f3a", lw=4, alpha=.55,
            solid_capstyle="butt", zorder=4)
    ax.set_xlim(a["x_min"] - 1.5, a["x_max"] + 1.5)
    ax.set_ylim(a["y_max"] + 1.5, a["y_min"] - 1.5)   # y down = image orientation
    ax.set_aspect("equal")


def fig_map_vs_reference(world, mapd):
    ref = cv2.cvtColor(cv2.imread(REF), cv2.COLOR_BGR2RGB)
    proj = floor_homography()
    over = ref.copy()
    for b in expanded_boxes(world):
        if b.name.startswith("wall_"):
            continue
        pts = np.array([proj(x, y) for x, y in box_corners_2d(b)], np.int32)
        cv2.polylines(over, [pts], True, (255, 215, 0), 2, cv2.LINE_AA)
    a = world.arena
    room = np.array([proj(x, y) for x, y in
                     [(a["x_min"], a["y_min"]), (a["x_max"], a["y_min"]),
                      (a["x_max"], a["y_max"]), (a["x_min"], a["y_max"])]], np.int32)
    cv2.polylines(over, [room], True, (0, 220, 255), 2, cv2.LINE_AA)
    tp = proj(*mapd["objects"]["target"]["pos"][:2]).astype(int)
    cv2.circle(over, tuple(tp), 9, (255, 0, 0), 3)

    fig, axes = plt.subplots(2, 1, figsize=(15, 13.5),
                             gridspec_kw={"height_ratios": [1, 1.15]})
    axes[0].imshow(over)
    axes[0].set_title("Reference screenshot with the reconstructed footprints "
                      "reprojected onto it\n(yellow = obstacle footprints, "
                      "cyan = arena, red = target)", fontsize=11)
    axes[0].axis("off")
    plan_axes(axes[1], world, mapd)
    axes[1].set_title("New map1 plan view — same orientation as the screenshot "
                      "(x right, y down)", fontsize=11)
    axes[1].set_xlabel("x [m]"); axes[1].set_ylabel("y [m]")
    axes[1].grid(alpha=.25, lw=.5)
    fig.tight_layout()
    out = os.path.join(DOCS, "map_vs_reference.png")
    fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)
    return out


def fig_elements(world, mapd):
    fig, ax = plt.subplots(figsize=(13, 9.5))
    plan_axes(ax, world, mapd)
    for b in expanded_boxes(world):
        if b.name.startswith(("wall_", "pilaster")):
            continue
        cx, cy = b.center[0], b.center[1]
        ax.annotate(f"{b.name}\n{b.half[0]*2:.1f}x{b.half[1]*2:.1f}x{b.half[2]*2:.1f}",
                    (cx, cy), fontsize=6.0, ha="center", va="center",
                    color="#111",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", alpha=.72, lw=0))
    t = mapd["objects"]["target"]["pos"]
    ax.annotate("target\n(26.8, 3.9)", (t[0], t[1]), xytext=(t[0] - 5.5, t[1] + 5.0),
                fontsize=8, color="#b00", arrowprops=dict(arrowstyle="->", color="#b00"))
    ax.annotate("success radius 12 m", (t[0] - 12, t[1]), xytext=(t[0] - 13.5, t[1] - 7),
                fontsize=8, color="#b00")
    sp = mapd["spawn"]
    ax.annotate(f"spawn line x=0,  y in [{sp['y_range'][0]:.0f}, {sp['y_range'][1]:.0f}],"
                f"  z in [{sp['z_range'][0]:.0f}, {sp['z_range'][1]:.0f}]",
                (0, 0), xytext=(1.0, -13.0), fontsize=8, color="#1a7f2a",
                rotation=90, va="center")
    a = world.arena
    ax.set_title(f"map1 arena elements — {a['x_max']-a['x_min']:.1f} x "
                 f"{a['y_max']-a['y_min']:.0f} m, ceiling {mapd['ceiling']['height']:.0f} m, "
                 f"{len(world.boxes)} collision boxes, tile {mapd['tile_size']} m")
    ax.set_xlabel("x [m]  (toward target)"); ax.set_ylabel("y [m]")
    ax.grid(alpha=.25, lw=.5)
    fig.tight_layout()
    out = os.path.join(DOCS, "arena_elements.png")
    fig.savefig(out, dpi=125, bbox_inches="tight"); plt.close(fig)
    return out


def fig_camera_views(world, mapd):
    cam = RayCamera(world, 84, 84)
    shots = [((2.0, 0.0, -2.5), "x=2  y=0  z=-2.5  (just after spawn)"),
             ((9.5, -13.0, -3.0), "x=9.5  y=-13  z=-3  (gap in partition 1)"),
             ((15.0, 6.0, -2.0), "x=15  y=6  z=-2  (mid arena, low)"),
             ((20.0, 4.0, -3.0), "x=20  y=4  z=-3  (target in view)")]
    ref = cv2.cvtColor(cv2.imread(REF), cv2.COLOR_BGR2RGB)[527:782, 1203:1650]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.3))
    axes[0].imshow(ref)
    axes[0].set_title("reference: AirSim FPV inset\n(from the screenshot)", fontsize=9)
    axes[0].axis("off")
    for ax, (pos, label) in zip(axes[1:], shots):
        world.set_pose("drone0", list(pos))
        for i, other in enumerate(world.drone_names[1:]):
            world.set_pose(other, [40.0, 40.0 + i, -4.0])     # park out of view
        img = cam.render("drone0")[:, :, ::-1]                 # BGR -> RGB
        ax.imshow(cv2.resize(img, (336, 336), interpolation=cv2.INTER_NEAREST))
        ax.set_title(f"local sim 84x84\n{label}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Drone camera observations — 84x84, 120 deg FOV, 2x2 supersampled "
                 "(shown nearest-neighbour upscaled)", fontsize=12, y=1.06)
    fig.tight_layout()
    out = os.path.join(DOCS, "camera_views.png")
    fig.savefig(out, dpi=115, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    os.makedirs(DOCS, exist_ok=True)
    with open(os.path.join(LOCAL, "scripts/sim/maps/map1.json")) as f:
        mapd = json.load(f)
    world = World(["drone0", "drone1", "drone2"])
    for fn in (fig_map_vs_reference, fig_elements, fig_camera_views):
        print("wrote", fn(world, mapd))


if __name__ == "__main__":
    main()
