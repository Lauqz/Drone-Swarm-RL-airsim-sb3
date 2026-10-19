import math

import numpy as np
import pytest

from scripts.sim.camera import RayCamera
from scripts.sim.world import World

TEST_MAP = {
    "arena": {"x_min": -50, "x_max": 50, "y_min": -50, "y_max": 50,
              "wall_height": 20, "wall_thickness": 0.5},
    "objects": {
        "target": {"pos": [40.0, 0.0, -2.0], "radius": 1.0,
                   "material": "target", "collidable": False}
    },
    "boxes": [],
    "materials": {
        "stone": {"color": [150, 160, 170], "pattern": "brick"},
        "floor": {"color": [105, 118, 132], "pattern": "tile"},
        "target": {"color": [225, 40, 40], "pattern": "plain", "emissive": True},
        "drone": {"color": [45, 45, 50], "pattern": "plain"},
        "white": {"color": [235, 235, 235], "pattern": "plain"},
    },
}


def make(boxes=(), drones=("drone0",)):
    data = dict(TEST_MAP)
    data["boxes"] = list(boxes)
    w = World(list(drones), map_data=data)
    return w, RayCamera(w)


def test_center_depth_matches_wall_distance():
    wall = {"center": [10.0, 0.0, -10.0], "size": [0.5, 40.0, 20.0], "material": "white"}
    w, cam = make([wall])
    w.set_pose("drone0", [0.0, 0.0, -2.0])
    depth = cam.render_depth("drone0")
    # camera sits 0.6 above the body; wall front face at x = 9.75
    assert depth[42, 42] == pytest.approx(9.75, abs=0.02)


def test_horizontal_fov_is_120_degrees():
    """A wall of width w at distance d must span 2*atan(w/2d) of the 120 deg image."""
    wall = {"center": [10.0, 0.0, -10.0], "size": [0.5, 10.0, 20.0], "material": "white"}
    w, cam = make([wall])
    w.set_pose("drone0", [0.0, 0.0, -2.0])
    depth = cam.render_depth("drone0")
    row = depth[42, :]
    hit = np.where(row < 50)[0]
    # expected half-width in pixels: tan(atan(5/9.75)) / tan(60deg) * 42
    expected = math.tan(math.atan2(5.0, 9.75)) / math.tan(math.radians(60)) * 42
    assert len(hit) == pytest.approx(2 * expected, abs=3)


def test_target_renders_red_and_emissive():
    w, cam = make()
    w.set_pose("drone0", [35.0, 0.0, -2.0])
    img = cam.render("drone0")  # BGR
    h, w_, _ = img.shape
    patch = img[h // 2 - 3:h // 2 + 3, w_ // 2 - 3:w_ // 2 + 3].astype(int)
    # red dominant: R channel (index 2 in BGR) well above B and G
    assert patch[:, :, 2].mean() > 180
    assert patch[:, :, 2].mean() > patch[:, :, 0].mean() + 80


def test_other_drones_are_visible():
    w, cam = make(drones=("drone0", "drone1"))
    w.set_pose("drone0", [0.0, 0.0, -2.0])
    w.set_pose("drone1", [3.0, 0.0, -2.0])
    with_other = cam.render("drone0").astype(int)
    w.set_pose("drone1", [0.0, 0.0, -2.0])  # same place as viewer: out of view
    without = cam.render("drone0").astype(int)
    assert np.abs(with_other - without).sum() > 0


def test_render_is_deterministic():
    w, cam = make([{"center": [8.0, 2.0, -3.0], "size": [2, 2, 6],
                    "rpy_deg": [0, 0, 25], "material": "stone"}])
    w.set_pose("drone0", [0.0, 0.0, -2.0])
    a = cam.render("drone0")
    b = cam.render("drone0")
    assert a.dtype == np.uint8 and a.shape == (84, 84, 3)
    assert np.array_equal(a, b)


def test_depth_perspective_not_planar():
    """DepthPerspective = distance along the ray, so off-axis pixels looking at
    a frontal wall must read farther than the center pixel."""
    wall = {"center": [10.0, 0.0, -10.0], "size": [0.5, 60.0, 20.0], "material": "white"}
    w, cam = make([wall])
    w.set_pose("drone0", [0.0, 0.0, -2.0])
    depth = cam.render_depth("drone0")
    assert depth[42, 10] > depth[42, 42] + 1.0
