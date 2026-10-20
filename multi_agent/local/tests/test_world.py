"""Physics tests.

Dynamics/collision behaviour is tested against a small synthetic map so the
assertions stay valid when map1 geometry is refined. A separate group asserts
the properties the *real* map must satisfy for the env to work at all.
"""

import math

import numpy as np
import pytest

from scripts.sim.world import World, rotation_matrix

NAMES = ["drone0", "drone1"]

# A bare 100x100 room with one pillar at (10, 0) and one thin pole at (20, 0).
SYNTH = {
    "arena": {"x_min": -50, "x_max": 50, "y_min": -50, "y_max": 50,
              "wall_height": 20, "wall_thickness": 0.5, "cap": 0.0},
    "ceiling": {"height": 12.0, "material": "floor"},
    "objects": {},
    "boxes": [
        {"name": "pillar", "center": [10.0, 0.0, -4.0], "size": [2.0, 2.0, 8.0],
         "rpy_deg": [0, 0, 0], "material": "stone"},
        {"name": "pole", "center": [20.0, 0.0, -4.0], "size": [0.18, 0.18, 8.0],
         "rpy_deg": [0, 0, 0], "material": "pole"},
    ],
    "materials": {
        "stone": {"color": [150, 160, 170], "pattern": "block"},
        "cap": {"color": [230, 230, 230], "pattern": "smooth"},
        "pole": {"color": [60, 45, 35], "pattern": "bark"},
        "floor": {"color": [120, 140, 160], "pattern": "tile"},
        "drone": {"color": [45, 45, 50], "pattern": "smooth"},
    },
}


def make_world(**kw):
    kw.setdefault("map_data", {k: (v.copy() if hasattr(v, "copy") else v)
                               for k, v in SYNTH.items()})
    return World(NAMES, **kw)


def park_other(w):
    w.set_pose("drone1", [0, 40, -4])


# ------------------------------------------------------------------ dynamics
def test_move_without_obstacle_matches_ramp():
    w = make_world(tau=0.25)
    w.set_pose("drone0", [0, 0, -4]); park_other(w)
    w.move("drone0", [2.0, 0.0, 0.0], duration=1.0)
    expected = 2.0 * (1.0 + 0.25 * (math.exp(-1 / 0.25) - 1.0))
    assert w.get_pose("drone0")[0] == pytest.approx(expected, abs=1e-9)
    assert w.collision_stamp("drone0") == 0


def test_velocity_is_carried_between_commands():
    # Second identical command starts at the velocity reached by the first,
    # so it covers more ground than a command issued from rest.
    w = make_world(tau=0.25)
    w.set_pose("drone0", [0, 0, -4]); park_other(w)
    w.move("drone0", [2.0, 0.0, 0.0], duration=1.0)
    first = w.get_pose("drone0")[0]
    assert w.get_velocity("drone0")[0] == pytest.approx(2.0 * (1 - math.exp(-4)), abs=1e-9)
    w.move("drone0", [2.0, 0.0, 0.0], duration=1.0)
    second = w.get_pose("drone0")[0] - first
    assert second > first
    assert second == pytest.approx(2.0, abs=0.05)


def test_stop_and_teleport_zero_velocity():
    w = make_world(tau=0.25)
    w.set_pose("drone0", [0, 0, -4]); park_other(w)
    w.move("drone0", [3.0, 1.0, 0.0], duration=1.0)
    w.stop("drone0")
    assert np.allclose(w.get_velocity("drone0"), 0.0)
    w.move("drone0", [3.0, 0.0, 0.0], duration=1.0)
    w.set_pose("drone0", [0, 0, -4])
    assert np.allclose(w.get_velocity("drone0"), 0.0)


def test_collision_zeroes_velocity():
    w = make_world(tau=0.0)
    w.set_pose("drone0", [8.0, 0, -4]); park_other(w)
    w.move("drone0", [3.0, 0.0, 0.0], duration=1.0)   # into the pillar
    assert w.collision_stamp("drone0") == 1
    assert np.allclose(w.get_velocity("drone0"), 0.0)


def test_pure_kinematics_when_tau_zero():
    w = make_world(tau=0.0)
    w.set_pose("drone0", [0, 0, -4]); park_other(w)
    w.move("drone0", [3.0, -1.0, 0.5], duration=1.0)
    assert np.allclose(w.get_pose("drone0"), [3.0, -1.0, -3.5])


# ---------------------------------------------------------------- collisions
def test_collision_with_pillar_stops_and_flags():
    w = make_world(tau=0.0)
    w.set_pose("drone0", [7.0, 0.0, -4.0]); park_other(w)
    stamp = w.collision_stamp("drone0")
    w.move("drone0", [3.0, 0.0, 0.0], duration=1.0)
    assert w.collision_stamp("drone0") != stamp
    assert w.get_pose("drone0")[0] < 9.0  # stopped before the pillar face


def test_no_tunneling_through_thin_pole_at_max_speed():
    w = make_world(tau=0.0)
    w.set_pose("drone0", [17.5, 0.0, -4.0]); park_other(w)
    stamp = w.collision_stamp("drone0")
    w.move("drone0", [3.0, 0.0, 0.0], duration=2.0)
    assert w.collision_stamp("drone0") != stamp, "flew through the pole"


def test_ground_collision():
    w = make_world(tau=0.0)
    w.set_pose("drone0", [0, 0, -1.0]); park_other(w)
    stamp = w.collision_stamp("drone0")
    w.move("drone0", [0.0, 0.0, 3.0], duration=1.0)   # +z = down
    assert w.collision_stamp("drone0") != stamp
    assert w.get_pose("drone0")[2] < -w.drone_radius


def test_ceiling_collision():
    """The invisible capture ceiling is solid."""
    w = make_world(tau=0.0)
    w.set_pose("drone0", [0, 0, -10.0]); park_other(w)
    stamp = w.collision_stamp("drone0")
    w.move("drone0", [0.0, 0.0, -3.0], duration=1.0)  # -z = up
    assert w.collision_stamp("drone0") != stamp
    assert w.get_pose("drone0")[2] > w.ceiling_z


def test_drone_drone_collision():
    w = make_world(tau=0.0)
    w.set_pose("drone0", [0, 0, -4]); w.set_pose("drone1", [0, 2.0, -4])
    stamp = w.collision_stamp("drone0")
    w.move("drone0", [0.0, 3.0, 0.0], duration=1.0)
    assert w.collision_stamp("drone0") != stamp


def test_wall_bounds():
    w = make_world(tau=0.0)
    w.set_pose("drone0", [0, 48.0, -4]); w.set_pose("drone1", [0, -40, -4])
    stamp = w.collision_stamp("drone0")
    w.move("drone0", [0.0, 3.0, 0.0], duration=1.0)
    assert w.collision_stamp("drone0") != stamp
    assert w.get_pose("drone0")[1] < 50.0


def test_collision_stamp_sticky_until_new_baseline():
    """Mirrors AirSim simGetCollisionInfo().time_stamp semantics."""
    w = make_world(tau=0.0)
    w.set_pose("drone0", [7.0, 0.0, -4.0]); park_other(w)
    baseline = w.collision_stamp("drone0")
    w.move("drone0", [3.0, 0.0, 0.0], duration=1.0)
    s1 = w.collision_stamp("drone0")
    assert s1 != baseline
    assert w.collision_stamp("drone0") == s1 != baseline


def test_rotation_matrix_yaw():
    r = rotation_matrix(0, 0, 90)
    assert np.allclose(r @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-12)


# ------------------------------------------------------- the real map1 arena
def test_map1_spawn_band_is_clear():
    """Every reachable spawn pose (x=0, y in [-18,18], z in [-8,-1]) must be free."""
    w = World(NAMES)
    w.set_pose("drone1", [99, 99, -4])
    for y in np.linspace(-18, 18, 145):
        for z in np.linspace(-8, -1, 29):
            assert not w._collides("drone0", np.array([0.0, y, z])), (y, z)


def test_map1_target_is_reachable_and_inside():
    w = World(NAMES)
    t = w.get_object_pose("target")
    a = w.arena
    assert a["x_min"] < t[0] < a["x_max"]
    assert a["y_min"] < t[1] < a["y_max"]


def test_map1_has_ceiling_above_spawn_ceiling():
    w = World(NAMES)
    assert w.ceiling_z is not None
    assert w.ceiling_z < -8.0, "ceiling must clear the -8 m spawn altitude"


def test_map1_caps_and_pilasters_expanded():
    w = World(NAMES)
    names = [b.name for b in w.boxes]
    assert any(n.endswith("_cap") for n in names), "wall caps not expanded"
    assert any(n.startswith("pilaster") for n in names), "pilasters not expanded"
    assert not any(n.startswith("pilaster_b") for n in names), \
        "back (spawn) wall should have no pilasters"
