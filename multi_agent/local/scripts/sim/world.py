"""Minimal drone world used in place of UE4+AirSim.

Replicates the subset of AirSim behavior the training env relies on:

- Vehicles are teleported to a start pose (``set_pose``) and then moved with
  body-frame velocity commands held for a fixed duration (``move``), matching
  ``moveByVelocityBodyFrameAsync(...).join()``. Yaw is never commanded by the
  env, so the body frame coincides with the world frame.
- SimpleFlight does not reach the commanded velocity instantly; the ramp is
  modeled as a first-order response with time constant ``tau`` (set tau=0 for
  pure kinematics).
- Collisions follow AirSim's ``simGetCollisionInfo().time_stamp`` semantics:
  ``collision_stamp`` returns a counter that changes when a new collision
  happens and stays changed afterwards, so comparing against a baseline taken
  at episode setup gives a sticky per-episode collision flag.
- Vehicles that stop receiving commands (done agents) stay where they are and
  remain physical obstacles for the others, as in AirSim.
- The arena is closed by an invisible ceiling, like the capture ceiling in the
  UE map: drones cannot fly through it and their cameras cannot see past it,
  but the browser viewer never draws it so the arena stays readable from above.

Coordinates are NED meters: x forward (toward the target), y right, z down
(negative z = altitude). The ground plane is z=0 and the ceiling z=-height.

Map JSON conveniences expanded here (so the map file stays declarative):
  * a box with ``"cap": t`` also emits a light coping slab of thickness t on
    top, slightly oversized — every wall and pillar in the reference image has
    one.
  * the ``pilasters`` block emits buttresses spaced along the perimeter walls.
"""

import json
import math
import os

import numpy as np

DEFAULT_MAP = os.path.join(os.path.dirname(__file__), "maps", "map1.json")


def rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """Intrinsic ZYX (yaw-pitch-roll) rotation, body-to-world."""
    r, p, y = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


class Box:
    __slots__ = ("name", "center", "half", "rot", "material", "axis_aligned",
                 "bound_r")

    def __init__(self, name, center, size, rpy_deg, material):
        self.name = name
        self.center = np.asarray(center, dtype=np.float64)
        self.half = np.asarray(size, dtype=np.float64) / 2.0
        self.rot = rotation_matrix(*rpy_deg)
        self.material = material
        self.axis_aligned = np.allclose(self.rot, np.eye(3))
        self.bound_r = float(np.linalg.norm(self.half))

    def closest_point_dist(self, p):
        """Distance from point p to the box surface (0 if inside)."""
        local = self.rot.T @ (p - self.center)
        clamped = np.clip(local, -self.half, self.half)
        return float(np.linalg.norm(local - clamped))


class World:
    def __init__(self, drone_names, map_path=None, map_data=None,
                 drone_radius=0.35, tau=0.25, substeps=10):
        if map_data is None:
            with open(map_path or DEFAULT_MAP, "r") as f:
                map_data = json.load(f)
        self.map_data = map_data
        self.drone_names = list(drone_names)
        self.drone_radius = float(drone_radius)
        self.tau = float(tau)
        self.substeps = int(substeps)

        self.materials = map_data["materials"]
        self.objects = map_data.get("objects", {})
        self.arena = map_data["arena"]
        self.ceiling_z = -float(map_data.get("ceiling", {}).get("height", 0.0)) \
            if map_data.get("ceiling") else None
        self.ceiling_material = map_data.get("ceiling", {}).get("material", "floor")

        self.boxes = []
        for i, b in enumerate(map_data.get("boxes", [])):
            self._add_box(b.get("name", f"box{i}"), b["center"], b["size"],
                          b.get("rpy_deg", [0, 0, 0]), b["material"], b.get("cap"))
        self._add_arena(self.arena)
        self._add_pilasters(map_data.get("pilasters"), self.arena)

        self.positions = {n: np.zeros(3) for n in self.drone_names}
        self._collision_counter = {n: 0 for n in self.drone_names}

    # ------------------------------------------------------- map expansion
    def _add_box(self, name, center, size, rpy, material, cap=None):
        self.boxes.append(Box(name, center, size, rpy, material))
        if cap:
            c = np.asarray(center, float)
            s = np.asarray(size, float)
            rot = rotation_matrix(*rpy)
            # sit the coping on the top face (top = -z in NED), slightly proud
            up = rot @ np.array([0.0, 0.0, -1.0])
            cc = c + up * (s[2] / 2.0 + cap / 2.0)
            self.boxes.append(Box(name + "_cap", cc,
                                  [s[0] + 0.18, s[1] + 0.18, cap], rpy, "cap"))

    def _add_arena(self, a):
        x0, x1, y0, y1 = a["x_min"], a["x_max"], a["y_min"], a["y_max"]
        h, t, cap = a["wall_height"], a["wall_thickness"], a.get("cap", 0.0)
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        lx, ly = (x1 - x0) + 2 * t, (y1 - y0) + 2 * t
        z = -h / 2.0
        for nm, c, s in (
            ("wall_back",  [x0 - t / 2, cy, z], [t, ly, h]),
            ("wall_far",   [x1 + t / 2, cy, z], [t, ly, h]),
            ("wall_left",  [cx, y0 - t / 2, z], [lx, t, h]),
            ("wall_right", [cx, y1 + t / 2, z], [lx, t, h]),
        ):
            self._add_box(nm, c, s, [0, 0, 0], "stone", cap or None)

    def _add_pilasters(self, pil, a):
        if not pil:
            return
        d, w, h, sp = pil["depth"], pil["width"], pil["height"], pil["spacing"]
        mat = pil.get("material", "stone")
        x0, x1, y0, y1 = a["x_min"], a["x_max"], a["y_min"], a["y_max"]
        z = -h / 2.0
        n = 0
        for y in np.arange(y0 + sp, y1 - sp / 2, sp):
            # the back (spawn) wall gets none: never in view, and they would
            # crowd the x=0 spawn line
            if not pil.get("skip_back_wall", False):
                self._add_box(f"pilaster_b{n}", [x0 + d / 2, y, z], [d, w, h],
                              [0, 0, 0], mat); n += 1
            self._add_box(f"pilaster_f{n}", [x1 - d / 2, y, z], [d, w, h],
                          [0, 0, 0], mat); n += 1
        for x in np.arange(x0 + sp, x1 - sp / 2, sp):
            self._add_box(f"pilaster_l{n}", [x, y0 + d / 2, z], [w, d, h],
                          [0, 0, 0], mat); n += 1
            self._add_box(f"pilaster_r{n}", [x, y1 - d / 2, z], [w, d, h],
                          [0, 0, 0], mat); n += 1

    # ------------------------------------------------------------- AirSim API
    def reset(self):
        for n in self.drone_names:
            self.positions[n] = np.zeros(3)

    def set_pose(self, name, pos):
        """Teleport, ignoring collisions (simSetVehiclePose ignore_collision)."""
        self.positions[name] = np.asarray(pos, dtype=np.float64).copy()

    def get_pose(self, name):
        return self.positions[name].copy()

    def get_object_pose(self, name):
        return np.asarray(self.objects[name]["pos"], dtype=np.float64).copy()

    def collision_stamp(self, name):
        return self._collision_counter[name]

    def move(self, name, velocity, duration=1.0):
        """Hold a body-frame velocity for `duration` seconds (yaw is 0, so the
        body frame equals the world frame). Stops at the first collision."""
        v = np.asarray(velocity, dtype=np.float64)
        pos = self.positions[name]
        dt = duration / self.substeps
        for k in range(self.substeps):
            step = v * self._ramp_integral(k * dt, (k + 1) * dt)
            candidate = pos + step
            if self._collides(name, candidate):
                self._collision_counter[name] += 1
                break
            pos = candidate
        self.positions[name] = pos

    # ----------------------------------------------------------------- guts
    def _ramp_integral(self, t0, t1):
        """Integral of the velocity fraction (1 - e^{-t/tau}) over [t0, t1]."""
        if self.tau <= 0.0:
            return t1 - t0
        tau = self.tau
        return (t1 - t0) + tau * (math.exp(-t1 / tau) - math.exp(-t0 / tau))

    def _collides(self, name, p):
        r = self.drone_radius
        if p[2] >= -r:  # ground plane (z down)
            return True
        if self.ceiling_z is not None and p[2] <= self.ceiling_z + r:
            return True  # invisible capture ceiling
        for box in self.boxes:
            if abs(p[0] - box.center[0]) > box.bound_r + r:
                continue
            if box.closest_point_dist(p) < r:
                return True
        obj = self.objects.get("target")
        if obj and obj.get("collidable", False):
            if np.linalg.norm(p - np.asarray(obj["pos"])) < r + obj["radius"]:
                return True
        for other, q in self.positions.items():
            if other != name and np.linalg.norm(p - q) < 2 * r:
                return True
        return False
