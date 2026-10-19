"""Vectorized CPU ray-cast camera reproducing the AirSim capture settings.

Matches the project's settings.json: 84x84, FOV_Degrees=120 (AirSim FOV is
horizontal; the image is square so the vertical FOV is identical), camera
mounted at body offset (0, 0, -0.6), pitch/roll/yaw = 0, looking along body +x.
Axes follow AirSim's camera convention: image right = body +y, image down =
body +z (NED).

Output matches ``simGetImages(ImageType.Scene)``: HxWx3 uint8 in BGR channel
order (the repo's own multi_rgb branch converts with COLOR_BGR2GRAY, i.e. the
AirSim byte stream is BGR). ``render_depth`` reproduces
``ImageType.DepthPerspective``: per-pixel distance along the ray in meters.

Image realism is tuned against the FPV inset of imgs/trajectory_map1_3drones.png:

* ``samples``x``samples`` supersampling per pixel (default 2 -> 4 rays/pixel),
  which is what removes the "8-bit" stair-stepping at 84x84.
* procedural detail textures evaluated at the hit point: ashlar blocks with
  mortar courses on the stone, tiles with grout and per-tile shade variation on
  the floor and the ceiling, planks, cobble, and knotted bark on the poles,
  plus a little value noise on everything.
* strong aerial-perspective haze (exponential blend toward a bright blue-white)
  which is the most visually dominant feature of the reference frame.
* a bright, low-contrast exposure curve with lifted blacks, matching the UE
  capture.
* the invisible capture ceiling is opaque to the camera and textured like the
  floor, so drones never see "sky" from inside the arena.

Speed: rays are float32 and boxes are culled per frame against the view cone
and the current nearest-hit bound, which keeps the cost roughly flat as map
detail grows.
"""

import math

import numpy as np

F32 = np.float32


class RayCamera:
    def __init__(self, world, width=84, height=84, fov_deg=120.0,
                 offset=(0.0, 0.0, -0.6), samples=2):
        self.world = world
        self.width = int(width)
        self.height = int(height)
        self.samples = max(1, int(samples))
        self.offset = np.asarray(offset, dtype=np.float64)

        s = self.samples
        half_w = math.tan(math.radians(fov_deg) / 2.0)
        half_h = half_w * (self.height / self.width)
        # sub-pixel sample centres
        sub = (np.arange(s) + 0.5) / s
        ux = ((np.arange(self.width)[:, None] + sub[None, :]).ravel()
              - self.width / 2.0) / (self.width / 2.0)
        vy = ((np.arange(self.height)[:, None] + sub[None, :]).ravel()
              - self.height / 2.0) / (self.height / 2.0)
        yy = np.tile(ux * half_w, self.height * s)
        zz = np.repeat(vy * half_h, self.width * s)
        xx = np.ones_like(yy)
        dirs = np.stack([xx, yy, zz], axis=1)
        self.dirs = (dirs / np.linalg.norm(dirs, axis=1, keepdims=True)).astype(F32)
        self.n_rays = self.dirs.shape[0]
        self.rows, self.cols = self.height * s, self.width * s
        self.dirs_2d = self.dirs.reshape(self.rows, self.cols, 3)
        self._half_w, self._half_h = half_w, half_h

        m = world.map_data
        self.sun = np.asarray(m.get("sun_dir", [0.32, -0.28, 0.90]), dtype=F32)
        self.sun /= np.linalg.norm(self.sun)
        self.ambient = F32(m.get("ambient", 0.62))
        sky = m.get("sky", {"horizon": [226, 236, 244], "zenith": [154, 196, 232]})
        self.sky_horizon = np.asarray(sky["horizon"], dtype=F32)
        self.sky_zenith = np.asarray(sky["zenith"], dtype=F32)
        hz = m.get("haze", {})
        self.haze_color = np.asarray(hz.get("color", [214, 230, 242]), dtype=F32)
        self.haze_k = F32(math.log(2.0) / max(1e-6, hz.get("half_distance", 26.0)))
        ex = m.get("exposure", {})
        self.gain = F32(ex.get("gain", 1.18))
        self.lift = F32(ex.get("lift", 0.10))
        self.tile = F32(m.get("tile_size", 3.45))
        self.ceiling_z = world.ceiling_z

        # per-box static data, packed for fast culling
        self.boxes = world.boxes
        self.b_center = np.array([b.center for b in self.boxes], dtype=F32) \
            if self.boxes else np.zeros((0, 3), F32)
        self.b_rad = np.array([b.bound_r for b in self.boxes], dtype=F32) \
            if self.boxes else np.zeros((0,), F32)
        # world-space corners, used for screen-space bounding boxes
        sgn = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1)
                        for sz in (-1, 1)], F32)
        self.b_corners = np.array(
            [(b.center + (b.rot @ (sgn * b.half).T).T) for b in self.boxes],
            dtype=F32) if self.boxes else np.zeros((0, 8, 3), F32)

    # ----------------------------------------------------------------- public
    def render(self, name):
        """RGB scene image for the named drone, (H, W, 3) uint8, BGR order."""
        origin = (self.world.get_pose(name) + self.offset).astype(F32)
        _, rgb = self._trace(origin, shade=True, exclude=name)
        img = self._downsample(rgb.reshape(-1, 3))
        return np.clip(img, 0, 255).astype(np.uint8)[:, :, ::-1]  # RGB -> BGR

    def render_depth(self, name, max_depth=100.0):
        """Perspective depth (distance along ray, meters), (H, W) float32."""
        origin = (self.world.get_pose(name) + self.offset).astype(F32)
        t, _ = self._trace(origin, shade=False, exclude=name)
        d = np.minimum(t, max_depth).reshape(-1, 1)
        return self._downsample(d)[:, :, 0].astype(np.float32)

    def _downsample(self, flat):
        """Average the supersampled rays back down to (H, W, C)."""
        s, H, W, C = self.samples, self.height, self.width, flat.shape[1]
        a = flat.reshape(H, s, W, s, C)
        return a.mean(axis=(1, 3)) if s > 1 else a.reshape(H, W, C)

    # ------------------------------------------------------------------ trace
    def _trace(self, o, shade, exclude):
        R, C = self.rows, self.cols
        d2 = self.dirs_2d
        best = np.full((R, C), np.inf, F32)
        rgb = self._sky(self.dirs).reshape(R, C, 3) if shade else None

        # --- ground plane z = 0 and the invisible capture ceiling -----------
        dz = d2[:, :, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t_g = np.where(dz > 1e-9, (-o[2]) / dz, np.inf).astype(F32)
        self._blit(o, d2, t_g, best, rgb, shade,
                   self.world.materials["floor"], np.array([0, 0, -1.0], F32))
        best = np.minimum(best, t_g)
        if self.ceiling_z is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                t_c = np.where(dz < -1e-9,
                               (self.ceiling_z - o[2]) / dz, np.inf).astype(F32)
            self._blit(o, d2, t_c, best, rgb, shade,
                       self.world.materials[self.world.ceiling_material],
                       np.array([0, 0, 1.0], F32))
            best = np.minimum(best, t_c)

        # --- boxes, each restricted to its screen-space bounding box --------
        for idx, r0, r1, c0, c1 in self._box_tiles(o):
            box = self.boxes[idx]
            sub_d = d2[r0:r1, c0:c1]
            sub_best = best[r0:r1, c0:c1]
            t, axis, sign = self._intersect_box(o, sub_d.reshape(-1, 3), box)
            t = t.reshape(sub_d.shape[:2])
            mask = t < sub_best
            if not mask.any():
                continue
            if shade:
                hp = o + t[mask, None] * sub_d[mask]
                ax = axis.reshape(sub_d.shape[:2])[mask]
                sg = sign.reshape(sub_d.shape[:2])[mask]
                normal = (box.rot[:, ax] * sg).T.astype(F32)
                rgb[r0:r1, c0:c1][mask] = self._shade(
                    self.world.materials[box.material], hp, normal, t[mask])
            best[r0:r1, c0:c1] = np.where(mask, t, sub_best)

        # --- spheres: target plus the other drones --------------------------
        for center, radius, mat in self._spheres(exclude):
            tile = self._sphere_tile(o, center, radius)
            if tile is None:
                continue
            r0, r1, c0, c1 = tile
            sub_d = d2[r0:r1, c0:c1]
            sub_best = best[r0:r1, c0:c1]
            t = self._intersect_sphere(o, sub_d.reshape(-1, 3), center,
                                       radius).reshape(sub_d.shape[:2])
            mask = t < sub_best
            if not mask.any():
                continue
            if shade:
                hp = o + t[mask, None] * sub_d[mask]
                nrm = hp - center
                nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
                rgb[r0:r1, c0:c1][mask] = self._shade(mat, hp,
                                                      nrm.astype(F32), t[mask])
            best[r0:r1, c0:c1] = np.where(mask, t, sub_best)

        return best.reshape(-1), (rgb.reshape(-1, 3) if shade else None)

    def _blit(self, o, d2, t, best, rgb, shade, material, normal):
        if not shade:
            return
        mask = t < best
        if not mask.any():
            return
        hp = o + t[mask, None] * d2[mask]
        n = np.repeat(normal[None, :], int(mask.sum()), axis=0)
        rgb[mask] = self._shade(material, hp, n, t[mask])

    # --- screen-space tiling ----------------------------------------------
    def _project_tile(self, rel):
        """Screen-space bbox (r0,r1,c0,c1) of a set of camera-relative points."""
        if rel[:, 0].min() <= 0.05:
            return 0, self.rows, 0, self.cols       # straddles the image plane
        ny = (rel[:, 1] / rel[:, 0]) / self._half_w
        nz = (rel[:, 2] / rel[:, 0]) / self._half_h
        c0 = int(np.floor((ny.min() + 1.0) * self.cols * 0.5)) - 1
        c1 = int(np.ceil((ny.max() + 1.0) * self.cols * 0.5)) + 1
        r0 = int(np.floor((nz.min() + 1.0) * self.rows * 0.5)) - 1
        r1 = int(np.ceil((nz.max() + 1.0) * self.rows * 0.5)) + 1
        c0, c1 = max(0, c0), min(self.cols, c1)
        r0, r1 = max(0, r0), min(self.rows, r1)
        if r1 <= r0 or c1 <= c0:
            return None
        return r0, r1, c0, c1

    def _box_tiles(self, o):
        out = []
        for i in range(len(self.boxes)):
            tile = self._project_tile(self.b_corners[i] - o)
            if tile is not None:
                out.append((i, *tile))
        # front-to-back keeps `best` tight, so later boxes reject sooner
        out.sort(key=lambda e: float(np.linalg.norm(self.b_center[e[0]] - o)))
        return out

    def _sphere_tile(self, o, center, radius):
        rel = center - o
        ax = np.eye(3, dtype=F32) * radius
        pts = rel[None, :] + np.concatenate([ax, -ax], axis=0)
        return self._project_tile(pts)

    def _spheres(self, exclude):
        out = []
        for obj in self.world.objects.values():
            out.append((np.asarray(obj["pos"], F32), F32(obj["radius"]),
                        self.world.materials[obj["material"]]))
        for other in self.world.drone_names:
            if other != exclude:
                out.append((self.world.get_pose(other).astype(F32),
                            F32(self.world.drone_radius),
                            self.world.materials["drone"]))
        return out

    @staticmethod
    def _intersect_box(o, d, box):
        """Slab test in the box's local frame. Returns (t, face_axis, face_sign)."""
        if box.axis_aligned:
            lo = (o - box.center).astype(F32)
            ld = d
        else:
            rot = box.rot.astype(F32)
            lo = (o - box.center).astype(F32) @ rot   # rot.T @ v, vectorized
            ld = d @ rot
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = F32(1.0) / ld
        half = box.half.astype(F32)
        t1 = (-half - lo) * inv
        t2 = (half - lo) * inv
        tn = np.minimum(t1, t2)
        tf = np.maximum(t1, t2)
        tnear = tn.max(axis=1)
        tfar = tf.min(axis=1)
        hit = (tnear <= tfar) & (tfar > 1e-5) & (tnear > 1e-5)
        t = np.where(hit, tnear, np.inf).astype(F32)
        axis = tn.argmax(axis=1)
        sign = -np.sign(ld[np.arange(len(axis)), axis])
        sign[sign == 0] = 1.0
        return t, axis, sign

    @staticmethod
    def _intersect_sphere(o, d, center, radius):
        oc = (o - center).astype(F32)
        b = d @ oc
        c = oc @ oc - radius * radius
        disc = b * b - c
        with np.errstate(invalid="ignore"):
            t = -b - np.sqrt(disc)
        return np.where((disc > 0) & (t > 1e-5), t, np.inf).astype(F32)

    # ---------------------------------------------------------------- shading
    def _shade(self, material, hp, normal, dist):
        base = np.asarray(material["color"], dtype=F32)
        color = np.repeat(base[None, :], hp.shape[0], axis=0)
        color = color * self._texture(material, hp, normal)[:, None]

        if material.get("emissive", False):
            return self._fog(np.clip(color, 0, 255), dist)

        lambert = np.clip(-(normal @ self.sun), 0.0, 1.0)
        shade = self.ambient + (1.0 - self.ambient) * lambert
        # a touch of sky bounce on upward-facing surfaces keeps floors airy
        shade = shade + 0.06 * np.clip(-normal[:, 2], 0.0, 1.0)
        out = color * shade[:, None]
        out = (out * self.gain) + self.lift * 255.0 * (1.0 - out / 255.0)
        return self._fog(np.clip(out, 0, 255), dist)

    def _fog(self, color, dist):
        f = 1.0 - np.exp(-self.haze_k * dist)
        return color * (1.0 - f)[:, None] + self.haze_color[None, :] * f[:, None]

    # --- procedural detail -------------------------------------------------
    @staticmethod
    def _hash2(a, b):
        """Cheap deterministic value noise in [0,1) from two integer grids."""
        h = np.sin(a * 127.1 + b * 311.7) * 43758.5453
        return h - np.floor(h)

    def _surface_uv(self, hp, normal):
        """Pick the two surface-tangent world axes for the dominant normal."""
        nx, ny, nz = np.abs(normal[:, 0]), np.abs(normal[:, 1]), np.abs(normal[:, 2])
        horiz = (nz >= nx) & (nz >= ny)
        u = np.where(horiz, hp[:, 0], np.where(nx >= ny, hp[:, 1], hp[:, 0]))
        v = np.where(horiz, hp[:, 1], -hp[:, 2])
        return u, v

    def _texture(self, material, hp, normal):
        pat = material.get("pattern", "smooth")
        u, v = self._surface_uv(hp, normal)
        f = np.ones(hp.shape[0], F32)

        if pat == "tile":
            s = self.tile
            gu, gv = u / s, v / s
            iu, iv = np.floor(gu), np.floor(gv)
            grout = (np.minimum(gu - iu, 1 - (gu - iu)) < 0.022) | \
                    (np.minimum(gv - iv, 1 - (gv - iv)) < 0.022)
            checker = ((iu + iv) % 2) * 0.085           # alternating tiles
            shade = self._hash2(iu, iv) * 0.10 - 0.05    # per-tile variation
            f = 0.985 + checker + shade
            f = np.where(grout, 0.70, f)
            # fine streaks, like the weathered UE floor
            f *= 1.0 + 0.045 * np.sin(u * 2.3 + v * 0.7) * np.sin(v * 3.1)
        elif pat == "block":
            ch, cw = 0.95, 1.9                           # ashlar course size
            row = np.floor(v / ch)
            uu = u + (row % 2) * (cw * 0.5)
            col = np.floor(uu / cw)
            fv = v / ch - row
            fu = uu / cw - col
            mortar = (np.minimum(fv, 1 - fv) < 0.05) | (np.minimum(fu, 1 - fu) < 0.028)
            f = 0.94 + self._hash2(col, row) * 0.16      # per-block shade
            f = np.where(mortar, 0.72, f)
            f *= 1.0 + 0.05 * np.sin(u * 5.0 + row * 2.0)
        elif pat == "cobble":
            gu, gv = u / 0.55, v / 0.55
            iu, iv = np.floor(gu), np.floor(gv)
            du = gu - iu - 0.5 + (self._hash2(iu, iv) - 0.5) * 0.45
            dv = gv - iv - 0.5 + (self._hash2(iu + 7, iv - 3) - 0.5) * 0.45
            r = np.sqrt(du * du + dv * dv)
            f = np.where(r > 0.42, 0.72, 0.92 + self._hash2(iu, iv) * 0.18)
        elif pat == "plank":
            row = np.floor(v / 0.42)
            fv = v / 0.42 - row
            seam = np.minimum(fv, 1 - fv) < 0.05
            f = 0.93 + self._hash2(row, np.zeros_like(row)) * 0.14
            f = np.where(seam, 0.78, f)
            f *= 1.0 + 0.05 * np.sin(u * 9.0 + row)
        elif pat == "bark":
            f = 0.82 + 0.18 * (0.5 + 0.5 * np.sin(v * 7.0))
            knot = np.floor(v / 0.8)
            f *= np.where(np.abs(v / 0.8 - knot - 0.5) < 0.09, 0.70, 1.0)
        else:  # smooth
            f = np.full(hp.shape[0], 0.99, F32)

        # a little broadband grain on everything, as in the UE textures
        f = f * (0.97 + 0.06 * self._hash2(np.floor(u * 9.0), np.floor(v * 9.0)))
        return f.astype(F32)

    def _sky(self, d):
        """Only reachable outside the arena (the ceiling closes it from inside)."""
        elev = np.clip(-d[:, 2], 0.0, 1.0) ** 0.7
        return (self.sky_horizon[None, :] * (1 - elev[:, None])
                + self.sky_zenith[None, :] * elev[:, None])
