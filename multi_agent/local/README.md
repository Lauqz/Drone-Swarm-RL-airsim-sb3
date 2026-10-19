# Local simulator (no UE4 / AirSim required)

A drop-in training backend that replaces Unreal Engine 4 + AirSim with a pure
Python simulator, so the multi-agent training algorithm can be developed and
tested on machines that cannot run UE4 (e.g. a laptop under WSL). The AirSim
variant in `../airsim` is unchanged; this folder is a parallel alternative.

## What is reproduced

`scripts/local_env.py` is a line-by-line port of
`../airsim/scripts/airsim_env.py` with the AirSim RPC client replaced by
`scripts/sim/` (kinematic flight + collisions + a vectorized CPU ray-cast
camera). Identical by construction, and covered by `tests/`:

- PettingZoo `ParallelEnv` interface and the SuperSuit chain
  (`black_death_v3` → `pettingzoo_env_to_vec_env_v1` → `concat_vec_envs_v1`).
- Observation spaces `{"cam": Box(0,255,(84,84,3),uint8), "pos": Box(±200,(3,),f32)}`
  and action space `Box([0,-3,-3],[3,3,3])` (body-frame velocity, 1 s hold).
- The full reward/termination logic — vicinity reward, −3 backward penalty,
  −100 collision, +100 success inside 12 m, −10 pairwise proximity (<2 m),
  ±100 group bonus, −200 timeout at step 100 — **including the original
  quirks**, kept on purpose so reward curves are comparable with AirSim runs
  (see the docstring in `scripts/local_env.py`).
- The camera: 84×84, 120° horizontal FOV pinhole at body offset (0,0,−0.6),
  BGR uint8 like AirSim's `ImageType.Scene`; `depth` mode reproduces
  `DepthPerspective` with the original 3.4 m threshold. Image quality is tuned
  against the FPV inset of the reference screenshot: 2×2 supersampling per
  pixel, procedural ashlar/tile/plank/cobble/bark textures evaluated at the hit
  point, strong aerial-perspective haze, and a bright low-contrast exposure
  curve. Resolution stays 84×84 — `samples` only changes how many rays are
  averaged into each pixel (`samples: 1` renders ~3× faster and blockier).
  See `docs/camera_views.png`.
- The map: `scripts/sim/maps/map1.json`, reconstructed photogrammetrically from
  `imgs/trajectory_map1_3drones.png` — see [Map reconstruction](#map-reconstruction).
  Edit the JSON to refine geometry; the sim, the renderer and the browser
  viewer all read the same file.

## Usage

Same workflow as the AirSim variant, minus UE4:

```bash
cd multi_agent/local
python train.py       # headless training (PPO + CustomCombinedExtractor)
python evaluate.py    # evaluates saved_policy/best_model, viewer on by default
python -m pytest tests/           # parity + API + physics + camera tests
python tools/render_map_figures.py  # regenerate the docs/ fidelity figures
```

Settings live in `config.yml`. Extra keys compared to the AirSim variant:

- `num_vec_envs`: independent arenas stepped in parallel (AirSim was pinned
  to one simulator).
- `viewer` / `viewer_port`: live browser view.
- `samples`: camera supersampling (rays per pixel = samples²).
- `verbose`: restores the original per-step debug printing.

## Browser visualization

Set `viewer: true` (or construct the env with `render_mode="browser"`), then
open `http://localhost:8777` — from Windows too, WSL2 forwards localhost. The
page (Three.js, vendored, works offline) shows the arena built from the same
map JSON the physics uses, drone markers with per-episode trajectory trails,
the target with its 12 m success ring, each drone's actual 84×84 camera
observation, and per-drone reward/status. Drag to orbit, wheel to zoom,
`T` toggles trails. The viewer runs in a daemon thread, is throttled to
~15 Hz per client, and never blocks training; headless runs never start it.

## Map reconstruction

The arena is not hand-placed by eye: the plan layout was recovered from the
reference screenshot.

1. The floor-plane homography was fitted from the left/right/top wall-base
   lines plus the near-wall cap line, giving the two ground vanishing points
   (`V_x`, `V_y`) and the image of the top-left floor corner.
2. Metric scale along x comes from the known target position (26.8, 3.9) in
   `env_config.yml`; the y-scale was fixed by requiring the floor tiles to be
   square, which puts them at **3.45 m** and the room at **28.7 × 40 m**.
3. Every obstacle footprint was then read by inverse-projecting its
   floor-contact pixels. This matters: tall objects sit far from where they
   appear — the big pillar's top face is near the far wall in the image but its
   base is at y ≈ −4.5.

`docs/map_vs_reference.png` is the check: the reconstructed footprints are
reprojected back onto the screenshot and land on the real obstacles, and the
target circle lands on the sphere. Regenerate all figures with:

```bash
python tools/render_map_figures.py      # writes docs/*.png
```

Known approximations, both deliberate:

- **Room depth.** The visible floor runs a few metres past y = +20, so the real
  near wall is probably set back further. The arena is kept at y ∈ [−20, 20]
  because the env spawns drones at y ∈ [−18, 18] and no drone ever reaches the
  difference.
- **Heights** are visual estimates. Single-view height recovery was not stable
  on this screenshot (the vertical vanishing point is ill-conditioned), so
  heights were chosen to match the image and to straddle the −8…−1 m spawn
  band. Plan positions are the measured part.

### Ceiling

The map is closed by the same invisible capture ceiling as the UE map, at
**12 m** (`ceiling.height`): drones cannot fly through it and their cameras
cannot see past it — from inside there is no sky, just a floor-textured
ceiling, which is exactly what the reference FPV frame shows. The browser
viewer never draws it, so the arena stays readable from above. 12 m is a guess
constrained from below by the −8 m spawn altitude.

## Fidelity notes

- The arena is closed by an invisible ceiling (see above); collisions also
  cover the floor, the perimeter walls, every obstacle and drone-drone contact.
- AirSim's SimpleFlight velocity ramp is modeled as a first-order response
  (`tau=0.25 s`) inside `World.move`; collisions use a 0.35 m sphere swept in
  10 substeps (no tunneling through the thin poles at 3 m/s). Done drones
  stay parked in the arena and remain physical obstacles, as in AirSim.
- Yaw is never commanded by the env (as in the original), so the body frame
  equals the world frame and the camera always faces +x.
- A policy trained here is not expected to transfer zero-shot to the UE4 map
  (different textures/exact geometry); the purpose is validating the training
  algorithm with comparable reward dynamics.
