# Multi-agent Deep Reinforcement Learning for Drone Swarms using UE4, AirSim, Stable-Baselines3, PettingZoo, SuperSuit

![3 drones trajectory](imgs/trajectory_map1_3drones.png)

## Overview

This repository contains the code and environments for my thesis on reinforcement learning of drone swarms using Unreal Engine 4 and Microsoft AirSim conducted at AIST, Tsukuba. It includes two separate implementations:

* **Single-Agent**: Training a single drone using PPO in a custom UE4 using StableBaseline3 and Gymnasium.
* **Multi-Agent**: Training a swarm of drones using PPO with a patched SuperSuit, to support stacked `Dict` observations (RGB camera plus position) across multiple agents. See [Library patches and upstream status](#library-patches-and-upstream-status).

All code is placed under the `airsim` folder and assumes AirSim and UE4 are installed following the [official AirSim documentation](https://microsoft.github.io/AirSim/).

## Repository Structure

```
.
├── README.md
├── imgs/                        # Images and gifs of the experiments
├── old_experiments/             # Old experiments for reference
├── single_agent/
│   ├── environment.yml          # Conda environment for single-agent setup
│   ├── settings/                # Single drone configuration for AirSim
│       └── ...
│   └── airsim/                  # Single-agent code to place inside AirSim PythonClient folder
│       └── ...
├── multi_agent/
│   ├── environment.yml          # Conda environment for multi-agent setup
│   ├── modified_libs/           # Patched packages: SuperSuit, Stable-Baselines3, msgpackrpc
│   │   └── PATCHES.md           # What changed in each file, and why
│   ├── archive/
│   │   └── unmodified_libs/     # Vendored but never modified; install from PyPI instead
│   ├── settings/                # Multi-drone configurations for AirSim
│       └── ...
│   └── airsim/                  # Multi-agent code to place inside AirSim PythonClient folder
│       └── ...
└── LICENSE
```

## Prerequisites
> ⚠️ **Warning:**  this client was developed and tested on **Windows only**. It may not install or run correctly on macOS or Linux.
> 
1. **Unreal Engine 4**: install via Epic Games Launcher. Ensure you have a UE4 project compatible with AirSim.
2. **Microsoft AirSim**: clone and build following the [AirSim installation guide](https://microsoft.github.io/AirSim/) *AirSim creates a `settings.json` file in your user Documents directory under `Documents/AirSim` on Windows or `~/Documents/AirSim` on Linux systems* ([microsoft.github.io](https://microsoft.github.io/AirSim/settings/)).
3. **Conda**: for managing Python environments.

## Single-Agent Setup

1. Navigate to the `single_agent` folder:

   ```bash
   cd single_agent
   ```
2. Create and activate the Conda environment:

   ```bash
   conda env create -f environment.yml     # Python 3.9.16 
   conda activate deeprl_single
   ```
3. Copy the `airsim` folder into your AirSim PythonClient directory:

   ```bash
   cp -r airsim /path/to/AirSim/PythonClient/
   ```
4. Launch UE4, open your project, and hit **Play** to run the environment (at this point, the drone will be stuck waiting for training/evaluation script).

## Multi-Agent Setup

1. Navigate to the `multi_agent` folder:

   ```bash
   cd multi_agent
   ```
2. Install the environment using:

   ```bash
   conda env create -f environment.yml     # Python 3.11.3 
   conda activate deeprl_multi
   ```
3. Install the patched libraries in editable mode:

   ```bash
   pip install -e modified_libs/supersuit
   pip install -e modified_libs/stable_baselines3
   pip install -e modified_libs/msgpackrpc
   ```

   Only these three are patched. Every vendored package was diffed against the
   version pinned in `environment.yml`: `pettingzoo`, `msgpack` and `gymnasium`
   turned out to be unmodified, so they moved to
   `archive/unmodified_libs/` and should be installed from PyPI as normal.

   `modified_libs/PATCHES.md` documents exactly what changed in each file and why.
4. **Place your custom `settings.json`** (with your desired number of drones) *into the folder created by AirSim at `Documents/AirSim` on Windows or `~/Documents/AirSim` on Linux*, replacing the default settings file ([microsoft.github.io](https://microsoft.github.io/AirSim/settings/)).
5. Copy the `airsim` folder into the AirSim Python Client folder:

   ```bash
   cp -r airsim /path/to/AirSim/PythonClient
   ```
6. Open UE4 and run your project; the multi-drone config in `settings.json` will enable different sized swarm spawn.

## Library patches and upstream status

The multi-agent setup needs a patched SuperSuit because two of its wrappers
reject `Dict` observation spaces, which is what this project uses (each drone
observes an RGB frame plus its own position).

Both gaps have been submitted upstream:

| wrapper | needed for | upstream PR |
|---|---|---|
| `black_death_v3` | **the default `train.py` chain**. Without it the run stops at `black_death_v3` with `observation spaces for black death must be Box spaces` | [SuperSuit#269](https://github.com/Farama-Foundation/SuperSuit/pull/269) |
| `frame_stack_v1/v2` | only if the `frame_stack_v2` line in `scripts/airsim_env.py` is uncommented; it is disabled by default | [SuperSuit#268](https://github.com/Farama-Foundation/SuperSuit/pull/268) |

Measured against stock SuperSuit 3.11:

| SuperSuit | default chain | with `frame_stack_v2` |
|---|---|---|
| unpatched | fails | fails |
| #269 only | **works** | fails |
| #268 only | fails | fails |
| both | works | works |

Once both merge, `modified_libs/supersuit` becomes unnecessary and the project can
run against a released SuperSuit.

### Verified against current versions

The environment was re-checked against much newer packages than the ones pinned
in `environment.yml`:

* passes PettingZoo's own `parallel_api_test` on **PettingZoo 1.26.1**
* the full wrapper chain from `train.py`
  (`black_death_v3` → `pettingzoo_env_to_vec_env_v1` → `concat_vec_envs_v1`)
  runs on **SuperSuit 3.11** with the two patches above, producing
  `{'cam': (3, 84, 84, 9), 'pos': (3, 9)}` with `frame_stack_v2(3)`
* runs on **NumPy 2.x** and **Gymnasium 1.3**

Two source changes were required for this and are already applied:
`np.fromstring` → `np.frombuffer` (the binary mode was removed in NumPy 2.0) and
an explicit `self.render_mode = None` (recent SuperSuit reads it).

Caveat: these checks used a mocked AirSim client, so they cover the
PettingZoo/SuperSuit API surface, not a full training run against the simulator.
`environment.yml` still pins the original, known-good versions.

## Usage

> **Important**: All coordinates, spawn positions, and environment parameters in the code must be adapted to your specific UE4 map. If you decide to create your own map, you must update any hard-coded positions and settings to match the map you create. If you want to use my maps, every drone will start from a random initial position along the same axis and will be trained to reach a point lying within a specified radius of the target asset.
>
* Launch training via:

  ```bash
  python airsim/train.py    # Single and multi-agent
  ```

* Adjust hyperparameters or environment settings directly in the Python scripts or `settings.json`.

## Monitoring Training

You can monitor training progress, rewards, and hyperparameters using **TensorBoard**.

After launching training, open a terminal and run:

```bash
tensorboard --logdir path/to/tb_logs/
```

This allows you to inspect:

* Rewards over time
* Episode lengths
* Custom metrics
* Learning rates and other hyperparameters

## Evaluate

You can evaluate the trajectory of the single and multi-agent training by running:

  ```bash
  python airsim/evaluate.py    # Single and multi-agent
  ```

It will evaluate the weights saved inside `saved_policy`.

Always be sure to be running the UE4 env before running training or evaluation, and that the settings are set correctly (and renamed to `settings.json`). If you want to modify the number of drones, you should modify the code.

![Demo](imgs/3drones.gif)

## Training Maps

> **Note:** The training maps used in this project are **~50 GB**.  
> They’re available on request; just let me know if you’d like a copy!
> They include:
> * A standard map populated with random assets.
> * A map featuring NPCs that move randomly.
> * A dense-forest map.
> * A composite map combining all the characteristics above.

## Acknowledgments

* Unreal Engine 4 (Epic Games)
* Microsoft AirSim
* PettingZoo & SuperSuit
* Prof. Akiya Kamimura & Prof. Andrea Roli

## Citation

If you find this work useful in your research, you are welcome to cite it. An official publication is currently in preparation and will be linked here once available. In the meantime, feel free to reference this repository and acknowledge the work.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
