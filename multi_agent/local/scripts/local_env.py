"""Local (no-AirSim) variant of multi_agent/airsim/scripts/airsim_env.py.

This is a line-by-line port of AirSimDroneEnv onto a pure-Python simulator
(scripts/sim). The PettingZoo interface, observation/action spaces, reward
function, termination logic and all their quirks are kept identical so that
reward curves are comparable with the AirSim runs. Deliberately preserved
quirks of the original:

- ``target_dist_prev`` mixes frames: it is the norm of
  [mean(y_spawn), mean(z_spawn)] - [x_target, y_target] (airsim_env.py:218).
  It only rescales the vicinity reward by a per-episode constant, and the
  AirSim runs were trained with it.
- ``truncations`` are raised together with terminations, and all flags are
  0/1 ints rather than bools.
- On timeout (step 100) every agent gets -200 and the episode ends.
- Rewards/done/info dicts always cover all possible agents, while obs only
  covers the agents alive at the start of the step (black_death fills in).

Differences from the AirSim variant (besides the backend):

- ``ip_address`` is accepted and ignored, so call sites are drop-in.
- Debug printing is gated behind ``verbose`` (the local sim runs thousands of
  steps per minute).
- ``render_mode="browser"`` streams a live view to http://localhost:<port>.
- The ``depth`` input mode is implemented per-agent (in the original it was a
  broken single-agent leftover); ``single_rgb`` is the parity-tested default.
- ``num_vec_envs`` in the petting_zoo() wrapper allows multiple independent
  arenas (with AirSim this was pinned to 1 simulator).
"""

import math
import os
from itertools import combinations
from typing import Optional, overload

import gymnasium
import numpy as np
import supersuit as ss
from gymnasium.utils import EzPickle, seeding
from pettingzoo import ParallelEnv

from .sim.camera import RayCamera
from .sim.world import DEFAULT_MAP, World

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def env(num_vec_envs=1, **kwargs):
    env = LocalDroneEnv(**kwargs)
    env = ss.black_death_v3(env)
    #env = ss.frame_stack_v2(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs, base_class="stable_baselines3")
    return env


petting_zoo = env


class LocalDroneEnv(ParallelEnv, EzPickle):
    metadata = {'name': 'drones', 'render_modes': ['human', 'browser']}

    def __init__(self,
                 image_shape,
                 input_mode,
                 num_drones,
                 ip_address=None,          # accepted for drop-in parity, unused
                 map_path=None,
                 render_mode=None,
                 viewer_port=8777,
                 verbose=False,
                 samples=2,
                 ):
        EzPickle.__init__(self,
                 image_shape=image_shape,
                 input_mode=input_mode,
                 num_drones=num_drones,
                 ip_address=ip_address,
                 map_path=map_path,
                 render_mode=render_mode,
                 viewer_port=viewer_port,
                 verbose=verbose,
                 samples=samples,
                 )

        # Settings
        self.image_shape = image_shape
        self.input_mode = input_mode
        self.num = num_drones
        self.verbose = verbose
        # SuperSuit's MarkovVectorEnv reads this; PettingZoo expects it on every env.
        self.render_mode = render_mode

        # PettingZoo variables
        self.possible_agents = ["drone"+str(i) for i in range(0, num_drones)]
        self.agents = self.possible_agents[:]
        self.truncations = None
        self.info = None
        self.collision_time = None
        self.reward = None
        self.done = None
        self.obj = None
        self.max_steps = 100
        self.current_step = None

        # Init local simulator (replaces airsim.MultirotorClient)
        self.map_path = map_path or DEFAULT_MAP
        self.world = World(self.possible_agents, map_path=self.map_path)
        cam_hw = (image_shape[0], image_shape[1])
        # samples=2 -> 2x2 rays per pixel. This is what removes the blocky
        # aliasing at 84x84; set samples=1 to trade detail for ~3x render speed.
        self.camera = RayCamera(self.world, width=cam_hw[1], height=cam_hw[0],
                                samples=samples)

        # Browser viewer (lazy import so headless runs never touch it)
        self.viewer = None
        if self.render_mode == "browser":
            from .viewer.server import DroneViewer
            self.viewer = DroneViewer(self.world.map_data, port=viewer_port)
        self._episode_count = 0

        # Observation space
        self.observation_spaces = gymnasium.spaces.Dict(
                {
                    id: gymnasium.spaces.Dict(
                        {
                            "cam": gymnasium.spaces.Box(
                                low=0,
                                high=255,
                                shape=self.image_shape,
                                dtype=np.uint8
                            ),
                            "pos": gymnasium.spaces.Box(
                                low=-200.0,
                                high=200.0,
                                shape=(3,),
                                dtype=np.float32
                            )
                        }
                    )
                    for id in self.possible_agents
                }
            )

        # Action space
        self.action_spaces = gymnasium.spaces.Dict(
                {
                    id: gymnasium.spaces.Box(
                        low=np.array([0.0, -3.0, -3.0]),
                        high=np.array([3.0, 3.0, 3.0]),
                        shape=(3,),
                        dtype=np.float32
                    )
                    for id in self.possible_agents
                }
            )

        # Setup flight and set seed
        self.setup_flight()
        self._seed(42)

    @overload
    def observation_space(self):
        return self.observation_space

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def step(self, action):
        self.current_step += 1
        self.reward = {agent: (self.reward[agent] if self.done[agent] != 1 else 0) for agent in self.possible_agents}

        for i in self.agents:
            self.do_action(action[i], i)
        obs, info = self.get_obs(self.done)
        self.reward, self.done = self.compute_reward(self.reward, self.done, action)

        # PettingZoo
        self.agents = [k for k in self.done.keys() if self.done[k] != 1]

        if self.verbose:
            print("##################################")
            print("########### Step debug ###########")
            print("##################################")
            print("Returned rewards", self.reward)
            print("Active agents (not dead/not success):", self.agents)
            print("Done?", self.done)
            print("Infos?", info)

        if self.viewer is not None:
            self._push_viewer_state(obs)

        return obs, self.reward, self.done, self.truncations, info

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None,):

        self.setup_flight()
        obs, infos = self.get_obs(self.done)

        self._episode_count += 1
        if self.viewer is not None:
            self._push_viewer_state(obs)

        return obs, infos

    def render(self):
        return self.get_obs(self.done)

    def generate_pos(self):
        # Start the agent at random section at a random yz position
        # Left border -18, right border 18
        # Upper border -8, lower border -1
        res = True
        while res:
            # Generate
            y = np.random.uniform(-18, 18, self.num)
            z = np.random.uniform(-8, -1, self.num)
            y_combos = combinations(y, 2)
            y_diff = [a-b for a, b in y_combos]

            # Check range
            if all(ele < -5 or ele > 5 for ele in y_diff):
                res = False

        return y, z

    # Multi agent start setup
    def setup_flight(self):
        self.world.reset()

        # Resetting data
        self.info = {i: 0 for i in self.possible_agents}
        self.reward = {i: 0 for i in self.possible_agents}
        self.done = {i: 0 for i in self.possible_agents}
        self.truncations = {i: 0 for i in self.possible_agents}
        self.obj = {i: 0 for i in self.possible_agents}
        self.current_step = 0

        # PettingZoo parameters
        self.agents = self.possible_agents[:]

        # Set x start and target
        self.agent_start_pos = 0
        x_t, y_t, _ = self.world.get_object_pose('target')
        self.target_pos = np.array([x_t, y_t])

        # Set y,z start at a min distance
        y_pos, z_pos = self.generate_pos()

        if self.verbose:
            print("Starting y positions:", y_pos)

        for i in range(0, self.num):
            self.world.set_pose(self.possible_agents[i],
                                [self.agent_start_pos, y_pos[i], z_pos[i]])

        # Get target distance with mean distance for reward calculation
        # (kept verbatim from the AirSim env, including its frame mixing)
        self.target_dist_prev = np.linalg.norm(
            np.array([np.mean(y_pos), np.mean(z_pos)]) - self.target_pos)

        if self.input_mode == "multi_rgb":
            self.obs_stack = np.zeros(self.image_shape)

        # Get collision time stamp
        self.collision_time = {i: self.world.collision_stamp(i) for i in self.possible_agents}

    def do_action(self, action, name):
        # Execute action: body-frame velocity held for 1s, then hover
        self.world.move(name,
                        [float(action[0]), float(action[1]), float(action[2])],
                        duration=1.0)

    # Multi agent observations as list of single obs
    def get_obs(self, done):
        obs = {}
        for i in self.agents:
            self.info[i] = {"collision": self.is_collision(i)}
            x, y, z = self.world.get_pose(i)

            if self.input_mode == "multi_rgb":
                # Kept for parity with the AirSim env; single-agent leftover.
                obs_t = self.get_rgb_image(i)
                import cv2
                obs_t_gray = cv2.cvtColor(obs_t, cv2.COLOR_BGR2GRAY)
                self.obs_stack[:, :, 0] = self.obs_stack[:, :, 1]
                self.obs_stack[:, :, 1] = self.obs_stack[:, :, 2]
                self.obs_stack[:, :, 2] = obs_t_gray
                obs = np.hstack((
                    self.obs_stack[:, :, 0],
                    self.obs_stack[:, :, 1],
                    self.obs_stack[:, :, 2]))
                obs = np.expand_dims(obs, axis=2)

            elif self.input_mode == "single_rgb":
                if done[i] != 1:
                    obs[i] = {}
                    obs[i]['cam'] = self.get_rgb_image(i)
                    obs[i]['pos'] = np.array([x, y, z], dtype=np.float32)
                else:
                    obs[i] = {}
                    obs[i]['cam'] = np.zeros((self.image_shape), dtype=np.uint8)
                    obs[i]['pos'] = np.zeros((3,), dtype=np.float32)

            elif self.input_mode == "depth":
                # Per-agent depth (properly multi-agent, unlike the original).
                if done[i] != 1:
                    obs[i] = {}
                    obs[i]['cam'] = self.get_depth_image(i, thresh=3.4)
                    obs[i]['pos'] = np.array([x, y, z], dtype=np.float32)
                else:
                    obs[i] = {}
                    obs[i]['cam'] = np.zeros((self.image_shape), dtype=np.uint8)
                    obs[i]['pos'] = np.zeros((3,), dtype=np.float32)

        if self.verbose:
            for i in obs:
                print("Position of", i, ":", obs[i]["pos"])
        return obs, self.info

    # Multi agent reward
    def compute_reward(self, reward, done, act):
        coord = {}

        if self.current_step >= self.max_steps:
            reward = {i: -200 for i in self.possible_agents}
            done = {i: 1 for i in self.possible_agents}
            self.truncations = {i: 1 for i in self.possible_agents}
            self.obj = {i: -1 for i in self.possible_agents}
            return reward, done

        for i in self.agents:
            if done[i] != 1:
                reward[i] = 0

                # Get agent position
                x, y, z = self.world.get_pose(i)
                coord[i] = (x, y, z)

                # Vicinity reward
                target_dist_curr = np.linalg.norm(np.array([x, y]) - self.target_pos)
                if act[i][0] > 0:
                    reward[i] += (self.target_dist_prev/target_dist_curr)*act[i][0]
                else:
                    reward[i] -= 3

                # Collision penalty
                if self.is_collision(i):
                    reward[i] = -100
                    done[i] = 1
                    self.truncations[i] = 1
                    self.obj[i] = -1

                # Check if agent almost arrived
                if target_dist_curr < 12:
                    reward[i] = 100
                    done[i] = 1
                    self.truncations[i] = 1
                    self.obj[i] = 1

        # Negative reward if drones get too near; might be slow
        if len(self.agents) > 1:
            for i in list(combinations(self.agents, 2)):
                if self.msd(coord[i[0]], coord[i[1]]) < 2:
                    reward[i[0]] -= 10
                    reward[i[1]] -= 10

        # Give another reward if all drones reach objective
        if all([k == 1 for k in self.obj.values()]):
            reward = {k: v+100 for k, v in reward.items()}
            if self.verbose:
                print("################### !!! ALL DRONES ARRIVED !!! ###################")
        elif all([k == -1 for k in self.obj.values()]):
            reward = {k: v-100 for k, v in reward.items()}
            if self.verbose:
                print("################### ALL DRONES CRASHED :( ###################")

        return reward, done

    def msd(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    # Multi agent collision
    def is_collision(self, name):
        current_collision_time = self.world.collision_stamp(name)
        return True if current_collision_time != self.collision_time[name] else False

    # Multi agent rgb view
    def get_rgb_image(self, name):
        return self.camera.render(name).reshape(self.image_shape)

    # Per-agent depth image, thresholded like the original and scaled to uint8
    def get_depth_image(self, name, thresh=2.0):
        depth_image = self.camera.render_depth(name)
        depth_image[depth_image > thresh] = thresh
        depth_image = ((depth_image/thresh)*255).astype(np.uint8)
        return depth_image.reshape(self.image_shape)

    # ------------------------------------------------------------- viewer glue
    def _push_viewer_state(self, obs):
        import base64
        import cv2
        drones = {}
        for i in self.possible_agents:
            p = self.world.get_pose(i)
            drones[i] = {
                "pos": [float(p[0]), float(p[1]), float(p[2])],
                "done": int(self.done[i]),
                "obj": int(self.obj[i]),
                "reward": float(self.reward[i]),
                "collision": bool(self.world.collision_stamp(i) != self.collision_time[i]),
            }
        cams = {}
        for i, o in obs.items():
            frame = o["cam"]
            if frame.shape[-1] == 1:
                frame = frame[:, :, 0]
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                cams[i] = base64.b64encode(buf.tobytes()).decode("ascii")
        self.viewer.push({
            "episode": self._episode_count,
            "step": self.current_step,
            "success_radius": 12,
            "drones": drones,
            "cams": cams,
        })
