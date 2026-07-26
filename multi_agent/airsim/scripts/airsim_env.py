import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import gymnasium
import supersuit as ss
from itertools import combinations
from . import airsim
from pettingzoo import ParallelEnv
from typing import Optional, overload
from gymnasium.utils import EzPickle, seeding

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def env(**kwargs):
    env = AirSimDroneEnv(**kwargs)
    env = ss.black_death_v3(env)
    #env = ss.frame_stack_v2(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    return env


petting_zoo = env


class AirSimDroneEnv(ParallelEnv, EzPickle):
    metadata = {'name': 'drones', 'render_modes': ['human']}

    def __init__(self,  
                 ip_address, 
                 image_shape, 
                 input_mode, 
                 num_drones
                 ):
        EzPickle.__init__(self,  
                 ip_address = ip_address, 
                 image_shape = image_shape, 
                 input_mode = input_mode, 
                 num_drones = num_drones
                 )
        
        # Settings
        self.image_shape = image_shape
        self.input_mode = input_mode
        self.num = num_drones
        # SuperSuit's MarkovVectorEnv reads this; PettingZoo expects it on every env.
        self.render_mode = None

        # Init
        self.drone = airsim.MultirotorClient(ip=ip_address)

        # PettingZoo variables
        self.possible_agents = ["drone"+str(i) for i in range(0,num_drones)]
        self.agents = self.possible_agents[:]
        self.truncations = None
        self.info = None
        self.collision_time = None
        self.reward = None
        self.done = None
        self.obj = None
        self.max_steps = 100
        self.current_step = None

        # Observation space
        self.observation_spaces = gymnasium.spaces.Dict(
                {
                    id:gymnasium.spaces.Dict(
                        {
                            "cam":gymnasium.spaces.Box(
                                low=0, 
                                high=255, 
                                shape=self.image_shape, 
                                dtype=np.uint8
                            ),
                            "pos":gymnasium.spaces.Box(
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
                    id:gymnasium.spaces.Box(
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
        self.reward = {agent:(self.reward[agent] if self.done[agent]!=1 else 0) for agent in self.possible_agents}

        for i in self.agents:
            self.do_action(action[i], i)
        obs, info = self.get_obs(self.done)
        self.reward, self.done = self.compute_reward(self.reward, self.done, action)

        # PettingZoo
        self.agents = [k for k in self.done.keys() if self.done[k]!=1]

        print("##################################")
        print("########### Step debug ###########")
        print("##################################")
        #print("Chosen actions for each drone:", action)
        #print("Obs len:", len(obs))
        print("Returned rewards", self.reward)
        print("Active agents (not dead/not success):", self.agents)
        print("Done?", self.done)
        print("Infos?", info)
        #print("Truncated?", self.truncations)

        return obs, self.reward, self.done, self.truncations, info

    def observe(self, agent):
        return self.get_ob(agent_id=agent)

    def reset(self,
              seed: Optional[int] = None,
              return_info: bool = False,
              options: Optional[dict] = None,):

        self.setup_flight()
        obs, infos = self.get_obs(self.done)

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
            #y -= np.array([i for i in range(len(y))]) * 2 # AirSim BUG: spawn offset must be considered!
            z = np.random.uniform(-8, -1, self.num)
            y_combos = combinations(y, 2)
            y_diff = [a-b for a,b in y_combos]

            # Check range
            if all(ele < -5 or ele > 5 for ele in y_diff):
                res = False

        return y,z

    # Multi agent start setup
    def setup_flight(self):
        self.drone.reset()

        # Resetting data
        self.info = {i: 0 for i in self.possible_agents}
        self.reward = {i: 0 for i in self.possible_agents}
        self.done = {i: 0 for i in self.possible_agents} 
        self.truncations = {i: 0 for i in self.possible_agents}     
        self.obj = {i: 0 for i in self.possible_agents}   
        self.current_step = 0

        # PettingZoo parameters
        self.agents = self.possible_agents[:]

        # For each drone
        for i in self.possible_agents:
            self.drone.enableApiControl(True, vehicle_name=i)
            self.drone.armDisarm(True, vehicle_name=i)

            # Prevent drone from falling after reset
            self.drone.moveToZAsync(-1, 1, vehicle_name=i)

        # Set x start and target
        self.agent_start_pos = 0
        x_t, y_t, _ = self.drone.simGetObjectPose('target').position
        self.target_pos = np.array([x_t, y_t])

        # Set y,z start at a min distance
        y_pos, z_pos = self.generate_pos()

        print("Starting y positions:", y_pos)

        for i in range(0,self.num):
            pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos,y_pos[i],z_pos[i]))
            self.drone.simSetVehiclePose(pose=pose, ignore_collision=True, vehicle_name=self.possible_agents[i])

        # Get target distance with mean distance for reward calculation
        self.target_dist_prev = np.linalg.norm(
            np.array([np.mean(y_pos), np.mean(z_pos)]) - self.target_pos)

        if self.input_mode == "multi_rgb":
            self.obs_stack = np.zeros(self.image_shape)

        # Get collision time stamp    
        self.collision_time = {i: self.drone.simGetCollisionInfo(vehicle_name=i).time_stamp for i in self.possible_agents}


    def do_action(self, action, name):
        # Execute action
        self.drone.moveByVelocityBodyFrameAsync(
            float(action[0]), float(action[1]), float(action[2]), duration=1, vehicle_name=name).join()

        # Prevent swaying
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1, vehicle_name=name)

    # Multi agent observations as list of single obs
    def get_obs(self,done):
        obs = {}
        for i in self.agents:
            self.info[i] = {"collision": self.is_collision(i)}
            x,y,z = self.drone.simGetVehiclePose(i).position
            #y += int(i[-1])*2 # AirSim BUG: spawn offset must be considered! 

            # Still to implement multi agent
            if self.input_mode == "multi_rgb":
                obs_t = self.get_rgb_image()	
                obs_t_gray = cv2.cvtColor(obs_t, cv2.COLOR_BGR2GRAY)
                self.obs_stack[:,:,0] = self.obs_stack[:,:,1]
                self.obs_stack[:,:,1] = self.obs_stack[:,:,2]
                self.obs_stack[:,:,2] = obs_t_gray
                obs = np.hstack((
                    self.obs_stack[:,:,0],
                    self.obs_stack[:,:,1],
                    self.obs_stack[:,:,2]))
                obs = np.expand_dims(obs, axis=2)

            elif self.input_mode == "single_rgb":
                if done[i]!=1:
                    obs[i]={}
                    obs[i]['cam'] = self.get_rgb_image(i)
                    obs[i]['pos'] = np.array([x,y,z], dtype=np.float32)
                else:
                    obs[i]={}
                    obs[i]['cam'] = np.zeros((self.image_shape), dtype=np.uint8)
                    obs[i]['pos'] = np.zeros((3,), dtype=np.float32)

            # Still to implement multi agent
            elif self.input_mode == "depth":
                obs = self.get_depth_image(thresh=3.4).reshape(self.image_shape)
                obs = ((obs/3.4)*255).astype(int)

        for i in obs:
            print("Position of",i,":",obs[i]["pos"])        
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
            if done[i]!=1:
                reward[i] = 0

                # Get agent position
                x,y,z = self.drone.simGetVehiclePose(i).position
                #y += int(i[-1])*2 # AirSim BUG: spawn offset must be considered! 
                coord[i] = (x,y,z)

                # Vicinity reward
                target_dist_curr = np.linalg.norm(np.array([x, y]) - self.target_pos)
                if act[i][0] > 0:
                    reward[i] += (self.target_dist_prev/target_dist_curr)*act[i][0]
                else:
                    reward[i] -= 3 #3
                    
                # Collision penalty
                if self.is_collision(i):
                    reward[i] = -100
                    done[i] = 1
                    self.truncations[i] = 1
                    self.obj[i] = -1

                # Check if agent almost arrived
                if target_dist_curr < 12: #17
                    reward[i] = 100
                    done[i] = 1
                    self.truncations[i] = 1
                    self.obj[i] = 1

        # Negative reward if drones get too near; might be slow
        if len(self.agents)>1:
            for i in list(combinations(self.agents, 2)):
                if self.msd(coord[i[0]],coord[i[1]])<2:
                    reward[i[0]]-=10
                    reward[i[1]]-=10

        # Give another reward if all drones reach objective
        if all([k==1 for k in self.obj.values()]):
            reward = {k:v+100 for k,v in reward.items()}
            print("################### !!! ALL DRONES ARRIVED !!! ###################")
        elif all([k==-1 for k in self.obj.values()]):
            reward = {k:v-100 for k,v in reward.items()}
            print("################### ALL DRONES CRASHED :( ###################")
        # elif all([k==1 for k in self.done.values()]) and any([k==-1 for k in self.obj.values()]) and any([k==1 for k in self.obj.values()]): #try to give a negative reward for each collided proportion to their number and try to give a positive for each arrived
        #     neg = -100*(len([k for k in self.obj.values() if k==-1])/self.num)
        #     pos = 100*(len([k for k in self.obj.values() if k==1])/self.num)
        #     tot = neg + pos
        #     reward = {k:v+tot for k,v in reward.items()}
        #     print("################### SOME ARRIVED, SOME CRUSHED ###################")

        # Debug
        # print("############# Drone n.", i,"#############")
        # print("Agents start pos", x, y, z, " and ", self.agent_start_pos)
        # print("Target pos", self.target_pos)
        # print("Distance origin to target", self.target_dist_prev)
        # print("Traveled x", agent_traveled_x)
        # print("Distance x,y to target", target_dist_curr)
        # print("Rewards", reward)
        # print("#########################################")

        return reward, done

    def msd(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

    # Multi agent collision
    def is_collision(self, name):
        current_collision_time = self.drone.simGetCollisionInfo(vehicle_name=name).time_stamp
        return True if current_collision_time != self.collision_time[name] else False
    
    # Multi agent rgb view
    def get_rgb_image(self, name):
        rgb_image_request = airsim.ImageRequest(
            0, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request], vehicle_name=name)
        # .copy() because np.frombuffer returns a read-only view, whereas the
        # np.fromstring it replaces returned a writable array. Without it this
        # function would return a read-only array on the normal path and a
        # writable one from the except branch below.
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).copy()
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3)) 

        # Sometimes no image returns from api
        try:
            return img2d.reshape(self.image_shape)
        except:
            return np.zeros((self.image_shape))

    # Still to implement multi agent
    def get_depth_image(self, thresh = 2.0):
        depth_image_request = airsim.ImageRequest(
            1, airsim.ImageType.DepthPerspective, True, False)
        responses = self.drone.simGetImages([depth_image_request])
        depth_image = np.array(responses[0].image_data_float, dtype=np.float32)
        depth_image = depth_image.reshape(responses[0].height, responses[0].width)
        depth_image[depth_image>thresh]=thresh

        if len(depth_image) == 0:
            depth_image = np.zeros(self.image_shape)

        return depth_image
