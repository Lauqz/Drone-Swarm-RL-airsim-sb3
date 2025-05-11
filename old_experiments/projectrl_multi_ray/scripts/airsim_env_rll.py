from . import airsim
import os
import tempfile
import gym
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ray

from ray import tune
from ray.tune import register_env

from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv

#from ray.rllib.algorithms.ppo import PPO
from ray.rllib.models import ModelCatalog

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class AirSimDroneEnv(ExternalMultiAgentEnv):
    def __init__(self, env_config: EnvContext, observation_space: gym.Space, action_space: gym.Space, max_concurrent: int, ip_address, image_shape, input_mode, num_drones):
        self.image_shape = image_shape
        self.input_mode = input_mode
        self.address = ip_address
        self.num = num_drones
        self.names = ["drone"+str(i) for i in range(0,num_drones)]
        #self.drone = airsim.MultirotorClient(ip=ip_address)

        # if self.input_mode == "multi_rgb":
        #     self.observation_space = gym.spaces.Box(
        #         low=0, high=255, 
        #         shape=(image_shape[0],image_shape[1]*3,1), 
        #         dtype=np.uint8)
        # else:
        #     self.observation_space = gym.spaces.Box(
        #         low=0, high=255, shape=self.image_shape, dtype=np.uint8)

        # self.action_space = gym.spaces.Box(
        #     low=-3.0, high=3.0, shape=(2,), dtype=np.float32) #0.6

        #self.info = {"collision": False}
        self.info = {i: False for i in range(0,num_drones)}
        self.collision_time = {i:0 for i in self.names}
        self.random_start = True
        #self.setup_flight()
        super().__init__(action_space=action_space, observation_space=observation_space, max_concurrent=max_concurrent)

    def run(self):
        done = {i:0 for i in range(0, self.num)} #try
        reward = {i:0 for i in range(0, self.num)} #try

        self.drone = airsim.MultirotorClient(self.address)
        self.drone.confirmConnection()
        self.drone.simPause(True)

        self.setup_flight()
        self.drone.simContinueForTime(2.0)

        steps_counter = 0
        episode_id = self.start_episode()
        print("starting episode id:", episode_id, "with client", self.address)
        
        while True:
            steps_counter += 1
            obs, info =  self.get_obs(done)
            actions = self.get_action(episode_id, obs)  # or self.log_action(episode_id, obs, action)
            for i in range(self.num):
                if done[i]!=1:
                    self.do_action(actions[i], self.names[i]) #actions[i]
            rewards, done = self.compute_reward(reward, done) #try
            self.log_returns(episode_id, rewards, info_dict=info) #multiagent_done_dict=done,
            if all(val==1 for val in done.values()):
                steps_counter = 0
                print("ending episode id:", episode_id, "with client", self.address)
                self.end_episode(episode_id, obs)

                done = {i:0 for i in range(0, self.num)} #try
                reward = {i:0 for i in range(0, self.num)} #try
                self.setup_flight()
                self.drone.simContinueForTime(2.0)

                episode_id = self.start_episode()
                print("starting episode id:", episode_id, "with client", self.address)


    def step(self, action):
        for i in self.names:
            self.do_action(action, i)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        return obs, reward, done, info

    def reset(self):
        self.setup_flight()
        obs, _ = self.get_obs()
        return obs

    def render(self):
        return self.get_obs()
    
    # Multi agent start setup
    def setup_flight(self):
        self.drone.reset()

        # For each drone
        for i in self.names:
            self.drone.enableApiControl(True, vehicle_name=i)
            self.drone.armDisarm(True, vehicle_name=i)

            # Prevent drone from falling after reset
            self.drone.moveToZAsync(-1, 1, vehicle_name=i)

            # Get collision time stamp
            self.collision_time[i] = self.drone.simGetCollisionInfo(vehicle_name=i).time_stamp

        self.agent_start_pos = 0 #section["offset"][0] #check what its
        x_t, y_t, _ = self.drone.simGetObjectPose('target').position
        self.target_pos = np.array([x_t, y_t])#section["target"]

        # Start the agent at random section at a random yz position
        # y_pos = np.array([8.0,8.2,8.4])
        # z_pos = np.array([-2.0,-2.0,-2.0])
        y_pos = ((np.random.rand(self.num)-0.5)*2*18) #+ 5 #(np.random.rand()-0.5)*2*2.5
        z_pos = ((np.random.rand(self.num)-1)*2*3.5) #(np.random.rand()-0.5)*2*2
        print(y_pos)
        for i in range(0,self.num):
            pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos,y_pos[i],z_pos[i]))
            self.drone.simSetVehiclePose(pose=pose, ignore_collision=True, vehicle_name=self.names[i])

        # Get target distance with mean distance for reward calculation
        self.target_dist_prev = np.linalg.norm(
            np.array([np.mean(y_pos), np.mean(z_pos)]) - self.target_pos)

        if self.input_mode == "multi_rgb":
            self.obs_stack = np.zeros(self.image_shape)

    # Multi agent action
    def do_action(self, action, name):
        # Execute action
        print(name, action)
        self.drone.moveByVelocityBodyFrameAsync(
            vx=1.0, vy=float(action[0]), vz=float(action[1]), duration=1, vehicle_name=name)#.join() #vy=float(action[0]), vz=float(action[1])
        self.drone.simContinueForTime(0.02)
        # Prevent swaying
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1, vehicle_name=name)
        self.drone.simContinueForTime(0.02)

    # Multi agent observations as list of single obs
    def get_obs(self,done):
        obs = {}
        for i in range(0,self.num):
            self.info[i] = self.is_collision(self.names[i])

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
                    obs[i] = self.get_rgb_image(self.names[i])
                else:
                    obs[i] = np.zeros((self.image_shape), dtype=np.uint8)

            # Still to implement multi agent
            elif self.input_mode == "depth":
                obs = self.get_depth_image(thresh=3.4).reshape(self.image_shape)
                obs = ((obs/3.4)*255).astype(int)
	
        return obs, self.info

    def compute_reward(self, reward, done):
        #reward = {i:0 for i in range(0, self.num)}
        #done = {i:0 for i in range(0, self.num)}

        for i in range(0, self.num):
            if done[i]!=1:
                print(done[i])
                # Get agent position
                x,y,z = self.drone.simGetVehiclePose(self.names[i]).position
                agent_traveled_x = np.abs(self.agent_start_pos - x)

                target_dist_curr = np.linalg.norm(np.array([x, y]) - self.target_pos)
                #target_dist_curr_3d = np.sqrt(np.square(target_dist_curr) + np.square(3.7-agent_traveled_x))

                # Vicinity reward
                reward[i] += (self.target_dist_prev/target_dist_curr) #np.exp(-target_dist_curr)*30

                # Debug
                print("############# Drone n.", i,"#############")
                print("Agents start pos", x, y, z, " and ", self.agent_start_pos)
                print("Target pos", self.target_pos)
                print("Distance origin to target", self.target_dist_prev)
                print("Traveled x", agent_traveled_x)
                print("Distance x,y to target", target_dist_curr)
                print("Rewards", reward)
                print("#########################################")

                # Collision penalty
                if self.is_collision(self.names[i]):
                    reward[i] += -100 #try
                    done[i] = 1

                # Check if agent almost arrived
                elif target_dist_curr < 15:
                    reward[i] += 100 #try
                    done[i] = 1

        return reward, done
    
    def is_collision(self, name):
        current_collision_time = self.drone.simGetCollisionInfo(vehicle_name=name).time_stamp
        return True if current_collision_time != self.collision_time[name] else False
    
    # Multi agent rgb view
    def get_rgb_image(self, name):
        #rgb_image_request = airsim.ImageRequest(
        #    0, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)], vehicle_name=name)
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3)) 
        self.save_image(img2d)
        # Sometimes no image returns from api
        try:
            return img2d.reshape(self.image_shape)
        except:
            return np.zeros((self.image_shape), dtype=np.uint8)

    def save_image(self, im):
        tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
        print ("Saving images to %s" % tmp_dir)
        try:
            os.makedirs(tmp_dir)
        except OSError:
            if not os.path.isdir(tmp_dir):
                raise
        filename = os.path.join(tmp_dir, "pic")
        cv2.imwrite(os.path.normpath(filename + '.png'), im)

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


class TestEnv(AirSimDroneEnv):
    def __init__(
        self, 
        ip_address, 
        image_shape, 
        env_config, 
        input_mode, 
        test_mode
    ):
    
        self.start_pos = -1

        super(TestEnv, self).__init__(
            ip_address, 
            image_shape, 
            env_config, 
            input_mode
        )
        
        self.test_mode = test_mode
        self.total_traveled = 0
        self.eps_n = 0
        self.eps_success = 0

        if self.test_mode == "sequential":
            print("Enter start position \n0: easy, 20: medium, 40: hard")
            self.start_pos = int(input())

    def setup_flight(self):
        super(TestEnv, self).setup_flight()

        if self.start_pos != -1:
            self.agent_start_pos = self.start_pos
        
        # Start the agent at a random yz position
        y_pos, z_pos = ((np.random.rand(1,2)-0.5)*2).squeeze()
        pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos,y_pos,z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)

    def do_action(self, action):
        speed = 1 #0.4

        x,_,_ = self.drone.simGetVehiclePose().position

        self.drone.moveByVelocityBodyFrameAsync(
            speed, float(action[0]), float(action[1]), duration=1
        ).join()

        # Prevent swaying
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)
        
    def compute_reward(self):
        reward = 0
        done = 0

        x,y,z = self.drone.simGetVehiclePose().position
        agent_traveled_x = np.abs(self.agent_start_pos - x)

        if self.is_collision():
            done = 1

        # Random test
        if self.test_mode == "random": 
            if agent_traveled_x > 3.7:
                self.eps_success += 1
                done = 1

            if done:
                self.eps_n += 1
                print("-----------------------------------")
                print("> Total episodes:", self.eps_n)
                print("> Holes reached: %d out of %d" % \
                    (self.eps_success, self.eps_n))
                print("> Success rate: %.2f%%" % (self.eps_success*100/self.eps_n))
                print("-----------------------------------\n")
            
        # Sequential test
        if self.test_mode == "sequential":
            if (agent_traveled_x+0.3)/4 > 5:
                done = 1
            
            if done:
                self.eps_n += 1
                self.total_traveled += agent_traveled_x + 0.3
                mean_traveled = self.total_traveled/self.eps_n

                print("-----------------------------------")
                print("> Total episodes:", self.eps_n)
                print("> Flight distance (mean): %.2f" % (mean_traveled))
                print("> Holes reached (mean): %d out of 5" % (int(mean_traveled//4)))
                print("-----------------------------------\n")

        return reward, done
