import supersuit as ss
from . import airsim
import os
import gym
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] ='True'


def env(**kwargs):
    env = AirSimDroneEnv(**kwargs)
    env = ss.black_death_v3(env)
    #env = ss.frame_stack_v2(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    return env


petting_zoo = env


class AirSimDroneEnv(gym.Env):
    def __init__(self, ip_address, image_shape, env_config, input_mode):
        self.image_shape = image_shape
        #self.sections = env_config["sections"]
        self.input_mode = input_mode

        self.drone = airsim.MultirotorClient(ip=ip_address)

        if self.input_mode == "multi_rgb":
            self.observation_space = gym.spaces.Box(
                low=0, high=255, 
                shape=(image_shape[0],image_shape[1]*3,1), 
                dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=self.image_shape, dtype=np.uint8)

        self.action_space = gym.spaces.Box(
            low=-3.0, high=3.0, shape=(2,), dtype=np.float32) #0.6

        self.info = {"collision": False}
        self.collision_time = 0
        self.random_start = True
        self.setup_flight()

    def step(self, action):
        self.do_action(action)
        obs, info = self.get_obs()
        reward, done = self.compute_reward()
        return obs, reward, done, info

    def reset(self):
        self.setup_flight()
        obs, _ = self.get_obs()
        return obs

    def render(self):
        return self.get_obs()

    def setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Prevent drone from falling after reset
        self.drone.moveToZAsync(-1, 1)

        # Get collision time stamp
        self.collision_time = self.drone.simGetCollisionInfo().time_stamp

        self.agent_start_pos = 0 #section["offset"][0] #check what its
        x_t, y_t, _ = self.drone.simGetObjectPose('target').position
        self.target_pos = np.array([x_t, y_t])#section["target"]

        # Start the agent at random section at a random yz position
        # y_pos, z_pos = ((np.random.rand(1,2)-0.5)*2).squeeze()
        y_pos = (np.random.rand()-0.5)*2*19 #+ 5 #(np.random.rand()-0.5)*2*2.5
        z_pos = (np.random.rand()-1)*2*4 #(np.random.rand()-0.5)*2*2
        pose = airsim.Pose(airsim.Vector3r(self.agent_start_pos,y_pos,z_pos))
        self.drone.simSetVehiclePose(pose=pose, ignore_collision=True)
        
        # Get target distance for reward calculation
        self.target_dist_prev = np.linalg.norm(
            np.array([y_pos, z_pos]) - self.target_pos)

        if self.input_mode == "multi_rgb":
            self.obs_stack = np.zeros(self.image_shape)

    def do_action(self, action):
        # Execute action
        self.drone.moveByVelocityBodyFrameAsync(
            1, float(action[0]), float(action[1]), duration=1).join()

        # Prevent swaying
        self.drone.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1)

    def get_obs(self):
        self.info["collision"] = self.is_collision()
        
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
            obs = self.get_rgb_image()

        elif self.input_mode == "depth":
            obs = self.get_depth_image(thresh=3.4).reshape(self.image_shape)
            obs = ((obs/3.4)*255).astype(int)
	
        return obs, self.info

    def compute_reward(self):
        reward = 0
        done = 0

        # Get agent position
        x,y,z = self.drone.simGetVehiclePose().position
        agent_traveled_x = np.abs(self.agent_start_pos - x)

        target_dist_curr = np.linalg.norm(np.array([x, y]) - self.target_pos)
        #target_dist_curr_3d = np.sqrt(np.square(target_dist_curr) + np.square(3.7-agent_traveled_x))

        # Vicinity reward
        reward += (self.target_dist_prev/target_dist_curr) #np.exp(-target_dist_curr)*30

        # Debug
        print("##########################")
        print("Agents start pos", x, y, z, " and ", self.agent_start_pos)
        print("Target pos", self.target_pos)
        print("Distance origin to target", self.target_dist_prev)
        print("Traveled x", agent_traveled_x)
        print("Distance x,y to target", target_dist_curr)
        print("Rewards", reward)
        print("##########################")

        # Collision penalty
        if self.is_collision():
            reward = -100
            done = 1

        # Check if agent almost arrived
        elif target_dist_curr < 15:
            reward = 100 
            done = 1

        return reward, done

    def is_collision(self):
        current_collision_time = self.drone.simGetCollisionInfo().time_stamp
        return True if current_collision_time != self.collision_time else False
    
    def get_rgb_image(self):
        rgb_image_request = airsim.ImageRequest(
            0, airsim.ImageType.Scene, False, False)
        responses = self.drone.simGetImages([rgb_image_request])
        img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img2d = np.reshape(img1d, (responses[0].height, responses[0].width, 3)) 

        # Sometimes no image returns from api
        try:
            return img2d.reshape(self.image_shape)
        except:
            return np.zeros((self.image_shape))

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

        # Go straight while passing through hole
        # t = (x + 0.3) % 4 
        # if t > 3.8 or t < 0.2:
        #     self.drone.moveByVelocityBodyFrameAsync(
        #         speed, 0.0, 0.0, duration=1
        #     ).join()
        # else:
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
