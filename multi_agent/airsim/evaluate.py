import time
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from scripts.network import NatureCNN
from stable_baselines3.common.env_checker import check_env
from pettingzoo.test import parallel_api_test

from scripts.airsim_env import AirSimDroneEnv, petting_zoo

# Load inference configs
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Determine input image shape
image_shape = (84,84,1) if config["train_mode"]=="depth" else (84,84,3)

env = petting_zoo(        
        ip_address="127.0.0.1", 
        image_shape=image_shape,
        input_mode=config["train_mode"],
        num_drones=config["num_drones"],
        prev_action_obs=config.get("prev_action_obs", False)
        )

model = PPO.load("saved_policy/best_model", env=env)

obs, info = env.reset()

# Evaluate the agents
episode_reward = 0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward = reward
    if terminated.all() or truncated.all(): # or info.get("is_success", False):
        print("Reward:", episode_reward)#, "Success?", info.get("is_success", False))
        episode_reward = 0
        obs, info = env.reset()
