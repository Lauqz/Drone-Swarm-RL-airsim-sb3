import time
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from scripts.network import CustomCombinedExtractor

from scripts.local_env import LocalDroneEnv, petting_zoo

# Load inference configs
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Determine input image shape
image_shape = (84,84,1) if config["train_mode"]=="depth" else (84,84,3)

env = petting_zoo(
        image_shape=image_shape,
        input_mode=config["train_mode"],
        num_drones=config["num_drones"],
        num_vec_envs=config.get("num_vec_envs", 1),
        render_mode="browser" if config.get("viewer", False) else None,
        viewer_port=config.get("viewer_port", 8777),
        verbose=config.get("verbose", False),
        samples=config.get("samples", 2),
        )

policy_kwargs = dict(
    features_extractor_class=CustomCombinedExtractor
)

# New model (same hyperparameters as the AirSim variant; device="auto" so the
# same script runs on CPU-only machines)
model = PPO(
    'MultiInputPolicy',
    env,
    learning_rate=0.0001, #0.0001, 0.00005, 0.00001
    batch_size=256,
    clip_range=0.2, #0.1
    verbose=1,
    seed=42,
    device="auto",
    tensorboard_log="./tb_logs/",
    policy_kwargs=policy_kwargs,
)

print('==========================================================')
print('Model Design:')
print(model.policy)
print('==========================================================')

# Evaluation callback
callbacks = []
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=230, verbose=1)
eval_callback = EvalCallback(
    env,
    callback_on_new_best=callback_on_best,
    n_eval_episodes=30,
    best_model_save_path="saved_policy",
    log_path=".",
    eval_freq=8192,
)

callbacks.append(eval_callback)
kwargs = {}
kwargs["callback"] = callbacks

log_name = "ppo_local_map1_" + str(time.time())

model.learn(
    total_timesteps=4000000,
    tb_log_name=log_name,
    **kwargs
)
