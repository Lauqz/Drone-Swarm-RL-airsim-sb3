import yaml

from stable_baselines3 import PPO

from scripts.local_env import LocalDroneEnv, petting_zoo

# Load inference configs
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Determine input image shape
image_shape = (84,84,1) if config["test_mode"]=="depth" else (84,84,3)

# Viewer on by default for evaluation: open http://localhost:<viewer_port>
env = petting_zoo(
        image_shape=image_shape,
        input_mode=config["test_mode"],
        num_drones=config["num_drones"],
        render_mode="browser" if config.get("viewer", True) is not False else None,
        viewer_port=config.get("viewer_port", 8777),
        verbose=config.get("verbose", False),
        samples=config.get("samples", 2),
        )

model = PPO.load("saved_policy/best_model", env=env)

obs, info = env.reset()

# Evaluate the agents
episode_reward = 0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward = reward
    if terminated.all() or truncated.all():
        print("Reward:", episode_reward)
        episode_reward = 0
        obs, info = env.reset()
