"""PettingZoo API compliance and SuperSuit/SB3 integration tests."""

import numpy as np
import pytest

from scripts.local_env import LocalDroneEnv, petting_zoo


def test_parallel_api():
    """API compliance of the surface SB3 consumes: the black_death-wrapped env.

    The raw env (like the original AirSimDroneEnv it mirrors) returns
    reward/termination dicts for *all* possible agents each step, which
    parallel_api_test rejects — that is precisely why the training chain wraps
    it in black_death_v3 and why train.py keeps parallel_api_test commented out.
    """
    import supersuit as ss
    from pettingzoo.test import parallel_api_test
    np.random.seed(0)
    env = LocalDroneEnv(image_shape=(84, 84, 3), input_mode="single_rgb", num_drones=2)
    parallel_api_test(ss.black_death_v3(env), num_cycles=120)


def test_supersuit_chain_shapes():
    """The patched SuperSuit/SB3 chain exposes the gymnasium-style vec API:
    reset -> (obs, info), step -> 5-tuple — same as the AirSim evaluate.py."""
    np.random.seed(0)
    env = petting_zoo(image_shape=(84, 84, 3), input_mode="single_rgb", num_drones=3)
    obs, info = env.reset()
    assert env.num_envs == 3  # 3 agents through MarkovVectorEnv
    assert set(obs.keys()) == {"cam", "pos"}
    assert obs["cam"].shape == (3, 84, 84, 3) and obs["cam"].dtype == np.uint8
    assert obs["pos"].shape == (3, 3) and obs["pos"].dtype == np.float32
    for _ in range(12):
        actions = np.stack([env.action_space.sample() for _ in range(env.num_envs)])
        obs, rewards, terminated, truncated, infos = env.step(actions)
    assert obs["cam"].shape == (3, 84, 84, 3)
    assert rewards.shape == (3,)
    env.close()


def test_num_vec_envs_multiplies_arenas():
    np.random.seed(0)
    env = petting_zoo(image_shape=(84, 84, 3), input_mode="single_rgb",
                      num_drones=2, num_vec_envs=2)
    obs, info = env.reset()
    assert env.num_envs == 4  # 2 drones x 2 arenas
    assert obs["cam"].shape == (4, 84, 84, 3)
    env.close()


def test_depth_mode_chain():
    np.random.seed(0)
    env = petting_zoo(image_shape=(84, 84, 1), input_mode="depth", num_drones=2)
    obs, info = env.reset()
    assert obs["cam"].shape == (2, 84, 84, 1) and obs["cam"].dtype == np.uint8
    env.close()


@pytest.mark.slow
def test_ppo_smoke():
    """End-to-end: the full chain trains with PPO + CustomCombinedExtractor."""
    from stable_baselines3 import PPO
    from scripts.network import CustomCombinedExtractor

    np.random.seed(0)
    env = petting_zoo(image_shape=(84, 84, 3), input_mode="single_rgb", num_drones=2)
    model = PPO(
        "MultiInputPolicy", env,
        learning_rate=1e-4, n_steps=32, batch_size=64, n_epochs=1,
        seed=42, device="cpu", verbose=0,
        policy_kwargs=dict(features_extractor_class=CustomCombinedExtractor),
    )
    model.learn(total_timesteps=128)
    env.close()
