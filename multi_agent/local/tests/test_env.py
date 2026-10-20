"""Reward/termination parity tests against multi_agent/airsim/scripts/airsim_env.py.

Every expected value here is hand-computed from the original compute_reward
(airsim_env.py:277-348), including its quirks (frame-mixed target_dist_prev,
truncations raised with terminations, -200 timeout for everyone).
"""

import numpy as np
import pytest

from scripts.local_env import LocalDroneEnv


@pytest.fixture
def env2():
    np.random.seed(123)
    e = LocalDroneEnv(image_shape=(84, 84, 3), input_mode="single_rgb", num_drones=2)
    e.reset()
    return e


def put(env, name, pos):
    env.world.set_pose(name, pos)


def act(vx=1.0, vy=0.0, vz=0.0):
    return np.array([vx, vy, vz], dtype=np.float32)


def test_spaces_match_airsim_env(env2):
    obs_sp = env2.observation_space("drone0")
    assert obs_sp["cam"].shape == (84, 84, 3) and obs_sp["cam"].dtype == np.uint8
    assert obs_sp["pos"].shape == (3,) and obs_sp["pos"].dtype == np.float32
    act_sp = env2.action_space("drone0")
    assert np.allclose(act_sp.low, [0.0, -3.0, -3.0])
    assert np.allclose(act_sp.high, [3.0, 3.0, 3.0])


def test_vicinity_reward_positive_vx(env2):
    put(env2, "drone0", [5.0, 0.0, -4.0])
    put(env2, "drone1", [0.0, 15.0, -4.0])
    reward, done = env2.compute_reward(
        {"drone0": 0, "drone1": 0}, {"drone0": 0, "drone1": 0},
        {"drone0": act(2.0), "drone1": act(1.0)})
    d0 = np.linalg.norm(np.array([5.0, 0.0]) - env2.target_pos)
    d1 = np.linalg.norm(np.array([0.0, 15.0]) - env2.target_pos)
    assert reward["drone0"] == pytest.approx((env2.target_dist_prev / d0) * 2.0)
    assert reward["drone1"] == pytest.approx((env2.target_dist_prev / d1) * 1.0)
    assert done == {"drone0": 0, "drone1": 0}


def test_nonpositive_vx_penalty(env2):
    put(env2, "drone0", [5.0, 0.0, -4.0])
    put(env2, "drone1", [0.0, 15.0, -4.0])
    reward, _ = env2.compute_reward(
        {"drone0": 0, "drone1": 0}, {"drone0": 0, "drone1": 0},
        {"drone0": act(0.0), "drone1": act(0.0)})
    assert reward["drone0"] == -3
    assert reward["drone1"] == -3


def test_success_within_12m(env2):
    tx, ty = env2.target_pos
    put(env2, "drone0", [tx - 11.0, ty, -4.0])
    put(env2, "drone1", [0.0, 15.0, -4.0])
    reward, done = env2.compute_reward(
        {"drone0": 0, "drone1": 0}, {"drone0": 0, "drone1": 0},
        {"drone0": act(1.0), "drone1": act(1.0)})
    assert reward["drone0"] == 100
    assert done["drone0"] == 1 and env2.truncations["drone0"] == 1
    assert env2.obj["drone0"] == 1
    assert done["drone1"] == 0


def test_collision_penalty(env2):
    # fly drone0 straight down into the floor so the world flags a collision
    put(env2, "drone0", [3.0, 0.0, -1.0])
    put(env2, "drone1", [0.0, 15.0, -4.0])
    env2.world.move("drone0", [0.0, 0.0, 3.0], duration=1.0)
    assert env2.is_collision("drone0")
    reward, done = env2.compute_reward(
        {"drone0": 0, "drone1": 0}, {"drone0": 0, "drone1": 0},
        {"drone0": act(1.0), "drone1": act(1.0)})
    assert reward["drone0"] == -100
    assert done["drone0"] == 1 and env2.truncations["drone0"] == 1
    assert env2.obj["drone0"] == -1


def test_proximity_penalty(env2):
    put(env2, "drone0", [5.0, 0.0, -4.0])
    put(env2, "drone1", [5.0, 1.5, -4.0])
    reward, _ = env2.compute_reward(
        {"drone0": 0, "drone1": 0}, {"drone0": 0, "drone1": 0},
        {"drone0": act(1.0), "drone1": act(1.0)})
    d0 = np.linalg.norm(np.array([5.0, 0.0]) - env2.target_pos)
    d1 = np.linalg.norm(np.array([5.0, 1.5]) - env2.target_pos)
    assert reward["drone0"] == pytest.approx((env2.target_dist_prev / d0) * 1.0 - 10)
    assert reward["drone1"] == pytest.approx((env2.target_dist_prev / d1) * 1.0 - 10)


def test_all_arrived_bonus(env2):
    tx, ty = env2.target_pos
    env2.obj = {"drone0": 1, "drone1": 0}  # drone0 arrived earlier
    env2.done = {"drone0": 1, "drone1": 0}
    env2.agents = ["drone1"]
    put(env2, "drone1", [tx - 11.0, ty, -4.0])
    reward, done = env2.compute_reward(
        {"drone0": 0, "drone1": 0}, {"drone0": 1, "drone1": 0},
        {"drone1": act(1.0)})
    assert env2.obj == {"drone0": 1, "drone1": 1}
    assert reward["drone1"] == 200  # 100 success + 100 group bonus
    assert reward["drone0"] == 100  # 0 carried + 100 group bonus


def test_timeout_truncates_everyone(env2):
    env2.current_step = env2.max_steps
    reward, done = env2.compute_reward(
        {"drone0": 0, "drone1": 0}, {"drone0": 0, "drone1": 0},
        {"drone0": act(1.0), "drone1": act(1.0)})
    assert reward == {"drone0": -200, "drone1": -200}
    assert done == {"drone0": 1, "drone1": 1}
    assert env2.truncations == {"drone0": 1, "drone1": 1}
    assert env2.obj == {"drone0": -1, "drone1": -1}


def test_target_dist_prev_keeps_original_frame_mixing(env2):
    """Deliberate parity with airsim_env.py:218-219."""
    # recompute from a controlled spawn
    np.random.seed(7)
    env2.setup_flight()
    # it must NOT equal the true spawn->target distance except by coincidence;
    # verify it equals the verbatim formula instead
    ys = [env2.world.get_pose(a)[1] for a in env2.possible_agents]
    zs = [env2.world.get_pose(a)[2] for a in env2.possible_agents]
    expected = np.linalg.norm(np.array([np.mean(ys), np.mean(zs)]) - env2.target_pos)
    assert env2.target_dist_prev == pytest.approx(expected)


def test_step_returns_all_agent_rewards_but_only_alive_obs(env2):
    env2.done = {"drone0": 1, "drone1": 0}
    env2.agents = ["drone1"]
    put(env2, "drone1", [3.0, 15.0, -4.0])
    obs, rew, done, trunc, info = env2.step(
        {"drone0": act(1.0), "drone1": act(1.0)})
    assert set(rew.keys()) == {"drone0", "drone1"}
    assert set(obs.keys()) == {"drone1"}
    assert rew["drone0"] == 0  # dead agent reward reset to 0


def test_dead_agent_obs_is_zeroed(env2):
    env2.done["drone0"] = 1
    obs, info = env2.get_obs(env2.done)
    assert (obs["drone0"]["cam"] == 0).all()
    assert (obs["drone0"]["pos"] == 0).all()


def test_spawn_positions_respect_min_gap():
    np.random.seed(0)
    e = LocalDroneEnv(image_shape=(84, 84, 3), input_mode="single_rgb", num_drones=4)
    for _ in range(5):
        e.setup_flight()
        ys = sorted(e.world.get_pose(a)[1] for a in e.possible_agents)
        gaps = np.diff(ys)
        assert (gaps > 5).all(), ys
        for a in e.possible_agents:
            p = e.world.get_pose(a)
            assert p[0] == 0.0
            assert -18 <= p[1] <= 18
            assert -8 <= p[2] <= -1


def test_episode_is_deterministic_given_global_seed():
    def rollout():
        np.random.seed(42)
        e = LocalDroneEnv(image_shape=(84, 84, 3), input_mode="single_rgb", num_drones=2)
        obs, _ = e.reset()
        total = []
        for k in range(10):
            a = {i: act(2.0, 0.3 * (-1) ** k, 0.0) for i in e.agents}
            if not e.agents:
                break
            obs, rew, done, trunc, info = e.step(a)
            total.append(tuple(sorted(rew.items())))
        return total

    assert rollout() == rollout()


# ------------------------------------------------------ flight model changes
def test_prev_action_obs_off_by_default(env2):
    assert "prev_act" not in env2.observation_space("drone0").spaces
    obs, _ = env2.reset()
    assert "prev_act" not in obs["drone0"]


def test_prev_action_obs_key():
    from scripts.local_env import LocalDroneEnv
    e = LocalDroneEnv(image_shape=(84, 84, 3), input_mode="single_rgb",
                      num_drones=2, samples=1, prev_action_obs=True)
    sp = e.observation_space("drone0")["prev_act"]
    assert sp.shape == (3,) and sp.dtype == np.float32
    assert np.allclose(sp.low, [0, -3, -3]) and np.allclose(sp.high, [3, 3, 3])
    obs, _ = e.reset()
    assert (obs["drone0"]["prev_act"] == 0).all()
    a = {"drone0": np.array([1.0, 0.5, -0.2], np.float32),
         "drone1": np.array([5.0, -9.0, 0.0], np.float32)}   # out of bounds
    obs, *_ = e.step(a)
    for n in obs:
        if e.done[n] != 1:
            expected = np.clip(a[n], sp.low, sp.high)
            assert np.allclose(obs[n]["prev_act"], expected)
        else:
            assert (obs[n]["prev_act"] == 0).all()


def test_velocity_carries_across_env_steps(env2):
    env2.reset()
    for n in env2.possible_agents:
        env2.world.set_pose(n, [0, 0 if n == "drone0" else 15, -4])
    env2.target_dist_prev = 1.0
    a = {n: np.array([2.0, 0.0, 0.0], np.float32) for n in env2.possible_agents}
    env2.step(a)
    assert env2.world.get_velocity("drone0")[0] > 1.9
    env2.step(a)
    # done drones are stopped explicitly
    for n in env2.possible_agents:
        if env2.done[n] == 1:
            assert np.allclose(env2.world.get_velocity(n), 0.0)
