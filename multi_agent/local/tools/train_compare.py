"""Old vs new flight model: train two fresh PPO models under a wall-clock
budget and compare learning curves and a final evaluation episode.

  python tools/train_compare.py --setup legacy --minutes 20   # old flight model
  python tools/train_compare.py --setup new    --minutes 20   # new defaults
  python tools/train_compare.py --plot                        # figures in docs/

legacy: ramp from rest + hard stop every step, log_std_init=0, no prev_act.
new:    velocity carried between steps, log_std_init / prev_action_obs from
        config.yml.
Everything else (seed, reward, network, PPO hyper-parameters, map, number of
drones, camera) is identical. Per-episode return (mean over drones of the sum
of rewards), success fraction and episode length are logged to
runs/<setup>/episodes.csv against SB3 timesteps (= env steps x num_drones).
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import scripts.local_env as le
from scripts.local_env import LocalDroneEnv, petting_zoo
from scripts.network import CustomCombinedExtractor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DOCS = os.path.join(ROOT, "docs")
RUNS = os.path.join(ROOT, "runs")


class EpisodeLogEnv(LocalDroneEnv):
    """Logs one row per episode and caches end-of-step poses."""
    log_path = None

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._env_steps = 0
        self._ep_return = 0.0
        self._ep_len = 0
        self.last_pose = {n: self.world.get_pose(n) for n in self.possible_agents}

    def step(self, action):
        out = super().step(action)
        self._env_steps += 1
        self._ep_len += 1
        self._ep_return += float(np.mean([self.reward[n] for n in self.possible_agents]))
        self.last_pose = {n: self.world.get_pose(n) for n in self.possible_agents}
        if all(self.done[n] == 1 for n in self.possible_agents):
            success = float(np.mean([self.obj[n] == 1 for n in self.possible_agents]))
            if self.log_path:
                with open(self.log_path, "a", newline="") as f:
                    csv.writer(f).writerow([self._env_steps * self.num, self._ep_return,
                                            success, self._ep_len])
            self._ep_return, self._ep_len = 0.0, 0
        return out


class LegacyEnv(EpisodeLogEnv):
    """Original flight model: ramp from rest every step, then a hard stop."""

    def do_actions(self, action):
        super().do_actions(action)
        for name in self.agents:
            self.world.stop(name)


class TimeBudget(BaseCallback):
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds
        self.t0 = None

    def _on_training_start(self):
        self.t0 = time.time()

    def _on_step(self):
        return (time.time() - self.t0) < self.seconds


def make_env(cls, cfg, prev_action_obs, log_path=None):
    cls.log_path = log_path

    def thunk(**kw):
        return cls(image_shape=(84, 84, 3), input_mode="single_rgb",
                   num_drones=cfg["num_drones"], samples=1,
                   prev_action_obs=prev_action_obs, **kw)
    orig = le.LocalDroneEnv
    le.LocalDroneEnv = thunk
    try:
        vec = petting_zoo(num_vec_envs=1)
    finally:
        le.LocalDroneEnv = orig
    raw = vec.venv.vec_envs[0].par_env.unwrapped if hasattr(vec, "venv") \
        else vec.vec_envs[0].par_env.unwrapped
    return vec, raw


def setup_params(setup, cfg):
    if setup == "legacy":
        return dict(cls=LegacyEnv, prev=False, log_std=0.0)
    return dict(cls=EpisodeLogEnv, prev=cfg.get("prev_action_obs", False),
                log_std=cfg.get("log_std_init", 0.0))


def eval_episode(model, vec, raw, seed):
    np.random.seed(seed)
    obs, _ = vec.reset()
    paths = {n: [raw.world.get_pose(n)] for n in raw.possible_agents}
    alive = {n: True for n in raw.possible_agents}
    rewards = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, _ = vec.step(action)
        ended = bool(term.all() or trunc.all())
        rewards.append(float(np.mean(rew)))
        for n in raw.possible_agents:
            if alive[n]:
                paths[n].append(raw.last_pose[n])
                if ended or raw.done[n] == 1:
                    alive[n] = False
        if ended:
            break
    # obj is still valid right after the terminal step (auto-reset re-creates it,
    # so read the cached copy made by the logging env before reset)
    return paths, sum(rewards)


def train(setup, minutes, seed):
    with open(os.path.join(ROOT, "config.yml")) as f:
        cfg = yaml.safe_load(f)
    out = os.path.join(RUNS, setup)
    os.makedirs(out, exist_ok=True)
    log_path = os.path.join(out, "episodes.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["timesteps", "ep_return", "success", "length"])

    p = setup_params(setup, cfg)
    np.random.seed(seed)
    vec, raw = make_env(p["cls"], cfg, p["prev"], log_path)
    model = PPO(
        "MultiInputPolicy", vec,
        learning_rate=0.0001, batch_size=256, clip_range=0.2,
        n_steps=512,              # shorter rollouts so a 20 min budget yields ~25 updates
        seed=seed, device="auto", verbose=0,
        tensorboard_log=os.path.join(out, "tb"),
        policy_kwargs=dict(features_extractor_class=CustomCombinedExtractor,
                           log_std_init=p["log_std"]),
    )
    print(f"[{setup}] flight={'stop every step' if setup == 'legacy' else 'carry velocity'} "
          f"log_std_init={p['log_std']} prev_action_obs={p['prev']} budget={minutes} min", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=10**9, callback=TimeBudget(minutes * 60), tb_log_name="ppo")
    print(f"[{setup}] trained {model.num_timesteps} timesteps in {(time.time()-t0)/60:.1f} min", flush=True)
    model.save(os.path.join(out, "final_model"))

    # one deterministic evaluation episode with the final model (fresh env, no logging)
    p["cls"].log_path = None
    vec_e, raw_e = make_env(p["cls"], cfg, p["prev"])
    paths, ret = eval_episode(model, vec_e, raw_e, seed=seed + 1)
    np.savez(os.path.join(out, "eval_episode.npz"), ret=ret, target=raw_e.target_pos,
             **{n: np.array(v) for n, v in paths.items()})
    print(f"[{setup}] eval episode return {ret:.1f}, steps {len(next(iter(paths.values())))-1}", flush=True)


def smooth(x, y, w):
    if len(y) < w:
        return x, y
    k = np.ones(w) / w
    return x[w - 1:], np.convolve(y, k, mode="valid")


def plot():
    setups = ["legacy", "new"]
    labels = {"legacy": "before (stop every step, log_std 0)",
              "new": "after (carry velocity, log_std -1)"}
    data = {}
    for s in setups:
        rows = np.genfromtxt(os.path.join(RUNS, s, "episodes.csv"), delimiter=",", names=True)
        data[s] = rows
    os.makedirs(DOCS, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for s in setups:
        d = data[s]
        w = max(5, len(d) // 15)
        x, y = smooth(d["timesteps"], d["ep_return"], w)
        axes[0].plot(x, y, label=f"{labels[s]} (n={len(d)} ep)")
        axes[0].plot(d["timesteps"], d["ep_return"], ".", ms=2, alpha=0.25, color=axes[0].lines[-1].get_color())
        x, y = smooth(d["timesteps"], d["success"], w)
        axes[1].plot(x, y, label=labels[s])
        x, y = smooth(d["timesteps"], d["length"], w)
        axes[2].plot(x, y, label=labels[s])
    axes[0].set(title="Mean per-drone episode return (rolling mean)", xlabel="timesteps", ylabel="return")
    axes[1].set(title="Success rate (fraction of drones reaching the target)", xlabel="timesteps", ylabel="success")
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].set(title="Episode length", xlabel="timesteps", ylabel="steps")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "learning_curves.png"), dpi=130)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for col, s in enumerate(setups):
        z = np.load(os.path.join(RUNS, s, "eval_episode.npz"))
        target = z["target"]
        names = [k for k in z.files if k.startswith("drone")]
        for d_i, n in enumerate(sorted(names)):
            p = z[n]
            c = plt.cm.tab10(d_i)
            axes[0, col].plot(p[:, 1], p[:, 0], "-o", color=c, ms=2, lw=1.2, label=n)
            axes[0, col].plot(p[0, 1], p[0, 0], "s", color=c, ms=6)
            axes[1, col].plot(p[:, 0], -p[:, 2], "-o", color=c, ms=2, lw=1.2)
        axes[0, col].add_patch(plt.Circle((target[1], target[0]), 12, fill=False, ls="--", color="k"))
        axes[0, col].plot(target[1], target[0], "k*", ms=12)
        axes[0, col].set(title=f"{labels[s]}\nfinal model, deterministic, return {float(z['ret']):.0f}",
                         xlabel="y [m]", ylabel="x [m]", aspect="equal")
        axes[0, col].set_xlim(-22, 22); axes[0, col].set_ylim(-2, 42); axes[0, col].legend(fontsize=8)
        axes[1, col].set(title="side view", xlabel="x [m]", ylabel="altitude [m]")
        axes[1, col].set_xlim(-2, 42); axes[1, col].set_ylim(0, 12.5)
        for ax in axes[:, col]:
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "eval_trajectories.png"), dpi=130)

    print(f"{'':34s}{'before':>12s}{'after':>12s}")
    def tail(d, key, frac=0.25):
        k = max(1, int(len(d) * frac)); return d[key][-k:].mean()
    for label, fn in [
        ("episodes", lambda d: len(d)),
        ("timesteps", lambda d: d["timesteps"][-1]),
        ("return, first 25% of episodes", lambda d: d["ep_return"][:max(1, len(d)//4)].mean()),
        ("return, last 25% of episodes", lambda d: tail(d, "ep_return")),
        ("success, last 25% of episodes", lambda d: tail(d, "success")),
        ("best rolling return (w=10)", lambda d: smooth(d["timesteps"], d["ep_return"], 10)[1].max()),
        ("episode length, last 25%", lambda d: tail(d, "length")),
    ]:
        print(f"{label:34s}{fn(data['legacy']):12.1f}{fn(data['new']):12.1f}")
    print("wrote docs/learning_curves.png and docs/eval_trajectories.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", choices=["legacy", "new"])
    ap.add_argument("--minutes", type=float, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        plot()
    elif a.setup:
        train(a.setup, a.minutes, a.seed)
    else:
        ap.error("give --setup legacy|new or --plot")
