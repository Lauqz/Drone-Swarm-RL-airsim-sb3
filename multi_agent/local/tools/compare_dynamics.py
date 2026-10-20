"""Before/after comparison of the flight model with an UNTRAINED PPO policy.

No training happens here. The same freshly initialised PPO (stochastic
actions) flies a few episodes in the local sim under two set-ups:

  before: the original flight model (each 1 s command starts from rest and is
          followed by a hard stop) with SB3's default exploration noise
          (log_std_init = 0 -> std 1 m/s on every axis) and no prev_act obs.
          The sequential one-drone-after-another flight of the old AirSim
          do_action() has no visible effect in the instantaneous local sim,
          so it is not modelled.
  after:  velocity carried between steps (no stop, explicit hover only when a
          drone is done) and the exploration noise / prev_action_obs from
          config.yml.

Outputs (in docs/):
  dynamics_trajectories.png   top-down and side views of every drone path
  dynamics_velocity.png       per-drone velocity variance and |dv| per step
and a metrics table on stdout.

Usage:  python tools/compare_dynamics.py [--episodes 5] [--seed 0]
"""

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from scripts.local_env import LocalDroneEnv, petting_zoo
from scripts.network import CustomCombinedExtractor

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")


class PoseLogEnv(LocalDroneEnv):
    """Caches the end-of-step poses, which the vec env's auto-reset would
    otherwise overwrite before the caller can read them."""

    def step(self, action):
        out = super().step(action)
        self.last_pose = {n: self.world.get_pose(n) for n in self.possible_agents}
        return out


class LegacyFlightEnv(PoseLogEnv):
    """Original flight model: ramp from rest every step, then a hard stop."""

    def do_actions(self, action):
        super().do_actions(action)
        for name in self.agents:
            self.world.stop(name)


def make_env(cls, num_drones, prev_action_obs):
    def thunk(**kw):
        return cls(image_shape=(84, 84, 3), input_mode="single_rgb",
                   num_drones=num_drones, samples=1,
                   prev_action_obs=prev_action_obs, **kw)
    # same SuperSuit chain as train.py, but with our env class
    import scripts.local_env as le
    orig = le.LocalDroneEnv
    le.LocalDroneEnv = thunk
    try:
        return petting_zoo(num_vec_envs=1)
    finally:
        le.LocalDroneEnv = orig


def rollout(vec_env, model, raw_env, episodes, seed):
    """Returns a list of episodes; each is {drone: (T+1, 3) positions}."""
    np.random.seed(seed)
    model.set_random_seed(seed)
    out = []
    for _ in range(episodes):
        obs, _ = vec_env.reset()
        paths = {n: [raw_env.world.get_pose(n)] for n in raw_env.possible_agents}
        alive = {n: True for n in raw_env.possible_agents}
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, rew, term, trunc, infos = vec_env.step(action)
            ended = bool(term.all() or trunc.all())   # vec env auto-resets after this
            for n in raw_env.possible_agents:
                if alive[n]:
                    # last_pose is cached by PoseLogEnv.step before auto-reset
                    paths[n].append(raw_env.last_pose[n])
                    if ended or raw_env.done[n] == 1:
                        alive[n] = False
            if ended:
                break
        out.append({n: np.array(p) for n, p in paths.items()})
    return out


def metrics(episodes):
    """Per-drone metrics over all episodes."""
    var, dv, zig, head = [], [], [], []
    for ep in episodes:
        for n, p in ep.items():
            v = np.diff(p, axis=0)              # realised velocity, m/s (1 s steps)
            if len(v) < 2:
                continue
            var.append(v.var(axis=0).sum())
            dv.append(np.linalg.norm(np.diff(v, axis=0), axis=1))
            net = np.linalg.norm(p[-1] - p[0])
            zig.append(np.linalg.norm(v, axis=1).sum() / max(net, 1e-6))
            nv = np.linalg.norm(v, axis=1)
            ok = (nv[:-1] > 1e-6) & (nv[1:] > 1e-6)
            cosang = (v[:-1] * v[1:]).sum(1)[ok] / (nv[:-1] * nv[1:])[ok]
            head.extend(np.degrees(np.arccos(np.clip(cosang, -1, 1))))
    return dict(var=np.array(var), dv=dv, zig=np.array(zig), head=np.array(head))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(HERE, "..", "config.yml")) as f:
        cfg = yaml.safe_load(f)
    n = cfg["num_drones"]

    setups = {
        "before": dict(cls=LegacyFlightEnv, prev=False, log_std=0.0),
        "after": dict(cls=PoseLogEnv, prev=cfg.get("prev_action_obs", False),
                      log_std=cfg.get("log_std_init", 0.0)),
    }
    results, mets = {}, {}
    for name, s in setups.items():
        vec = make_env(s["cls"], n, s["prev"])
        raw = vec.venv.vec_envs[0].par_env.unwrapped if hasattr(vec, "venv") else None
        if raw is None:
            raw = vec.vec_envs[0].par_env.unwrapped
        model = PPO("MultiInputPolicy", vec, seed=42, device="cpu", verbose=0,
                    policy_kwargs=dict(features_extractor_class=CustomCombinedExtractor,
                                       log_std_init=s["log_std"]))
        print(f"[{name}] log_std_init={s['log_std']} prev_action_obs={s['prev']} "
              f"flight={'stop every step' if s['cls'] is LegacyFlightEnv else 'carry velocity'}")
        results[name] = rollout(vec, model, raw, args.episodes, args.seed)
        mets[name] = metrics(results[name])
        vec.close()

    target = raw.target_pos
    os.makedirs(DOCS, exist_ok=True)

    # ------------------------------------------------------------ trajectories
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for col, name in enumerate(setups):
        for ep_i, ep in enumerate(results[name]):
            for d_i, (dn, p) in enumerate(ep.items()):
                c = plt.cm.tab10(d_i)
                axes[0, col].plot(p[:, 1], p[:, 0], "-", color=c, lw=1, alpha=0.8)
                axes[0, col].plot(p[0, 1], p[0, 0], "o", color=c, ms=3)
                axes[1, col].plot(p[:, 0], -p[:, 2], "-", color=c, lw=1, alpha=0.8)
        axes[0, col].add_patch(plt.Circle((target[1], target[0]), 12, fill=False, ls="--", color="k"))
        axes[0, col].plot(target[1], target[0], "k*", ms=12)
        axes[0, col].set(title=f"{name}: top-down ({args.episodes} episodes, {n} drones, untrained PPO)",
                         xlabel="y [m]", ylabel="x [m] (towards target)", aspect="equal")
        axes[0, col].set_xlim(-22, 22); axes[0, col].set_ylim(-2, 42)
        axes[1, col].set(title=f"{name}: side view", xlabel="x [m]", ylabel="altitude [m]")
        axes[1, col].set_xlim(-2, 42); axes[1, col].set_ylim(0, 12.5)
        for ax in axes[:, col]:
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "dynamics_trajectories.png"), dpi=130)

    # ---------------------------------------------------------------- velocity
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].boxplot([mets[k]["var"] for k in setups], tick_labels=list(setups))
    axes[0].set(title="Realised velocity variance per drone-episode\n(sum over x,y,z)",
                ylabel="var [m²/s²]")
    T = 100
    for name in setups:
        rows = [np.pad(d, (0, T - len(d)), constant_values=np.nan)[:T] for d in mets[name]["dv"]]
        m = np.nanmean(np.array(rows), axis=0)
        axes[1].plot(np.arange(1, T + 1), m, label=name)
    axes[1].set(title="Velocity change between consecutive steps", xlabel="step",
                ylabel="mean |Δv| [m/s]"); axes[1].legend()
    axes[2].hist([mets[k]["head"] for k in setups], bins=36, range=(0, 180),
                 label=list(setups), density=True)
    axes[2].set(title="Heading change between consecutive 1 s segments",
                xlabel="angle [deg]", ylabel="density"); axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "dynamics_velocity.png"), dpi=130)

    # ------------------------------------------------------------------ table
    print()
    print(f"{'metric':46s} {'before':>10s} {'after':>10s}")
    rows = [
        ("velocity variance per drone-episode [m²/s²]", lambda m: m["var"].mean()),
        ("mean |Δv| between steps [m/s]", lambda m: np.concatenate(m["dv"]).mean()),
        ("median heading change [deg]", lambda m: np.median(m["head"])),
        ("path length / net displacement", lambda m: np.median(m["zig"])),
    ]
    for label, fn in rows:
        print(f"{label:46s} {fn(mets['before']):10.2f} {fn(mets['after']):10.2f}")
    steps = {k: np.mean([len(next(iter(ep.values()))) - 1 for ep in results[k]]) for k in setups}
    print(f"{'mean episode length [steps]':46s} {steps['before']:10.1f} {steps['after']:10.1f}")
    print("\nwrote", os.path.join(DOCS, "dynamics_trajectories.png"),
          "and", os.path.join(DOCS, "dynamics_velocity.png"))


if __name__ == "__main__":
    main()
