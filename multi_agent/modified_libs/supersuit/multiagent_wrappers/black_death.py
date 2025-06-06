import gymnasium
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

from supersuit.utils.wrapper_chooser import WrapperChooser


class black_death_par(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)

    def _check_valid_for_black_death(self):
        for agent in self.agents:
            space = self.observation_space(agent)

            # Trying to fix
            assert isinstance(
                space, gymnasium.spaces.Box
            ), f"observation sapces for black death must be Box spaces, is {space}"

    def reset(self, seed=None, options=None): #!!!!!!!!!!!!!
        obss, infos = self.env.reset(seed=seed, options=options)

        self.agents = self.env.agents[:]
        #self._check_valid_for_black_death()
        if isinstance(self.observation_space(self.agents[0]), gymnasium.spaces.Box):
            black_obs = {
                agent: np.zeros_like(self.observation_space(agent).low)
                for agent in self.agents
                if agent not in obss
            }
        elif isinstance(self.observation_space(self.agents[0]), gymnasium.spaces.Dict):
            black_obs = {
                agent: {
                    k: np.zeros(v) for k,v in self.observation_space(agent).items()
                }
                for agent in self.agents
                if agent not in obss
            }
        return {**obss, **black_obs}, infos

    def step(self, actions): #!!!!!!!!!!!!!
        active_actions = {agent: actions[agent] for agent in self.env.agents}
        obss, rews, terms, truncs, infos = self.env.step(active_actions)
        if isinstance(self.observation_space(self.agents[0]), gymnasium.spaces.Box):
            black_obs = {
                agent: np.zeros_like(self.observation_space(agent).low)
                for agent in self.agents
                if agent not in obss
            }
        elif isinstance(self.observation_space(self.agents[0]), gymnasium.spaces.Dict):
            black_obs = {
                agent: {
                    'cam':np.zeros((84,84,3), dtype=np.uint8),
                    'pos':np.zeros((3,), dtype=np.float32)
                }
                for agent in self.agents
                if agent not in obss
            }
        black_rews = {agent: 0.0 for agent in self.agents if agent not in obss}
        black_infos = {agent: {} for agent in self.agents if agent not in obss}
        terminations = np.fromiter(terms.values(), dtype=bool)
        truncations = np.fromiter(truncs.values(), dtype=bool)
        env_is_done = (terminations & truncations).all()
        total_obs = {**black_obs, **obss}
        total_rews = {**black_rews, **rews}
        total_infos = {**black_infos, **infos}
        total_dones = {agent: env_is_done for agent in self.agents}
        if env_is_done:
            self.agents.clear()
        return total_obs, total_rews, total_dones, total_dones, total_infos


black_death_v3 = WrapperChooser(parallel_wrapper=black_death_par)
