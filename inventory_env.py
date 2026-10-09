"""
Gymnasium wrapper around the tested InventorySim (Version 0: stationary Poisson demand).

Design choices
  * Demand for the whole episode is drawn at reset() from self.np_random, so the demand
    sequence depends ONLY on the seed -- never on the policy. This gives paired comparisons.
  * Observation is scaled: (I_t/scale, P_t/scale, t/T), float32, scale = max_order.
  * reward = profit (higher is better). The positive cost = -reward is kept in info["cost"].
  * terminated is always False; truncated=True when t reaches T.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from inventory_sim import InventorySim, Params


class InventoryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, params=None, demand_mean=20.0, init_inventory=25):
        super().__init__()
        self.sim = InventorySim(params or Params())
        p = self.sim.p
        self.demand_mean = demand_mean
        self.init_inventory = init_inventory
        self.scale = float(p.max_order)

        self.action_space = spaces.Discrete(p.max_order + 1)
        # loose but finite upper bound on on-hand inventory / pipeline entries
        max_stock = (init_inventory + p.T * p.max_order) / self.scale
        high = np.array([max_stock] + [p.max_order / self.scale] * (p.L - 1) + [1.0],
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros_like(high), high=high, dtype=np.float32)

    def _obs(self):
        raw = self.sim.observation()          # (I, P..., t/T)
        arr = np.array(raw, dtype=np.float32)
        arr[:-1] /= self.scale
        return arr

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.demands = self.np_random.poisson(self.demand_mean, size=self.sim.p.T)
        self.sim.reset(init_inventory=self.init_inventory)
        return self._obs(), {}

    def step(self, action):
        d = int(self.demands[self.sim.t])
        _, r, terminated, truncated, info = self.sim.step(int(action), d)
        info["cost"] = -r
        return self._obs(), float(r), terminated, truncated, info