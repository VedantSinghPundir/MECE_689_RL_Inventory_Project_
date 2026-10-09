# """
# Gymnasium wrapper around the tested InventorySim (Version 0: stationary Poisson demand).

# Design choices
#   * Demand for the whole episode is drawn at reset() from self.np_random, so the demand
#     sequence depends ONLY on the seed -- never on the policy. This gives paired comparisons.
#   * Observation is scaled: (I_t/scale, P_t/scale, t/T), float32, scale = max_order.
#   * reward = profit (higher is better). The positive cost = -reward is kept in info["cost"].
#   * terminated is always False; truncated=True when t reaches T.
# """
# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces

# from inventory_sim import InventorySim, Params


# class InventoryEnv(gym.Env):
#     metadata = {"render_modes": []}

#     def __init__(self, params=None, demand_mean=20.0, init_inventory=25):
#         super().__init__()
#         self.sim = InventorySim(params or Params())
#         p = self.sim.p
#         self.demand_mean = demand_mean
#         self.init_inventory = init_inventory
#         self.scale = float(p.max_order)

#         self.action_space = spaces.Discrete(p.max_order + 1)
#         # loose but finite upper bound on on-hand inventory / pipeline entries
#         max_stock = (init_inventory + p.T * p.max_order) / self.scale
#         high = np.array([max_stock] + [p.max_order / self.scale] * (p.L - 1) + [1.0],
#                         dtype=np.float32)
#         self.observation_space = spaces.Box(low=np.zeros_like(high), high=high, dtype=np.float32)

#     def _obs(self):
#         raw = self.sim.observation()          # (I, P..., t/T)
#         arr = np.array(raw, dtype=np.float32)
#         arr[:-1] /= self.scale
#         return arr

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.demands = self.np_random.poisson(self.demand_mean, size=self.sim.p.T)
#         self.sim.reset(init_inventory=self.init_inventory)
#         return self._obs(), {}

#     def step(self, action):
#         d = int(self.demands[self.sim.t])
#         _, r, terminated, truncated, info = self.sim.step(int(action), d)
#         info["cost"] = -r
#         return self._obs(), float(r), terminated, truncated, info

"""
Gymnasium wrapper around the tested InventorySim.

Demand regime is pluggable (see demand.py). The whole episode's demand (plus k days of
pre-episode history) is drawn at reset() from self.np_random, so demand depends ONLY on the
seed, never on the policy -> paired comparisons.

Observation (float32), identical layout for every regime:
    [ I_t/scale,
      P_t[0..L-2]/scale,                 pipeline still in transit after today's arrivals
      D_{t-1}/scale, ..., D_{t-k}/scale, last k observed demands (most recent first)
      t/T,                               episode progress
      sin(2 pi t/period), cos(2 pi t/period) ]   calendar/seasonal features
Only decision-time information is exposed; today's and future demand are never visible.

reward = profit (higher is better); info["cost"] = -reward. terminated is always False;
truncated=True when t reaches T.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from inventory_sim import InventorySim, Params
from demand import StationaryPoisson


class InventoryEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, params=None, demand=None, k=7, init_inventory=25, season_period=30):
        super().__init__()
        self.sim = InventorySim(params or Params())
        p = self.sim.p
        self.demand_model = demand or StationaryPoisson()
        self.k = k
        self.init_inventory = init_inventory
        self.season_period = season_period
        self.scale = float(p.max_order)
        self.demand_cap = 500.0          # observation clip for demand lags (>> any plausible demand)

        self.action_space = spaces.Discrete(p.max_order + 1)
        max_stock = (init_inventory + p.T * p.max_order) / self.scale
        low = np.array([0.0] * (1 + (p.L - 1) + k) + [0.0, -1.0, -1.0], dtype=np.float32)
        high = np.array([max_stock] + [p.max_order / self.scale] * (p.L - 1)
                        + [self.demand_cap / self.scale] * k + [1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # ---- observation ----------------------------------------------------
    def _obs(self):
        t, p = self.sim.t, self.sim.p
        lags = [self._all[self.k + t - 1 - i] for i in range(self.k)]   # D_{t-1} ... D_{t-k}
        ang = 2 * np.pi * t / self.season_period
        vec = [self.sim.inventory / self.scale]
        vec += [x / self.scale for x in self.sim.pipeline]
        vec += [min(d, self.demand_cap) / self.scale for d in lags]
        vec += [t / p.T, np.sin(ang), np.cos(ang)]
        return np.array(vec, dtype=np.float32)

    # ---- gym API --------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        T = self.sim.p.T
        # k days of history (calendar index -k..-1) followed by the T episode days
        self._all = self.demand_model.sample(self.np_random, T + self.k, t0=-self.k)
        self.history = self._all[:self.k]
        self.demands = self._all[self.k:]
        self.sim.reset(init_inventory=self.init_inventory)
        return self._obs(), {}

    def step(self, action):
        d = int(self.demands[self.sim.t])
        _, r, terminated, truncated, info = self.sim.step(int(action), d)
        info["cost"] = -r
        return self._obs(), float(r), terminated, truncated, info