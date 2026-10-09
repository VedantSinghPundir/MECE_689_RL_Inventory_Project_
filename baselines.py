# """Baseline policies + evaluation on paired seeds.

# Policies receive the RAW decision-time state (on-hand I_t, pipeline P_t), i.e. the same
# information the RL agent gets (just unscaled). No future demand is visible.
# """
# import numpy as np
# from inventory_env import InventoryEnv

# VAL_SEEDS = list(range(1000, 1050))     # used ONLY to tune B
# TEST_SEEDS = list(range(2000, 2100))    # held-out; never used for tuning


# class AlwaysOrder:
#     def __init__(self, q): self.q = q; self.name = f"always-{q}"
#     def reset(self): pass
#     def act(self, inv, pipe, max_order): return min(self.q, max_order)


# class RandomOrder:
#     def __init__(self, seed=0): self.seed = seed; self.name = "random"; self.reset()
#     def reset(self): self.rng = np.random.default_rng(self.seed)   # own RNG: demand stream untouched
#     def act(self, inv, pipe, max_order): return int(self.rng.integers(0, max_order + 1))


# class BaseStock:
#     """Order up to level B using inventory position IP = on-hand + pipeline."""
#     def __init__(self, B): self.B = B; self.name = f"base-stock(B={B})"
#     def reset(self): pass
#     def act(self, inv, pipe, max_order):
#         ip = inv + sum(pipe)
#         return int(np.clip(self.B - ip, 0, max_order))


# def run_episode(env, policy, seed):
#     env.reset(seed=seed)
#     policy.reset()
#     sim = env.unwrapped.sim
#     tot = dict(profit=0.0, demand=0, sales=0, unmet=0, holding=0.0, inv_sum=0.0, stockout_days=0)
#     trunc = False
#     while not trunc:
#         a = policy.act(sim.inventory, tuple(sim.pipeline), sim.p.max_order)
#         _, r, term, trunc, info = env.step(a)
#         tot["profit"] += r
#         tot["demand"] += info["demand"]; tot["sales"] += info["sales"]; tot["unmet"] += info["unmet"]
#         tot["holding"] += sim.p.h * info["end_inventory"]
#         tot["inv_sum"] += info["end_inventory"]
#         tot["stockout_days"] += int(info["unmet"] > 0)
#     T = sim.p.T
#     return dict(profit=tot["profit"], fill_rate=tot["sales"] / tot["demand"],
#                 stockout_day_freq=tot["stockout_days"] / T, unmet=tot["unmet"],
#                 holding_cost=tot["holding"], avg_end_inv=tot["inv_sum"] / T)


# def evaluate(env, policy, seeds):
#     rows = [run_episode(env, policy, s) for s in seeds]
#     out = {}
#     for k in rows[0]:
#         v = np.array([r[k] for r in rows])
#         out[k] = (v.mean(), v.std(ddof=1), 1.96 * v.std(ddof=1) / np.sqrt(len(v)))
#     return out


# def tune_base_stock(env, B_grid, seeds=VAL_SEEDS):
#     scores = {B: evaluate(env, BaseStock(B), seeds)["profit"][0] for B in B_grid}
#     best = max(scores, key=scores.get)
#     return best, scores

"""Baseline policies + evaluation on paired seeds.

Policies receive the RAW decision-time state (on-hand I_t, pipeline P_t), i.e. the same
information the RL agent gets (just unscaled). No future demand is visible.
"""
import numpy as np
from inventory_env import InventoryEnv

VAL_SEEDS = list(range(1000, 1050))     # used ONLY to tune control parameters (B, safety stock)
TEST_SEEDS = list(range(2000, 2100))    # held-out; never used for fitting or tuning
TRAIN_SEEDS = list(range(3000, 3300))   # used ONLY to fit the Random Forest forecaster


class AlwaysOrder:
    def __init__(self, q): self.q = q; self.name = f"always-{q}"
    def reset(self): pass
    def act(self, inv, pipe, max_order, obs=None): return min(self.q, max_order)


class RandomOrder:
    def __init__(self, seed=0): self.seed = seed; self.name = "random"; self.reset()
    def reset(self): self.rng = np.random.default_rng(self.seed)   # own RNG: demand stream untouched
    def act(self, inv, pipe, max_order, obs=None): return int(self.rng.integers(0, max_order + 1))


class BaseStock:
    """Order up to level B using inventory position IP = on-hand + pipeline."""
    def __init__(self, B): self.B = B; self.name = f"base-stock(B={B})"
    def reset(self): pass
    def act(self, inv, pipe, max_order, obs=None):
        ip = inv + sum(pipe)
        return int(np.clip(self.B - ip, 0, max_order))


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    policy.reset()
    if hasattr(policy, "prepare"):       # optional hook (used by the RF policy for batch forecasts)
        policy.prepare(env)
    sim = env.unwrapped.sim
    tot = dict(profit=0.0, demand=0, sales=0, unmet=0, holding=0.0, inv_sum=0.0, stockout_days=0)
    trunc = False
    while not trunc:
        a = policy.act(sim.inventory, tuple(sim.pipeline), sim.p.max_order, obs)
        obs, r, term, trunc, info = env.step(a)
        tot["profit"] += r
        tot["demand"] += info["demand"]; tot["sales"] += info["sales"]; tot["unmet"] += info["unmet"]
        tot["holding"] += sim.p.h * info["end_inventory"]
        tot["inv_sum"] += info["end_inventory"]
        tot["stockout_days"] += int(info["unmet"] > 0)
    T = sim.p.T
    return dict(profit=tot["profit"], fill_rate=tot["sales"] / tot["demand"],
                stockout_day_freq=tot["stockout_days"] / T, unmet=tot["unmet"],
                holding_cost=tot["holding"], avg_end_inv=tot["inv_sum"] / T)


def evaluate(env, policy, seeds):
    rows = [run_episode(env, policy, s) for s in seeds]
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows])
        out[k] = (v.mean(), v.std(ddof=1), 1.96 * v.std(ddof=1) / np.sqrt(len(v)))
    return out


def tune_base_stock(env, B_grid, seeds=VAL_SEEDS):
    scores = {B: evaluate(env, BaseStock(B), seeds)["profit"][0] for B in B_grid}
    best = max(scores, key=scores.get)
    return best, scores


def profits(env, policy, seeds):
    """Per-seed profit array (for paired comparisons on identical demand)."""
    return np.array([run_episode(env, policy, s)["profit"] for s in seeds])


def paired_diff(a, b):
    """mean, 95% CI half-width of (a - b) over matched seeds."""
    d = np.asarray(a) - np.asarray(b)
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))