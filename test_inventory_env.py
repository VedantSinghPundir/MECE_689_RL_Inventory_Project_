import numpy as np
from gymnasium.utils.env_checker import check_env

from inventory_env import InventoryEnv
from inventory_sim import Params
from baselines import AlwaysOrder, RandomOrder, BaseStock, run_episode


def test_check_env():
    check_env(InventoryEnv(Params(T=30)))


def test_same_seed_same_demand():
    e1, e2 = InventoryEnv(), InventoryEnv()
    e1.reset(seed=7); e2.reset(seed=7)
    assert (e1.demands == e2.demands).all()
    e2.reset(seed=8)
    assert not (e1.demands == e2.demands).all()


def test_demand_independent_of_policy():
    """Paired comparison: the realised demand must not depend on the policy."""
    env = InventoryEnv(Params(T=30))
    seen = []
    for pol in (AlwaysOrder(0), AlwaysOrder(40), RandomOrder(1), BaseStock(60)):
        env.reset(seed=123); pol.reset()
        sim = env.unwrapped.sim
        ds, trunc = [], False
        while not trunc:
            a = pol.act(sim.inventory, tuple(sim.pipeline), 40)
            _, _, _, trunc, info = env.step(a)
            ds.append(info["demand"])
        seen.append(ds)
    assert all(s == seen[0] for s in seen)


def test_demand_statistics_poisson20():
    env = InventoryEnv(Params(T=90))
    allD = np.concatenate([(env.reset(seed=s), env.demands)[1] for s in range(300)])
    assert abs(allD.mean() - 20) < 0.3
    assert abs(allD.var() - 20) < 1.5


def test_flags_and_cost_info():
    env = InventoryEnv(Params(T=5))
    env.reset(seed=0)
    for i in range(5):
        obs, r, term, trunc, info = env.step(5)
        assert term is False
        assert trunc == (i == 4)
        assert info["cost"] == -r
        assert env.observation_space.contains(obs)


def test_observation_scaled_and_in_space_under_extreme_policy():
    env = InventoryEnv(Params(T=90))
    env.reset(seed=3)
    trunc = False
    while not trunc:
        obs, _, _, trunc, _ = env.step(40)     # always-40 -> inventory piles up
        assert env.observation_space.contains(obs)


def test_terminal_rule_through_env():
    env = InventoryEnv(Params(T=5, L=2))
    env.reset(seed=0)
    executed = [env.step(10)[4]["executed_order"] for _ in range(5)]
    assert executed == [10, 10, 10, 0, 0]


def test_base_stock_orders_up_to_B():
    pol = BaseStock(60)
    assert pol.act(20, (10,), 40) == 30     # 60 - (20+10)
    assert pol.act(50, (30,), 40) == 0      # IP above B -> order 0
    assert pol.act(0, (0,), 40) == 40       # clipped to max action