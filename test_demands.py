import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from demand import StationaryPoisson, NegBin, SeasonalShock
from inventory_env import InventoryEnv
from inventory_sim import Params
from baselines import AlwaysOrder, RandomOrder, BaseStock

ALL = [StationaryPoisson, NegBin, SeasonalShock]


def rng(seed=0):
    return np.random.default_rng(seed)


# ---------------- distribution checks ----------------
def test_poisson_mean_var():
    d = StationaryPoisson().sample(rng(1), 200_000)
    assert abs(d.mean() - 20) < 0.1 and abs(d.var() - 20) < 0.5


def test_negbin_mean_var_matches_proposal():
    d = NegBin().sample(rng(2), 400_000)
    assert abs(d.mean() - 20) < 0.1          # mean = r(1-p)/p = 20
    assert abs(d.var() - 60) < 1.5           # var  = r(1-p)/p^2 = 60
    assert d.var() / d.mean() > 2.5          # clearly overdispersed vs Poisson


def test_seasonal_pattern_without_shocks():
    m = SeasonalShock(shock_prob=0.0)
    rows = np.array([m.sample(rng(s), 30, t0=0) for s in range(4000)])
    mean_by_day = rows.mean(axis=0)
    expected = 20 + 6 * np.sin(2 * np.pi * np.arange(30) / 30)
    assert np.abs(mean_by_day - expected).max() < 0.6
    assert mean_by_day[7] - mean_by_day[22] > 10     # peak (day ~7.5) vs trough (day ~22.5)


def test_seasonal_rate_positive_and_shocks_only_add():
    m = SeasonalShock()
    r = m.rates(rng(3), 5000, t0=0)
    base = m.seasonal(5000, t0=0)
    assert (r > 0).all()
    assert (r >= base - 1e-9).all()                  # shocks are positive only


def test_shock_process_properties():
    m = SeasonalShock()
    J = m.shocks(rng(4), 200_000)
    assert J.min() == 0 and J.max() <= 16 and J[J > 0].min() >= 8
    assert 0.12 < (J > 0).mean() < 0.22              # approx 5p/(5p+(1-p)) ~ 0.17
    # shocks last at least 3 days (the minimum duration)
    runs, cur = [], 0
    for x in J:
        if x > 0: cur += 1
        elif cur: runs.append(cur); cur = 0
    assert min(runs) >= 3


def test_shocks_raise_mean_and_variance():
    plain = SeasonalShock(shock_prob=0.0).sample(rng(5), 100_000)
    shocked = SeasonalShock().sample(rng(5), 100_000)
    assert shocked.mean() > plain.mean() + 1.0
    assert shocked.var() > plain.var()


@pytest.mark.parametrize("cls", ALL)
def test_generators_deterministic_and_integer(cls):
    a, b = cls().sample(rng(9), 500, t0=-7), cls().sample(rng(9), 500, t0=-7)
    assert (a == b).all() and (a >= 0).all() and np.issubdtype(a.dtype, np.integer)


# ---------------- env-level checks for every regime ----------------
@pytest.mark.parametrize("cls", ALL)
def test_check_env_each_regime(cls):
    check_env(InventoryEnv(Params(T=30), demand=cls()))


@pytest.mark.parametrize("cls", ALL)
def test_demand_independent_of_policy_each_regime(cls):
    env = InventoryEnv(Params(T=30), demand=cls())
    seen = []
    for pol in (AlwaysOrder(0), AlwaysOrder(40), RandomOrder(1), BaseStock(60)):
        env.reset(seed=5); pol.reset()
        sim = env.unwrapped.sim
        ds, trunc = [], False
        while not trunc:
            a = pol.act(sim.inventory, tuple(sim.pipeline), 40)
            _, _, _, trunc, info = env.step(a)
            ds.append(info["demand"])
        seen.append(ds)
    assert all(s == seen[0] for s in seen)


def test_observation_layout_and_no_leakage():
    k, L = 7, 2
    env = InventoryEnv(Params(T=20, L=L), k=k)
    obs, _ = env.reset(seed=11)
    assert obs.shape == (1 + (L - 1) + k + 3,)
    # t=0: lags are the k history days, most recent first
    assert np.allclose(obs[2:2 + k] * 40, env.history[::-1])
    assert obs[-3] == 0.0 and obs[-2] == pytest.approx(0.0) and obs[-1] == pytest.approx(1.0)
    obs1, _, _, _, info = env.step(10)
    # after day 0, D_{t-1} (first lag entry) must be exactly day 0's demand
    assert obs1[2] * 40 == pytest.approx(info["demand"])
    # the observation at t=0 must NOT contain day 0's demand unless by coincidence in history
    assert env.observation_space.contains(obs) and env.observation_space.contains(obs1)


def test_observation_seasonal_features_follow_calendar():
    env = InventoryEnv(Params(T=40), season_period=30)
    env.reset(seed=0)
    for _ in range(10):
        obs, *_ = env.step(0)
    t = 10
    assert obs[-2] == pytest.approx(np.sin(2 * np.pi * t / 30), abs=1e-6)
    assert obs[-1] == pytest.approx(np.cos(2 * np.pi * t / 30), abs=1e-6)


@pytest.mark.parametrize("cls", ALL)
def test_obs_in_space_extreme_policy(cls):
    env = InventoryEnv(Params(T=90), demand=cls())
    env.reset(seed=3)
    trunc = False
    while not trunc:
        obs, _, _, trunc, _ = env.step(40)
        assert env.observation_space.contains(obs)