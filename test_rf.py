import numpy as np
import pytest

from demand import StationaryPoisson, NegBin, SeasonalShock
from inventory_env import InventoryEnv
from inventory_sim import Params
from baselines import (BaseStock, run_episode, evaluate, VAL_SEEDS, TEST_SEEDS, TRAIN_SEEDS)
from rf_policy import (make_features, features_from_obs, targets_from_demands, build_dataset,
                       RFForecaster, RFOrderUpTo)

T, L, K, PERIOD = 90, 2, 7, 30


def test_seed_partitions_disjoint():
    tr, va, te = set(TRAIN_SEEDS), set(VAL_SEEDS), set(TEST_SEEDS)
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_targets_are_protection_period_sums():
    all_ = np.arange(T + K + L, dtype=float)
    y = targets_from_demands(all_, T, K, L)
    for t in (0, 1, 50, T - 1):
        assert y[t] == all_[K + t] + all_[K + t + 1] + all_[K + t + 2]   # days t, t+1, t+2


def test_features_use_only_past_demand():
    rng = np.random.default_rng(0)
    all_ = rng.poisson(20, T + K + L).astype(float)
    base = make_features(all_, T, K, PERIOD)
    for t in (0, 10, 45, T - 1):
        changed = all_.copy()
        changed[K + t:] = 999.0              # scramble today and the whole future
        assert np.allclose(make_features(changed, T, K, PERIOD)[t], base[t])


@pytest.mark.parametrize("model", [StationaryPoisson(), SeasonalShock()])
def test_training_features_equal_serving_features(model):
    env = InventoryEnv(Params(T=T, L=L), demand=model, k=K)
    obs, _ = env.reset(seed=4)
    batch = make_features(env._all, T, K, PERIOD)
    for t in range(T):
        assert np.allclose(features_from_obs(obs, L, K, env.scale), batch[t], atol=1e-4), f"t={t}"
        obs, *_ = env.step(10)


def test_dataset_shapes():
    X, y = build_dataset(StationaryPoisson(), range(5), T, L, K, PERIOD)
    assert X.shape == (5 * T, K + 3) and y.shape == (5 * T,)
    assert abs(y.mean() - 60) < 3          # 3 days of Poisson(20)


def test_forecaster_deterministic():
    a = RFForecaster(n_estimators=10).fit(StationaryPoisson(), range(3000, 3020))
    b = RFForecaster(n_estimators=10).fit(StationaryPoisson(), range(3000, 3020))
    env = InventoryEnv(Params(T=T, L=L), k=K); env.reset(seed=1)
    assert np.allclose(a.forecast_path(env), b.forecast_path(env))


class ConstForecaster:
    """Stub: always forecasts 60 -> RF policy with s=8 must equal BaseStock(68)."""
    T = T
    def forecast_path(self, env): return np.full(T, 60.0)


def test_order_up_to_rule_matches_base_stock_exactly():
    env = InventoryEnv(Params(T=T, L=L), demand=NegBin())
    for seed in (1, 2, 3):
        a = run_episode(env, RFOrderUpTo(ConstForecaster(), 8), seed)
        b = run_episode(env, BaseStock(68), seed)
        assert a == b


@pytest.mark.parametrize("model", [StationaryPoisson(), NegBin(), SeasonalShock()])
def test_rf_policy_runs_valid_actions_all_regimes(model):
    fc = RFForecaster(n_estimators=10).fit(model, range(3000, 3030))
    env = InventoryEnv(Params(T=T, L=L), demand=model)
    pol = RFOrderUpTo(fc, 8)
    obs, _ = env.reset(seed=7); pol.reset(); pol.prepare(env)
    sim = env.unwrapped.sim
    trunc, executed = False, []
    while not trunc:
        a = pol.act(sim.inventory, tuple(sim.pipeline), 40, obs)
        assert 0 <= a <= 40
        obs, _, _, trunc, info = env.step(a)
        executed.append(info["executed_order"])
    assert executed[-1] == 0 and executed[-2] == 0       # terminal rule still applied


def test_batch_forecast_equals_stepwise_prediction():
    """The cached batch forecast must equal predicting one step at a time from observations."""
    model = SeasonalShock()
    fc = RFForecaster(n_estimators=20).fit(model, range(3000, 3040))
    env = InventoryEnv(Params(T=T, L=L), demand=model)
    obs, _ = env.reset(seed=9)
    batch = fc.forecast_path(env)
    for t in range(0, T, 9):
        x = features_from_obs(obs, L, K, env.scale)[None, :]
        step_pred = fc.rf.predict(x)[0]
        assert step_pred == pytest.approx(batch[t], abs=1e-3)
        for _ in range(9):
            if env.unwrapped.sim.t < T:
                obs, *_ = env.step(10)