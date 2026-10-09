"""Random Forest + order-up-to vs tuned base-stock, per demand regime.

Partitions:  RF fit -> TRAIN_SEEDS | tuning of B and s -> VAL_SEEDS | final numbers -> TEST_SEEDS
"""
import numpy as np
from inventory_env import InventoryEnv
from inventory_sim import Params
from demand import StationaryPoisson, NegBin, SeasonalShock
from baselines import (BaseStock, evaluate, tune_base_stock, profits, paired_diff,
                       TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS)
from rf_policy import RFForecaster, RFOrderUpTo, tune_safety, build_dataset

REGIMES = [("poisson", StationaryPoisson()), ("negbin", NegBin()), ("seasonal_shock", SeasonalShock())]
T, L, K, PERIOD = 90, 2, 7, 30
params = Params(T=T, L=L)


def row(name, r):
    pm, _, ci = r["profit"]
    return (f"{name:<24}{pm:>8.0f} ± {ci:<7.0f}{r['fill_rate'][0]:<11.3f}"
            f"{r['stockout_day_freq'][0]:<15.3f}{r['avg_end_inv'][0]:<12.1f}")


for name, model in REGIMES:
    env = InventoryEnv(params, demand=model, k=K)
    print(f"\n=== {name} ===")

    # 1) fit forecaster on TRAIN seeds, report forecast quality on TEST seeds
    fc = RFForecaster(T=T, L=L, k=K, period=PERIOD).fit(model, TRAIN_SEEDS)
    Xte, yte = build_dataset(model, TEST_SEEDS, T, L, K, PERIOD)
    pred = fc.predict_features(Xte)
    rmse_rf = np.sqrt(np.mean((pred - yte) ** 2))
    rmse_const = np.sqrt(np.mean((fc.train_target_mean - yte) ** 2))
    print(f"3-day demand forecast RMSE on test seeds: RF = {rmse_rf:.2f} | "
          f"constant-mean forecast = {rmse_const:.2f}  (R^2 = {1 - (rmse_rf/rmse_const)**2:.3f})")

    # 2) tune control parameters on VAL seeds only
    B, _ = tune_base_stock(env, range(0, 161, 2), VAL_SEEDS)
    s, _ = tune_safety(env, fc, range(-10, 41, 2), VAL_SEEDS)
    print(f"tuned on validation seeds: base-stock B* = {B}, RF safety s* = {s}")

    # 3) held-out evaluation
    bs, rf = BaseStock(B), RFOrderUpTo(fc, s)
    print(f"{len(TEST_SEEDS)} held-out test episodes, T={T}")
    print(f"{'policy':<24}{'profit (mean ±95%CI)':<17}{'fill rate':<11}{'stockout days':<15}{'avg end inv':<12}")
    print(row(bs.name, evaluate(env, bs, TEST_SEEDS)))
    print(row(rf.name, evaluate(env, rf, TEST_SEEDS)))
    d, ci = paired_diff(profits(env, rf, TEST_SEEDS), profits(env, bs, TEST_SEEDS))
    verdict = "RF better" if d - ci > 0 else ("base-stock better" if d + ci < 0 else "no clear difference")
    print(f"paired difference (RF - base-stock): {d:+.0f} ± {ci:.0f}  -> {verdict}")