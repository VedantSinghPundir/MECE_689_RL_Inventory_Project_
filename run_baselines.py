# from inventory_env import InventoryEnv
# from inventory_sim import Params
# from baselines import (AlwaysOrder, RandomOrder, BaseStock, evaluate,
#                        tune_base_stock, TEST_SEEDS, VAL_SEEDS)

# env = InventoryEnv(Params(T=90, L=2))

# best_B, scores = tune_base_stock(env, range(0, 121, 2), VAL_SEEDS)
# print(f"Base-stock tuned on validation seeds {VAL_SEEDS[0]}-{VAL_SEEDS[-1]}: B* = {best_B}")
# print("  validation profit vs B (every 10):",
#       {B: round(s) for B, s in scores.items() if B % 10 == 0})

# policies = [AlwaysOrder(0), AlwaysOrder(40), RandomOrder(0), BaseStock(best_B)]
# print(f"\nHeld-out test seeds {TEST_SEEDS[0]}-{TEST_SEEDS[-1]} ({len(TEST_SEEDS)} episodes, T=90)")
# print(f"{'policy':<22}{'profit (mean ±95%CI)':<26}{'fill rate':<11}{'stockout days':<15}{'avg end inv':<12}")
# for pol in policies:
#     r = evaluate(env, pol, TEST_SEEDS)
#     pm, _, pci = r["profit"]
#     print(f"{pol.name:<22}{pm:>9.0f} ± {pci:<12.0f}{r['fill_rate'][0]:<11.3f}"
#           f"{r['stockout_day_freq'][0]:<15.3f}{r['avg_end_inv'][0]:<12.1f}")
"""Baselines on all three demand regimes.

For each regime: tune base-stock level B on validation seeds, then evaluate on held-out test seeds.
Also shows 'transfer': the B tuned on stationary Poisson applied unchanged to the other regimes.
"""
from inventory_env import InventoryEnv
from inventory_sim import Params
from demand import StationaryPoisson, NegBin, SeasonalShock
from baselines import (AlwaysOrder, RandomOrder, BaseStock, evaluate,
                       tune_base_stock, TEST_SEEDS, VAL_SEEDS)

REGIMES = [("poisson", StationaryPoisson()), ("negbin", NegBin()), ("seasonal_shock", SeasonalShock())]
B_GRID = range(0, 161, 2)
params = Params(T=90, L=2)


def row(name, r):
    pm, _, pci = r["profit"]
    return (f"{name:<24}{pm:>9.0f} ± {pci:<8.0f}{r['fill_rate'][0]:<11.3f}"
            f"{r['stockout_day_freq'][0]:<15.3f}{r['avg_end_inv'][0]:<12.1f}")


header = f"{'policy':<24}{'profit (mean ±95%CI)':<20}{'fill rate':<11}{'stockout days':<15}{'avg end inv':<12}"
tuned = {}
for name, model in REGIMES:
    env = InventoryEnv(params, demand=model)
    B, scores = tune_base_stock(env, B_GRID, VAL_SEEDS)
    tuned[name] = B
    print(f"\n=== {name} === base-stock tuned on val seeds: B* = {B}")
    print(f"{len(TEST_SEEDS)} held-out test episodes, T=90")
    print(header)
    for pol in [AlwaysOrder(0), AlwaysOrder(40), RandomOrder(0), BaseStock(B)]:
        print(row(pol.name, evaluate(env, pol, TEST_SEEDS)))

print("\n=== transfer: B tuned on Poisson, applied unchanged ===")
print(f"{'regime':<18}{'B=Poisson-tuned':<26}{'B=own-tuned':<26}{'profit lost'}")
for name, model in REGIMES:
    env = InventoryEnv(params, demand=model)
    own = evaluate(env, BaseStock(tuned[name]), TEST_SEEDS)["profit"]
    tr = evaluate(env, BaseStock(tuned["poisson"]), TEST_SEEDS)["profit"]
    print(f"{name:<18}{tr[0]:>8.0f} (B={tuned['poisson']})      {own[0]:>8.0f} (B={tuned[name]})      {own[0]-tr[0]:>6.0f}")