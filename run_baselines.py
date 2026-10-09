from inventory_env import InventoryEnv
from inventory_sim import Params
from baselines import (AlwaysOrder, RandomOrder, BaseStock, evaluate,
                       tune_base_stock, TEST_SEEDS, VAL_SEEDS)

env = InventoryEnv(Params(T=90, L=2))

best_B, scores = tune_base_stock(env, range(0, 121, 2), VAL_SEEDS)
print(f"Base-stock tuned on validation seeds {VAL_SEEDS[0]}-{VAL_SEEDS[-1]}: B* = {best_B}")
print("  validation profit vs B (every 10):",
      {B: round(s) for B, s in scores.items() if B % 10 == 0})

policies = [AlwaysOrder(0), AlwaysOrder(40), RandomOrder(0), BaseStock(best_B)]
print(f"\nHeld-out test seeds {TEST_SEEDS[0]}-{TEST_SEEDS[-1]} ({len(TEST_SEEDS)} episodes, T=90)")
print(f"{'policy':<22}{'profit (mean ±95%CI)':<26}{'fill rate':<11}{'stockout days':<15}{'avg end inv':<12}")
for pol in policies:
    r = evaluate(env, pol, TEST_SEEDS)
    pm, _, pci = r["profit"]
    print(f"{pol.name:<22}{pm:>9.0f} ± {pci:<12.0f}{r['fill_rate'][0]:<11.3f}"
          f"{r['stockout_day_freq'][0]:<15.3f}{r['avg_end_inv'][0]:<12.1f}")