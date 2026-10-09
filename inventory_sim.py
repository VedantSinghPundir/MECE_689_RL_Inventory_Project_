"""
Version-0 inventory MDP (plain Python, no Gymnasium / NumPy yet).

Timing convention (one day t):
  1. Orders due at the start of day t arrive.           -> I_t (on-hand after arrivals)
  2. Agent observes s_t = (I_t, P_t, t/T) and proposes q_t in {0..max_order}.
  3. Terminal rule: executed q_t = 0 if t + L >= T (it could never arrive in-episode).
  4. Demand D_t is realised.
  5. Sales S_t = min(I_t, D_t); unmet U_t = D_t - S_t (lost sales, no backlog).
  6. Reward r_t = v*S_t - c*q_exec - h*I_end - b*U_t, with I_end = I_t - S_t.
  7. Pipeline advances. Order placed on day t arrives at the START of day t + L.

Pipeline representation
  P_t has L-1 entries: P_t[j] arrives at the start of day t+1+j  (already-placed orders).
  Internally, after ordering, the queue is P_t + [q_exec] (length L);
  queue[0] arrives at the start of day t+1.

Termination vs truncation: the episode never terminates; it is truncated after T days.
"""
from dataclasses import dataclass


@dataclass
class Params:
    T: int = 90            # horizon (days)
    L: int = 2             # deterministic lead time (days), L >= 1
    max_order: int = 40    # action space {0..max_order}
    v: float = 10.0        # unit selling price
    c: float = 4.0         # unit procurement cost
    h: float = 1.0         # holding cost per unit of end-of-day inventory
    b: float = 6.0         # extra shortage penalty per unmet unit


class InventorySim:
    def __init__(self, params=None):
        self.p = params or Params()
        assert self.p.L >= 1 and self.p.T > 0

    # ---- helpers -------------------------------------------------------
    def reset(self, init_inventory=25, init_pipeline=None):
        L = self.p.L
        pipe = list(init_pipeline) if init_pipeline is not None else [0] * (L - 1)
        assert len(pipe) == L - 1, "pipeline must have L-1 entries"
        self.t = 0
        self.inventory = init_inventory      # I_0
        self.pipeline = pipe                 # P_0
        self.total_executed_orders = 0
        self.total_arrived = 0
        self.initial_stock = init_inventory + sum(pipe)
        return self.observation()

    def observation(self):
        """s_t = (I_t, P_t[0..L-2], t/T)  -- only decision-time information."""
        return (self.inventory, *self.pipeline, self.t / self.p.T)

    def executed_order(self, action):
        """Common terminal rule: zero executed order if it cannot arrive in-episode."""
        return 0 if self.t + self.p.L >= self.p.T else action

    # ---- one step ------------------------------------------------------
    def step(self, action, demand):
        p = self.p
        assert 0 <= action <= p.max_order and float(action).is_integer(), "invalid action"
        assert demand >= 0
        assert self.t < p.T, "episode already truncated"

        q = self.executed_order(action)
        sales = min(self.inventory, demand)
        unmet = demand - sales
        end_inv = self.inventory - sales
        reward = p.v * sales - p.c * q - p.h * end_inv - p.b * unmet

        info = dict(day=self.t, inventory_at_decision=self.inventory,
                    pipeline_at_decision=tuple(self.pipeline),
                    proposed_order=action, executed_order=q, demand=demand,
                    sales=sales, unmet=unmet, end_inventory=end_inv, reward=reward)

        # advance pipeline: queue[0] arrives at start of next day
        queue = self.pipeline + [q]
        arrival, self.pipeline = queue[0], queue[1:]
        self.inventory = end_inv + arrival
        self.total_executed_orders += q
        self.total_arrived += arrival
        self.t += 1

        terminated = False                    # the warehouse never "ends"
        truncated = self.t >= p.T             # time limit only
        return self.observation(), reward, terminated, truncated, info