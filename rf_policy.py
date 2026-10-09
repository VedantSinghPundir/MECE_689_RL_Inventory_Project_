"""
Random Forest (RF) + inventory control comparator.

Pipeline (as in the proposal):
  1. FORECAST   RF predicts total demand over the protection period (days t .. t+L, i.e. lead
                time + 1 review day) from decision-time information only:
                last k demands, their mean, and sin/cos seasonal features.
  2. CONTROL    order-up-to rule:  B_t = forecast_t + s,   q_t = clip(round(B_t - IP_t), 0, 40)
                with inventory position IP_t = on-hand + pipeline.
  3. PARTITIONS RF is fitted on TRAIN seeds; the safety margin s is tuned on VAL seeds;
                evaluation uses held-out TEST seeds (see baselines.py).

Efficiency note: demand is action-independent, so the forecast at day t depends only on past
demand, not on the policy. prepare(env) computes all T forecasts of an episode in ONE batch,
using only lags available at each decision time (tests verify no future leakage and that the
batch equals what you would get stepwise from the observation).
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor


# ---------------------------------------------------------------- features / targets
def make_features(all_, T, k, period):
    """Features for decision days t = 0..T-1 from a demand path with k history days in front.
    Day t's features use all_[t : t+k] only (the k demands before day t)."""
    t = np.arange(T)
    lags = np.stack([all_[k + t - 1 - i] for i in range(k)], axis=1).astype(float)  # D_{t-1}..D_{t-k}
    ang = 2 * np.pi * t / period
    return np.column_stack([lags, lags.mean(axis=1), np.sin(ang), np.cos(ang)])


def features_from_obs(obs, L, k, scale, period=None):
    """Same features, rebuilt from a single env observation (checks train/serve consistency)."""
    lags = np.asarray(obs[L:L + k], dtype=float) * scale
    return np.concatenate([lags, [lags.mean()], [obs[-2], obs[-1]]])


def targets_from_demands(all_, T, k, L):
    """Target for day t = sum of demand on days t..t+L (protection period)."""
    csum = np.concatenate([[0.0], np.cumsum(all_)])
    start = k + np.arange(T)
    return csum[start + L + 1] - csum[start]


def build_dataset(demand_model, seeds, T, L, k, period):
    X, y = [], []
    for s in seeds:
        all_ = demand_model.sample(np.random.default_rng(s), T + k + L, t0=-k)
        X.append(make_features(all_, T, k, period))
        y.append(targets_from_demands(all_, T, k, L))
    return np.vstack(X), np.concatenate(y)


# ---------------------------------------------------------------- forecaster
class RFForecaster:
    def __init__(self, T=90, L=2, k=7, period=30, n_estimators=100, min_samples_leaf=10, seed=0):
        self.T, self.L, self.k, self.period = T, L, k, period
        self.rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                        random_state=seed, n_jobs=-1)
        self._cache = {}

    def fit(self, demand_model, train_seeds):
        X, y = build_dataset(demand_model, train_seeds, self.T, self.L, self.k, self.period)
        self.rf.fit(X, y)
        self.train_target_mean = float(y.mean())
        self._cache = {}
        return self

    def predict_features(self, X):
        return self.rf.predict(X)

    def forecast_path(self, env):
        """Forecasts for decision days 0..T-1 of the env's current episode (batch, past-only)."""
        all_ = env.unwrapped._all
        key = all_.tobytes()
        if key not in self._cache:
            X = make_features(all_, self.T, self.k, self.period)
            self._cache[key] = self.rf.predict(X)
        return self._cache[key]


# ---------------------------------------------------------------- policy
class RFOrderUpTo:
    def __init__(self, forecaster, safety):
        self.f, self.s = forecaster, safety
        self.name = f"RF+order-up-to(s={safety})"

    def reset(self): pass

    def prepare(self, env):
        self.F = self.f.forecast_path(env)

    def act(self, inv, pipe, max_order, obs=None):
        t = int(round(float(obs[-3]) * self.f.T))        # day index from the observation
        ip = inv + sum(pipe)
        return int(np.clip(round(self.F[t] + self.s - ip), 0, max_order))


def tune_safety(env, forecaster, s_grid, seeds):
    from baselines import evaluate
    scores = {s: evaluate(env, RFOrderUpTo(forecaster, s), seeds)["profit"][0] for s in s_grid}
    best = max(scores, key=scores.get)
    return best, scores