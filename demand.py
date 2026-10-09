"""
Demand generators (all action-independent, all driven by a passed-in NumPy Generator).

Each generator has sample(rng, n, t0) -> int array of n daily demands, where day i has
calendar index t0 + i (t0 may be negative: pre-episode history).

1. StationaryPoisson   D ~ Poisson(20)
2. NegBin              D ~ NegBin(r=10, p=1/3)  -> mean 20, variance 60
                       (NumPy parameterisation: number of failures before r successes)
3. SeasonalShock       D ~ Poisson(lambda_t),
                       lambda_t = max(eps, 20 + 6 sin(2 pi t / 30) + J_t)

Shock process J_t  (PROVISIONAL -- the proposal only says "temporary positive demand-rate
shock", so these values are my assumptions and should be confirmed/changed):
  * on a day with no active shock, a new shock starts with probability shock_prob = 0.04
  * its size is Uniform(8, 16) extra demand-rate units, its duration is 3..7 days (uniform)
  * shocks do not overlap; J_t = 0 when no shock is active
"""
import numpy as np


class StationaryPoisson:
    name = "poisson"

    def __init__(self, mean=20.0):
        self.mean = mean

    def sample(self, rng, n, t0=0):
        return rng.poisson(self.mean, size=n)


class NegBin:
    name = "negbin"

    def __init__(self, r=10, p=1 / 3):
        self.r, self.p = r, p            # mean = r(1-p)/p, var = r(1-p)/p^2

    def sample(self, rng, n, t0=0):
        return rng.negative_binomial(self.r, self.p, size=n)


class SeasonalShock:
    name = "seasonal_shock"

    def __init__(self, base=20.0, amp=6.0, period=30, eps=1e-6,
                 shock_prob=0.04, shock_size=(8.0, 16.0), shock_dur=(3, 7)):
        self.base, self.amp, self.period, self.eps = base, amp, period, eps
        self.shock_prob, self.shock_size, self.shock_dur = shock_prob, shock_size, shock_dur

    def seasonal(self, n, t0=0):
        t = t0 + np.arange(n)
        return self.base + self.amp * np.sin(2 * np.pi * t / self.period)

    def shocks(self, rng, n):
        J = np.zeros(n)
        remaining, size = 0, 0.0
        for i in range(n):
            if remaining == 0 and rng.random() < self.shock_prob:
                size = rng.uniform(*self.shock_size)
                remaining = int(rng.integers(self.shock_dur[0], self.shock_dur[1] + 1))
            if remaining > 0:
                J[i] = size
                remaining -= 1
        return J

    def rates(self, rng, n, t0=0):
        return np.maximum(self.eps, self.seasonal(n, t0) + self.shocks(rng, n))

    def sample(self, rng, n, t0=0):
        return rng.poisson(self.rates(rng, n, t0))


REGIMES = {"poisson": StationaryPoisson, "negbin": NegBin, "seasonal_shock": SeasonalShock}