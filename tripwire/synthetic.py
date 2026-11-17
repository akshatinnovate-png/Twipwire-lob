"""Synthetic 10-level book with known regime changes (the 2x2 benchmark).

The latent mid moves by  dm_{t+1} = (1/h) * sum_{k<h} s_{t-k} + sigma_n * eps,
with signal s_t = beta_imb * imbalance_t + beta_ofi * ofi_t computed from the
generated book, so the h-event forward change contains beta*s_t plus noise.
Quoted prices sit on a tick grid, so the observed mid moves in half ticks,
as in a real large-tick book. Signal-to-noise is set so the out-of-sample R^2
is roughly 5-10 %, an order of magnitude above real LOBSTER data but far from
the near-noiseless toy that makes every method look perfect.

Regime knobs:
  scale   - size of order-flow / depth fluctuations and spread (P(X) shift)
  beta_*  - how the book maps into future price moves     (P(Y|X) shift)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List

import numpy as np


@dataclass
class Regime:
    name: str = "calm"
    beta_imb: float = 0.6
    beta_ofi: float = 0.6
    scale: float = 1.0
    spread_mean: float = 1.4
    noise: float = 0.75
    regime_id: int = 0


CALM = Regime()
COVARIATE = replace(CALM, name="covariate", scale=2.2, spread_mean=2.6, regime_id=0)
CONCEPT = replace(CALM, name="concept", beta_imb=-0.6, beta_ofi=-0.6, regime_id=1)
COUPLED = replace(CALM, name="coupled", beta_imb=-0.6, beta_ofi=-0.6, scale=2.2, spread_mean=2.6, regime_id=1)


class SyntheticBook:
    """Streaming generator; `regime` can be changed between calls (the live
    cockpit's shock buttons do exactly that)."""

    def __init__(self, horizon: int = 10, tick: float = 0.01, seed: int = 0, regime: Regime = CALM):
        self.h = horizon
        self.tick = tick
        self.rng = np.random.default_rng(seed)
        self.regime = regime
        self.base = 400.0 * np.exp(0.12 * np.arange(10))
        self.logdev_b = np.zeros(10)
        self.logdev_a = np.zeros(10)
        self.sig_hist = np.zeros(horizon)
        self.mid = 10000.0
        self.n = 0
        self.spread = 1
        self.prev_bs = None
        self.prev_as = None

    def step(self):
        g = self.regime
        rng = self.rng
        # depth: AR(1) log-deviations with a common (imbalance) factor plus idiosyncratic noise
        common = rng.normal() * 0.35 * g.scale
        decay = np.exp(-0.12 * np.arange(10))
        self.logdev_b = 0.94 * self.logdev_b + 0.10 * g.scale * rng.normal(size=10) + common * decay * 0.25
        self.logdev_a = 0.94 * self.logdev_a + 0.10 * g.scale * rng.normal(size=10) - common * decay * 0.25
        bs = np.round(self.base * np.exp(np.clip(self.logdev_b, -3, 3)))
        as_ = np.round(self.base * np.exp(np.clip(self.logdev_a, -3, 3)))
        imb = (bs[0] - as_[0]) / (bs[0] + as_[0])
        if self.prev_bs is None:
            ofi = 0.0
        else:
            ofi = ((bs[0] - self.prev_bs[0]) - (as_[0] - self.prev_as[0])) / self.base[0]
        self.prev_bs, self.prev_as = bs, as_
        s = g.beta_imb * imb * 0.55 + g.beta_ofi * ofi * 0.40
        self.sig_hist[self.n % self.h] = s
        drift = float(self.sig_hist.sum()) / self.h if self.n >= self.h else 0.0
        self.mid += drift + g.noise * rng.normal()
        # spread in whole ticks, wider and noisier when scale is high
        if rng.random() < 0.05:
            self.spread = max(1, int(round(rng.exponential(g.spread_mean - 0.5) + 0.5)))
        half = 0.5 * self.spread
        bid1 = np.floor(self.mid - half + 0.5)       # tick grid
        ask1 = bid1 + self.spread
        lv = np.arange(10)
        bp = (bid1 - lv) * self.tick
        ap = (ask1 + lv) * self.tick
        self.n += 1
        return bp, bs, ap, as_, s


def scenario_regimes(name: str):
    """Returns list of (start_fraction, Regime)."""
    if name == "calm":
        return [(0.0, CALM)]
    if name == "covariate":
        return [(0.0, CALM), (0.5, COVARIATE)]
    if name == "concept":
        return [(0.0, CALM), (0.5, CONCEPT)]
    if name == "coupled":
        return [(0.0, CALM), (0.5, COUPLED)]
    if name == "recurrence":
        return [(0.0, CALM), (0.25, CONCEPT), (0.45, COVARIATE), (0.65, CONCEPT), (0.85, CALM)]
    raise ValueError(name)


SCENARIOS = ["calm", "covariate", "concept", "coupled"]


def generate(name: str, T: int = 40000, horizon: int = 10, seed: int = 0) -> Dict:
    plan = scenario_regimes(name)
    starts = [int(f * T) for f, _ in plan]
    gen = SyntheticBook(horizon=horizon, seed=seed, regime=plan[0][1])
    bp = np.empty((T, 10)); bs = np.empty((T, 10)); ap = np.empty((T, 10)); as_ = np.empty((T, 10))
    rid = np.empty(T, dtype=int)
    sig = np.empty(T)
    k = 0
    for t in range(T):
        while k + 1 < len(plan) and t >= starts[k + 1]:
            k += 1
            gen.regime = plan[k][1]
        bp[t], bs[t], ap[t], as_[t], sig[t] = gen.step()
        rid[t] = plan[k][1].regime_id
    changes: List[int] = starts[1:]
    kinds = [plan[i][1].name for i in range(1, len(plan))]
    return dict(bid_p=bp, bid_s=bs, ask_p=ap, ask_s=as_, regime_id=rid, signal=sig,
                change_points=changes, change_kinds=kinds, tick=0.01, name=name)
