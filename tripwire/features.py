"""Stage A: streaming feature extraction from the top 10 levels of the book.

Every feature at event t uses only the book at events <= t (causal by
construction). The same class drives the batch evaluation and the live
cockpit, so there is one implementation and no train/serve skew.

Book arrays use the LOBSTER convention for one side: prices[0] is the best
level. Prices are in currency units, sizes in shares/contracts.
"""

from __future__ import annotations

import math

import numpy as np

FEATURE_NAMES = [
    "ofi_l1", "ofi_l2", "ofi_l3", "ofi_l4", "ofi_l5",  # multi-level OFI / trailing depth (EWMA over ~h events)
    "ofi_deep_6_10",                                  # aggregated deep-book OFI, levels 6-10
    "microprice_minus_mid",                           # ticks
    "imb_l1", "imb_l1_3", "imb_l1_5", "imb_l1_10",    # depth imbalances
    "bid_logdepth_slope", "ask_logdepth_slope",       # log-depth slope across 10 levels
    "log_spread_ticks",
    "vol_ratio_20_200",                               # fast/slow realised mid volatility
    "past_return_h",                                  # mid_t - mid_{t-h} in ticks (already known)
]
N_FEATURES = len(FEATURE_NAMES)

_LOG_LEVELS = np.log(np.arange(1.0, 11.0))
_LOG_LEVELS_C = _LOG_LEVELS - _LOG_LEVELS.mean()
_LOG_LEVELS_SS = float(np.sum(_LOG_LEVELS_C ** 2))


def level_ofi(bp, bs, ap, as_, pbp, pbs, pap, pas):
    """Per-level order-flow imbalance (Cont, Kukanov & Stoikov 2014; Xu, Gould &
    Howison 2019). Vectorised over the 10 levels."""
    e_bid = np.where(bp >= pbp, bs, 0.0) - np.where(bp <= pbp, pbs, 0.0)
    e_ask = np.where(ap <= pap, as_, 0.0) - np.where(ap >= pap, pas, 0.0)
    return e_bid - e_ask


class FeatureExtractor:
    """Streaming 16-feature extractor. Call ``update`` once per book event."""

    def __init__(self, tick_size: float = 0.01, horizon: int = 10,
                 depth_halflife: int = 5000):
        self.tick = float(tick_size)
        self.h = int(horizon)
        self.a_ofi = 1.0 - math.exp(-1.0 / max(self.h, 1))        # OFI integrated over ~h events
        self.a_depth = 1.0 - math.exp(-math.log(2) / depth_halflife)
        self.a_fast = 2.0 / 21.0
        self.a_slow = 2.0 / 201.0
        self.reset()

    def reset(self) -> None:
        self.prev = None
        self.depth = None
        self.ofi_ewma = np.zeros(10)
        self.var_fast = 0.0
        self.var_slow = 0.0
        self.mid_hist = np.zeros(self.h + 1)
        self.n = 0
        self.x = np.zeros(N_FEATURES)

    def update(self, bid_p, bid_s, ask_p, ask_s):
        """Returns (features[16], mid_in_ticks, spread_in_ticks)."""
        bid_p = np.asarray(bid_p, dtype=float)
        bid_s = np.asarray(bid_s, dtype=float)
        ask_p = np.asarray(ask_p, dtype=float)
        ask_s = np.asarray(ask_s, dtype=float)
        tick = self.tick
        mid = 0.5 * (bid_p[0] + ask_p[0]) / tick
        spread = max((ask_p[0] - bid_p[0]) / tick, 1e-9)
        cur_depth = np.maximum(0.5 * (bid_s + ask_s), 1.0)

        if self.prev is None:
            self.depth = cur_depth.copy()
            ofi = np.zeros(10)
            dmid = 0.0
        else:
            pbp, pbs, pap, pas, pmid = self.prev
            ofi = level_ofi(bid_p, bid_s, ask_p, ask_s, pbp, pbs, pap, pas)
            self.depth += self.a_depth * (cur_depth - self.depth)
            dmid = mid - pmid
        self.prev = (bid_p, bid_s, ask_p, ask_s, mid)

        self.ofi_ewma += self.a_ofi * (ofi / self.depth - self.ofi_ewma)
        self.var_fast += self.a_fast * (dmid * dmid - self.var_fast)
        self.var_slow += self.a_slow * (dmid * dmid - self.var_slow)

        x = self.x
        x[0:5] = self.ofi_ewma[0:5]
        x[5] = float(np.mean(self.ofi_ewma[5:10]))
        v1 = bid_s[0] + ask_s[0]
        micro = (bid_s[0] * ask_p[0] + ask_s[0] * bid_p[0]) / max(v1, 1e-9)
        x[6] = micro / tick - mid
        cb = np.cumsum(bid_s)
        ca = np.cumsum(ask_s)
        x[7] = (cb[0] - ca[0]) / max(cb[0] + ca[0], 1e-9)
        x[8] = (cb[2] - ca[2]) / max(cb[2] + ca[2], 1e-9)
        x[9] = (cb[4] - ca[4]) / max(cb[4] + ca[4], 1e-9)
        x[10] = (cb[9] - ca[9]) / max(cb[9] + ca[9], 1e-9)
        x[11] = float(_LOG_LEVELS_C @ np.log(np.maximum(bid_s, 1.0))) / _LOG_LEVELS_SS
        x[12] = float(_LOG_LEVELS_C @ np.log(np.maximum(ask_s, 1.0))) / _LOG_LEVELS_SS
        x[13] = math.log(spread)
        x[14] = math.sqrt((self.var_fast + 1e-6) / (self.var_slow + 1e-6))
        self.mid_hist[self.n % (self.h + 1)] = mid
        if self.n >= self.h:
            x[15] = mid - self.mid_hist[(self.n - self.h) % (self.h + 1)]
        else:
            x[15] = 0.0
        self.n += 1
        return x.copy(), mid, spread


def extract_all(bid_p, bid_s, ask_p, ask_s, tick_size=0.01, horizon=10):
    """Run the streaming extractor over a whole (T, 10) book. Returns X, mid, spread."""
    T = bid_p.shape[0]
    fx = FeatureExtractor(tick_size=tick_size, horizon=horizon)
    X = np.empty((T, N_FEATURES))
    mid = np.empty(T)
    spr = np.empty(T)
    for t in range(T):
        X[t], mid[t], spr[t] = fx.update(bid_p[t], bid_s[t], ask_p[t], ask_s[t])
    return X, mid, spr
