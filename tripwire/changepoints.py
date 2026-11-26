"""Offline (hindsight) change points on realised volatility: event labels that never come
from TRIPWIRE. Binary segmentation of the per-minute log realised volatility for changes in
mean, with a BIC-style penalty. Used only to label evaluation windows, never by the guard."""

from __future__ import annotations

import numpy as np


def minute_log_vol(ts: np.ndarray, mid: np.ndarray, bucket: float = 60.0):
    """Per-bucket log realised volatility of mid changes. Returns (bucket_start_times, values)."""
    if len(ts) < 3:
        return np.zeros(0), np.zeros(0)
    b = np.floor((ts - ts[0]) / bucket).astype(int)
    r2 = np.r_[0.0, np.diff(mid) ** 2]
    nb = b[-1] + 1
    rv = np.bincount(b, weights=r2, minlength=nb)
    return ts[0] + np.arange(nb) * bucket, np.log(rv + 1e-12)


def binary_segmentation(x: np.ndarray, min_size: int = 10, penalty: float = None, max_cp: int = 50):
    """Mean-shift change points of x. penalty defaults to the BIC 2*sigma^2*log(n)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * min_size:
        return []
    sigma2 = np.var(np.diff(x)) / 2 + 1e-12          # robust to the mean shifts themselves
    pen = 2 * sigma2 * np.log(n) if penalty is None else penalty
    cps = []

    def best_split(a, b):
        seg = x[a:b]
        m = len(seg)
        if m < 2 * min_size:
            return None, 0.0
        cs = np.cumsum(seg)
        k = np.arange(min_size, m - min_size + 1)
        left = cs[k - 1] / k
        right = (cs[-1] - cs[k - 1]) / (m - k)
        gain = k * (m - k) / m * (left - right) ** 2
        i = int(np.argmax(gain))
        return a + int(k[i]), float(gain[i])

    segs = [(0, n)]
    while segs and len(cps) < max_cp:
        cand = [(best_split(a, b), (a, b)) for a, b in segs]
        cand = [(c, s) for c, s in cand if c[0] is not None]
        if not cand:
            break
        (cp, gain), (a, b) = max(cand, key=lambda z: z[0][1])
        if gain < pen:
            break
        cps.append(cp)
        segs.remove((a, b))
        segs += [(a, cp), (cp, b)]
    return sorted(cps)


def volatility_change_times(ts, mid, bucket=60.0, min_minutes=10):
    t0, lv = minute_log_vol(ts, mid, bucket)
    return [float(t0[i]) for i in binary_segmentation(lv, min_size=min_minutes)]
