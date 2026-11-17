"""Evaluation metrics (all computed on the walk-forward live period only)."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .pool import ridge_fit


def labels_from_level(level: np.ndarray, h: int) -> np.ndarray:
    y = np.zeros_like(level)
    y[:-h] = level[h:] - level[:-h]
    return y


def level_from_labels(y: np.ndarray, h: int) -> np.ndarray:
    """Inverse of labels_from_level: any label sequence is realisable as h
    interleaved chains level[t+h] = level[t] + y[t]."""
    lv = np.zeros(len(y))
    for t in range(len(y) - h):
        lv[t + h] = lv[t] + y[t]
    return lv


def regime_oracle(Z: np.ndarray, y: np.ndarray, h: int, segments: Sequence[int], t0: int, l2: float = 10.0) -> np.ndarray:
    """In-hindsight ridge fitted separately on each regime segment (uses the
    future: it is a reference line for 'cost of delay', not a competitor)."""
    T = len(y)
    out = np.zeros(T)
    bounds = [t0] + [s for s in segments if s > t0] + [T - h]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 50:
            continue
        th, _ = ridge_fit(Z[a:b], y[a:b], l2)
        out[a:b] = Z[a:b] @ th
    return out


def mcc(pred: np.ndarray, truth: np.ndarray) -> float:
    tp = np.sum((pred > 0) & (truth > 0)); tn = np.sum((pred < 0) & (truth < 0))
    fp = np.sum((pred > 0) & (truth < 0)); fn = np.sum((pred < 0) & (truth > 0))
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float((tp * tn - fp * fn) / den) if den > 0 else 0.0


def forecast_metrics(y, yhat, q, calls, spread, t0, t1) -> Dict[str, float]:
    s = slice(t0, t1)
    yy, ff = y[s], yhat[s]
    e = yy - ff
    out = {
        "r2_oos": float(1 - np.sum(e ** 2) / max(np.sum(yy ** 2), 1e-12)),
        "mae": float(np.mean(np.abs(e))),
        "mse": float(np.mean(e ** 2)),
    }
    if q is not None:
        out["coverage"] = float(np.mean(np.abs(e) <= q[s]))
        out["mean_width"] = float(np.mean(2 * q[s]))
    if calls is not None:
        c = calls[s]
        m = c != 0
        nz = m & (yy != 0)
        out["call_rate"] = float(np.mean(m))
        out["hit_rate"] = float(np.mean(np.sign(yy[nz]) == c[nz])) if nz.any() else float("nan")
        out["mcc_calls"] = mcc(c[nz], yy[nz]) if nz.any() else float("nan")
        out["gross_edge_ticks"] = float(np.mean(c[m] * yy[m])) if m.any() else float("nan")
        out["half_spread_ticks"] = float(np.mean(0.5 * spread[s][m])) if m.any() else float("nan")
        out["wrong_calls"] = int(np.sum(nz & (np.sign(yy) != c)))
    return out


def gate_calls(yhat, q, t_ref0, t_live, budget):
    """Same ratio gate as the engine, for baselines (fixed interval width q)."""
    ratio = np.abs(yhat) / np.maximum(q, 1e-12)
    k = np.quantile(ratio[t_ref0:t_live], 1 - budget)
    calls = np.where(ratio > k, np.sign(yhat), 0.0)
    calls[:t_live] = 0.0
    return calls


def detection(alarms: List[int], tau: int, window: int) -> Optional[int]:
    hits = [a for a in alarms if tau <= a < tau + window]
    return (hits[0] - tau) if hits else None


def cost_of_delay(y, yhat, oracle, tau, W) -> float:
    s = slice(tau, tau + W)
    return float(np.sum((y[s] - yhat[s]) ** 2 - (y[s] - oracle[s]) ** 2))


def silent_failure_time(y, yhat, alarms, t_live, tau, t_end, win=200, ratio=1.5) -> Optional[float]:
    """Events from rolling MAE first exceeding `ratio` x its pre-change median
    to the first alarm. None if MAE never degrades. If no alarm: time to end."""
    ae = np.abs(y - yhat)
    cs = np.cumsum(np.r_[0, ae])
    roll = (cs[win:] - cs[:-win]) / win             # roll[i] = mean ae[i : i+win]
    pre = roll[t_live: max(tau - win, t_live + 1)]
    if pre.size == 0:
        return None
    base = np.median(pre)
    post_idx = np.arange(tau - win + 1, min(t_end - win, len(roll)))
    post_idx = post_idx[post_idx >= 0]
    bad = post_idx[roll[post_idx] > ratio * base]
    if bad.size == 0:
        return None
    t_fail = int(bad[0] + win)
    nxt = [a for a in alarms if a >= tau]
    t_alarm = nxt[0] if nxt else t_end
    return float(max(0, t_alarm - t_fail))
