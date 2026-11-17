"""Baselines B1-B5. All see the same standardised features, the same labels
with the same h-event delay, the same walk-forward start and the same call
gate as TRIPWIRE, so differences come from the method, not the plumbing.

B1 frozen ridge (fit block only)
B2 sliding-window ridge, refit every `refit_every` events on the last `window` matured labels
B3 rolling two-sample KS on every feature (Bonferroni) -> retrain on post-alarm data
B4 RLS with one fixed forgetting factor; the best factor is picked *in hindsight* (favours B4)
B5 oracle-timed reset: knows the true change times, restarts learning there (not available live)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .pool import ridge_fit


def ks_crit(n: int, m: int, alpha: float) -> float:
    c = math.sqrt(-0.5 * math.log(alpha / 2.0))
    return c * math.sqrt((n + m) / (n * m))


def ks_stat(a_sorted: np.ndarray, b: np.ndarray) -> float:
    b = np.sort(b)
    allv = np.concatenate([a_sorted, b])
    fa = np.searchsorted(a_sorted, allv, side="right") / a_sorted.size
    fb = np.searchsorted(b, allv, side="right") / b.size
    return float(np.max(np.abs(fa - fb)))


def _rls_path(Z, y, h, t0, lam, theta0, P0, resets=(), l2=10.0, y_clip=1e9):
    """Online RLS predictions with h-delayed updates; optional reset times."""
    T, d = Z.shape
    th = theta0.copy()
    P = P0.copy()
    yhat = np.zeros(T)
    resets = set(int(r) for r in resets)
    for t in range(t0, T):
        if t in resets:
            th = np.zeros(d)
            P = np.eye(d) / l2
        j = t - h
        if j >= t0 - h and j >= 0 and (not resets or j >= max([r for r in resets if r <= t], default=-1)):
            x = Z[j]
            Px = P @ x
            g = Px / (lam + x @ Px)
            err = max(-y_clip, min(y_clip, y[j] - th @ x))
            th = th + g * err
            P = (P - np.outer(g, Px)) / lam
            tr = np.trace(P)
            if tr > 10 * d:
                P *= 10 * d / tr
        yhat[t] = th @ Z[t]
    return yhat


def run_baselines(Z: np.ndarray, y: np.ndarray, h: int, t_fit: int, n_fit: int,
                  change_points: Optional[List[int]] = None, l2: float = 10.0,
                  lambdas=(0.99, 0.998, 0.9995, 0.9999, 0.99995, 1.0),
                  window: int = 3000, refit_every: int = 200,
                  ks_ref: int = 3000, ks_cur: int = 600, ks_every: int = 100, ks_alpha: float = 0.01,
                  t_eval: int = 0) -> Dict[str, Dict]:
    """Z: standardised features (T,d); y: label of event t (level[t+h]-level[t]),
    only used at event t+h. t_fit: first event with predictions; n_fit: fit block length."""
    T, d = Z.shape
    theta0, P0 = ridge_fit(Z[:n_fit], y[:n_fit], l2)
    y_clip = 8.0 * float(np.sqrt(np.mean(y[:n_fit] ** 2)))
    out: Dict[str, Dict] = {}

    # B1
    yh = Z @ theta0
    yh[:t_fit] = 0.0
    out["B1 frozen ridge"] = {"yhat": yh, "alarms": []}

    # B2
    yh = np.zeros(T)
    th = theta0.copy()
    for t in range(t_fit, T):
        if (t - t_fit) % refit_every == 0 and t - h > n_fit:
            lo = max(0, t - h - window)
            th, _ = ridge_fit(Z[lo:t - h], y[lo:t - h], l2)
        yh[t] = th @ Z[t]
    out["B2 sliding-window ridge"] = {"yhat": yh, "alarms": []}

    # B3: rolling KS on every feature, Bonferroni over d features
    yh = np.zeros(T)
    th = theta0.copy()
    alarms = []
    crit = ks_crit(ks_ref, ks_cur, ks_alpha / d)
    last_alarm = None
    cool = 0
    for t in range(t_fit, T):
        if t >= ks_ref + ks_cur and (t - t_fit) % ks_every == 0 and cool <= 0:
            ref = Z[t - ks_ref - ks_cur: t - ks_cur]
            cur = Z[t - ks_cur: t]
            for k in range(d):
                if ks_stat(np.sort(ref[:, k]), cur[:, k]) > crit:
                    alarms.append(t)
                    last_alarm = t
                    cool = ks_ref          # wait for a clean post-change reference
                    break
        cool -= 1
        if last_alarm is not None and (t - last_alarm) % refit_every == 0:
            n_post = t - h - last_alarm
            if n_post >= 300:
                lo = max(last_alarm, t - h - window)
                th, _ = ridge_fit(Z[lo:t - h], y[lo:t - h], l2)
        yh[t] = th @ Z[t]
    out["B3 rolling-KS + retrain"] = {"yhat": yh, "alarms": alarms}

    # B4: best fixed forgetting factor in hindsight
    best = None
    for lam in lambdas:
        p = _rls_path(Z, y, h, t_fit, lam, theta0, P0, l2=l2, y_clip=y_clip)
        a = max(t_eval, t_fit)
        mse = float(np.mean((y[a:T - h] - p[a:T - h]) ** 2))
        if best is None or mse < best[0]:
            best = (mse, lam, p)
    out["B4 best fixed-forgetting RLS"] = {"yhat": best[2], "alarms": [], "lambda": best[1]}

    # B5: oracle-timed reset
    if change_points:
        p = _rls_path(Z, y, h, t_fit, 1.0, theta0, P0, resets=change_points, l2=l2, y_clip=y_clip)
        out["B5 oracle-timed reset"] = {"yhat": p, "alarms": list(change_points)}
    return out
