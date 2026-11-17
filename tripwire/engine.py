"""TRIPWIRE-LOB engine: one call per book event.

Causality contract (tested in tests/test_causality.py):
* the label for event t is level[t+h] - level[t]; it is used only at event t+h
  ("matured label"). Expert updates, fixed-share weights, Monitor R and the
  interval never see an unmatured label.
* Monitor F uses only the current feature vector, so it can react inside the
  label delay.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, Optional

import numpy as np

from .aci import QuantileTracker
from .config import TripwireConfig
from .conformal import ConformalSR
from .pool import ExpertPool, ridge_fit

OUT_FIELDS = ["yhat", "q", "call", "logR_f", "logR_r", "alarm_f", "alarm_r",
              "y_matured", "live", "infl"]


class TripwireEngine:
    def __init__(self, cfg: Optional[TripwireConfig] = None, d: int = 16):
        self.cfg = cfg or TripwireConfig()
        self.d = d
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        c = self.cfg
        h = c.horizon
        self.t = 0
        self.L = h + 1
        self.buf_x = np.zeros((self.L, self.d))
        self.buf_preds = np.zeros((self.L, 1 + len(c.rls_lambdas) + c.max_spawned))
        self.buf_yhat = np.zeros(self.L)
        self.buf_q = np.zeros(self.L)
        self.buf_level = np.zeros(self.L)
        self.fit_X = []
        self.fit_level = []
        self.fitted = False
        self.t_live = None
        self.pool = ExpertPool(self.d, c.rls_lambdas, c.max_spawned, c.spawn_lambda,
                               c.fs_eta, c.fs_alpha, c.loss_clip)
        A_f, A_r = c.thresholds()
        sf, sr = c.strides()
        self.mon_f = ConformalSR(A_f, c.f_ref_size, c.f_ref_gap, c.f_min_ref, sf, seed=c.seed + 11, enabled=c.enable_f, min_ref_rearm=c.f_min_ref_rearm)
        self.mon_r = ConformalSR(A_r, c.ref_size, c.ref_gap, c.min_ref, sr, seed=c.seed + 23, enabled=c.enable_r, min_ref_rearm=c.min_ref_rearm)
        self.recent_X = deque(maxlen=c.spawn_window)
        self.recent_y = deque(maxlen=c.spawn_window)
        self.boost_left = 0
        self.infl_left = 0
        self.tracker = None
        self.k_star = None
        self.gate_ratios = []
        self.var_y = 1.0
        self.var_yhat = 1e-6
        self.var_f = 1e-6
        self.ref_slot = 1 + int(np.argmax(c.rls_lambdas))   # longest-memory RLS expert
        self.a_scale = 1.0 - math.exp(-math.log(2) / c.scale_halflife)
        self.alarms_f = []
        self.alarms_r = []
        self.out = np.zeros(len(OUT_FIELDS))

    def _standardise(self, x_raw: np.ndarray) -> np.ndarray:
        z = (x_raw - self.mu) / self.sd
        return np.clip(z, -self.cfg.clip_z, self.cfg.clip_z)

    def _fit(self) -> None:
        c = self.cfg
        h = c.horizon
        Xr = np.asarray(self.fit_X[: c.n_fit])
        lv = np.asarray(self.fit_level)
        y = lv[h: c.n_fit + h] - lv[: c.n_fit]
        self.mu = Xr.mean(0)
        self.sd = Xr.std(0) + 1e-9
        Z = np.clip((Xr - self.mu) / self.sd, -c.clip_z, c.clip_z)
        theta, Pinv = ridge_fit(Z, y, c.ridge_l2)
        self.pool.init_from_ridge(theta, Pinv)
        cov = np.cov(Z.T)
        shrunk = 0.7 * cov + 0.3 * np.diag(np.diag(cov)) + 1e-3 * np.eye(self.d)
        self.prec = np.linalg.inv(shrunk)
        self.var_y = float(np.mean(y ** 2)) + 1e-9
        res = np.abs(y - Z @ theta)
        self.var_yhat = float(np.mean((Z @ theta) ** 2)) + 1e-12
        self.var_f = self.var_yhat
        q0 = float(np.quantile(res, 1 - c.alpha))
        self.tracker = QuantileTracker(c.alpha, c.aci_gamma if c.enable_aci else 0.0, q0, float(res.mean()))
        self.y_clip = 8.0 * math.sqrt(self.var_y)
        # standardise the buffered not-yet-matured events too
        for j in range(self.L):
            self.buf_x[j] = self._standardise(self.buf_x[j])
        self.fitted = True
        self.t_fit_done = self.t
        self.fit_X = None

    # ------------------------------------------------------------------ step
    def step(self, x_raw: np.ndarray, mid: float, spread: float, level: Optional[float] = None) -> np.ndarray:
        """Process one book event. `level` is the series being forecast
        (defaults to mid, in ticks). Returns self.out (see OUT_FIELDS)."""
        c = self.cfg
        h = c.horizon
        t = self.t
        lvl = mid if level is None else level
        ptr = t % self.L
        out = self.out
        out[:] = 0.0
        alarm_f = alarm_r = False

        if not self.fitted:
            self.fit_X.append(np.array(x_raw, dtype=float))
            self.fit_level.append(lvl)
            self.buf_x[ptr] = x_raw
            self.buf_level[ptr] = lvl
            if t + 1 >= c.n_fit + h:
                self._fit()
            self.t += 1
            return out

        z = self._standardise(np.asarray(x_raw, dtype=float))
        pool = self.pool

        # ---- 1. matured label (event t-h): supervised updates only here
        if t - h >= self.t_fit_done - h:
            j = (t - h) % self.L
            y = lvl - self.buf_level[j]
            out[7] = y
            x_old = self.buf_x[j]
            preds_old = self.buf_preds[j]
            yhat_old = self.buf_yhat[j]
            r = y - yhat_old
            self.var_y += self.a_scale * (y * y - self.var_y)
            self.var_yhat += self.a_scale * (yhat_old * yhat_old - self.var_yhat)
            sy = math.sqrt(self.var_y)

            alpha = c.fs_alpha_boost if self.boost_left > 0 else c.fs_alpha
            pool.fixed_share(preds_old, y, sy, alpha)
            pool.rls_update(x_old, y, self.y_clip)
            self.recent_X.append(x_old.copy())
            self.recent_y.append(y)

            if c.monitor_r_score == "abs":
                score = abs(r) / sy
            else:
                # orthogonality: a calibrated least-squares forecaster has E[(y - f) f] = 0.
                # A harmful concept shift makes the errors anti-aligned with the forecast.
                f = preds_old[self.ref_slot] if c.monitor_r_ref == "slow" else yhat_old
                self.var_f += self.a_scale * (f * f - self.var_f)
                score = -((y - f) * f) / (sy * math.sqrt(self.var_f) + 1e-12)
            if self.mon_r.update(score):
                alarm_r = True
                self.alarms_r.append(t)
                self.boost_left = c.boost_events
                pool.spawn(np.asarray(self.recent_X), np.asarray(self.recent_y), c.ridge_l2,
                           c.spawn_prior_weight, t)
                k = int(np.flatnonzero(pool.spawn_birth == t)[0])
                self.buf_preds[:, k] = self.buf_x @ pool.theta[k]
            if self.boost_left > 0:
                self.boost_left -= 1
            self.tracker.update(abs(r), self.buf_q[j], c.aci_gamma_boost if self.boost_left > 0 else 1.0)

        # ---- 2. Monitor F on the current features (label-free)
        dz = z
        score_f = float(dz @ self.prec @ dz)
        if self.mon_f.update(score_f):
            alarm_f = True
            self.alarms_f.append(t)
            self.infl_left = c.f_hold_events or (3 * h + 200)

        # ---- 3. forecast, interval, call
        preds = pool.predict_all(z)
        yhat = pool.aggregate(preds)
        infl = c.f_inflation if (self.infl_left > 0 and c.enable_aci) else 1.0
        if self.infl_left > 0:
            self.infl_left -= 1
        q = self.tracker.q * infl if c.enable_aci else self.tracker.q

        live = self.t_live is not None
        if not live:
            self.gate_ratios.append(abs(yhat) / max(q, 1e-12))
            if self.mon_r.state == "monitoring" and len(self.gate_ratios) >= c.n_ref:
                self.k_star = float(np.quantile(self.gate_ratios, 1 - c.call_budget))
                self.t_live = t + 1
            call = 0.0
        else:
            cost = 0.5 * spread
            if c.gate == "abstract":
                ok = abs(yhat) - q > cost
            elif c.enable_aci:
                ok = abs(yhat) > self.k_star * q and abs(yhat) > c.cost_mult * cost
            else:  # no interval: plain magnitude threshold from calibration
                ok = abs(yhat) > self.k_star * self.tracker.q and abs(yhat) > c.cost_mult * cost
            call = (1.0 if yhat > 0 else -1.0) if ok else 0.0

        self.buf_x[ptr] = z
        self.buf_preds[ptr] = preds
        self.buf_yhat[ptr] = yhat
        self.buf_q[ptr] = q
        self.buf_level[ptr] = lvl

        out[0] = yhat
        out[1] = q
        out[2] = call
        out[3] = self.mon_f.log10_R
        out[4] = self.mon_r.log10_R
        out[5] = 1.0 if alarm_f else 0.0
        out[6] = 1.0 if alarm_r else 0.0
        out[8] = 1.0 if live else 0.0
        out[9] = infl
        self.t += 1
        return out

    # ------------------------------------------------------------------ batch
    def run(self, X: np.ndarray, mid: np.ndarray, spread: np.ndarray,
            level: Optional[np.ndarray] = None, record_weights: bool = True) -> Dict[str, np.ndarray]:
        T = X.shape[0]
        res = np.zeros((T, len(OUT_FIELDS)))
        W = np.zeros((T, self.pool.K)) if record_weights else None
        lat = np.zeros(T)
        lv = mid if level is None else level
        pc = time.perf_counter_ns
        for t in range(T):
            t0 = pc()
            o = self.step(X[t], mid[t], spread[t], lv[t])
            lat[t] = (pc() - t0) * 1e-3
            res[t] = o
            if W is not None:
                W[t] = self.pool.w
        out = {k: res[:, i] for i, k in enumerate(OUT_FIELDS)}
        out["weights"] = W
        out["latency_us"] = lat
        out["t_live"] = self.t_live if self.t_live is not None else T
        out["alarms_f"] = np.array(self.alarms_f, dtype=int)
        out["alarms_r"] = np.array(self.alarms_r, dtype=int)
        return out
