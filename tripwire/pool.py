"""Stage B: expert pool (frozen ridge + RLS experts + alarm-spawned regime
heads) aggregated with Herbster-Warmuth fixed-share."""

from __future__ import annotations

import numpy as np


def ridge_fit(X: np.ndarray, y: np.ndarray, l2: float):
    d = X.shape[1]
    A = X.T @ X + l2 * np.eye(d)
    theta = np.linalg.solve(A, X.T @ y)
    return theta, np.linalg.inv(A)


class ExpertPool:
    """Slot 0: frozen ridge. Slots 1..R: RLS with fixed forgetting factors.
    Remaining slots: spawned on Monitor-R alarms; the newest learns, older ones
    are frozen and kept as regime heads (so a recurring regime can win weight back)."""

    def __init__(self, d: int, rls_lambdas, max_spawned: int, spawn_lambda: float,
                 eta: float, alpha: float, loss_clip: float):
        self.d = d
        self.n_rls = len(rls_lambdas)
        self.K = 1 + self.n_rls + max_spawned
        self.first_spawn = 1 + self.n_rls
        self.theta = np.zeros((self.K, d))
        self.P = np.tile(np.eye(d), (self.K, 1, 1))
        self.lam = np.ones(self.K)
        self.lam[1:1 + self.n_rls] = rls_lambdas
        self.lam[self.first_spawn:] = spawn_lambda
        self.active = np.zeros(self.K, dtype=bool)
        self.active[: self.first_spawn] = True
        self.learning = np.zeros(self.K, dtype=bool)
        self.learning[1: self.first_spawn] = True
        self.w = np.where(self.active, 1.0, 0.0)
        self.w /= self.w.sum()
        self.eta = eta
        self.alpha = alpha
        self.loss_clip = loss_clip
        self.spawn_count = 0
        self.spawn_birth = np.full(self.K, -1, dtype=np.int64)

    def init_from_ridge(self, theta: np.ndarray, Pinv: np.ndarray) -> None:
        self.theta[: self.first_spawn] = theta
        self.P[: self.first_spawn] = Pinv

    def predict_all(self, x: np.ndarray) -> np.ndarray:
        p = self.theta @ x
        p[~self.active] = 0.0
        return p

    def aggregate(self, preds: np.ndarray) -> float:
        return float(self.w @ preds)

    def rls_update(self, x: np.ndarray, y: float, y_clip: float) -> None:
        idx = np.flatnonzero(self.learning)
        if idx.size == 0:
            return
        P = self.P[idx]                       # (k,d,d)
        lam = self.lam[idx]
        Px = P @ x                            # (k,d)
        denom = lam + Px @ x
        gain = Px / denom[:, None]
        err = np.clip(y - self.theta[idx] @ x, -y_clip, y_clip)
        self.theta[idx] += gain * err[:, None]
        P = (P - gain[:, :, None] * Px[:, None, :]) / lam[:, None, None]
        P = 0.5 * (P + np.transpose(P, (0, 2, 1)))
        # keep P bounded (forgetting windup when x is uninformative)
        tr = np.trace(P, axis1=1, axis2=2)
        cap = 10.0 * self.d
        big = tr > cap
        if np.any(big):
            P[big] *= (cap / tr[big])[:, None, None]
        self.P[idx] = P

    def fixed_share(self, preds_delayed: np.ndarray, y: float, scale: float, alpha: float) -> None:
        a = self.active
        loss = np.minimum(((y - preds_delayed[a]) / scale) ** 2, self.loss_clip)
        v = self.w[a] * np.exp(-self.eta * (loss - loss.min()))
        v /= v.sum()
        n = int(a.sum())
        self.w[:] = 0.0
        self.w[a] = (1.0 - alpha) * v + alpha / n

    def spawn(self, Xr: np.ndarray, yr: np.ndarray, l2: float, prior_w: float, t: int) -> int:
        """Freeze the currently learning spawned head, start a new one warm-started
        on the recent matured window. Returns the slot used."""
        sp = np.arange(self.first_spawn, self.K)
        self.learning[sp] = False                      # older heads become frozen regime heads
        free = sp[~self.active[sp]]
        if free.size:
            k = int(free[0])
        else:                                           # recycle the lowest-weight head
            k = int(sp[np.argmin(self.w[sp])])
        theta, Pinv = ridge_fit(Xr, yr, l2)
        self.theta[k] = theta
        self.P[k] = Pinv
        self.active[k] = True
        self.learning[k] = True
        self.spawn_birth[k] = t
        self.w[k] = 0.0
        self.w *= (1.0 - prior_w) / max(self.w.sum(), 1e-12)
        self.w[k] = prior_w
        self.spawn_count += 1
        return k
