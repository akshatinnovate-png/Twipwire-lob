"""'The desk's model': a small neural network the guard protects from the outside.

A 16-32-1 tanh MLP trained with Adam on the first n events of a stream (labels
fully matured inside that block) and then frozen, like a production model that
is retrained overnight. NumPy only, deterministic given the seed.
"""

from __future__ import annotations

import numpy as np


class MLPDeskModel:
    def __init__(self, hidden: int = 32, l2: float = 1e-3, epochs: int = 40, lr: float = 3e-3,
                 batch: int = 512, seed: int = 0, clip_z: float = 6.0):
        self.hidden, self.l2, self.epochs, self.lr, self.batch = hidden, l2, epochs, lr, batch
        self.seed, self.clip_z = seed, clip_z

    def _z(self, X):
        return np.clip((X - self.mu) / self.sd, -self.clip_z, self.clip_z)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPDeskModel":
        rng = np.random.default_rng(self.seed)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = self._z(X)
        self.ys = float(np.std(y) + 1e-9)
        yt = np.clip(y / self.ys, -6, 6)
        d, H = Z.shape[1], self.hidden
        P = {"W1": rng.normal(0, 1 / np.sqrt(d), (d, H)), "b1": np.zeros(H),
             "W2": rng.normal(0, 1 / np.sqrt(H), (H, 1)), "b2": np.zeros(1)}
        m = {k: np.zeros_like(v) for k, v in P.items()}
        v = {k: np.zeros_like(w) for k, w in P.items()}
        step = 0
        n = len(Z)
        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for a in range(0, n, self.batch):
                b = idx[a:a + self.batch]
                x, t = Z[b], yt[b][:, None]
                h = np.tanh(x @ P["W1"] + P["b1"])
                out = h @ P["W2"] + P["b2"]
                g_out = 2 * (out - t) / len(b)
                grads = {"W2": h.T @ g_out + self.l2 * P["W2"], "b2": g_out.sum(0)}
                g_h = (g_out @ P["W2"].T) * (1 - h * h)
                grads["W1"] = x.T @ g_h + self.l2 * P["W1"]
                grads["b1"] = g_h.sum(0)
                step += 1
                for k in P:
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]
                    v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                    mh = m[k] / (1 - 0.9 ** step)
                    vh = v[k] / (1 - 0.999 ** step)
                    P[k] -= self.lr * mh / (np.sqrt(vh) + 1e-8)
        self.P = P
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(self._z(np.atleast_2d(X)) @ self.P["W1"] + self.P["b1"])
        return ((h @ self.P["W2"] + self.P["b2"])[:, 0]) * self.ys
