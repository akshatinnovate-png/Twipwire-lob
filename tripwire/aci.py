"""Stage D: online quantile tracking of |residual| (adaptive conformal inference,
Gibbs & Candes 2021; quantile tracking form of Angelopoulos, Candes & Tibshirani
2023). With a fixed step the long-run miscoverage converges to alpha for any
sequence; label delay and the post-alarm step boost are outside the textbook
theorem, so coverage is measured, not assumed."""

from __future__ import annotations


class QuantileTracker:
    def __init__(self, alpha: float, gamma: float, q0: float, scale0: float):
        self.alpha = alpha
        self.gamma = gamma
        self.q = max(q0, 1e-6)
        self.scale = max(scale0, 1e-6)

    def update(self, abs_res: float, q_used: float, boost: float = 1.0) -> None:
        err = 1.0 if abs_res > q_used else 0.0
        self.scale += 0.001 * (abs_res - self.scale)
        self.q += boost * self.gamma * self.scale * (err - self.alpha)
        if self.q < 1e-6:
            self.q = 1e-6
