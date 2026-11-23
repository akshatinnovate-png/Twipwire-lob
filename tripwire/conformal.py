"""Stage C: conformal p-values driving a Shiryaev-Roberts change detector.

Theory used (and its limits):
* If p_1, p_2, ... are independent U(0,1), the power-mixture bet
  b(p) = int_0^1 eps * p^(eps-1) d eps integrates to 1, so R_t - t with
  R_t = (1 + R_{t-1}) b(p_t) is a martingale. Optional stopping then gives
  E[stopping time] >= A for the rule "alarm when R_t >= A" (Pollak 1985;
  Vovk 2021, conformal test martingales).
* Conformal p-values against a *rolling* reference are only approximately
  uniform and independent on real data: overlapping labels and volatility
  clustering break exchangeability. We thin the stream (stride) and keep a
  gap between the newest score and the reference, then *measure* the false
  alarm rate (scripts/run_benchmarks.py: ARL calibration).
"""

from __future__ import annotations

import bisect
import math
from collections import deque

import numpy as np


def power_mixture_bet(p: float) -> float:
    """b(p) = (1 - p + p ln p) / (p (ln p)^2); b -> 1/2 as p -> 1, int_0^1 b = 1."""
    if p >= 1.0 - 1e-9:
        return 0.5
    p = max(p, 1e-12)
    lp = math.log(p)
    return (1.0 - p + p * lp) / (p * lp * lp)


class ConformalSR:
    """Block-mean conformal Shiryaev-Roberts detector with restart after alarm.

    Scores are averaged over non-overlapping blocks of `stride` events before
    testing: block means of an MA(h-1) score sequence are close to independent
    and use every event (thinning would discard h-1 of every h scores).

    States: "calibrating" (collecting a reference, cannot alarm) and
    "monitoring". After an alarm the reference is flushed and rebuilt from
    fresh scores, so the detector re-anchors on the new regime.

    Instant re-arm (``instant_rearm=True``): instead of going blind while the
    new reference fills, each post-alarm block is tested against the blocks
    seen since the alarm and then added to them (no gap). That is the plain
    online conformal p-value, which is exactly uniform and independent under
    exchangeability, so the run-length guarantee is unchanged; the test simply
    has little power for the first few blocks. Once the reference reaches
    ``min_ref_rearm`` blocks the usual gapped rolling reference resumes.
    """

    def __init__(self, threshold: float, ref_size: int = 800, ref_gap: int = 400,
                 min_ref: int = 200, stride: int = 1, seed: int = 0, enabled: bool = True,
                 min_ref_rearm: int = 0, instant_rearm: bool = False, two_sided: bool = False):
        self.A = float(threshold)
        self.ref_size = int(ref_size)
        self.ref_gap = int(ref_gap)
        self.min_ref_initial = int(min_ref)
        self.min_ref = int(min_ref)
        self.min_ref_rearm = int(min_ref_rearm or min_ref)
        self.stride = max(int(stride), 1)
        self.enabled = enabled
        self.instant_rearm = bool(instant_rearm)
        self.two_sided = bool(two_sided)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.min_ref = self.min_ref_initial
        self.sorted_ref: list = []
        self.ref_fifo: deque = deque()
        self.pending: deque = deque()
        self.R = 0.0
        self.counter = 0
        self.block_sum = 0.0
        self.n_tests = 0
        self.n_alarms = 0
        self.last_p = 0.5
        self.rearming = False
        self.last_quiet_test = 0       # last test at which R was below 1 (no evidence yet)
        self.tests_since_quiet_at_alarm = 0

    @property
    def state(self) -> str:
        if self.rearming:
            return "re-arming"
        return "monitoring" if len(self.sorted_ref) >= self.min_ref else "calibrating"

    def _push_reference(self, s: float) -> None:
        bisect.insort(self.sorted_ref, s)
        self.ref_fifo.append(s)
        if len(self.ref_fifo) > self.ref_size:
            old = self.ref_fifo.popleft()
            i = bisect.bisect_left(self.sorted_ref, old)
            del self.sorted_ref[i]

    def pvalue(self, s: float) -> float:
        ref = self.sorted_ref
        n = len(ref)
        lo = bisect.bisect_left(ref, s)
        hi = bisect.bisect_right(ref, s)
        greater = n - hi
        equal = hi - lo
        u = self.rng.random()
        p_up = (greater + u * (equal + 1)) / (n + 1)
        if not self.two_sided:
            return p_up
        # two-sided: unusually large OR unusually small scores; 2*min(p, 1-p) is
        # still super-uniform under the null, so the run-length guarantee holds.
        p_lo = (lo + (1.0 - u) * (equal + 1)) / (n + 1)
        return min(1.0, 2.0 * min(p_up, p_lo))

    def update(self, score: float) -> bool:
        """Feed one score (every event). Returns True on alarm."""
        if not math.isfinite(score):
            return False
        self.counter += 1
        self.block_sum += score
        if self.counter % self.stride != 0:
            return False
        score = self.block_sum / self.stride      # non-overlapping block mean
        self.block_sum = 0.0
        alarm = False
        if self.rearming and len(self.sorted_ref) >= self.min_ref:
            self.rearming = False                 # enough fresh blocks: back to the gapped reference
        if self.state in ("monitoring", "re-arming"):
            p = self.pvalue(score)
            self.last_p = p
            self.R = (1.0 + self.R) * power_mixture_bet(p)
            self.n_tests += 1
            if self.R < 1.0:
                self.last_quiet_test = self.n_tests
            if self.enabled and self.R >= self.A:
                alarm = True
                self.n_alarms += 1
                # change-point estimate: evidence has been building since R last sat below 1
                self.tests_since_quiet_at_alarm = self.n_tests - self.last_quiet_test
                self.R = 0.0
                self.sorted_ref = []
                self.ref_fifo.clear()
                self.pending.clear()
                self.min_ref = self.min_ref_rearm
                self.rearming = self.instant_rearm
        # new score enters the reference after `ref_gap` further tests
        # (during calibration it enters immediately)
        if (self.state in ("calibrating", "re-arming")) and not alarm:
            self._push_reference(score)
        elif not alarm:
            self.pending.append(score)
            if len(self.pending) > self.ref_gap:
                self._push_reference(self.pending.popleft())
        return alarm

    @property
    def log10_R(self) -> float:
        return math.log10(1.0 + self.R)
