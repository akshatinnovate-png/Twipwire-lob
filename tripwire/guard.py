"""Model-risk guard: turns TRIPWIRE's alarms into a trust state a quoting
engine can act on.

States (see docs/PREREGISTRATION.md, settings fixed there):

  TRUST     no alarm                     skew x1, width x1,   size x1,   inventory x1
  CAUTION   Monitor F alarm (held)       skew x1, width x1.5, size x0.5, inventory x1
  DISTRUST  Monitor R alarm / feed gap   skew 0,  width x2,   size x0.5, inventory x0.5
  RESTORE   guarded model's matured R^2 > 0 over the last 1,000 labels
            -> trust ramps linearly back to 1 over 2,000 events

The guard never looks at prices or P&L; it consumes only TRIPWIRE's alarms
and the guarded model's matured errors, so it works for any desk model.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class GuardPolicy:
    caution_hold: int = 350              # events a CAUTION is held (3h+200 at h=50)
    caution_width: float = 1.5
    caution_size: float = 0.5
    distrust_width: float = 2.0
    distrust_size: float = 0.5
    distrust_inventory: float = 0.5
    restore_window: int = 1000           # matured labels used to test the guarded model again
    restore_ramp: int = 2000             # events to ramp trust from 0 back to 1
    failsafe_gap_s: float = 5.0          # live only: no event for this long -> DISTRUST
    failsafe_resume: int = 200           # ... until this many events after the feed resumes
    use_f: bool = True                   # ablations
    use_r: bool = True


class TrustGuard:
    STATES = ("TRUST", "CAUTION", "DISTRUST", "RESTORE")

    def __init__(self, policy: Optional[GuardPolicy] = None):
        self.p = policy or GuardPolicy()
        self.state = "TRUST"
        self.trust = 1.0
        self.caution_left = 0
        self.ramp_left = 0
        self.failsafe_left = 0
        self.last_ts = None
        self.err2 = deque(maxlen=self.p.restore_window)
        self.y2 = deque(maxlen=self.p.restore_window)
        self.reason = "start"
        self.transitions = []            # (event_index, from, to, reason)
        self.n = 0

    # ------------------------------------------------------------------ internals
    def _go(self, new, reason):
        if new != self.state:
            self.transitions.append((self.n, self.state, new, reason))
            self.state = new
        self.reason = reason

    def _distrust(self, reason):
        self.trust = 0.0
        self.ramp_left = 0
        self.err2.clear()
        self.y2.clear()
        self._go("DISTRUST", reason)

    # ------------------------------------------------------------------ public
    def update(self, alarm_f: bool = False, alarm_r: bool = False,
               y_matured: Optional[float] = None, f_matured: Optional[float] = None,
               ts: Optional[float] = None) -> dict:
        """One book event. y_matured / f_matured: the label that matured now and the
        guarded model's forecast for it (None while nothing has matured)."""
        p = self.p
        self.n += 1
        # fail-safe on feed gaps (live)
        if ts is not None:
            if self.last_ts is not None and ts - self.last_ts > p.failsafe_gap_s:
                self._distrust(f"feed gap {ts - self.last_ts:.1f}s (fail-safe)")
                self.failsafe_left = p.failsafe_resume
            self.last_ts = ts
        if alarm_r and p.use_r:
            self._distrust("Monitor R: the model's errors line up against its forecasts")
        if alarm_f and p.use_f:
            self.caution_left = p.caution_hold
            if self.state == "TRUST":
                self._go("CAUTION", "Monitor F: the book distribution moved")
        if self.caution_left > 0:
            self.caution_left -= 1
            if self.caution_left == 0 and self.state == "CAUTION":
                self._go("TRUST", "caution window over")
        if self.failsafe_left > 0:
            self.failsafe_left -= 1

        # evidence for restoring trust: the guarded model's matured R^2 since the alarm
        if self.state in ("DISTRUST", "RESTORE") and y_matured is not None and f_matured is not None:
            self.err2.append((y_matured - f_matured) ** 2)
            self.y2.append(y_matured ** 2)
        if (self.state == "DISTRUST" and self.failsafe_left == 0 and len(self.err2) == p.restore_window
                and sum(self.err2) < sum(self.y2)):
            self.ramp_left = p.restore_ramp
            self._go("RESTORE", f"model R² > 0 again over {p.restore_window} matured labels")
        if self.state == "RESTORE":
            self.ramp_left -= 1
            self.trust = 1.0 - self.ramp_left / p.restore_ramp
            if self.ramp_left <= 0:
                self.trust = 1.0
                self._go("CAUTION" if self.caution_left > 0 else "TRUST", "trust restored")
        return self.decision()

    def decision(self) -> dict:
        p = self.p
        t = self.trust if self.state in ("DISTRUST", "RESTORE") else 1.0
        width = 1.0 + (p.distrust_width - 1.0) * (1.0 - t)
        size = 1.0 - (1.0 - p.distrust_size) * (1.0 - t)
        inv = 1.0 - (1.0 - p.distrust_inventory) * (1.0 - t)
        if self.caution_left > 0:
            width = max(width, p.caution_width)
            size = min(size, p.caution_size)
        return {"state": self.state, "skew": t, "width": width, "size": size, "inventory": inv,
                "reason": self.reason}


def guard_series(alarm_f, alarm_r, y_matured, f_matured, live_mask, policy: Optional[GuardPolicy] = None):
    """Batch helper: run the guard over engine output arrays. f_matured[t] is the guarded
    model's forecast for the label maturing at t. Returns arrays skew/width/size/inventory
    and the guard (for its transitions)."""
    import numpy as np
    g = TrustGuard(policy)
    T = len(alarm_f)
    out = {k: np.ones(T) for k in ("skew", "width", "size", "inventory")}
    state = np.empty(T, dtype=object)
    for t in range(T):
        if not live_mask[t]:
            d = g.decision()
        else:
            d = g.update(bool(alarm_f[t]), bool(alarm_r[t]), y_matured[t], f_matured[t])
        for k in out:
            out[k][t] = d[k]
        state[t] = d["state"]
    out["state"] = state
    return out, g
