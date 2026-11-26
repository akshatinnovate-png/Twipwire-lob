"""Small market-making desks for the live cockpit: S0 neutral, S1 always-on alpha, S2 guarded.

Works on any price path in ticks (synthetic, LOBSTER replay, BTC). Fills are 'touch' fills:
our bid fills when the next level trades at or below it, our ask when it trades at or above
it. That is cruder than the queue-accurate simulator in tripwire/mm.py (used by every number
in results/desk/), and the cockpit says so; it exists so the guard's effect is visible live.
Same quoting rule as mm.DeskSim: reservation = level + theta*skew*forecast - kappa*inventory.
"""

from __future__ import annotations

import math
from collections import deque

from .guard import GuardPolicy, TrustGuard

DESKS = ("S0", "S1", "S2")


class _Book:
    def __init__(self, max_inv):
        self.cash = 0.0
        self.inv = 0
        self.fills = 0
        self.toxic = 0.0
        self.max_inv = max_inv


class MiniDesks:
    def __init__(self, horizon, theta=1.0, kappa=0.25, max_inv=10, policy: GuardPolicy | None = None):
        self.h = horizon
        self.theta, self.kappa = theta, kappa
        self.guard = TrustGuard(policy or GuardPolicy(caution_hold=3 * horizon + 200))
        self.books = {d: _Book(max_inv) for d in DESKS}
        self.quotes = {d: None for d in DESKS}
        self.marks = deque()                 # (t_due, desk, side, price) for 30-event markouts
        self.decision = self.guard.decision()
        self.t = 0
        self.pnl_hist = deque(maxlen=4000)

    def reset_pnl(self):
        for d in DESKS:
            self.books[d] = _Book(self.books[d].max_inv)
        self.quotes = {d: None for d in DESKS}
        self.marks.clear()
        self.pnl_hist.clear()

    def step(self, level, spread, forecast, alarm_f=False, alarm_r=False, y_mat=None, f_mat=None, live=True, ts=None):
        """level/spread/forecast in ticks. Returns the guard decision."""
        self.t += 1
        # 1) fills against the quotes resting since the previous event
        for d, q in self.quotes.items():
            if q is None:
                continue
            bid, ask, size = q
            b = self.books[d]
            if bid is not None and level <= bid and b.inv < b.max_inv:
                n = min(size, b.max_inv - b.inv)
                b.inv += n; b.cash -= n * bid; b.fills += 1
                self.marks.append((self.t + 30, d, +1, bid, n))
            if ask is not None and level >= ask and b.inv > -b.max_inv:
                n = min(size, b.max_inv + b.inv)
                b.inv -= n; b.cash += n * ask; b.fills += 1
                self.marks.append((self.t + 30, d, -1, ask, n))
        while self.marks and self.marks[0][0] <= self.t:
            _, d, side, px, n = self.marks.popleft()
            mo = side * n * (level - px)
            if mo < 0:
                self.books[d].toxic -= mo
        # 2) guard
        if live:
            self.decision = self.guard.update(alarm_f, alarm_r, y_mat, f_mat, ts)
        g = self.decision
        # 3) new quotes
        ctl = {"S0": (0.0, 1.0, 1.0), "S1": (1.0, 1.0, 1.0), "S2": (g["skew"], g["width"], g["size"])}
        for d, (skew, width, size) in ctl.items():
            b = self.books[d]
            r = level + self.theta * skew * forecast - self.kappa * b.inv
            half = max(spread / 2.0, 1.0) * width
            lots = max(1, int(round(size * 1)))
            inv_cap = b.max_inv * (g["inventory"] if d == "S2" else 1.0)
            bid = math.floor(r - half) if b.inv < inv_cap else None
            ask = math.ceil(r + half) if b.inv > -inv_cap else None
            self.quotes[d] = (bid, ask, lots)
        self.pnl_hist.append([round(self.pnl(d, level), 2) for d in DESKS])
        return g

    def pnl(self, d, level):
        b = self.books[d]
        return b.cash + b.inv * level

    def snapshot(self, level):
        g = self.decision
        return {
            "state": g["state"], "trust": round(float(g["skew"]), 3), "reason": g.get("reason", ""),
            "desks": {d: {"pnl": round(self.pnl(d, level), 1), "inv": self.books[d].inv, "fills": self.books[d].fills,
                          "toxic": round(self.books[d].toxic, 1)} for d in DESKS},
            "transitions": [list(x) for x in self.guard.transitions[-6:]],
        }
