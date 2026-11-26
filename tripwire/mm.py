"""Event-driven market-making desk simulator.

One desk = one virtual bid and one virtual ask resting in a real order book.
Settings follow docs/PREREGISTRATION.md.

Quoting (Avellaneda-Stoikov style, stationary form):
    reservation r = mid + theta * skew * forecast_ticks * tick - kappa * inventory_lots * tick
    half-spread d = max(spread / 2, tick) * width
    bid = floor((r - d) / tick) * tick,  ask = ceil((r + d) / tick) * tick, never crossing the
    opposite best; no bid at the long inventory limit, no ask at the short limit.

Fills
  * order-level data (LOBSTER): our order joins the BACK of the queue at its price. A visible
    execution at our price consumes the queue ahead of us first; an execution or a marketable
    incoming order at a price beyond ours (i.e. we were the best, invisibly) fills us first.
    Cancellations at our price shrink the queue ahead pro-rata ("pro_rata") or not at all
    ("back", conservative).
  * trade data (crypto): traded volume at our price must exceed the queue ahead; trades through
    our price fill us. Queue ahead is shrunk pro-rata by level-size drops that trades don't explain.
Latency: a new quote becomes live `latency` seconds after the decision; until then the
previous quote keeps working. Re-quoting at an unchanged price keeps queue position.

Accounting (marked to mid): P&L = spread captured + position P&L (+ fees/rebates).
Adverse selection = side * qty * (mid(t_fill + k) - mid_fill) for k in {1, 5, 30} s.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


@dataclass
class DeskParams:
    tick: float = 0.01
    lot: float = 100.0                   # shares (LOBSTER) / coins (crypto)
    theta: float = 1.0                   # alpha skew strength (forecast in ticks)
    kappa: float = 0.25                  # inventory skew, ticks per lot held
    max_inventory_lots: float = 10.0
    latency: float = 0.001               # seconds from decision to live quote
    cancel_model: str = "pro_rata"       # "pro_rata" | "back"
    maker_fee_per_unit: float = -0.002   # negative = rebate (LOBSTER: +0.20 c/share rebate)
    maker_fee_bps: float = 0.0           # crypto: fee in basis points of notional
    markouts: tuple = (1.0, 5.0, 30.0)


@dataclass
class _Order:
    side: int            # +1 bid, -1 ask
    price: float
    qty: float
    ahead: float
    live_at: float


@dataclass
class Fill:
    t: int
    ts: float
    side: int
    price: float
    qty: float
    mid: float


class DeskSim:
    def __init__(self, params: DeskParams):
        self.p = params
        self.inv = 0.0
        self.cash = 0.0
        self.fees = 0.0
        self.orders: Dict[int, Optional[_Order]] = {+1: None, -1: None}
        # in-flight quote messages per side, applied in order when each one's latency has elapsed
        self.pending: Dict[int, List[_Order]] = {+1: [], -1: []}
        self.fills: List[Fill] = []

    # ------------------------------------------------------------------ fills
    def _fill(self, o: _Order, qty: float, t: int, ts: float, mid: float):
        qty = min(qty, o.qty)
        if qty <= 0:
            return
        o.qty -= qty
        self.inv += o.side * qty
        self.cash -= o.side * qty * o.price
        p = self.p
        self.fees += qty * p.maker_fee_per_unit + qty * o.price * p.maker_fee_bps * 1e-4
        self.fills.append(Fill(t, ts, o.side, o.price, qty, mid))
        if o.qty <= 1e-12:
            self.orders[o.side] = None

    def on_message(self, t, ts, mtype, msize, mprice, mdir, level_size_before, mid):
        """LOBSTER message (applied to the book BEFORE it). mdir: +1 buy-side order, -1 sell."""
        for side in (+1, -1):
            o = self.orders[side]
            if o is None or ts < o.live_at:
                continue
            same_side = (mdir == side)
            if mtype == 4 and same_side:
                # an aggressor traded against resting `side` orders at mprice
                better = (mprice < o.price) if side == +1 else (mprice > o.price)
                if better:                      # we were the (invisible) best: filled first
                    self._fill(o, msize, t, ts, mid)
                elif abs(mprice - o.price) < 1e-9:
                    if msize > o.ahead:
                        self._fill(o, msize - o.ahead, t, ts, mid)
                    o.ahead = max(0.0, o.ahead - msize)
            elif mtype == 1 and not same_side:
                # an incoming opposite order marketable against our quote (we were inside the book)
                cross = (mprice <= o.price) if side == +1 else (mprice >= o.price)
                if cross:
                    self._fill(o, msize, t, ts, mid)
            elif mtype in (2, 3) and same_side and abs(mprice - o.price) < 1e-9:
                if self.p.cancel_model == "pro_rata" and level_size_before > 0:
                    o.ahead = max(0.0, o.ahead - msize * min(1.0, o.ahead / level_size_before))

    def on_trade(self, t, ts, price, qty, aggressor_sell: bool, mid):
        """Crypto trade print. A sell aggressor hits bids, a buy aggressor lifts asks."""
        side = +1 if aggressor_sell else -1
        o = self.orders[side]
        if o is None or ts < o.live_at:
            return
        better = (price < o.price) if side == +1 else (price > o.price)
        if better:
            self._fill(o, qty, t, ts, mid)
        elif abs(price - o.price) < 1e-9:
            if qty > o.ahead:
                self._fill(o, qty - o.ahead, t, ts, mid)
            o.ahead = max(0.0, o.ahead - qty)

    def on_level_change(self, side, old_size, new_size, traded):
        """Crypto snapshots: unexplained size drop at our price = cancellations (pro-rata)."""
        o = self.orders[side]
        if o is None or self.p.cancel_model != "pro_rata" or old_size <= 0:
            return
        drop = max(0.0, old_size - new_size - traded)
        o.ahead = max(0.0, o.ahead - drop * min(1.0, o.ahead / old_size))

    # ------------------------------------------------------------------ quoting
    def activate(self, ts, level_size_at):
        """Promote pending quotes whose latency has elapsed."""
        for side in (+1, -1):
            queue = self.pending[side]
            while queue and queue[0].live_at <= ts:
                q = queue.pop(0)
                cur = self.orders[side]
                if cur is not None and q.qty > 0 and abs(cur.price - q.price) < 1e-9:
                    cur.qty = q.qty                # same price: keep queue position
                elif q.qty > 0:
                    q.ahead = level_size_at(side, q.price)
                    self.orders[side] = q
                else:
                    self.orders[side] = None

    def decide(self, ts, best_bid, best_ask, forecast_ticks, skew=1.0, width=1.0, size=1.0, inventory=1.0):
        p = self.p
        mid = 0.5 * (best_bid + best_ask)
        spread = best_ask - best_bid
        lots = self.inv / p.lot
        r = mid + p.theta * skew * forecast_ticks * p.tick - p.kappa * lots * p.tick
        d = max(spread / 2.0, p.tick) * width
        bid = math.floor((r - d) / p.tick + 1e-9) * p.tick
        ask = math.ceil((r + d) / p.tick - 1e-9) * p.tick
        bid = min(bid, best_ask - p.tick)
        ask = max(ask, best_bid + p.tick)
        qty = max(p.lot * size, 1e-9)
        lim = p.max_inventory_lots * inventory * p.lot
        for side, px in ((+1, bid), (-1, ask)):
            allowed = (self.inv < lim) if side == +1 else (self.inv > -lim)
            q = min(qty, lim - self.inv) if side == +1 else min(qty, lim + self.inv)
            want = _Order(side, round(px, 10), max(q, 0.0) if allowed else 0.0, 0.0, ts + p.latency)
            # compare with what the market will show once in-flight messages land
            last = self.pending[side][-1] if self.pending[side] else self.orders[side]
            same = (last is not None and abs(last.price - want.price) < 1e-9 and abs(last.qty - want.qty) < 1e-9) or \
                   (last is None and want.qty == 0)
            if not same:
                self.pending[side].append(want)

    def mark(self, mid):
        return self.cash + self.inv * mid - self.fees


# ---------------------------------------------------------------------------- runners
def _level_lookup(bid_p, bid_s, ask_p, ask_s):
    def f(side, price):
        P, S = (bid_p, bid_s) if side == +1 else (ask_p, ask_s)
        hit = np.flatnonzero(np.abs(P - price) < 1e-9)
        return float(S[hit[0]]) if hit.size else 0.0
    return f


def run_lobster_desk(book: Dict[str, np.ndarray], forecast: np.ndarray, controls: Dict[str, np.ndarray],
                     params: DeskParams, start: int, end: int, offset: Optional[np.ndarray] = None,
                     pnl_guard: Optional[Callable] = None) -> Dict:
    """Run one desk over LOBSTER events [start, end). `controls` holds per-event arrays
    skew/width/size/inventory (from the guard). `offset` (ticks) shifts every price, used to
    inject a break consistently into book, messages and fills. `pnl_guard(ts, pnl)` may return
    a skew multiplier (drawdown guard S4)."""
    p = params
    bp, bs, ap, as_ = book["bid_p"], book["bid_s"], book["ask_p"], book["ask_s"]
    ts_all, mt, ms, mpx, md = book["time"], book["msg_type"], book["msg_size"], book["msg_price"], book["msg_dir"]
    off = np.zeros(len(ts_all)) if offset is None else offset * p.tick
    sim = DeskSim(p)
    n = end - start
    pnl = np.zeros(n)
    inv = np.zeros(n)
    mids = np.zeros(n)
    for k, t in enumerate(range(start, end)):
        ts = ts_all[t]
        o = off[t]
        prev = t - 1 if t > start else t
        pbp, pbs, pap, pas = bp[prev] + off[prev], bs[prev], ap[prev] + off[prev], as_[prev]
        mid_prev = 0.5 * (pbp[0] + pap[0])
        # 1. the message hits the book as it stood before it (shifted consistently)
        if k > 0 and mt[t] in (1, 2, 3, 4):
            price = mpx[t] + off[prev]
            P, S = (pbp, pbs) if md[t] == 1 else (pap, pas)
            hit = np.flatnonzero(np.abs(P - price) < 1e-9)
            lvl = float(S[hit[0]]) if hit.size else 0.0
            sim.on_message(t, ts, int(mt[t]), float(ms[t]), price, int(md[t]), lvl, mid_prev)
        # 2. book after the message; pending quotes go live; decide the next quote
        cb, cbs, ca, cas = bp[t] + o, bs[t], ap[t] + o, as_[t]
        sim.activate(ts, _level_lookup(cb, cbs, ca, cas))
        mid = 0.5 * (cb[0] + ca[0])
        skew = controls["skew"][t]
        if pnl_guard is not None:
            skew *= pnl_guard(ts, sim.mark(mid))
        sim.decide(ts, cb[0], ca[0], forecast[t], skew, controls["width"][t], controls["size"][t],
                   controls["inventory"][t])
        pnl[k] = sim.mark(mid)
        inv[k] = sim.inv
        mids[k] = mid
    return summarize(sim, ts_all[start:end], mids, pnl, inv, params)


def summarize(sim: DeskSim, ts: np.ndarray, mids: np.ndarray, pnl: np.ndarray, inv: np.ndarray,
              params: DeskParams) -> Dict:
    f = sim.fills
    res = {"pnl": pnl, "inventory": inv, "n_fills": len(f), "total": float(pnl[-1]) if len(pnl) else 0.0,
           "fees": float(sim.fees), "fills": f}
    if not f:
        res.update(spread_capture=0.0, markout={h: 0.0 for h in params.markouts}, toxic_share=0.0,
                   max_drawdown=float(np.max(np.maximum.accumulate(pnl) - pnl)) if len(pnl) else 0.0)
        return res
    side = np.array([x.side for x in f]); qty = np.array([x.qty for x in f])
    price = np.array([x.price for x in f]); fmid = np.array([x.mid for x in f]); fts = np.array([x.ts for x in f])
    res["spread_capture"] = float(np.sum(side * qty * (fmid - price)))
    mk = {}
    tox = None
    for h in params.markouts:
        j = np.clip(np.searchsorted(ts, fts + h), 0, len(ts) - 1)
        m = side * qty * (mids[j] - fmid)
        mk[h] = float(np.sum(m))
        if h == max(params.markouts):
            tox = float(np.mean(m < 0))
    res["markout"] = mk
    res["toxic_share"] = tox
    res["position_pnl"] = float(res["total"] - res["spread_capture"] + sim.fees)
    res["max_drawdown"] = float(np.max(np.maximum.accumulate(pnl) - pnl))
    return res


class DrawdownGuard:
    """S4, 'the human': pull the alpha for `pause_s` after the desk's P&L over the last
    `window_s` falls by more than `mult` x its median 10-minute range so far."""

    def __init__(self, window_s=600.0, mult=3.0, pause_s=600.0):
        self.window_s, self.mult, self.pause_s = window_s, mult, pause_s
        self.hist = []          # (ts, pnl) samples, thinned
        self.ranges = []
        self.paused_until = -1.0
        self.last_sample = -1e18

    def __call__(self, ts, pnl):
        if ts - self.last_sample >= 1.0:
            self.hist.append((ts, pnl))
            self.last_sample = ts
            while self.hist and ts - self.hist[0][0] > self.window_s:
                self.hist.pop(0)
            if len(self.hist) > 10:
                vals = [v for _, v in self.hist]
                drop = vals[-1] - max(vals)
                rng = max(vals) - min(vals)
                if len(self.ranges) > 30 and drop < -self.mult * float(np.median(self.ranges)):
                    self.paused_until = ts + self.pause_s
                self.ranges.append(rng)
        return 0.0 if ts < self.paused_until else 1.0


def run_trades_desk(data: Dict[str, np.ndarray], forecast: np.ndarray, controls: Dict[str, np.ndarray],
                    params: DeskParams, start: int, end: int, pnl_guard: Optional[Callable] = None) -> Dict:
    """Crypto desk over book snapshots + trade prints. data: time, bid_p/bid_s/ask_p/ask_s (T,10),
    trade_ts, trade_px, trade_qty, trade_sell (aggressor is the seller), sorted by time.
    Trades with timestamp in (ts[t-1], ts[t]] are applied against the book as of t-1."""
    p = params
    bp, bs, ap, as_, ts_all = data["bid_p"], data["bid_s"], data["ask_p"], data["ask_s"], data["time"]
    tts, tpx, tq, tsell = data["trade_ts"], data["trade_px"], data["trade_qty"], data["trade_sell"]
    sim = DeskSim(p)
    n = end - start
    pnl = np.zeros(n); inv = np.zeros(n); mids = np.zeros(n)
    j = int(np.searchsorted(tts, ts_all[start], side="right"))
    for k, t in enumerate(range(start, end)):
        ts = ts_all[t]
        if k > 0:
            mid_prev = 0.5 * (bp[t - 1, 0] + ap[t - 1, 0])
            traded = {+1: 0.0, -1: 0.0}
            while j < len(tts) and tts[j] <= ts:
                side = +1 if tsell[j] else -1
                o = sim.orders[side]
                if o is not None and abs(tpx[j] - o.price) < 1e-9:
                    traded[side] += tq[j]
                sim.on_trade(t, tts[j], tpx[j], tq[j], bool(tsell[j]), mid_prev)
                j += 1
            for side, P0, S0, P1, S1 in ((+1, bp[t - 1], bs[t - 1], bp[t], bs[t]), (-1, ap[t - 1], as_[t - 1], ap[t], as_[t])):
                o = sim.orders[side]
                if o is None:
                    continue
                h0 = np.flatnonzero(np.abs(P0 - o.price) < 1e-9); h1 = np.flatnonzero(np.abs(P1 - o.price) < 1e-9)
                old = float(S0[h0[0]]) if h0.size else 0.0
                new = float(S1[h1[0]]) if h1.size else 0.0
                sim.on_level_change(side, old, new, traded[side])
        sim.activate(ts, _level_lookup(bp[t], bs[t], ap[t], as_[t]))
        mid = 0.5 * (bp[t, 0] + ap[t, 0])
        skew = controls["skew"][t]
        if pnl_guard is not None:
            skew *= pnl_guard(ts, sim.mark(mid))
        sim.decide(ts, bp[t, 0], ap[t, 0], forecast[t], skew, controls["width"][t], controls["size"][t],
                   controls["inventory"][t])
        pnl[k] = sim.mark(mid); inv[k] = sim.inv; mids[k] = mid
    return summarize(sim, ts_all[start:end], mids, pnl, inv, params)
