"""Live paper market-making desk with the TRIPWIRE guard (evidence C in the pre-registration).

    python scripts/paper_desk.py                         # BTCUSDT + ETHUSDT, runs until stopped
    python scripts/paper_desk.py --symbols BTCUSDT --minutes 60

For every symbol it follows the live Binance book and trade prints and runs three paper desks
on identical fills: S0 neutral, S1 always-on alpha, S2 guarded alpha. Nothing is ever sent to
an exchange. Everything is logged under data/paper/<SYMBOL>/:
  ticks_<date>.csv      every book event: time, mid, guard state, P&L and inventory per desk
  fills_<date>.jsonl    every paper fill (desk, side, price, qty, mid)
  events_<date>.jsonl   guard transitions, alarms, feed problems, restarts
  raw/<date>_*.csv      the raw book + trades (warm start on restart, exact replay later)
Make a track record any time with: python scripts/track_record.py

Robustness: a feed gap > 5 s puts the guard in DISTRUST (fail-safe); any crash restarts the
symbol's desk after 10 s, warm-started from its own raw recording. Run it under the desktop
app, `nohup`, Task Scheduler or systemd to keep it going for days.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np

CODE = getattr(sys, "_MEIPASS", None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")   # bundle root when frozen
ROOT = os.environ.get("TRIPWIRE_HOME") or CODE   # writable: data/, results/
sys.path.insert(0, CODE)

from tripwire import FeatureExtractor, TripwireConfig, TripwireEngine  # noqa: E402
from tripwire.guard import GuardPolicy, TrustGuard  # noqa: E402
from tripwire.livefeed import BookFeed  # noqa: E402
from tripwire.mm import DeskParams, DeskSim, _level_lookup  # noqa: E402

LOTS = {"BTCUSDT": 0.01, "ETHUSDT": 0.1}
H = 20
DESK_NAMES = ("S0", "S1", "S2")
STATE_CODE = {"TRUST": 0, "CAUTION": 1, "DISTRUST": 2, "RESTORE": 3}


def engine_config(seed=0):
    return TripwireConfig(horizon=H, n_fit=2000, n_ref=2000, arl_events=20_000, stride_f=100, f_min_ref=30, seed=seed)


class Desk:
    """One symbol: feature extractor + engine + guard + three paper desks."""

    def __init__(self, symbol, out_dir, hosts=None, interval=0.2, latency=0.25):
        self.symbol = symbol
        self.dir = out_dir
        os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
        self.fx = FeatureExtractor(0.01, H)
        self.engine = TripwireEngine(engine_config())
        self.guard = TrustGuard(GuardPolicy(caution_hold=3 * H + 200))
        lot = LOTS.get(symbol, 0.01)
        # paper desk on REST polling: decisions reach the "exchange" a poll later -> 250 ms latency
        self.params = DeskParams(tick=0.01, lot=lot, latency=latency, maker_fee_per_unit=0.0)
        self.desks = {k: DeskSim(self.params) for k in DESK_NAMES}
        self.feed = BookFeed(symbol, interval=interval, hosts=hosts, trades=True)
        self.prev_book = None
        self.mid_hist = deque(maxlen=H + 1)
        self.f_hist = deque(maxlen=H + 1)
        self.n = 0
        self.n_fills = {k: 0 for k in DESK_NAMES}
        self.day = None
        self._files = {}
        self.last_state = "TRUST"

    # ------------------------------------------------------------------ logging
    def _file(self, kind, ext):
        day = time.strftime("%Y-%m-%d", time.gmtime())
        if day != self.day:
            for f in self._files.values():
                f.close()
            self._files = {}
            self.day = day
        key = (kind, ext)
        if key not in self._files:
            path = os.path.join(self.dir, f"{kind}_{day}.{ext}") if kind != "raw" else os.path.join(self.dir, "raw", f"{day}.{ext}")
            new = not os.path.exists(path)
            self._files[key] = open(path, "a", buffering=1)
            if kind == "ticks" and new:
                self._files[key].write("ts,mid,state,trust,pnl_s0,pnl_s1,pnl_s2,inv_s0,inv_s1,inv_s2,forecast\n")
        return self._files[key]

    def log_event(self, kind, **kw):
        self._file("events", "jsonl").write(json.dumps({"ts": time.time(), "kind": kind, **kw}) + "\n")

    # ------------------------------------------------------------------ warm start
    def warm_start(self, max_events=20_000):
        """Fast-forward the engine on this desk's own most recent raw recording."""
        raw = os.path.join(self.dir, "raw")
        files = sorted(f for f in os.listdir(raw) if f.endswith(".book.csv"))
        rows = []
        for f in reversed(files):
            with open(os.path.join(raw, f)) as fh:
                rows = fh.read().splitlines()[-max_events:] + rows
            if len(rows) >= max_events:
                break
        for line in rows[-max_events:]:
            v = np.array(line.split(","), dtype=float)
            book = (v[1:41:4], v[2:41:4], v[3:41:4], v[4:41:4])
            self._model_step(book)
        if rows:
            self.log_event("warm_start", events=len(rows[-max_events:]), live=self.engine.t_live is not None)

    # ------------------------------------------------------------------ core step
    def _model_step(self, book, ts=None):
        bp, bs, ap, as_ = book
        x, mid, spr = self.fx.update(bp, bs * 1000.0, ap, as_ * 1000.0)   # sizes in milli-coins for features
        o = self.engine.step(x, mid, spr, mid)
        self.n += 1
        f = float(o[0])
        y_mat = f_mat = None
        if len(self.mid_hist) >= H:                     # the label of event n-H matures now
            y_mat = mid - self.mid_hist[-H]
            f_mat = self.f_hist[-H]
        self.mid_hist.append(mid)
        self.f_hist.append(f)
        live = self.engine.t_live is not None
        d = self.guard.update(bool(o[5]), bool(o[6]), y_mat, f_mat, ts) if live else self.guard.decision()
        if o[5] or o[6]:
            self.log_event("alarm", monitor="F" if o[5] else "R", event=self.n)
        return f, d, live

    def on_book(self, book, ts, trades):
        bp, bs, ap, as_ = book
        mid_now = 0.5 * (bp[0] + ap[0])
        # 1. fills from trade prints since the last book, against the last book
        if self.prev_book is not None:
            pbp, pbs, pap, pas = self.prev_book
            mid_prev = 0.5 * (pbp[0] + pap[0])
            for name, sim in self.desks.items():
                traded = {+1: 0.0, -1: 0.0}
                nf = len(sim.fills)
                for (tts, px, q, sell) in trades:
                    side = +1 if sell else -1
                    o = sim.orders[side]
                    if o is not None and abs(px - o.price) < 1e-9:
                        traded[side] += q
                    sim.on_trade(self.n, tts, px, q, sell, mid_prev)
                for side, P0, S0, P1, S1 in ((+1, pbp, pbs, bp, bs), (-1, pap, pas, ap, as_)):
                    o = sim.orders[side]
                    if o is not None:
                        h0 = np.flatnonzero(np.abs(P0 - o.price) < 1e-9); h1 = np.flatnonzero(np.abs(P1 - o.price) < 1e-9)
                        sim.on_level_change(side, float(S0[h0[0]]) if h0.size else 0.0, float(S1[h1[0]]) if h1.size else 0.0, traded[side])
                for fl in sim.fills[nf:]:
                    self.n_fills[name] += 1
                    self._file("fills", "jsonl").write(json.dumps({"ts": fl.ts, "desk": name, "side": fl.side, "price": fl.price,
                                                                   "qty": fl.qty, "mid": fl.mid}) + "\n")
        self.prev_book = book
        # 2. raw record (book + trades) for warm starts and exact replay
        self._file("raw", "book.csv").write(f"{ts:.3f}," + ",".join(f"{a:.8g},{b:.8g},{c:.8g},{e:.8g}" for a, b, c, e in zip(bp, bs, ap, as_)) + "\n")
        for tr in trades:
            self._file("raw", "trades.csv").write(f"{tr[0]:.3f},{tr[1]},{tr[2]},{1 if tr[3] else 0}\n")
        # 3. model + guard, then each desk quotes
        f, d, live = self._model_step(book, ts)
        if d["state"] != self.last_state:
            self.log_event("state", frm=self.last_state, to=d["state"], reason=d["reason"])
            self.last_state = d["state"]
        look = _level_lookup(bp, bs, ap, as_)
        for name, sim in self.desks.items():
            sim.activate(ts, look)
            if not live:
                continue
            skew = 0.0 if name == "S0" else (1.0 if name == "S1" else d["skew"])
            ctl = (1.0, 1.0, 1.0) if name != "S2" else (d["width"], d["size"], d["inventory"])
            sim.decide(ts, bp[0], ap[0], f, skew, *ctl)
        s = self.desks
        self._file("ticks", "csv").write(
            f"{ts:.3f},{mid_now:.8g},{STATE_CODE[d['state']]},{d['skew']:.3f},"
            + ",".join(f"{s[k].mark(mid_now):.4f}" for k in DESK_NAMES) + ","
            + ",".join(f"{s[k].inv:.6g}" for k in DESK_NAMES) + f",{f:.4f}\n")

    # ------------------------------------------------------------------ loop
    def run(self, stop: threading.Event, status: dict):
        self.warm_start()
        self.feed.start()
        self.log_event("start", warm=self.engine.t_live is not None)
        gap_logged = False
        while not stop.is_set():
            item = self.feed.get(timeout=0.5)
            now = time.time()
            if item is None:
                if self.feed.status == "down" and not gap_logged:
                    self.log_event("feed_down", error=self.feed.last_error)
                    gap_logged = True
                    self.guard.update(ts=now)                   # fail-safe: big gap -> DISTRUST
                continue
            gap_logged = False
            book, ts, trades = item
            self.on_book(book, ts, trades)
            s = self.desks
            status[self.symbol] = dict(events=self.n, state=self.guard.state, live=self.engine.t_live is not None,
                                       feed=self.feed.status, host=self.feed.host,
                                       pnl={k: round(s[k].mark(0.5 * (book[0][0] + book[2][0])), 4) for k in DESK_NAMES},
                                       fills=dict(self.n_fills))
        self.feed.stop()


def supervise(symbol, out_root, stop, status, hosts, interval):
    while not stop.is_set():
        desk = None
        try:
            desk = Desk(symbol, os.path.join(out_root, symbol), hosts=hosts, interval=interval)
            desk.run(stop, status)
        except Exception as e:  # restart after a crash, warm-started from the raw recording
            msg = f"{symbol}: {e!r}"
            print("paper desk error:", msg, flush=True)
            traceback.print_exc()
            if desk is not None:
                try:
                    desk.log_event("crash", error=msg)
                except Exception:
                    pass
            stop.wait(10.0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    ap.add_argument("--minutes", type=float, default=0, help="0 = run until stopped")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "paper"))
    ap.add_argument("--hosts", default=None, help="comma-separated feed hosts (default: Binance public hosts)")
    ap.add_argument("--interval", type=float, default=0.2)
    a = ap.parse_args(argv)
    stop = threading.Event()
    status = {}
    hosts = a.hosts.split(",") if a.hosts else None
    threads = [threading.Thread(target=supervise, args=(s, a.out, stop, status, hosts, a.interval), daemon=True)
               for s in a.symbols.split(",")]
    for t in threads:
        t.start()
    t0 = time.time()
    print(f"paper desk running for {a.symbols} -> {os.path.abspath(a.out)}   (Ctrl+C to stop)", flush=True)
    try:
        while not a.minutes or time.time() - t0 < 60 * a.minutes:
            time.sleep(10)
            line = " | ".join(f"{k}: {v['state']} ev={v['events']} feed={v['feed']} pnl S1={v['pnl']['S1']} S2={v['pnl']['S2']}"
                              for k, v in sorted(status.items()))
            print(time.strftime("%H:%M:%S"), line or "waiting for the first book...", flush=True)
    except KeyboardInterrupt:
        pass
    stop.set()
    for t in threads:
        t.join(timeout=5)
    return status


if __name__ == "__main__":
    main()
