"""Live TRIPWIRE-LOB cockpit. Standard library only, works offline.

    python scripts/cockpit.py                 # then open http://localhost:8765
    python demo.py --source AAPL              # one-command launcher (opens the browser)

The engine runs live in this process; the page receives frames over
Server-Sent Events. Shock buttons (or keys 1/2/3/0) change the data while it
streams:
  * synthetic: switch the generator's regime (concept / covariate / both / calm)
  * AAPL / AMZN / MSFT (real LOBSTER events) and BTC: inject a concept flip
    into the real label stream (P(X) untouched), as in the benchmark protocol.

Beside TRIPWIRE the cockpit runs the standard baseline live: a rolling
two-sample KS test on every feature (Bonferroni), exactly as B3 in the
benchmark, so viewers can watch it alarm on calm data.

BTC source: warm-starts from the newest recording in data/btc/ (make one with
scripts/record_binance.py), then follows Binance's live 10-level book. If the
network fails it falls back to replaying the recording and says so on screen.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

CODE = getattr(sys, "_MEIPASS", None) or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")   # bundle root when frozen
ROOT = os.environ.get("TRIPWIRE_HOME") or CODE   # writable: data/, results/
sys.path.insert(0, CODE)

from tripwire import FEATURE_NAMES, FeatureExtractor, TripwireConfig, TripwireEngine  # noqa: E402
from tripwire.baselines import ks_crit, ks_stat  # noqa: E402
from tripwire.lobster import load_lobster  # noqa: E402
from tripwire.minidesk import DESKS, MiniDesks  # noqa: E402
from tripwire.pool import ridge_fit  # noqa: E402
from tripwire.synthetic import CALM, CONCEPT, COUPLED, COVARIATE, SyntheticBook  # noqa: E402

REGIMES = {"calm": CALM, "concept": CONCEPT, "covariate": COVARIATE, "coupled": COUPLED}
COCKPIT = os.path.join(CODE, "cockpit")
PRETTY = {
    "ofi_l1": "order flow L1", "ofi_l2": "order flow L2", "ofi_l3": "order flow L3", "ofi_l4": "order flow L4",
    "ofi_l5": "order flow L5", "ofi_deep_6_10": "deep order flow", "microprice_minus_mid": "micro-price",
    "imb_l1": "L1 imbalance", "imb_l1_3": "L1-3 imbalance", "imb_l1_5": "L1-5 imbalance",
    "imb_l1_10": "L1-10 imbalance", "bid_logdepth_slope": "bid depth slope", "ask_logdepth_slope": "ask depth slope",
    "log_spread_ticks": "spread", "vol_ratio_20_200": "volatility ratio", "past_return_h": "recent return",
}
SLOT_NAMES = ["frozen ridge", "RLS .998", "RLS .9995", "RLS .99995", "RLS λ=1", "head 1", "head 2", "head 3", "head 4"]
KS_REF, KS_CUR, KS_EVERY = 3000, 600, 100
# Monitor R tests the long-memory expert; it needs a settled history before a shock (a flip
# ~7k events after warm-up can go undetected). Pre-run this many calm events at full speed.
SYNTH_PRERUN = 20_000
BTC_DIR = os.path.join(ROOT, "data", "btc")
FEED_HOSTS = None          # overridden by --btc-hosts (tests point this at a local mock server)


def btc_recordings():
    return sorted(glob.glob(os.path.join(BTC_DIR, "*.csv")), key=os.path.getmtime)


def btc_recordings_ok():
    return [p for p in btc_recordings() if os.path.basename(p).startswith("btcusdt")
            and not p.endswith(("_times.csv", "_trades.csv"))]


class Session:
    """One live stream. Thread-safe enough for a single-presenter demo."""

    def __init__(self, source="synthetic", seed=None):
        self.lock = threading.Lock()
        self.source = source
        self.speed = 400.0          # events per second (replays; live BTC runs at the market's pace)
        self.paused = False
        self.dead = False
        self.log = deque(maxlen=60)
        self.seed = int(seed if seed is not None else time.time()) % 100000
        self.feed = None
        self._setup()

    # ------------------------------------------------------------------ data sources
    def _setup(self):
        self.t = 0
        self.beta = 1.0
        self.book = None
        if self.source == "synthetic":
            self.h = 10
            self.cfg = TripwireConfig(horizon=10, arl_events=50_000, seed=self.seed)
            self.gen = SyntheticBook(horizon=10, seed=self.seed, regime=CALM)
            self.fx = FeatureExtractor(0.01, 10)
            self.regime = "calm"
        elif self.source == "BTC":
            self._setup_btc()
        else:
            path = os.path.join(ROOT, "data", "cache", f"{self.source}_feat_h50.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} missing: run python scripts/fetch_lobster.py then python scripts/build_cache.py")
            z = np.load(path)
            self.h = 50
            self.cfg = TripwireConfig(horizon=50, n_fit=20_000, n_ref=10_000, arl_events=60_000, seed=self.seed)
            self.X, self.mid, self.spr = z["X"], z["mid"], z["spr"]
            self.time = z["time"] if "time" in z.files else None
            n = len(self.mid)
            y = np.zeros(n)
            y[:-self.h] = self.mid[self.h:] - self.mid[:-self.h]
            Zs = (self.X - self.X.mean(0)) / (self.X.std(0) + 1e-9)
            th, _ = ridge_fit(Zs[: n - self.h], y[: n - self.h], 10.0)
            self.g = Zs @ th                       # the real signal, reversed by a concept shock
            self.y_real = y
            self.lv_hist = np.zeros(n)
            self.beta_hist = np.ones(n)
            self.i = 0
            self.regime = "real (no injection)"
        if self.source != "BTC":
            self.engine = TripwireEngine(self.cfg)
        self._reset_displays()
        self.event(f"session started: source={self.source}, h={self.h}, seed={self.seed}")

    def _reset_displays(self):
        self.cov_win = deque(maxlen=2000)
        self.call_win = deque(maxlen=2000)
        self.hit_win = deque(maxlen=2000)
        self.err_tw = deque(maxlen=2000)
        self.err_b1 = deque(maxlen=2000)
        self.lat = deque(maxlen=5000)
        self.pending = deque()
        self.frames = deque(maxlen=400)
        self.ks_buf = deque(maxlen=KS_REF + KS_CUR)
        self.ks_cool = 0
        self.ks_alarms = 0
        self.ks_crit = ks_crit(KS_REF, KS_CUR, 0.01 / len(FEATURE_NAMES))
        self.shock_t = None
        self.dmg_tw = 0.0
        self.dmg_b1 = 0.0
        self.dmg_n = 0
        self.prev_w = None
        # desk view: the frozen model quoted by three desks (S0 neutral, S1 always-on, S2 guarded)
        self.desk = MiniDesks(self.h)
        self.desk_base = {d: 0.0 for d in DESKS}
        self.desk_state = "TRUST"

    # ---------------------------------------------------------------- BTC (live)
    def _setup_btc(self):
        from tripwire.livefeed import BookFeed
        self.h = 20
        # short warm-up so a 30-45 minute recording is enough (live book ~4-5 events/s)
        self.cfg = TripwireConfig(horizon=20, n_fit=2000, n_ref=2000, arl_events=20_000, stride_f=100,
                                  f_min_ref=30, seed=self.seed)
        self.engine = TripwireEngine(self.cfg)
        self.fx = FeatureExtractor(0.01, self.h)
        self.mid_ring = deque(maxlen=self.h)       # events t-h .. t-1
        self.lvl_ring = deque(maxlen=self.h)
        self.beta_ring = deque(maxlen=self.h)
        recs = btc_recordings_ok()
        self.rec = None
        self.rec_i = 0
        self.mode = "live"
        if recs:
            d = load_lobster(recs[-1])
            self.rec = (d["bid_p"], d["bid_s"], d["ask_p"], d["ask_s"])
            n = len(self.rec[0])
            for i in range(n):                         # fast-forward the warm-up on recorded data
                self._process_book(*(a[i] for a in self.rec), warm=True)
            self.event(f"warm-started on recording {os.path.basename(recs[-1])} ({n:,} book events)")
            if self.engine.t_live is None:
                self.event("recording too short to finish the warm-up: keep the feed running (or record longer)")
        else:
            self.event("no BTC recording in data/btc/: warming up on the live feed (takes ~30 min)")
        self.feed = BookFeed(hosts=FEED_HOSTS).start()
        self.regime = "live BTC-USDT (no injection)"

    def _process_book(self, bp, bs, ap, as_, warm=False):
        """Feature step + label injection for a BTC book. Returns (x, mid, spread, level).
        Sizes arrive in BTC (often < 1); the feature code floors sizes at 1, so work in milli-BTC."""
        bs = np.asarray(bs, dtype=float) * 1000.0
        as_ = np.asarray(as_, dtype=float) * 1000.0
        x, mid, spr = self.fx.update(bp, bs, ap, as_)
        eng = self.engine
        if len(self.mid_ring) == self.h:
            # the label of event j = t-h matures now; a concept shock adds (beta-1) x the frozen
            # ridge's forecast for event j (the book's 'signal'), which leaves P(X) untouched
            j = self.t - self.h
            g_j = float(eng.buf_preds[j % eng.L][0]) if eng.fitted else 0.0
            y_j = mid - self.mid_ring[0] + (self.beta_ring[0] - 1.0) * g_j
            level = self.lvl_ring[0] + y_j
            x[15] = y_j
        else:
            level = mid
        self.mid_ring.append(mid)
        self.lvl_ring.append(level)
        self.beta_ring.append(self.beta)
        if warm:
            eng.step(x, level, spr, level)
            self.t += 1
            return None
        self.book = (bp, bs, ap, as_)
        return x, level, spr, level

    def _next_btc(self):
        feed = self.feed
        if feed.status == "live":
            if self.mode != "live":
                self.mode = "live"
                self.event(f"BTC feed is LIVE again ({feed.host})")
            b = feed.get()
            return None if b is None else self._process_book(*b)
        if self.rec is not None and feed.status == "down":
            if self.mode != "replay":
                self.mode = "replay"
                self.event(f"BTC feed down ({feed.last_error[:80]}): replaying the recording, clearly labelled")
            n = len(self.rec[0])
            i = self.rec_i % n
            self.rec_i += 1
            return self._process_book(*(a[i] for a in self.rec))
        return None

    # ------------------------------------------------------------------ events
    def event(self, msg):
        self.log.append({"t": self.t, "msg": msg, "wall": time.strftime("%H:%M:%S")})

    def ready(self):
        return self.engine.t_live is not None and (self.source != "synthetic" or self.t >= SYNTH_PRERUN)

    def shock(self, kind):
        with self.lock:
            self.shock_t = self.t
            self.dmg_tw = self.dmg_b1 = 0.0
            self.dmg_n = 0
            lv = self.desk.pnl_hist[-1] if self.desk.pnl_hist else [0.0] * len(DESKS)
            self.desk_base = dict(zip(DESKS, lv))
            if self.source == "synthetic":
                self.gen.regime = REGIMES[kind]
                self.regime = kind
                self.event(f"SHOCK injected at event {self.t}: regime -> {kind.upper()} (ground truth, hidden from the model)")
            else:
                self.beta = {"calm": 1.0, "concept": -3.0, "coupled": -3.0, "covariate": 1.0}[kind]
                base = "live BTC-USDT" if self.source == "BTC" else "real"
                self.regime = f"{base} + reversed signal (β=-3)" if self.beta != 1.0 else f"{base} (no injection)"
                if kind == "covariate":
                    self.event("real data: a covariate shock can't be injected without faking the book; use CONCEPT FLIP")
                else:
                    self.event(f"SHOCK at event {self.t}: real-label signal multiplier β -> {self.beta:g}")

    def _next(self):
        if self.source == "synthetic":
            bp, bs, ap, as_, _ = self.gen.step()
            x, mid, spr = self.fx.update(bp, bs, ap, as_)
            self.book = (bp, bs, ap, as_)
            return x, mid, spr, mid
        if self.source == "BTC":
            return self._next_btc()
        i = self.i % len(self.mid)
        if i == 0 and self.i > 0:
            self.event("end of day reached: restarting replay")
            self._setup()
            return self._next()
        h = self.h
        self.beta_hist[i] = self.beta
        if i >= h:
            j = i - h
            yj = self.y_real[j] + (self.beta_hist[j] - 1.0) * self.g[j]
            self.lv_hist[i] = self.lv_hist[j] + yj
        level = self.lv_hist[i]
        x = self.X[i].copy()
        if i >= h:
            x[15] = level - self.lv_hist[i - h]
        self.book = None
        self.i += 1
        return x, level, self.spr[i], level

    # ------------------------------------------------------------------ live baseline: rolling KS
    def _ks_step(self, x):
        eng = self.engine
        if not eng.fitted:
            return 0
        self.ks_buf.append(eng._standardise(np.asarray(x, dtype=float)))
        self.ks_cool -= 1
        if eng.t_live is None or self.t % KS_EVERY or self.ks_cool > 0 or len(self.ks_buf) < KS_REF + KS_CUR:
            return 0
        Z = np.asarray(self.ks_buf)
        ref, cur = Z[:KS_REF], Z[KS_REF:]
        for k in range(Z.shape[1]):
            if ks_stat(np.sort(ref[:, k]), cur[:, k]) > self.ks_crit:
                self.ks_cool = KS_REF
                self.ks_alarms += 1
                calm = self.regime in ("calm",) or "no injection" in self.regime
                self.event(f"rolling-KS ALARM on '{PRETTY[FEATURE_NAMES[k]]}' -> the standard baseline would retrain now"
                           + (" (nothing changed: false alarm)" if calm and self.shock_t is None and self.source == "synthetic" else ""))
                return 1
        return 0

    def _explain_f(self):
        c = self.engine.f_contrib
        top = np.argsort(-c)[:3]
        tot = float(np.sum(np.maximum(c, 0))) + 1e-12
        return ", ".join(f"{PRETTY[FEATURE_NAMES[k]]} {100 * max(c[k], 0) / tot:.0f}%" for k in top)

    def _watch_recurrence(self):
        """Log when an expert from an earlier regime (the frozen ridge = opening calm regime,
        or a spawned regime head) falls out of use and later wins the weight back."""
        pool = self.engine.pool
        w = pool.w
        if self.prev_w is None:
            self.prev_w = np.zeros(len(w), dtype=bool)            # latch: armed once an expert is unused
        for k in [0] + [j for j in range(pool.first_spawn, pool.K) if pool.active[j]]:
            if w[k] < 0.10:
                self.prev_w[k] = True
            elif self.prev_w[k] and w[k] >= 0.30:
                self.prev_w[k] = False
                who = "frozen ridge (fit on the opening calm regime)" if k == 0 else f"regime {SLOT_NAMES[k]}"
                self.event(f"REGIME RECURRENCE: {who} won back {100 * w[k]:.0f}% of the weight -> an earlier regime is back and its saved expert takes over")

    # ------------------------------------------------------------------ main loop
    def advance(self, n):
        rows = []
        for _ in range(n):
            nxt = self._next()
            if nxt is None:                       # live feed: no new book yet
                break
            x, mid, spr, level = nxt
            t0 = time.perf_counter_ns()
            o = self.engine.step(x, mid, spr, level)
            lat = (time.perf_counter_ns() - t0) * 1e-3
            eng = self.engine
            self.t += 1
            live = o[8] > 0
            if live:
                self.lat.append(lat)
            ks = self._ks_step(x)
            yhat, q, call = float(o[0]), float(o[1]), float(o[2])
            b1_now = float(eng.buf_preds[(self.t - 1) % eng.L][0]) if eng.fitted else 0.0
            self.pending.append((self.t - 1, level, yhat, q, call, b1_now))
            y_m = f_m = None
            while self.pending and self.pending[0][0] <= self.t - 1 - self.h:
                t_old, lv_old, yh_old, q_old, c_old, b1_old = self.pending.popleft()
                if t_old >= (eng.t_live or 10**12):
                    y = level - lv_old
                    self.cov_win.append(1.0 if abs(y - yh_old) <= q_old else 0.0)
                    self.call_win.append(1.0 if c_old != 0 else 0.0)
                    if c_old != 0 and y != 0:
                        self.hit_win.append(1.0 if (y > 0) == (c_old > 0) else 0.0)
                    self.err_tw.append((y - yh_old) ** 2)
                    self.err_b1.append((y - b1_old) ** 2)
                    y_m, f_m = y, b1_old
                    if self.shock_t is not None and t_old >= self.shock_t:
                        self.dmg_tw += (y - yh_old) ** 2
                        self.dmg_b1 += (y - b1_old) ** 2
                        self.dmg_n += 1
            if live:
                g = self.desk.step(level, spr, b1_now, bool(o[5]), bool(o[6]), y_m, f_m, live=True)
                if g["state"] != self.desk_state:
                    self.event(f"DESK GUARD {self.desk_state} -> {g['state']} at event {self.t}: {g['reason']}")
                    self.desk_state = g["state"]
            if o[5]:
                self.event(f"MONITOR F ALARM at event {self.t}: the book looks different (top movers: {self._explain_f()}) (evidence crossed A = {eng.mon_f.A:,.0f}) -> interval x{self.cfg.f_inflation}, model untouched")
            if o[6]:
                self.event(f"MONITOR R ALARM at event {self.t}: matured forecast errors lined up against the forecast "
                           f"(evidence crossed A = {eng.mon_r.A:,.0f}) -> new regime head, faster re-weighting")
            if live and self.t % 25 == 0:
                self._watch_recurrence()
            if eng.t_live is not None and self.t == eng.t_live:
                self.event(f"warm-up done at event {self.t}: walk-forward forecasting is LIVE")
            if self.source == "synthetic" and self.t == SYNTH_PRERUN:
                self.event(f"pre-ran {SYNTH_PRERUN:,} calm events so the long-memory expert has settled: ready for shocks")
            rows.append((self.t, mid, yhat, q, call, float(o[3]), float(o[4]), int(o[5]), int(o[6]), live, ks))
        return rows

    def frame(self, rows):
        eng = self.engine
        A_f, A_r = self.cfg.thresholds()
        last = rows[-1]
        def m(d):
            return round(float(np.mean(d)), 4) if len(d) else None
        fr = {
            "t": last[0], "source": self.source, "h": self.h, "regime": self.regime,
            "live": bool(last[9]),
            "phase": ("live" if last[9] and not (self.source == "synthetic" and self.t < SYNTH_PRERUN)
                      else ("fitting" if not eng.fitted else ("calibrating detectors" if not last[9] else "settling (pre-run)"))),
            "pts": [[r[0], round(r[1], 3), round(r[2], 4), round(r[3], 4), r[4], round(r[5], 3), round(r[6], 3), r[7], r[8], r[10]] for r in rows],
            "thr_f": math.log10(1 + A_f), "thr_r": math.log10(1 + A_r),
            "weights": [round(float(w), 4) for w in eng.pool.w],
            "active": [bool(a) for a in eng.pool.active],
            "learning": [bool(a) for a in eng.pool.learning],
            "mon_f_state": eng.mon_f.state, "mon_r_state": eng.mon_r.state,
            "coverage": m(self.cov_win), "call_rate": m(self.call_win), "hit_rate": m(self.hit_win),
            "mse_tw": m(self.err_tw), "mse_b1": m(self.err_b1),
            "lat_p50": round(float(np.percentile(self.lat, 50)), 1) if self.lat else None,
            "lat_p99": round(float(np.percentile(self.lat, 99)), 1) if self.lat else None,
            "n_alarms_f": len(eng.alarms_f), "n_alarms_r": len(eng.alarms_r), "spawns": eng.pool.spawn_count,
            "n_alarms_ks": self.ks_alarms,
            "dmg": None if self.shock_t is None else {"tw": round(self.dmg_tw, 1), "b1": round(self.dmg_b1, 1), "n": self.dmg_n},
            "desk": self._desk_frame(rows[-1][1]),
            "log": list(self.log)[-14:], "speed": self.speed, "paused": self.paused,
        }
        if self.book is not None:
            bp, bs, ap, as_ = self.book
            fr["book"] = {"bid_p": [round(float(v), 2) for v in bp], "bid_s": [float(v) for v in bs],
                          "ask_p": [round(float(v), 2) for v in ap], "ask_s": [float(v) for v in as_]}
        if self.source == "BTC":
            fr["feed"] = {"status": self.feed.status if self.feed else "none", "mode": self.mode,
                          "host": (self.feed.host or "").replace("https://", "") if self.feed else ""}
            fr["clock_text"] = time.strftime("%H:%M:%S") + (" · LIVE BTC-USDT" if self.mode == "live" else " · REPLAY (feed down)")
        elif self.source != "synthetic" and self.time is not None:
            ii = min(max(self.i - 1, 0), len(self.time) - 1)
            fr["clock"] = float(self.time[ii])
        return fr

    def _desk_frame(self, level):
        snap = self.desk.snapshot(level)
        for d in DESKS:
            snap["desks"][d]["since_shock"] = round(snap["desks"][d]["pnl"] - self.desk_base[d], 1)
        snap["pnl"] = list(self.desk.pnl_hist)[-1] if self.desk.pnl_hist else None
        return snap

    def run_forever(self):
        tick = 0.05
        while not self.dead:
            t0 = time.time()
            with self.lock:
                if not self.paused:
                    warm = self.source != "BTC" and (self.engine.t_live is None or
                                                     (self.source == "synthetic" and self.t < SYNTH_PRERUN))
                    if self.source == "BTC":
                        n = 50 if self.mode == "live" else max(1, int(min(self.speed, 20) * tick) or 1)
                    else:
                        n = 4000 if warm else max(1, int(self.speed * tick))
                    rows = self.advance(n)
                    if warm:
                        rows = rows[-200:]
                    if rows:
                        self.frames.append(self.frame(rows))
            time.sleep(max(0.0, tick - (time.time() - t0)))

    def close(self):
        self.dead = True
        self.paused = True
        if self.feed is not None:
            self.feed.stop()


STATE = {"session": None, "thread": None}


def get_session():
    return STATE["session"]


def start_session(source="synthetic", seed=None):
    s = Session(source, seed)
    old = STATE["session"]
    if old is not None:
        old.close()
    STATE["session"] = s

    def loop():
        while not s.dead:
            try:
                s.run_forever()
            except Exception as e:  # keep the demo alive
                s.event(f"error: {e!r}")
                time.sleep(0.5)
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    STATE["thread"] = th
    return s


def available_sources():
    av = ["synthetic"] + [s for s in ("AAPL", "AMZN", "MSFT")
                          if os.path.exists(os.path.join(ROOT, "data", "cache", f"{s}_feat_h50.npz"))]
    av.append("BTC")
    return av


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self._file(os.path.join(COCKPIT, "index.html"), "text/html; charset=utf-8")
        if path == "/replays.json":
            return self._file(os.path.join(COCKPIT, "replays.json"), "application/json")
        if path == "/summary.json":
            return self._file(os.path.join(COCKPIT, "summary.json"), "application/json")
        if path.startswith("/figures/"):
            return self._file(os.path.join(CODE, "results", "figures", os.path.basename(path)), "image/png")
        if path == "/sources":
            s = get_session()
            return self._send(200, json.dumps({"sources": available_sources(), "current": s.source if s else None,
                                               "btc_recordings": len(btc_recordings_ok())}))
        if path == "/stream":
            return self._stream()
        self._send(404, "{}")

    def _file(self, p, ctype):
        if not os.path.exists(p):
            return self._send(404, json.dumps({"error": f"{os.path.basename(p)} not found; run scripts/run_benchmarks.py"}))
        with open(p, "rb") as fh:
            self._send(200, fh.read(), ctype)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_id = None
        try:
            while True:
                s = get_session()
                fr = s.frames[-1] if s and s.frames else None
                if fr is not None and id(fr) != last_id:
                    last_id = id(fr)
                    self.wfile.write(b"data: " + json.dumps(fr).encode() + b"\n\n")
                    self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return  # browser closed or refreshed the page: normal

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0) or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        s = get_session()
        act = body.get("action")
        if act == "shock":
            s.shock(body.get("kind", "calm"))
        elif act == "speed":
            s.speed = float(max(20, min(20000, body.get("value", 600))))
        elif act == "pause":
            s.paused = not s.paused
        elif act == "source":
            try:
                start_session(body.get("value", "synthetic"))
            except FileNotFoundError as e:
                return self._send(400, json.dumps({"error": str(e)}))
        elif act == "reset":
            start_session(s.source)
        return self._send(200, json.dumps({"ok": True}))


def serve(host="127.0.0.1", port=8765, source="synthetic"):
    start_session(source)
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv


def main():
    global FEED_HOSTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--source", default="synthetic", help="synthetic | AAPL | AMZN | MSFT | BTC")
    ap.add_argument("--btc-hosts", default=None, help="comma-separated feed hosts (default: Binance public hosts)")
    a = ap.parse_args()
    if a.btc_hosts:
        FEED_HOSTS = a.btc_hosts.split(",")
    srv = serve(a.host, a.port, a.source)
    print(f"TRIPWIRE-LOB cockpit -> http://{a.host}:{a.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
