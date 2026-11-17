"""Live TRIPWIRE-LOB cockpit. Standard library only, works offline.

    python scripts/cockpit.py            # then open http://localhost:8765

The engine runs live in this process; the page receives frames over
Server-Sent Events. Shock buttons change the data-generating process while it
streams:
  * synthetic source: switch the generator's regime (concept / covariate / both / calm)
  * AAPL / AMZN / MSFT source (real LOBSTER events): inject a concept flip into the
    real label stream (P(X) untouched), as in the benchmark's injection protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from tripwire import FeatureExtractor, TripwireConfig, TripwireEngine  # noqa: E402
from tripwire.pool import ridge_fit  # noqa: E402
from tripwire.synthetic import CALM, CONCEPT, COUPLED, COVARIATE, SyntheticBook  # noqa: E402

REGIMES = {"calm": CALM, "concept": CONCEPT, "covariate": COVARIATE, "coupled": COUPLED}
COCKPIT = os.path.join(ROOT, "cockpit")


class Session:
    """One live stream. Thread-safe enough for a single-presenter demo."""

    def __init__(self, source="synthetic", seed=None):
        self.lock = threading.Lock()
        self.source = source
        self.speed = 400.0          # events per second
        self.paused = False
        self.log = deque(maxlen=60)
        self.seed = int(seed if seed is not None else time.time()) % 100000
        self._setup()

    # ------------------------------------------------------------------ data sources
    def _setup(self):
        if self.source == "synthetic":
            self.h = 10
            self.cfg = TripwireConfig(horizon=10, arl_events=50_000, seed=self.seed)
            self.gen = SyntheticBook(horizon=10, seed=self.seed, regime=CALM)
            self.fx = FeatureExtractor(0.01, 10)
            self.regime = "calm"
        else:
            path = os.path.join(ROOT, "data", "cache", f"{self.source}_feat_h50.npz")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} missing: run scripts/run_benchmarks.py or fetch LOBSTER data first")
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
            self.g = Zs @ th                       # real signal used for injection
            self.y_real = y
            self.lv_hist = np.zeros(n)
            self.beta = 1.0
            self.beta_hist = np.ones(n)
            self.i = 0
            self.regime = "real (no injection)"
        self.engine = TripwireEngine(self.cfg)
        self.t = 0
        self.mid_now = 0.0
        self.book = None
        self.cov_win = deque(maxlen=2000)
        self.call_win = deque(maxlen=2000)
        self.hit_win = deque(maxlen=2000)
        self.err_tw = deque(maxlen=2000)
        self.err_b1 = deque(maxlen=2000)
        self.lat = deque(maxlen=5000)
        self.pending = deque()
        self.frames = deque(maxlen=400)
        self.event(f"session started: source={self.source}, h={self.h}, seed={self.seed}")

    def event(self, msg):
        self.log.append({"t": self.t, "msg": msg, "wall": time.strftime("%H:%M:%S")})

    def shock(self, kind):
        with self.lock:
            if self.source == "synthetic":
                self.gen.regime = REGIMES[kind]
                self.regime = kind
                self.event(f"SHOCK injected at event {self.t}: regime -> {kind.upper()} (ground truth, hidden from the model)")
            else:
                self.beta = {"calm": 1.0, "concept": -3.0, "coupled": -3.0, "covariate": 1.0}[kind]
                self.regime = "real + reversed signal (β=-3)" if self.beta != 1.0 else "real (no injection)"
                self.event(f"SHOCK at event {self.t}: real-label signal multiplier β -> {self.beta:g}")

    # ------------------------------------------------------------------ one event
    def _next(self):
        if self.source == "synthetic":
            bp, bs, ap, as_, _ = self.gen.step()
            x, mid, spr = self.fx.update(bp, bs, ap, as_)
            self.book = (bp, bs, ap, as_)
            level = mid
        else:
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
            spr = self.spr[i]
            mid = level
            self.book = None
            self.i += 1
        return x, mid, spr, level

    def advance(self, n):
        rows = []
        for _ in range(n):
            x, mid, spr, level = self._next()
            t0 = time.perf_counter_ns()
            o = self.engine.step(x, mid, spr, level)
            lat = (time.perf_counter_ns() - t0) * 1e-3
            eng = self.engine
            self.t += 1
            self.mid_now = mid
            live = o[8] > 0
            if live:
                self.lat.append(lat)
            yhat, q, call = float(o[0]), float(o[1]), float(o[2])
            # bookkeeping for display metrics, evaluated when the label matures
            self.pending.append((self.t - 1, level, yhat, q, call, float(eng.buf_preds[(self.t - 1) % eng.L][0]) if eng.fitted else 0.0))
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
            if o[5]:
                self.event(f"MONITOR F ALARM at event {self.t}: feature distribution moved -> interval x{self.cfg.f_inflation} (weights untouched)")
            if o[6]:
                self.event(f"MONITOR R ALARM at event {self.t}: forecasts became unreliable -> spawned regime head, switching rate boosted")
            if eng.t_live is not None and self.t == eng.t_live:
                self.event(f"warm-up done at event {self.t}: walk-forward forecasting is LIVE")
            rows.append((self.t, mid, yhat, q, call, float(o[3]), float(o[4]), int(o[5]), int(o[6]), live))
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
            "phase": ("live" if last[9] else ("fitting" if not eng.fitted else "calibrating detectors")),
            "pts": [[r[0], round(r[1], 3), round(r[2], 4), round(r[3], 4), r[4], round(r[5], 3), round(r[6], 3), r[7], r[8]] for r in rows],
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
            "log": list(self.log)[-14:], "speed": self.speed, "paused": self.paused,
        }
        if self.book is not None:
            bp, bs, ap, as_ = self.book
            fr["book"] = {"bid_p": [round(v, 2) for v in bp], "bid_s": bs.tolist(), "ask_p": [round(v, 2) for v in ap], "ask_s": as_.tolist()}
        if self.source != "synthetic" and self.time is not None:
            ii = min(max(self.i - 1, 0), len(self.time) - 1)
            fr["clock"] = float(self.time[ii])
        return fr

    def run_forever(self):
        tick = 0.05
        while True:
            t0 = time.time()
            with self.lock:
                if not self.paused:
                    warm = self.engine.t_live is None
                    n = 4000 if warm else max(1, int(self.speed * tick))
                    rows = self.advance(n)
                    if warm:
                        rows = rows[-200:]
                    self.frames.append(self.frame(rows))
            time.sleep(max(0.0, tick - (time.time() - t0)))


STATE = {"session": None, "thread": None}


def get_session():
    return STATE["session"]


def start_session(source="synthetic", seed=None):
    s = Session(source, seed)
    old = STATE["session"]
    if old is not None:
        old.paused = True
        old.dead = True
    STATE["session"] = s
    def loop():
        while not getattr(s, "dead", False):
            try:
                s.run_forever()
            except Exception as e:  # keep the demo alive
                s.event(f"error: {e!r}")
                time.sleep(0.5)
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    STATE["thread"] = th
    return s


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
            return self._file(os.path.join(ROOT, "results", "figures", os.path.basename(path)), "image/png")
        if path == "/sources":
            av = ["synthetic"] + [s for s in ("AAPL", "AMZN", "MSFT")
                                  if os.path.exists(os.path.join(ROOT, "data", "cache", f"{s}_feat_h50.npz"))]
            return self._send(200, json.dumps(av))
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
        except (BrokenPipeError, ConnectionResetError):
            return

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--source", default="synthetic")
    a = ap.parse_args()
    start_session(a.source)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"TRIPWIRE-LOB cockpit -> http://{a.host}:{a.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
