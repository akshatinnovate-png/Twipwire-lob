"""Live paper desk end to end against a mock exchange (depth + aggTrades), then the track record."""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from tripwire.synthetic import SyntheticBook  # noqa: E402


class MockExchange:
    def __init__(self):
        self.gen = SyntheticBook(horizon=20, seed=4)
        self.book = self.gen.step()
        self.tid = 1000
        self.rng = np.random.default_rng(0)
        lock = threading.Lock()
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                u = urlparse(self.path)
                with lock:
                    if u.path.endswith("/depth"):
                        outer.book = outer.gen.step()
                        bp, bs, ap, as_, _ = outer.book
                        body = {"bids": [[f"{p:.2f}", f"{q / 1000:.5f}"] for p, q in zip(bp, bs)],
                                "asks": [[f"{p:.2f}", f"{q / 1000:.5f}"] for p, q in zip(ap, as_)]}
                    else:
                        bp, _, ap, _, _ = outer.book
                        out = []
                        for _ in range(int(outer.rng.integers(1, 4))):
                            outer.tid += 1
                            sell = bool(outer.rng.random() < 0.5)
                            out.append({"a": outer.tid, "p": f"{bp[0] if sell else ap[0]:.2f}",
                                        "q": f"{outer.rng.uniform(0.2, 1.0):.4f}", "T": int(time.time() * 1000), "m": sell})
                        body = out[-1:] if "fromId" not in parse_qs(u.query) else out
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


def _seed_recording(sym_dir, n=9000):
    os.makedirs(os.path.join(sym_dir, "raw"), exist_ok=True)
    gen = SyntheticBook(horizon=20, seed=11)
    day = time.strftime("%Y-%m-%d", time.gmtime())
    with open(os.path.join(sym_dir, "raw", f"{day}.book.csv"), "w") as f:
        t = time.time() - n * 0.2
        for i in range(n):
            bp, bs, ap, as_, _ = gen.step()
            f.write(f"{t + 0.2 * i:.3f}," + ",".join(f"{a:.8g},{b / 1000:.8g},{c:.8g},{e / 1000:.8g}" for a, b, c, e in zip(bp, bs, ap, as_)) + "\n")


def test_paper_desk_runs_logs_and_reports(tmp_path):
    import paper_desk
    import track_record
    mock = MockExchange()
    out = tmp_path / "paper"
    _seed_recording(str(out / "BTCUSDT"))
    try:
        status = paper_desk.main(["--symbols", "BTCUSDT", "--minutes", "0.35", "--hosts", mock.url,
                                  "--interval", "0.01", "--out", str(out)])
    finally:
        mock.stop()
    st = status["BTCUSDT"]
    assert st["live"] and st["events"] > 9000 + 100
    assert sum(st["fills"].values()) > 0
    sym = out / "BTCUSDT"
    ticks = list(sym.glob("ticks_*.csv"))
    assert ticks and sum(1 for _ in open(ticks[0])) > 100
    ev = [json.loads(l) for f in sym.glob("events_*.jsonl") for l in open(f)]
    assert any(e["kind"] == "warm_start" and e["live"] for e in ev)
    md = track_record.report(root=str(out), calendar=str(tmp_path / "none.csv"), out_dir=str(tmp_path / "res"))
    assert "BTCUSDT" in md and "H5, placebo hours" in md
    assert (tmp_path / "res" / "track_BTCUSDT.png").exists()
