"""Cockpit tests: live baseline, damage meter, explanations, and the BTC source
against a local mock of Binance's depth endpoint (no internet needed)."""

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import cockpit  # noqa: E402
from tripwire.livefeed import parse_depth  # noqa: E402
from tripwire.synthetic import SyntheticBook  # noqa: E402


def _book_json(gen):
    bp, bs, ap, as_, _ = gen.step()
    return {"lastUpdateId": gen.n,
            "bids": [[f"{p:.2f}", f"{q / 1000:.5f}"] for p, q in zip(bp, bs)],
            "asks": [[f"{p:.2f}", f"{q / 1000:.5f}"] for p, q in zip(ap, as_)]}


class _MockBinance:
    def __init__(self):
        gen = SyntheticBook(horizon=20, seed=5)
        lock = threading.Lock()

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                with lock:
                    body = json.dumps(_book_json(gen)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


def _write_recording(path, n):
    gen = SyntheticBook(horizon=20, seed=9)
    with open(path, "w") as f:
        for _ in range(n):
            bp, bs, ap, as_, _ = gen.step()
            row = []
            for l in range(10):
                row += [round(ap[l] * 1e4), as_[l] / 1000, round(bp[l] * 1e4), bs[l] / 1000]
            f.write(",".join(str(v) for v in row) + "\n")


def test_parse_depth_rejects_bad_books():
    gen = SyntheticBook(seed=1)
    bp, bs, ap, as_ = parse_depth(_book_json(gen))
    assert bp.shape == (10,) and ap[0] > bp[0]
    with pytest.raises(ValueError):
        parse_depth({"bids": [["1", "1"]] * 3, "asks": [["2", "1"]] * 3})


def test_synthetic_session_runs_baseline_damage_and_explanations():
    s = cockpit.Session("synthetic", seed=3)
    s.advance(cockpit.SYNTH_PRERUN)                      # what the cockpit pre-runs before the first shock
    assert s.ready()
    s.shock("concept")
    rows = s.advance(5000)
    fr = s.frame(rows)
    assert fr["dmg"] is not None and fr["dmg"]["n"] > 1000
    assert fr["dmg"]["b1"] > fr["dmg"]["tw"]            # frozen model takes more damage after a flip
    assert isinstance(fr["n_alarms_ks"], int)
    assert len(fr["pts"][0]) == 10                       # ks flag column present
    assert any("MONITOR R ALARM" in e["msg"] for e in s.log)
    expl = s._explain_f()
    assert expl.count("%") == 3


@pytest.mark.parametrize("seed", [3, 4, 7, 11])
def test_demo_flip_is_caught_after_prerun(seed):
    """Guards the live demo: a concept flip pressed right after the pre-run must be detected."""
    s = cockpit.Session("synthetic", seed=seed)
    s.advance(cockpit.SYNTH_PRERUN)
    n0 = len(s.engine.alarms_r)
    s.shock("concept")
    s.advance(5000)
    assert len(s.engine.alarms_r) > n0


def test_btc_source_warm_start_live_and_fallback(tmp_path, monkeypatch):
    rec = tmp_path / "btcusdt_test.csv"
    _write_recording(rec, 9000)
    mock = _MockBinance()
    monkeypatch.setattr(cockpit, "BTC_DIR", str(tmp_path))
    monkeypatch.setattr(cockpit, "FEED_HOSTS", [mock.url])
    s = cockpit.Session("BTC", seed=1)
    try:
        assert s.engine.t_live is not None, "warm start from the recording should finish the warm-up"
        t0 = s.t
        deadline = time.time() + 8
        while time.time() < deadline and s.t < t0 + 10:
            s.advance(50)
            time.sleep(0.05)
        assert s.feed.status == "live" and s.mode == "live"
        assert s.t >= t0 + 10, "live events should flow from the mock exchange"
        s.shock("concept")
        s.advance(50)
        fr = s.frame([(s.t, 0, 0, 1, 0, 0, 0, 0, 0, True, 0)])
        assert fr["feed"]["mode"] == "live" and "LIVE" in fr["clock_text"]
        mock.stop()                                      # venue Wi-Fi dies
        deadline = time.time() + 15
        while time.time() < deadline and s.mode != "replay":
            s.advance(5)
            time.sleep(0.1)
        assert s.mode == "replay", "should fall back to replaying the recording"
        t1 = s.t
        s.advance(20)
        assert s.t > t1
    finally:
        s.close()
