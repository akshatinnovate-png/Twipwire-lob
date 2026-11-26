"""Live 10-level order-book feed (Binance spot REST, standard library only).

A background thread polls ``/api/v3/depth?limit=10`` and queues a snapshot
whenever the book changes. Several public hosts are tried in turn
(``data-api.binance.vision`` is Binance's market-data-only mirror and is
usually reachable where the main API is geo-blocked). ``status`` is one of
``connecting``, ``live`` or ``down``; the cockpit falls back to a recording
when the feed is down, so a bad venue network never blanks the demo.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request
from typing import List, Optional

import numpy as np

DEFAULT_HOSTS = ["https://data-api.binance.vision", "https://api.binance.com",
                 "https://api1.binance.com", "https://api.binance.us"]


def parse_depth(js) -> tuple:
    """Binance depth JSON -> (bid_p, bid_s, ask_p, ask_s) arrays of 10 levels."""
    bids = js["bids"][:10]
    asks = js["asks"][:10]
    if len(bids) < 10 or len(asks) < 10:
        raise ValueError("fewer than 10 levels")
    bp = np.array([float(p) for p, _ in bids])
    bs = np.array([float(q) for _, q in bids])
    ap = np.array([float(p) for p, _ in asks])
    as_ = np.array([float(q) for _, q in asks])
    if not (ap[0] > bp[0]):
        raise ValueError("crossed book")
    return bp, bs, ap, as_


class BookFeed:
    def __init__(self, symbol: str = "BTCUSDT", interval: float = 0.2,
                 hosts: Optional[List[str]] = None, timeout: float = 3.0, trades: bool = False):
        self.symbol = symbol
        self.interval = interval
        self.hosts = list(hosts or DEFAULT_HOSTS)
        self.timeout = timeout
        self.q: "queue.Queue" = queue.Queue(maxsize=5000)
        self.status = "connecting"
        self.host = None
        self.last_ok = 0.0
        self.last_error = ""
        self.n_events = 0
        self._stop = threading.Event()
        self._last = None
        self.trades = trades
        self._last_trade_id = None
        self._pending_trades = []
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _fetch(self, host):
        url = f"{host}/api/v3/depth?symbol={self.symbol}&limit=10"
        with urllib.request.urlopen(url, timeout=self.timeout) as r:
            return json.load(r)

    def _fetch_trades(self, host):
        """Aggregated trades since the last seen id: (ts_s, price, qty, aggressor_is_seller)."""
        q = f"{host}/api/v3/aggTrades?symbol={self.symbol}&limit=1000"
        if self._last_trade_id is not None:
            q += f"&fromId={self._last_trade_id + 1}"
        else:
            q = f"{host}/api/v3/aggTrades?symbol={self.symbol}&limit=1"
        with urllib.request.urlopen(q, timeout=self.timeout) as r:
            rows = json.load(r)
        out = []
        for a in rows:
            if self._last_trade_id is not None and a["a"] <= self._last_trade_id:
                continue
            out.append((a["T"] / 1000.0, float(a["p"]), float(a["q"]), bool(a["m"])))
            self._last_trade_id = a["a"]
        if self._last_trade_id is None and rows:
            self._last_trade_id = rows[-1]["a"]
            return []
        return out

    def _run(self):
        hi = 0
        fails = 0
        while not self._stop.is_set():
            t0 = time.time()
            host = self.hosts[hi % len(self.hosts)]
            try:
                book = parse_depth(self._fetch(host))
                if self.trades:
                    self._pending_trades += self._fetch_trades(host)
                key = tuple(np.concatenate(book).round(8))
                if key != self._last:
                    self._last = key
                    trades, self._pending_trades = self._pending_trades, []
                    try:
                        self.q.put_nowait((book, time.time(), trades) if self.trades else book)
                        self.n_events += 1
                    except queue.Full:
                        pass
                self.host = host
                self.status = "live"
                self.last_ok = time.time()
                fails = 0
            except Exception as e:  # network, HTTP 451/403, bad JSON: try the next host
                self.last_error = f"{host}: {e!r}"[:200]
                fails += 1
                hi += 1
                if time.time() - self.last_ok > 5.0:
                    self.status = "down"
                if fails >= len(self.hosts):
                    self._stop.wait(2.0)          # every host failed: back off
                    fails = 0
            self._stop.wait(max(0.0, self.interval - (time.time() - t0)))

    def get(self, timeout: float = 0.0):
        try:
            return self.q.get(timeout=timeout) if timeout > 0 else self.q.get_nowait()
        except queue.Empty:
            return None
