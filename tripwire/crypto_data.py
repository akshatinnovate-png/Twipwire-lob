"""Historical and recorded crypto order-book data -> the arrays the engine and desk use.

Sources
  * Tardis.dev free samples: the first day of every month is downloadable without an API key:
      https://datasets.tardis.dev/v1/<exchange>/<data_type>/<YYYY>/<MM>/01/<SYMBOL>.csv.gz
    data types used: book_snapshot_25 (top-25 book on every change) and trades.
    (scripts/fetch_crypto.py downloads them.)
  * Our own recordings (scripts/record_binance.py / the paper desk): LOBSTER-format book rows
    plus a trades CSV.
Snapshots whose top-10 book is unchanged are dropped, so one row = one visible book change.
"""

from __future__ import annotations

import csv
import gzip
import io
import os
from typing import Dict, Optional

import numpy as np


def _open(path):
    return io.TextIOWrapper(gzip.open(path, "rb")) if path.endswith(".gz") else open(path, newline="")


def load_tardis(book_path: str, trades_path: Optional[str] = None, max_rows: Optional[int] = None,
                levels: int = 10) -> Dict[str, np.ndarray]:
    ts, B, BS, A, AS = [], [], [], [], []
    last = None
    with _open(book_path) as f:
        r = csv.reader(f)
        hdr = next(r)
        col = {h: i for i, h in enumerate(hdr)}
        ip = [(col[f"bids[{l}].price"], col[f"bids[{l}].amount"], col[f"asks[{l}].price"], col[f"asks[{l}].amount"]) for l in range(levels)]
        it = col["timestamp"]
        for row in r:
            try:
                vals = tuple(float(row[c]) for q in ip for c in q)
            except (ValueError, IndexError):
                continue                          # fewer than `levels` levels in this snapshot
            if vals == last:
                continue
            last = vals
            v = np.array(vals).reshape(levels, 4)
            if not v[0, 2] > v[0, 0]:
                continue
            ts.append(int(row[it]) * 1e-6)        # tardis timestamps are microseconds
            B.append(v[:, 0]); BS.append(v[:, 1]); A.append(v[:, 2]); AS.append(v[:, 3])
            if max_rows and len(ts) >= max_rows:
                break
    out = dict(time=np.array(ts), bid_p=np.array(B), bid_s=np.array(BS), ask_p=np.array(A), ask_s=np.array(AS))
    out.update(load_trades_tardis(trades_path, out["time"][-1] if len(ts) else None) if trades_path else _no_trades())
    return out


def _no_trades():
    e = np.zeros(0)
    return dict(trade_ts=e, trade_px=e, trade_qty=e, trade_sell=np.zeros(0, dtype=bool))


def load_trades_tardis(path: str, until: Optional[float] = None) -> Dict[str, np.ndarray]:
    T, P, Q, S = [], [], [], []
    with _open(path) as f:
        r = csv.reader(f)
        col = {h: i for i, h in enumerate(next(r))}
        for row in r:
            t = int(row[col["timestamp"]]) * 1e-6
            if until is not None and t > until:
                break
            T.append(t); P.append(float(row[col["price"]])); Q.append(float(row[col["amount"]]))
            S.append(row[col["side"]] == "sell")      # tardis 'side' = aggressor side
    return dict(trade_ts=np.array(T), trade_px=np.array(P), trade_qty=np.array(Q), trade_sell=np.array(S, dtype=bool))


def load_recording(book_csv: str) -> Dict[str, np.ndarray]:
    """Our own recording: LOBSTER-format book rows + <name>_times.csv (+ <name>_trades.csv)."""
    ob = np.loadtxt(book_csv, delimiter=",", ndmin=2)
    out = dict(ask_p=ob[:, 0::4] / 1e4, ask_s=ob[:, 1::4], bid_p=ob[:, 2::4] / 1e4, bid_s=ob[:, 3::4])
    tp = book_csv.replace(".csv", "_times.csv")
    out["time"] = np.loadtxt(tp, ndmin=1) if os.path.exists(tp) else np.arange(len(ob), dtype=float)
    tr = book_csv.replace(".csv", "_trades.csv")
    if os.path.exists(tr) and os.path.getsize(tr) > 0:
        t = np.loadtxt(tr, delimiter=",", ndmin=2)
        out.update(trade_ts=t[:, 0], trade_px=t[:, 1], trade_qty=t[:, 2], trade_sell=t[:, 3] > 0)
    else:
        out.update(_no_trades())
    return out
