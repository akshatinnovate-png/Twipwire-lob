"""LOBSTER loader (https://lobsterdata.com, 10-level sample files, 21 June 2012).

orderbook_10.csv columns, per level l = 1..10:
  ask_price_l, ask_size_l, bid_price_l, bid_size_l   (prices in USD * 10000)
message_10.csv: time, type, order_id, size, price, direction.
We drop type-5 (hidden execution: book unchanged) and type-7 (halt) events, so
every remaining event changes the visible book. Empty levels (LOBSTER dummy
prices +-9999999999) are filled with the nearest real level +- 1 tick, size 0.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np


def load_lobster(orderbook_csv: str, message_csv: Optional[str] = None,
                 levels: int = 10, cache: Optional[str] = None) -> Dict[str, np.ndarray]:
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    ob = np.loadtxt(orderbook_csv, delimiter=",", dtype=np.float64)
    ap = ob[:, 0::4][:, :levels] / 1e4
    asz = ob[:, 1::4][:, :levels].copy()
    bp = ob[:, 2::4][:, :levels] / 1e4
    bsz = ob[:, 3::4][:, :levels].copy()
    keep = np.ones(ob.shape[0], dtype=bool)
    times = np.arange(ob.shape[0], dtype=float)
    if message_csv:
        msg = np.loadtxt(message_csv, delimiter=",", dtype=np.float64)
        times = msg[:, 0]
        keep &= (msg[:, 1] != 5) & (msg[:, 1] != 7)
    # dummy prices on empty levels
    for side_p, side_s, sign in ((ap, asz, 1.0), (bp, bsz, -1.0)):
        bad = (np.abs(side_p) > 1e5) | (side_p <= 0)
        for l in range(1, levels):
            fix = bad[:, l]
            side_p[fix, l] = side_p[fix, l - 1] + sign * 0.01
            side_s[fix, l] = 0.0
    keep &= (ap[:, 0] > bp[:, 0]) & (np.abs(ap[:, 0]) < 1e5) & (bp[:, 0] > 0)
    out = dict(bid_p=bp[keep], bid_s=bsz[keep], ask_p=ap[keep], ask_s=asz[keep], time=times[keep])
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        np.savez_compressed(cache, **out)
    return out
