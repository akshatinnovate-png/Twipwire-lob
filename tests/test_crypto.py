"""Evidence-B pipeline on a synthetic Tardis-format day (book_snapshot_25 + trades, gzipped)."""

import gzip
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from tripwire.synthetic import SyntheticBook  # noqa: E402


def _write_day(d, n=34000):
    os.makedirs(d, exist_ok=True)
    gen = SyntheticBook(horizon=20, seed=3)
    rng = np.random.default_rng(0)
    t0 = 1_700_000_000_000_000
    hdr = ["exchange", "symbol", "timestamp", "local_timestamp"]
    for l in range(25):
        hdr += [f"asks[{l}].price", f"asks[{l}].amount", f"bids[{l}].price", f"bids[{l}].amount"]
    with gzip.open(os.path.join(d, "2026-11-01_book.csv.gz"), "wt") as fb, \
            gzip.open(os.path.join(d, "2026-11-01_trades.csv.gz"), "wt") as ft:
        fb.write(",".join(hdr) + "\n")
        ft.write("exchange,symbol,timestamp,local_timestamp,id,side,price,amount\n")
        for i in range(n):
            bp, bs, ap, as_, _ = gen.step()
            ts = t0 + i * 200_000
            row = ["x", "BTCUSDT", str(ts), str(ts)]
            for l in range(25):
                k = min(l, 9)
                row += [f"{ap[k] + 0.01 * max(0, l - 9):.2f}", f"{as_[k] / 1000:.4f}", f"{bp[k] - 0.01 * max(0, l - 9):.2f}", f"{bs[k] / 1000:.4f}"]
            fb.write(",".join(row) + "\n")
            if rng.random() < 0.5:
                sell = rng.random() < 0.5
                ft.write(f"x,BTCUSDT,{ts + 1},{ts + 1},{i},{'sell' if sell else 'buy'},{bp[0] if sell else ap[0]:.2f},{rng.uniform(0.05, 0.5):.4f}\n")


def test_crypto_benchmark_runs_on_tardis_format(tmp_path):
    import crypto_benchmarks
    _write_day(str(tmp_path / "binance-futures" / "BTCUSDT"))
    crypto_benchmarks.OUT = str(tmp_path / "out")
    rows = crypto_benchmarks.main(["--data", str(tmp_path), "--max-rows", "40000"])
    assert len(rows) == 1 and "skipped" not in rows[0], rows
    r = rows[0]
    assert set(r["desks"]) == set(crypto_benchmarks.DESKS)
    assert r["desks"]["S1 always-on"]["fills"] > 0
    assert (tmp_path / "out" / "crypto_tables.md").exists()
