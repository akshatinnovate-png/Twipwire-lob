"""Download free historical crypto order-book data (Tardis.dev monthly free samples).

The first day of every month is free without an API key:
  https://datasets.tardis.dev/v1/<exchange>/<book_snapshot_25|trades>/<YYYY>/<MM>/01/<SYMBOL>.csv.gz

    python scripts/fetch_crypto.py                                  # BTCUSDT + ETHUSDT, last 3 months, binance-futures
    python scripts/fetch_crypto.py --months 6 --exchange binance --symbols BTCUSDT

Files go to data/crypto/<exchange>/<SYMBOL>/<YYYY-MM-01>_{book,trades}.csv.gz. A BTCUSDT futures
book file is large (often 1+ GB compressed per day); use --months to control the download.
Not reachable from the build sandbox; run on your laptop.
"""

import argparse
import datetime as dt
import os
import sys
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BASE = "https://datasets.tardis.dev/v1"


def month_starts(n, today=None):
    d = (today or dt.date.today()).replace(day=1)
    out = []
    for _ in range(n):
        d = (d - dt.timedelta(days=1)).replace(day=1)       # previous month's first day
        out.append(d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exchange", default="binance-futures")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    ap.add_argument("--months", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "crypto"))
    a = ap.parse_args(argv)
    for sym in a.symbols.split(","):
        d_out = os.path.join(a.out, a.exchange, sym)
        os.makedirs(d_out, exist_ok=True)
        for day in month_starts(a.months):
            for kind, name in (("book_snapshot_25", "book"), ("trades", "trades")):
                url = f"{BASE}/{a.exchange}/{kind}/{day:%Y/%m/%d}/{sym}.csv.gz"
                dst = os.path.join(d_out, f"{day:%Y-%m-%d}_{name}.csv.gz")
                if os.path.exists(dst) and os.path.getsize(dst) > 1000:
                    print("have", dst)
                    continue
                print("downloading", url, flush=True)
                try:
                    urllib.request.urlretrieve(url, dst + ".part")
                    os.replace(dst + ".part", dst)
                except Exception as e:
                    print("  failed:", e, file=sys.stderr)
    print("done ->", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
