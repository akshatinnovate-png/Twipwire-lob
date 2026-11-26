"""Record a BTC-USDT 10-level book (Binance spot REST, stdlib only) into a
LOBSTER-format CSV so it runs through the identical pipeline.

    python scripts/record_binance.py --minutes 45          # before the demo: lets the cockpit's BTC source start live instantly

Polls /api/v3/depth?limit=10 every --interval seconds (REST weight 2; 0.25 s
stays far inside Binance's limit). Only snapshots that differ from the previous
one are written (event time = book change). Prices are written *10000 like
LOBSTER; load with tripwire.lobster.load_lobster(path) and tick_size=0.01.
NOTE: not exercised in our CI sandbox (exchange hosts are blocked there);
run it on a laptop. Hosts are tried in turn (data-api.binance.vision is Binance's
public market-data mirror; api.binance.us works from the US).
"""

import argparse
import json
import os
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--interval", type=float, default=0.25)
    ap.add_argument("--hosts", default="data-api.binance.vision,api.binance.com,api1.binance.com,api.binance.us")
    ap.add_argument("--out", default=None, help="default data/btc/<symbol>_<date>.csv (the cockpit's BTC source reads btcusdt_*)")
    a = ap.parse_args()
    a.out = a.out or f"data/btc/{a.symbol.lower()}_{time.strftime('%Y%m%d_%H%M')}.csv"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    hosts = [h if h.startswith("http") else f"https://{h}" for h in a.hosts.split(",")]
    hi = 0
    end = time.time() + 60 * a.minutes
    last = None
    n = 0
    last_tid = None
    with open(a.out, "w") as f, open(a.out.replace(".csv", "_times.csv"), "w") as ft, \
            open(a.out.replace(".csv", "_trades.csv"), "w") as fr:
      try:
        while time.time() < end:
            t0 = time.time()
            try:
                url = f"{hosts[hi % len(hosts)]}/api/v3/depth?symbol={a.symbol}&limit=10"
                with urllib.request.urlopen(url, timeout=5) as r:
                    book = json.load(r)
                row = []
                for l in range(10):
                    ap_, as_ = book["asks"][l]
                    bp_, bs_ = book["bids"][l]
                    row += [round(float(ap_) * 1e4), float(as_), round(float(bp_) * 1e4), float(bs_)]
                # trades since the last poll (aggressor side: m = buyer is maker -> seller aggressed)
                tq = f"{hosts[hi % len(hosts)]}/api/v3/aggTrades?symbol={a.symbol}&limit=1000" + (f"&fromId={last_tid + 1}" if last_tid is not None else "")
                with urllib.request.urlopen(tq, timeout=5) as r:
                    for tr in json.load(r):
                        if last_tid is not None and tr["a"] <= last_tid:
                            continue
                        if last_tid is not None:
                            fr.write(f"{tr['T'] / 1000:.3f},{tr['p']},{tr['q']},{1 if tr['m'] else 0}\n")
                        last_tid = tr["a"]
                if row != last:
                    f.write(",".join(str(v) for v in row) + "\n")
                    ft.write(f"{t0:.3f}\n")
                    last = row
                    n += 1
                    if n % 100 == 0:
                        f.flush()
                        ft.flush()
                        fr.flush()
                    if n % 500 == 0:
                        print(f"  {n} book events recorded", flush=True)
            except Exception as e:  # keep recording through transient errors; try the next host
                print("warn:", hosts[hi % len(hosts)], e)
                hi += 1
            time.sleep(max(0.0, a.interval - (time.time() - t0)))
      except KeyboardInterrupt:
        print("stopped early")
    print(f"wrote {n} book events to {a.out}")


if __name__ == "__main__":
    main()
