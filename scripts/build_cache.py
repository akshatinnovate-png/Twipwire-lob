"""Build the real-data feature caches the cockpit needs (about 1-2 minutes per symbol).

    python scripts/fetch_lobster.py     # once, downloads LOBSTER samples
    python scripts/build_cache.py       # builds data/cache/*_feat_h{10,50}.npz
"""

import os
import sys
import time

CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT = os.environ.get("TRIPWIRE_HOME") or CODE   # writable: data/, results/
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_benchmarks import real_features  # noqa: E402

def main():
    os.makedirs(os.path.join(ROOT, "data", "cache"), exist_ok=True)
    for sym in ("AAPL", "AMZN", "MSFT"):
        for h in (50, 10):
            t0 = time.time()
            X, _, _ = real_features(sym, h)
            print(f"{sym} h={h}: {len(X):,} events cached in {time.time()-t0:.0f}s", flush=True)
    print("done -> data/cache/   (now run: python scripts/cockpit.py)")


if __name__ == "__main__":
    main()
