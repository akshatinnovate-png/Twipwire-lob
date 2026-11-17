"""Download the LOBSTER 10-level sample files (AAPL, AMZN, MSFT; 21 June 2012).

Official source: https://lobsterdata.com/info/DataSamples.php (free sample zips).
If that site is unreachable, this script pulls byte-identical copies that public
GitHub repositories host via Git LFS. Usage:  python scripts/fetch_lobster.py
"""

import os
import sys
import urllib.request

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lobster")
M = "https://media.githubusercontent.com/media"
SRC = {
    "AAPL": f"{M}/satelliteprogrammer/trading-systems/beff0fca9d8b3946a954fcdf72a165d95ed75f3f/data/lobster",
    "MSFT": f"{M}/satelliteprogrammer/trading-systems/beff0fca9d8b3946a954fcdf72a165d95ed75f3f/data/lobster",
    "AMZN": f"{M}/kpetridis24/lobsim/0cb48ed89a9cd5568e974d988214cfbebf51ca51/sample_data",
}


def main():
    os.makedirs(ROOT, exist_ok=True)
    for sym, base in SRC.items():
        for kind in ("orderbook", "message"):
            name = f"{sym}_2012-06-21_34200000_57600000_{kind}_10.csv"
            dst = os.path.join(ROOT, name)
            if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
                print("have", name)
                continue
            print("downloading", name, "...", flush=True)
            urllib.request.urlretrieve(f"{base}/{name}", dst)
            if os.path.getsize(dst) < 1_000_000:
                sys.exit(f"{name}: got a Git-LFS pointer, not data; use lobsterdata.com instead")
    print("done ->", os.path.abspath(ROOT))


if __name__ == "__main__":
    main()
