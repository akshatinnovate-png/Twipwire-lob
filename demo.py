"""One command for the stage.

    python demo.py                    # synthetic source
    python demo.py --source AAPL      # real LOBSTER replay (AAPL / AMZN / MSFT)
    python demo.py --source BTC       # live Binance book (warm-starts from data/btc/)

Checks the essentials, starts the cockpit, waits until the engine is warmed up,
then opens the browser on the cockpit. Nothing to type in front of the judges:
keys 1 / 2 / 3 / 0 inject shocks, space pauses, s switches source, r restarts.
"""

import argparse
import os
import socket
import sys
import threading
import time
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def free_port(start):
    for p in range(start, start + 20):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
        finally:
            s.close()
    raise SystemExit("no free port between %d and %d" % (start, start + 19))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic", help="synthetic | AAPL | AMZN | MSFT | BTC")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    src = a.source if a.source in ("synthetic", "BTC") else a.source.upper()

    print("TRIPWIRE-LOB demo launcher")
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise SystemExit("  numpy missing -> pip install -r requirements.txt")
    import cockpit  # scripts/cockpit.py

    if src in ("AAPL", "AMZN", "MSFT") and not os.path.exists(os.path.join(ROOT, "data", "cache", f"{src}_feat_h50.npz")):
        raise SystemExit(f"  no data for {src} -> python scripts/fetch_lobster.py && python scripts/build_cache.py")
    if src == "BTC" and not cockpit.btc_recordings_ok():
        print("  warning: no recording in data/btc/ -> the engine will warm up on the live feed (~30 min).")
        print("           record beforehand: python scripts/record_binance.py --minutes 45")

    port = free_port(a.port)
    srv = cockpit.serve("127.0.0.1", port, src)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/#cockpit"

    t0 = time.time()
    sess = cockpit.get_session()
    print(f"  warming up the engine on {src} ...", end="", flush=True)
    while not sess.ready() and time.time() - t0 < (25 if src == "BTC" else 90):
        time.sleep(0.25)
    print(" ready" if sess.ready() else " still warming (the page shows progress)")
    print(f"\n  cockpit: {url}")
    print("  keys: 1 concept flip · 2 covariate shock · 3 both · 0 calm · space pause · s next source · r restart")
    print("  Ctrl+C to stop\n")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
