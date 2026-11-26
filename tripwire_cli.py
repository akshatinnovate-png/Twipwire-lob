"""tripwire: one command for everything (also the backend the desktop app bundles).

    tripwire serve [--source synthetic|AAPL|AMZN|MSFT|BTC] [--port 8765] [--open]
    tripwire paper [--symbols BTCUSDT,ETHUSDT] [--minutes 0]    # live paper desk (0 = run until stopped)
    tripwire report                                              # track record of the paper desk (H4/H5)
    tripwire record [--minutes 45]                               # record the live BTC book for warm starts
    tripwire fetch lobster | crypto [--months 3]                 # download free sample data
    tripwire cache                                               # build LOBSTER feature caches
    tripwire bench desk | crypto | forecast                      # reproduce the result tables
    tripwire preflight                                           # stage checklist
    tripwire home                                                # print the data/results folder

Data and results live in a writable home folder: $TRIPWIRE_HOME, or --home, or the repository
when run from source (the desktop app passes its user-data folder).
"""

from __future__ import annotations

import argparse
import os
import sys


def _code_root():
    if getattr(sys, "frozen", False):                      # PyInstaller bundle
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


CODE = _code_root()
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "scripts"))


def _ssl_certs():
    """Bundled Python may not find the OS certificate store (macOS especially): use certifi's."""
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass


def _set_home(home):
    if home:
        os.environ["TRIPWIRE_HOME"] = os.path.abspath(home)
    elif getattr(sys, "frozen", False) and not os.environ.get("TRIPWIRE_HOME"):
        os.environ["TRIPWIRE_HOME"] = os.path.join(os.path.expanduser("~"), "TripwireLOB")
    h = os.environ.get("TRIPWIRE_HOME") or CODE
    for sub in ("data", "results"):
        os.makedirs(os.path.join(h, sub), exist_ok=True)
    cal_src, cal_dst = os.path.join(CODE, "data", "events.csv"), os.path.join(h, "data", "events.csv")
    if os.path.exists(cal_src) and not os.path.exists(cal_dst):
        import shutil
        shutil.copy(cal_src, cal_dst)                      # editable copy of the events calendar
    return h


def cmd_serve(a):
    import threading
    import time
    import webbrowser
    import cockpit
    if a.btc_hosts:
        cockpit.FEED_HOSTS = a.btc_hosts.split(",")
    src = a.source if a.source in ("synthetic", "BTC") else a.source.upper()
    srv = cockpit.serve(a.host, a.port, src)
    url = f"http://{a.host}:{srv.server_address[1]}"
    # machine-readable line: the desktop app waits for it
    print(f"TRIPWIRE_READY {url}", flush=True)
    if a.open:
        def opener():
            s = cockpit.get_session()
            for _ in range(600):
                if s and s.ready():
                    break
                time.sleep(0.5)
            webbrowser.open(url)
        threading.Thread(target=opener, daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def cmd_paper(a):
    import paper_desk
    sys.argv = ["paper_desk"]
    argv = ["--symbols", a.symbols]
    if a.minutes:
        argv += ["--minutes", str(a.minutes)]
    if a.hosts:
        argv += ["--hosts", a.hosts]
    paper_desk.main(argv)


def cmd_report(a):
    import track_record
    print(track_record.report())


def cmd_record(a):
    import record_binance
    sys.argv = ["record_binance", "--minutes", str(a.minutes)]
    record_binance.main()


def cmd_fetch(a):
    if a.what == "lobster":
        import fetch_lobster
        sys.argv = ["fetch_lobster"]
        fetch_lobster.main()
    else:
        import fetch_crypto
        fetch_crypto.main(["--months", str(a.months), "--symbols", a.symbols])


def cmd_cache(a):
    import build_cache
    sys.argv = ["build_cache"]
    build_cache.main()


def cmd_bench(a):
    if a.what == "desk":
        import desk_benchmarks
        sys.argv = ["desk_benchmarks"]
        desk_benchmarks.main()
    elif a.what == "crypto":
        import crypto_benchmarks
        crypto_benchmarks.main([])
    else:
        import run_benchmarks
        sys.argv = ["run_benchmarks"]
        run_benchmarks.main()


def cmd_preflight(a):
    sys.argv = ["preflight"]
    import preflight  # noqa: F401  (runs the checklist on import and exits)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tripwire", description="TRIPWIRE-LOB: model-risk guard for order-book trading models")
    ap.add_argument("--home", default=None, help="folder for data/ and results/ (default: $TRIPWIRE_HOME or the repo)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="live cockpit")
    s.add_argument("--source", default="synthetic")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=0, help="0 = any free port")
    s.add_argument("--open", action="store_true", help="open the browser once warmed up")
    s.add_argument("--btc-hosts", default=None)
    s.set_defaults(fn=cmd_serve)
    s = sub.add_parser("paper", help="live paper desk on Binance books")
    s.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    s.add_argument("--minutes", type=float, default=0)
    s.add_argument("--hosts", default=None, help="comma-separated feed hosts (default: Binance public hosts)")
    s.set_defaults(fn=cmd_paper)
    sub.add_parser("report", help="paper-desk track record").set_defaults(fn=cmd_report)
    s = sub.add_parser("record", help="record the live BTC book")
    s.add_argument("--minutes", type=float, default=45)
    s.set_defaults(fn=cmd_record)
    s = sub.add_parser("fetch", help="download sample data")
    s.add_argument("what", choices=["lobster", "crypto"])
    s.add_argument("--months", type=int, default=3)
    s.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    s.set_defaults(fn=cmd_fetch)
    sub.add_parser("cache", help="build LOBSTER feature caches").set_defaults(fn=cmd_cache)
    s = sub.add_parser("bench", help="reproduce result tables")
    s.add_argument("what", choices=["desk", "crypto", "forecast"])
    s.set_defaults(fn=cmd_bench)
    sub.add_parser("preflight", help="stage checklist").set_defaults(fn=cmd_preflight)
    sub.add_parser("home", help="print the data folder").set_defaults(fn=lambda a: print(os.environ.get("TRIPWIRE_HOME") or CODE))
    a = ap.parse_args(argv)
    _ssl_certs()
    _set_home(a.home)
    a.fn(a)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()                       # benchmarks use a process pool
    main()
