"""Run this the morning of the demo. It checks everything the presentation
depends on and prints exactly what to fix.

    python scripts/preflight.py
"""

import os
import socket
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
ok = True


def check(name, cond, fix=""):
    global ok
    print(("  [ok]   " if cond else "  [FAIL] ") + name + ("" if cond else f"\n         -> {fix}"))
    ok &= bool(cond)


print("TRIPWIRE-LOB preflight")
check("python >= 3.9", sys.version_info >= (3, 9), "install Python 3.9+")
try:
    import numpy  # noqa: F401
    check("numpy importable", True)
except ImportError:
    check("numpy importable", False, "pip install -r requirements.txt")
from tripwire import TripwireConfig, TripwireEngine  # noqa: E402
from tripwire.features import extract_all  # noqa: E402
from tripwire.synthetic import generate  # noqa: E402

t0 = time.time()
d = generate("concept", T=40000, seed=0)
X, mid, spr = extract_all(d["bid_p"], d["bid_s"], d["ask_p"], d["ask_s"], 0.01, 10)
o = TripwireEngine(TripwireConfig()).run(X, mid, spr)
tau = d["change_points"][0]
hit = [a for a in o["alarms_r"] if tau <= a < tau + 3000]
check(f"engine detects the synthetic concept flip (delay {hit[0]-tau if hit else None} events)", bool(hit), "engine regression: run pytest")
import numpy as np  # noqa: E402
p99 = float(np.percentile(o["latency_us"], 99))
check(f"p99 latency {p99:.0f} us < 1000 us", p99 < 1000, "close other programs / plug in the laptop")
for s in ("AAPL", "AMZN", "MSFT"):
    check(f"real-data cache for {s}", os.path.exists(os.path.join(ROOT, "data", "cache", f"{s}_feat_h50.npz")),
          "python scripts/fetch_lobster.py && python scripts/run_benchmarks.py")
for f in ("cockpit/index.html", "cockpit/replays.json", "cockpit/summary.json", "results/tables.md",
          "results/figures/recurrence.png"):
    check(f"{f} present", os.path.exists(os.path.join(ROOT, f)), "python scripts/run_benchmarks.py")
s = socket.socket()
free = s.connect_ex(("127.0.0.1", 8765)) != 0
s.close()
check("port 8765 free for the cockpit", free, "another cockpit is running (fine) or use --port 8766")
r = subprocess.run([sys.executable, "-m", "pytest", "-q", os.path.join(ROOT, "tests")], capture_output=True, text=True)
check("unit tests pass: " + (r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"), r.returncode == 0, r.stdout[-800:])
print(f"\n{'READY' if ok else 'NOT READY'}  ({time.time()-t0:.0f}s).  Start the demo:  python scripts/cockpit.py")
sys.exit(0 if ok else 1)
