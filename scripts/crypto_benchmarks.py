"""Evidence B: the desk use case on historical crypto order books, with REAL regime events.

    python scripts/fetch_crypto.py          # once (Tardis free samples)
    python scripts/crypto_benchmarks.py     # every downloaded day -> results/crypto/

For each day: a frozen MLP desk model is trained on the first 20,000 book events; the guard,
desks S0-S4 and fills from real trade prints (queue-aware) run on the rest. Event windows come
from an OFFLINE change-point method on realised volatility (never from TRIPWIRE): 5 min before
to 30 min after each change point, plus the full-day totals. Descriptive (pre-registration
lists crypto backtests as evidence B, without a pass/fail bar). P&L is gross of fees; a fee
sensitivity (maker +2 bp, 0, -0.5 bp) is reported from the fill notional.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from tripwire import FeatureExtractor, TripwireEngine  # noqa: E402
from tripwire.changepoints import volatility_change_times  # noqa: E402
from tripwire.crypto_data import load_tardis  # noqa: E402
from tripwire.desk_models import MLPDeskModel  # noqa: E402
from tripwire.guard import GuardPolicy, guard_series  # noqa: E402
from tripwire.metrics import labels_from_level  # noqa: E402
from tripwire.mm import DeskParams, DrawdownGuard, run_trades_desk  # noqa: E402

from paper_desk import H, LOTS, engine_config  # noqa: E402
from desk_benchmarks import ks_controls  # noqa: E402

OUT = os.path.join(ROOT, "results", "crypto")
N_TRAIN = 20_000
DESKS = ("S0 neutral", "S1 always-on", "S2 guarded", "S3 KS-guard", "S4 drawdown-guard")


def features(d):
    fx = FeatureExtractor(0.01, H)
    T = len(d["time"])
    X = np.empty((T, 16)); mid = np.empty(T); spr = np.empty(T)
    for t in range(T):
        X[t], mid[t], spr[t] = fx.update(d["bid_p"][t], d["bid_s"][t] * 1000.0, d["ask_p"][t], d["ask_s"][t] * 1000.0)
    return X, mid, spr


def run_day(args):
    book_path, trades_path, max_rows = args
    sym = os.path.basename(os.path.dirname(book_path))
    day = os.path.basename(book_path)[:10]
    d = load_tardis(book_path, trades_path, max_rows=max_rows)
    T = len(d["time"])
    if T < N_TRAIN + 10_000:
        return dict(symbol=sym, day=day, skipped=f"only {T} book events")
    X, mid, spr = features(d)
    y = labels_from_level(mid, H)
    model = MLPDeskModel().fit(X[:N_TRAIN - H], y[:N_TRAIN - H])
    f = model.predict(X)
    eng = TripwireEngine(replace(engine_config(), monitor_r_ref="external"))
    o = eng.run(X, mid, spr, mid, record_weights=False, f_ext=f)
    start = max(int(o["t_live"]), N_TRAIN)
    live = np.arange(T) >= start
    f_mat = np.r_[np.zeros(H), f[:T - H]]
    g, guard = guard_series(o["alarm_f"], o["alarm_r"], o["y_matured"], f_mat, live, GuardPolicy(caution_hold=3 * H + 200))
    ones = {k: np.ones(T) for k in ("skew", "width", "size", "inventory")}
    Z = np.clip((X - eng.mu) / eng.sd, -6, 6)
    ks_skew, _ = ks_controls(Z, T, start)
    controls = {"S0 neutral": dict(ones, skew=np.zeros(T)), "S1 always-on": ones, "S2 guarded": g,
                "S3 KS-guard": dict(ones, skew=ks_skew), "S4 drawdown-guard": ones}
    # tick inferred from the data: smallest positive price step in the top of book
    steps = np.diff(np.unique(np.round(d["bid_p"][:5000, :3].ravel(), 8)))
    tick = float(np.round(steps[steps > 0].min(), 8)) if np.any(steps > 0) else 0.01
    params = DeskParams(tick=tick, lot=LOTS.get(sym, 0.01), latency=0.001, maker_fee_per_unit=0.0)
    f_ticks = f * 0.01 / tick                       # model works in 0.01 units; desk in exchange ticks
    res = {k: run_trades_desk(d, f_ticks, controls[k], params, start, T,
                              pnl_guard=DrawdownGuard() if k.startswith("S4") else None) for k in DESKS}
    ts = d["time"]
    cps = [c for c in volatility_change_times(ts[start:], mid[start:] * 0.01) if c > ts[start] + 300]
    out = dict(symbol=sym, day=day, events=T - start, desk_r2=float(1 - np.sum((y[start:T - H] - f[start:T - H]) ** 2) / np.sum(y[start:T - H] ** 2)),
               n_changepoints=len(cps), distrust_share=float(np.mean(g["state"][start:] == "DISTRUST")), desks={})
    for k, r in res.items():
        tt = ts[start:T]
        win = []
        for c in cps:
            i, j = np.searchsorted(tt, c - 300), min(np.searchsorted(tt, c + 1800), len(tt) - 1)
            win.append(float(r["pnl"][j] - r["pnl"][max(i - 1, 0)]))
        notional = sum(x.qty * x.price for x in r["fills"])
        out["desks"][k] = dict(total=r["total"], fills=r["n_fills"], markout30=r.get("markout", {}).get(30.0, 0.0),
                               max_dd=r["max_drawdown"], window_pnl=float(np.sum(win)), notional=notional)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "crypto"))
    ap.add_argument("--max-rows", type=int, default=400_000, help="book events per day (memory/time cap)")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for book in sorted(glob.glob(os.path.join(a.data, "*", "*", "*_book.csv.gz"))):
        tr = book.replace("_book.csv.gz", "_trades.csv.gz")
        jobs.append((book, tr if os.path.exists(tr) else None, a.max_rows))
    if not jobs:
        print("no data: run python scripts/fetch_crypto.py first")
        return []
    t0 = time.time()
    with Pool() as p:
        rows = p.map(run_day, jobs, chunksize=1)
    L = ["# Desk use case on historical crypto books (evidence B)", "",
         "Event windows = offline volatility change points (5 min before → 30 min after), never from TRIPWIRE. "
         "P&L in quote currency, gross of fees; fee sensitivity uses fill notional.", "",
         "| day | symbol | desk | day P&L | P&L in event windows | fills | markout 30 s | fees @ +2 bp | fees @ −0.5 bp |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "skipped" in r:
            L.append(f"| {r['day']} | {r['symbol']} | skipped: {r['skipped']} | | | | | | |")
            continue
        for k, v in r["desks"].items():
            L.append(f"| {r['day']} | {r['symbol']} | {k} | {v['total']:,.2f} | {v['window_pnl']:,.2f} | {v['fills']:,} | {v['markout30']:,.2f} | "
                     f"{-2e-4 * v['notional']:,.2f} | {0.5e-4 * v['notional']:,.2f} |")
        L.append(f"| {r['day']} | {r['symbol']} | *{r['n_changepoints']} change points, desk-model R² {r['desk_r2']:.3f}, DISTRUST {100 * r['distrust_share']:.0f} % of the day* | | | | | | |")
    ok = [r for r in rows if "skipped" not in r]
    if ok:
        d = np.array([r["desks"]["S2 guarded"]["window_pnl"] - r["desks"]["S1 always-on"]["window_pnl"] for r in ok])
        L += ["", f"Across {len(ok)} days: S2 − S1 P&L inside change-point windows: mean {d.mean():,.2f}, S2 ahead on {int(np.sum(d > 0))}/{len(d)} days."]
    md = "\n".join(L)
    with open(os.path.join(OUT, "crypto_tables.md"), "w") as f:
        f.write(md)
    with open(os.path.join(OUT, "crypto_results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    print(md)
    print(f"done in {time.time() - t0:.0f}s")
    return rows


if __name__ == "__main__":
    main()
