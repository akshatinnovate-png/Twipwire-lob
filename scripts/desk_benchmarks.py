"""Desk use case: does the TRIPWIRE guard save a market maker money? Runs exactly the
experiments fixed in docs/PREREGISTRATION.md and grades H1-H3 against their pass criteria.

    python scripts/desk_benchmarks.py            # ~15 min on 4 cores
    python scripts/desk_benchmarks.py --quick    # 1 slice per stock, no sensitivity grid

Writes results/desk/desk_results.json, results/desk/desk_tables.md, results/desk/*.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..")
ROOT = os.environ.get("TRIPWIRE_HOME") or CODE   # writable: data/, results/
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

from tripwire import TripwireEngine  # noqa: E402
from tripwire.baselines import ks_crit, ks_stat  # noqa: E402
from tripwire.desk_models import MLPDeskModel  # noqa: E402
from tripwire.guard import GuardPolicy, guard_series  # noqa: E402
from tripwire.lobster import load_lobster  # noqa: E402
from tripwire.metrics import labels_from_level, level_from_labels  # noqa: E402
from tripwire.mm import DeskParams, DrawdownGuard, run_lobster_desk  # noqa: E402
from tripwire.pool import ridge_fit  # noqa: E402

import run_benchmarks as rb  # noqa: E402

OUT = os.path.join(ROOT, "results", "desk")
SYMS = ("AAPL", "AMZN", "MSFT")
H = 50
N_TRAIN = 20_000                    # desk model trained on the first 20k events, then frozen
WINDOW_S = 1800.0                   # 30 minutes after the break
DESKS = ("S0 neutral", "S1 always-on", "S2 guarded", "S2-F only", "S2-R only", "S3 KS-guard", "S4 drawdown-guard")


def load_book(sym):
    cache = os.path.join(ROOT, "data", "cache", f"{sym}_desk.npz")
    base = os.path.join(ROOT, "data", "lobster", f"{sym}_2012-06-21_34200000_57600000_")
    if os.path.exists(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    d = load_lobster(base + "orderbook_10.csv", base + "message_10.csv")
    np.savez(cache, **d)
    return d


def ks_controls(Z, T, start):
    """S3: rolling two-sample KS on every feature (as baseline B3); alpha pulled 3,000 events per alarm."""
    skew = np.ones(T)
    ref, cur, every = 3000, 600, 100
    crit = ks_crit(ref, cur, 0.01 / Z.shape[1])
    cool = 0
    t = start
    alarms = []
    while t < T:
        if t >= ref + cur and cool <= 0:
            R, C = Z[t - ref - cur:t - cur], Z[t - cur:t]
            if any(ks_stat(np.sort(R[:, k]), C[:, k]) > crit for k in range(Z.shape[1])):
                alarms.append(t)
                skew[t:t + 3000] = 0.0
                cool = 3000
        cool -= every
        t += every
    return skew, alarms


def desk_model(X, y, seed):
    m = MLPDeskModel(seed=seed).fit(X[:N_TRAIN - H], y[:N_TRAIN - H])
    return m.predict(X)


def run_stream(book, X, level, spr, tag, desks=DESKS, alpha="mlp", params=None, seed=0, end=None):
    """Engine + guard + every desk on one stream. Returns per-desk summaries and series."""
    T = len(level) if end is None else end
    y = labels_from_level(level, H)
    if alpha == "mlp":
        f = desk_model(X, y, seed)
        cfg = replace(rb.REAL_CFG, monitor_r_ref="external", seed=seed)
        eng = TripwireEngine(cfg)
        o = eng.run(X[:T], level[:T], spr[:T], level[:T], record_weights=False, f_ext=f[:T])
    else:                                   # secondary: TRIPWIRE's own adaptive pool is the alpha
        eng = TripwireEngine(replace(rb.REAL_CFG, seed=seed))
        o = eng.run(X[:T], level[:T], spr[:T], level[:T], record_weights=False)
        f = o["yhat"]
    start = int(o["t_live"])
    live = np.arange(T) >= start
    f_mat = np.r_[np.zeros(H), f[:T - H]]                  # forecast for the label maturing at t
    y_mat = o["y_matured"]
    controls = {}
    for name, pol in (("S2 guarded", GuardPolicy()), ("S2-F only", GuardPolicy(use_r=False)),
                      ("S2-R only", GuardPolicy(use_f=False))):
        if name in desks:
            controls[name], _g = guard_series(o["alarm_f"][:T], o["alarm_r"][:T], y_mat, f_mat, live, pol)
            if name == "S2 guarded":
                transitions = _g.transitions
    ones = {k: np.ones(T) for k in ("skew", "width", "size", "inventory")}
    controls["S0 neutral"] = dict(ones, skew=np.zeros(T))
    controls["S1 always-on"] = ones
    controls["S4 drawdown-guard"] = ones
    if "S3 KS-guard" in desks:
        Z = np.clip((X[:T] - eng.mu) / eng.sd, -6, 6)
        ks_skew, ks_alarms = ks_controls(Z, T, start)
        controls["S3 KS-guard"] = dict(ones, skew=ks_skew)
    params = params or DeskParams()
    res = {}
    for name in desks:
        pg = DrawdownGuard() if name == "S4 drawdown-guard" else None
        res[name] = run_lobster_desk(book, f, controls[name], params, start, T, offset=book.get("_offset"), pnl_guard=pg)
    states = controls.get("S2 guarded", {}).get("state")
    return res, dict(start=start, alarms_f=o["alarms_f"].tolist(), alarms_r=o["alarms_r"].tolist(),
                     transitions=[(int(a), b, c, d) for a, b, c, d in transitions] if "S2 guarded" in desks else [],
                     desk_r2=float(1 - np.sum((y[start:T - H] - f[start:T - H]) ** 2) / np.sum(y[start:T - H] ** 2)),
                     states=states)


def window_stats(r, ts, a, b):
    """P&L change and adverse selection (30 s markout) of fills inside [a, b) seconds."""
    tt = ts[-len(r["pnl"]):]
    i, j = np.searchsorted(tt, a), min(np.searchsorted(tt, b), len(tt) - 1)
    fills = [x for x in r["fills"] if a <= x.ts < b]
    if fills:
        side = np.array([x.side for x in fills]); q = np.array([x.qty for x in fills])
        fm = np.array([x.mid for x in fills]); ft = np.array([x.ts for x in fills])
        mids = r["_mids"]
        k = np.clip(np.searchsorted(tt, ft + 30.0), 0, len(tt) - 1)
        adverse = float(-np.sum(side * q * (mids[k] - fm)))          # positive = money lost to toxic flow
    else:
        adverse = 0.0
    return dict(pnl=float(r["pnl"][j] - r["pnl"][max(i - 1, 0)]), adverse=adverse, fills=len(fills))


def _mids(book, start, T, off):
    m = 0.5 * (book["bid_p"][start:T, 0] + book["ask_p"][start:T, 0])
    if off is not None:
        m = m + off[start:T] * 0.01
    return m


# ------------------------------------------------------------------------- experiments
def normal_day(sym):
    book = load_book(sym)
    X, mid, spr = rb.real_features(sym, H)
    res, info = run_stream(book, X, mid, spr, sym)
    out = {k: dict(total=v["total"], fills=v["n_fills"], spread=v.get("spread_capture", 0.0),
                   markout={str(h): m for h, m in v.get("markout", {}).items()}, toxic=v.get("toxic_share"),
                   max_dd=v["max_drawdown"], fees=v["fees"]) for k, v in res.items()}
    pnl_series = {k: v["pnl"][::200].tolist() for k, v in res.items() if k in ("S0 neutral", "S1 always-on", "S2 guarded")}
    return dict(symbol=sym, desks=out, start=info["start"], desk_r2=info["desk_r2"], n_alarms_f=len(info["alarms_f"]),
                n_alarms_r=len(info["alarms_r"]), transitions=info["transitions"][:200], pnl_series=pnl_series)


def injected(args):
    sym, off0, beta, n, alpha, params, desks = args
    book_full = load_book(sym)
    X, mid, spr = rb.real_features(sym, H)
    X = X[off0:off0 + n].copy(); spr = spr[off0:off0 + n]; mid = mid[off0:off0 + n]
    book = {k: v[off0:off0 + n] for k, v in book_full.items()}
    y = labels_from_level(mid, H)
    tau = n // 2 + 10_000
    Zs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    th, _ = ridge_fit(Zs[: n - H], y[: n - H], 10.0)
    g = Zs @ th
    # inject the break into the PRICE PATH, smoothly: from tau on, each event's mid move gets
    # (beta-1)/h times the signal of the last h events, so the h-event-ahead move carries
    # (beta-1) x signal (as in the synthetic generator) and the shifted book stays realistic.
    inc = np.zeros(n)
    cs = np.cumsum(np.r_[0.0, g])
    for t in range(tau, n - 1):
        lo = max(tau, t - H + 1)
        inc[t + 1] = (beta - 1.0) * (cs[t + 1] - cs[lo]) / H
    off = np.round(np.cumsum(inc))                 # whole ticks: the shifted book stays on the price grid
    rel = mid - mid[0]
    lv = rel + off
    X[H:, 15] = lv[H:] - lv[:-H]
    book["_offset"] = off
    ts = book["time"]
    t_end = min(int(np.searchsorted(ts, ts[tau] + WINDOW_S)) + 1, n)
    res, info = run_stream(book, X, lv, spr, f"{sym}@{off0}", desks=desks, alpha=alpha, params=params, end=t_end)
    out = {}
    for k, v in res.items():
        v["_mids"] = _mids(book, info["start"], t_end, off)
        out[k] = window_stats(v, ts[:t_end], ts[tau], ts[tau] + WINDOW_S)
    r_after = [a for a in info["alarms_r"] if a >= tau]
    return dict(symbol=sym, offset=off0, beta=beta, alpha=alpha, tau=tau, desks=out,
                first_r_alarm_after=(r_after[0] - tau) if r_after else None,
                params=dict(latency=params.latency, cancel=params.cancel_model) if params else None)


# ------------------------------------------------------------------------- grading
def grade(R):
    G = {}
    nd = R["normal_days"]
    ok = [d["desks"]["S2 guarded"]["total"] >= d["desks"]["S1 always-on"]["total"] - 0.05 * abs(d["desks"]["S1 always-on"]["total"]) for d in nd]
    G["H1"] = dict(passed=sum(ok) >= 2, detail=f"{sum(ok)}/{len(ok)} days within 5 % of always-on (or better)")
    inj = [r for r in R["injected"] if r["params"]["latency"] == 0.001 and r["params"]["cancel"] == "pro_rata" and r["alpha"] == "mlp"]
    a1 = np.mean([r["desks"]["S1 always-on"]["adverse"] for r in inj])
    a2 = np.mean([r["desks"]["S2 guarded"]["adverse"] for r in inj])
    red = 1 - a2 / a1 if a1 > 0 else float("nan")
    G["H2a"] = dict(passed=bool(a1 > 0 and red >= 0.30), detail=f"adverse selection S1 {a1:,.0f} vs S2 {a2:,.0f} $ (reduction {100 * red:.0f} %)")
    wins = sum(r["desks"]["S2 guarded"]["pnl"] > r["desks"]["S1 always-on"]["pnl"] for r in inj)
    G["H2b"] = dict(passed=wins >= 9, detail=f"S2 beats S1 in {wins}/{len(inj)} slices")
    m = {k: float(np.mean([r["desks"][k]["pnl"] for r in inj])) for k in ("S2 guarded", "S3 KS-guard", "S4 drawdown-guard")}
    G["H3"] = dict(passed=m["S2 guarded"] >= m["S3 KS-guard"] and m["S2 guarded"] >= m["S4 drawdown-guard"],
                   detail=f"mean post-break P&L: S2 {m['S2 guarded']:,.0f}, S3 {m['S3 KS-guard']:,.0f}, S4 {m['S4 drawdown-guard']:,.0f} $")
    return G


def tables(R, G):
    L = ["# Desk use case: results", "",
         "Settings and pass criteria were fixed in `docs/PREREGISTRATION.md` before any of this ran. "
         "P&L is in USD per desk, gross of fees except the LOBSTER rebate; one lot is 100 shares.", "",
         "## Pre-registered hypotheses", "", "| hypothesis | result | detail |", "|---|---|---|"]
    for k, v in G.items():
        L.append(f"| {k} | {'PASS' if v['passed'] else 'FAIL'} | {v['detail']} |")
    L += ["", "## H1: full real days, no injected break (desk model = frozen MLP)", "",
          "| symbol | desk | P&L $ | fills | spread captured $ | markout 30 s $ | toxic fills % | max drawdown $ |", "|---|---|---|---|---|---|---|---|"]
    for d in R["normal_days"]:
        for k, v in d["desks"].items():
            L.append(f"| {d['symbol']} | {k} | {v['total']:,.0f} | {v['fills']:,} | {v['spread']:,.0f} | {v['markout'].get('30.0', 0):,.0f} | "
                     f"{100 * (v['toxic'] or 0):.0f} | {v['max_dd']:,.0f} |")
        L.append(f"| {d['symbol']} | *desk model out-of-sample R²* | {d['desk_r2']:.4f} | | | | | |")
    for alpha in ("mlp", "pool"):
        inj = [r for r in R["injected"] if r["alpha"] == alpha and r["params"]["latency"] == 0.001 and r["params"]["cancel"] == "pro_rata"]
        if not inj:
            continue
        L += ["", f"## H2/H3: 30 minutes after a real-data break (β = −3), desk model = {'frozen MLP' if alpha == 'mlp' else 'TRIPWIRE adaptive pool'}", "",
              "| desk | mean P&L $ | mean adverse selection $ (lost) | mean fills | wins vs S1 |", "|---|---|---|---|---|"]
        for k in inj[0]["desks"]:
            pn = [r["desks"][k]["pnl"] for r in inj]
            ad = [r["desks"][k]["adverse"] for r in inj]
            fl = [r["desks"][k]["fills"] for r in inj]
            w = sum(r["desks"][k]["pnl"] > r["desks"]["S1 always-on"]["pnl"] for r in inj)
            L.append(f"| {k} | {np.mean(pn):,.0f} ± {1.96 * np.std(pn) / np.sqrt(len(pn)):,.0f} | {np.mean(ad):,.0f} | {np.mean(fl):,.0f} | {w}/{len(inj)} |")
        det = [r["first_r_alarm_after"] for r in inj]
        L.append("")
        L.append(f"Monitor R fired after the break in {sum(d is not None for d in det)}/{len(det)} slices "
                 f"(median {np.median([d for d in det if d is not None]) if any(d is not None for d in det) else float('nan'):.0f} events).")
    sens = [r for r in R["injected"] if r["alpha"] == "mlp" and not (r["params"]["latency"] == 0.001 and r["params"]["cancel"] == "pro_rata")]
    if sens:
        L += ["", "## Sensitivity: fill and latency assumptions (S2 − S1, mean post-break P&L $)", "",
              "| latency | cancellations | S1 P&L | S2 P&L | S2 − S1 | S2 wins |", "|---|---|---|---|---|---|"]
        keys = sorted({(r["params"]["latency"], r["params"]["cancel"]) for r in R["injected"] if r["alpha"] == "mlp"})
        for lat, can in keys:
            rs = [r for r in R["injected"] if r["alpha"] == "mlp" and r["params"]["latency"] == lat and r["params"]["cancel"] == can]
            s1 = np.mean([r["desks"]["S1 always-on"]["pnl"] for r in rs]); s2 = np.mean([r["desks"]["S2 guarded"]["pnl"] for r in rs])
            w = sum(r["desks"]["S2 guarded"]["pnl"] > r["desks"]["S1 always-on"]["pnl"] for r in rs)
            L.append(f"| {1000 * lat:.0f} ms | {can} | {s1:,.0f} | {s2:,.0f} | {s2 - s1:,.0f} | {w}/{len(rs)} |")
    return "\n".join(L)


def figures(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, HERE)
    from report import _style, INK, ACCENT
    _style(plt)
    inj = [r for r in R["injected"] if r["alpha"] == "mlp" and r["params"]["latency"] == 0.001 and r["params"]["cancel"] == "pro_rata"]
    if inj:
        ks = list(inj[0]["desks"].keys())
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        pn = [np.mean([r["desks"][k]["pnl"] for r in inj]) for k in ks]
        ad = [np.mean([r["desks"][k]["adverse"] for r in inj]) for k in ks]
        cols = [INK if k == "S2 guarded" else (ACCENT if k.startswith("S2") else "#aaa") for k in ks]
        ax[0].barh(ks, pn, color=cols); ax[0].invert_yaxis(); ax[0].set_title("mean desk P&L, 30 min after the break ($)")
        ax[1].barh(ks, ad, color=cols); ax[1].invert_yaxis(); ax[1].set_title("money lost to toxic fills ($, lower is better)")
        fig.tight_layout(); fig.savefig(os.path.join(OUT, "desk_break.png"), dpi=130); plt.close(fig)
    for d in R["normal_days"]:
        fig, ax = plt.subplots(figsize=(11, 3.5))
        for k, c in (("S0 neutral", "#aaa"), ("S1 always-on", ACCENT), ("S2 guarded", INK)):
            ax.plot(d["pnl_series"][k], color=c, label=k, lw=1.3)
        ax.set_title(f"{d['symbol']} full real day: cumulative desk P&L ($), every 200 events")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, f"desk_day_{d['symbol']}.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    base = DeskParams()
    jobs = []
    for sym in SYMS:
        n_all = len(rb.real_features(sym, H)[1])
        offs = np.linspace(10_000, n_all - 120_000, 1 if a.quick else 4).astype(int)
        for off in offs:
            jobs.append((sym, int(off), -3.0, 120_000, "mlp", base, DESKS))
            if not a.quick:
                jobs.append((sym, int(off), -3.0, 120_000, "pool", base, ("S0 neutral", "S1 always-on", "S2 guarded")))
                for lat in (0.0, 0.010):
                    jobs.append((sym, int(off), -3.0, 120_000, "mlp", replace(base, latency=lat), ("S1 always-on", "S2 guarded")))
                jobs.append((sym, int(off), -3.0, 120_000, "mlp", replace(base, cancel_model="back"), ("S1 always-on", "S2 guarded")))
    with Pool() as p:
        for s in SYMS:
            load_book(s)
        nd_async = p.map_async(normal_day, SYMS, chunksize=1)
        inj = p.map(injected, jobs, chunksize=1)
        nd = nd_async.get()
    R = dict(normal_days=nd, injected=inj, elapsed_s=time.time() - t0)
    G = grade(R)
    R["grades"] = G
    with open(os.path.join(OUT, "desk_results.json"), "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (float(o) if isinstance(o, np.floating) else (int(o) if isinstance(o, np.integer) else str(o))))
    md = tables(R, G)
    with open(os.path.join(OUT, "desk_tables.md"), "w") as f:
        f.write(md)
    figures(R)
    print(md)
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
