"""Reproduce every number in README / docs from scratch.

    python scripts/run_benchmarks.py                # full run (~25 min on 4 cores)
    python scripts/run_benchmarks.py --quick        # smoke run (~3 min)

Writes results/benchmarks.json, results/tables.md and results/figures/*.png.
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

CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ROOT = os.environ.get("TRIPWIRE_HOME") or CODE   # writable: data/, results/
sys.path.insert(0, CODE)

from tripwire import TripwireConfig, TripwireEngine  # noqa: E402
from tripwire.baselines import run_baselines  # noqa: E402
from tripwire.features import extract_all  # noqa: E402
from tripwire.lobster import load_lobster  # noqa: E402
from tripwire.metrics import (cost_of_delay, detection, forecast_metrics, gate_calls,  # noqa: E402
                              labels_from_level, level_from_labels, regime_oracle,
                              silent_failure_time)
from tripwire.pool import ridge_fit  # noqa: E402
from tripwire.synthetic import SCENARIOS, generate  # noqa: E402

RES = os.path.join(ROOT, "results")
DET_WINDOW = 10_000      # an alarm within this many events after a change counts as a detection
POST = 3_000             # "post-change" evaluation window
SYN_T = 40_000
SYN_CFG = TripwireConfig(horizon=10)
REAL_CFG = TripwireConfig(horizon=50, n_fit=20_000, n_ref=10_000, arl_events=60_000)

ABLATIONS = {
    "A1 no Monitor F": dict(enable_f=False),
    "A2 no Monitor R": dict(enable_r=False),
    "A3 no adaptive interval": dict(enable_aci=False),
    "A4 Monitor R on |residual| (abstract v1)": dict(monitor_r_score="abs", monitor_r_ref="pool"),
    "A5 pool only (no detectors)": dict(enable_f=False, enable_r=False),
}


# ----------------------------------------------------------------------------- core
def evaluate_stream(X, level, spread, cfg, change_points=(), ablations=ABLATIONS,
                    baselines=True, keep_series=False, b4_lambdas=None):
    h = cfg.horizon
    T = len(level)
    y = labels_from_level(level, h)
    end = T - h
    models = {}

    eng = TripwireEngine(cfg)
    o = eng.run(X, level, spread, level)
    t_live = int(o["t_live"])
    t_first = eng.t_fit_done + 1
    models["TRIPWIRE-LOB"] = dict(yhat=o["yhat"], q=o["q"], calls=o["call"],
                                  alarms=sorted(o["alarms_f"].tolist() + o["alarms_r"].tolist()),
                                  alarms_f=o["alarms_f"].tolist(), alarms_r=o["alarms_r"].tolist(),
                                  latency=o["latency_us"], weights=o["weights"], logR_f=o["logR_f"],
                                  logR_r=o["logR_r"], spawns=int(eng.pool.spawn_count))
    for name, kw in (ablations or {}).items():
        e2 = TripwireEngine(replace(cfg, **kw))
        o2 = e2.run(X, level, spread, level, record_weights=False)
        models[name] = dict(yhat=o2["yhat"], q=o2["q"], calls=o2["call"],
                            alarms=sorted(o2["alarms_f"].tolist() + o2["alarms_r"].tolist()),
                            alarms_f=o2["alarms_f"].tolist(), alarms_r=o2["alarms_r"].tolist())

    Z = np.clip((X - eng.mu) / eng.sd, -cfg.clip_z, cfg.clip_z)
    if baselines:
        th0, _ = ridge_fit(Z[:cfg.n_fit], y[:cfg.n_fit], cfg.ridge_l2)
        q_fixed = float(np.quantile(np.abs(y[:cfg.n_fit] - Z[:cfg.n_fit] @ th0), 1 - cfg.alpha))
        kw = {} if b4_lambdas is None else {"lambdas": b4_lambdas}
        bl = run_baselines(Z, y, h, t_first, cfg.n_fit, list(change_points), l2=cfg.ridge_l2, t_eval=t_live, **kw)
        for name, b in bl.items():
            q = np.full(T, q_fixed)
            calls = gate_calls(b["yhat"], q, t_first, t_live, cfg.call_budget)
            models[name] = dict(yhat=b["yhat"], q=q, calls=calls, alarms=b["alarms"], **({"lambda": b["lambda"]} if "lambda" in b else {}))

    oracle = regime_oracle(Z, y, h, list(change_points), t_live, cfg.ridge_l2)
    out = {"t_live": t_live, "T": T, "models": {}}
    for name, m in models.items():
        r = {"overall": forecast_metrics(y, m["yhat"], m["q"], m["calls"], spread, t_live, end)}
        if "lambda" in m:
            r["lambda"] = m["lambda"]
        r["n_alarms"] = len([a for a in m["alarms"] if a >= t_live])
        if "alarms_f" in m:
            r["alarms_f"] = [int(a) for a in m["alarms_f"]]
            r["alarms_r"] = [int(a) for a in m["alarms_r"]]
        per = []
        for tau in change_points:
            if tau <= t_live or tau + 500 > end:
                continue
            d = {"tau": int(tau)}
            d["post"] = forecast_metrics(y, m["yhat"], m["q"], m["calls"], spread, tau, min(tau + POST, end))
            d["cod_500"] = cost_of_delay(y, m["yhat"], oracle, tau, 500)
            d["cod_3000"] = cost_of_delay(y, m["yhat"], oracle, tau, min(POST, end - tau))
            d["sft"] = silent_failure_time(y, m["yhat"], m["alarms"], t_live, tau, end)
            d["delay_any"] = detection(m["alarms"], tau, DET_WINDOW)
            if "alarms_f" in m:
                d["delay_f"] = detection(m["alarms_f"], tau, DET_WINDOW)
                d["delay_r"] = detection(m["alarms_r"], tau, DET_WINDOW)
            per.append(d)
        r["changes"] = per
        # false alarms: before the first change, or anywhere if there is none
        first = min([c for c in change_points if c > t_live], default=end)
        def _fa(al):
            return len([a for a in al if t_live <= a < first])
        r["false_alarm_events"] = first - t_live
        r["false_alarms"] = _fa(m["alarms"])
        if "alarms_f" in m:
            r["false_alarms_f"] = _fa(m["alarms_f"])
            r["false_alarms_r"] = _fa(m["alarms_r"])
        out["models"][name] = r
    tw = models["TRIPWIRE-LOB"]
    lat = tw["latency"][t_live:]
    out["latency_us"] = {k: float(np.percentile(lat, p)) for k, p in (("p50", 50), ("p95", 95), ("p99", 99), ("max", 100))}
    if keep_series:
        out["series"] = dict(y=y, yhat=tw["yhat"], q=tw["q"], calls=tw["calls"], weights=tw["weights"],
                             logR_f=tw["logR_f"], logR_r=tw["logR_r"], oracle=oracle,
                             alarms_f=tw["alarms_f"], alarms_r=tw["alarms_r"], spread=spread,
                             b1=models.get("B1 frozen ridge", {}).get("yhat"),
                             b3_alarms=models.get("B3 rolling-KS + retrain", {}).get("alarms", []),
                             b4=models.get("B4 best fixed-forgetting RLS", {}).get("yhat"))
    return out


# ----------------------------------------------------------------------------- synthetic
def syn_task(args):
    scenario, seed, T = args
    d = generate(scenario, T=T, horizon=SYN_CFG.horizon, seed=seed)
    X, mid, spr = extract_all(d["bid_p"], d["bid_s"], d["ask_p"], d["ask_s"], 0.01, SYN_CFG.horizon)
    r = evaluate_stream(X, mid, spr, replace(SYN_CFG, seed=seed), d["change_points"])
    r["scenario"] = scenario
    r["seed"] = seed
    return r


def arl_task(args):
    kind, seed, T = args
    if kind == "synthetic":
        d = generate("calm", T=T, horizon=SYN_CFG.horizon, seed=10_000 + seed)
        X, mid, spr = extract_all(d["bid_p"], d["bid_s"], d["ask_p"], d["ask_s"], 0.01, SYN_CFG.horizon)
        cfg = replace(SYN_CFG, seed=seed)
        lv = mid
    else:  # block-shuffled real data: exchangeable by construction, real marginals
        X, mid, spr = real_features(kind, REAL_CFG.horizon)
        h = REAL_CFG.horizon
        y = labels_from_level(mid, h)[: len(mid) - h]
        X, spr = X[: len(y)], spr[: len(y)]
        rng = np.random.default_rng(seed)
        B = 500   # shuffle 500-event blocks: keeps short-range dependence, removes time-of-day regimes
        nb = len(y) // B
        order = rng.permutation(nb)
        idx = np.concatenate([np.arange(b * B, (b + 1) * B) for b in order])
        X, spr, y = X[idx].copy(), spr[idx], y[idx]
        lv = level_from_labels(np.r_[y, np.zeros(h)], h)
        X = np.vstack([X, np.repeat(X[-1:], h, 0)]); spr = np.r_[spr, np.repeat(spr[-1], h)]
        X[h:, 15] = lv[h:] - lv[:-h]
        cfg = replace(REAL_CFG, seed=seed)
    eng = TripwireEngine(cfg)
    o = eng.run(X, lv, spr, lv, record_weights=False)
    L = int(o["t_live"])
    return dict(kind=kind, seed=seed, events=len(lv) - L, arl_events=cfg.arl_events,
                alarms_f=int(np.sum(o["alarms_f"] >= L)), alarms_r=int(np.sum(o["alarms_r"] >= L)))


# ----------------------------------------------------------------------------- real data
def real_features(sym, h):
    cache = os.path.join(ROOT, "data", "cache", f"{sym}_feat_h{h}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["X"], z["mid"], z["spr"]
    base = os.path.join(ROOT, "data", "lobster", f"{sym}_2012-06-21_34200000_57600000_")
    d = load_lobster(base + "orderbook_10.csv", base + "message_10.csv",
                     cache=os.path.join(ROOT, "data", "cache", f"{sym}.npz"))
    X, mid, spr = extract_all(d["bid_p"], d["bid_s"], d["ask_p"], d["ask_s"], 0.01, h)
    np.savez(cache, X=X, mid=mid, spr=spr, time=d["time"])
    return X, mid, spr


def real_times(sym):
    z = np.load(os.path.join(ROOT, "data", "cache", f"{sym}.npz"))
    return z["time"]


def real_forecast_task(args):
    sym, h, target = args
    X, mid, spr = real_features(sym, h)
    cfg = replace(REAL_CFG, horizon=h)
    level = mid if target == "mid" else spr
    abl = {k: v for k, v in ABLATIONS.items() if k.startswith(("A2", "A3", "A4"))} if target == "mid" else {}
    r = evaluate_stream(X, level, spr, cfg, (), ablations=abl, keep_series=(h == 50 and target == "mid"))
    r.update(symbol=sym, h=h, target=target)
    if "series" in r:
        tm = real_times(sym)
        r["series"]["time"] = tm
    return r


def inject_task(args):
    sym, offset, beta, n = args
    h = REAL_CFG.horizon
    X, mid, spr = real_features(sym, h)
    X = X[offset: offset + n].copy(); spr = spr[offset: offset + n]
    y = labels_from_level(mid[offset: offset + n], h)
    tau = n // 2 + 10_000
    # in-slice best linear predictor of the real label = the 'true' signal we reverse after tau
    Zs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    th, _ = ridge_fit(Zs[: n - h], y[: n - h], 10.0)
    g = Zs @ th
    y2 = y.copy()
    y2[tau:] = y[tau:] + (beta - 1.0) * g[tau:]
    y2[n - h:] = 0.0
    lv = level_from_labels(y2, h)
    X[h:, 15] = lv[h:] - lv[:-h]
    abl = {k: v for k, v in ABLATIONS.items() if k.startswith(("A2", "A4"))}
    cps = [tau] if beta != 1.0 else []
    r = evaluate_stream(X, lv, spr, REAL_CFG, cps, ablations=abl,
                        b4_lambdas=(0.998, 0.9995, 0.9999, 0.99995, 1.0))
    if beta == 1.0:  # placebo: report what happens after the would-be change point
        for m in r["models"].values():
            m["placebo_alarms_after_tau"] = None
        tw = r["models"]["TRIPWIRE-LOB"]
        tw["placebo_alarms_r_after_tau"] = len([a for a in tw["alarms_r"] if tau <= a < tau + DET_WINDOW])
    r.update(symbol=sym, offset=offset, beta=beta, tau=tau, signal_r2=float(np.var(g) / np.var(y)))
    return r


def splice_task(args):
    """Natural regime change: the end of one real stock's stream is spliced onto another's."""
    symA, symB, a0, b0 = args if args else ("AAPL", "MSFT", 100_000, 100_000)
    h = REAL_CFG.horizon
    XA, mA, sA = real_features(symA, h)
    XB, mB, sB = real_features(symB, h)
    n = 60_000
    yA = labels_from_level(mA[a0:a0 + n], h)[: n - h]
    yB = labels_from_level(mB[b0:b0 + n], h)[: n - h]
    y = np.r_[yA, yB, np.zeros(h)]
    X = np.vstack([XA[a0:a0 + n - h], XB[b0:b0 + n - h], XB[b0 + n - h: b0 + n]])
    spr = np.r_[sA[a0:a0 + n - h], sB[b0:b0 + n - h], sB[b0 + n - h: b0 + n]]
    lv = level_from_labels(y, h)
    X[h:, 15] = lv[h:] - lv[:-h]
    r = evaluate_stream(X, lv, spr, REAL_CFG, [n - h], ablations={}, keep_series=args is None,
                        b4_lambdas=(0.998, 0.9995, 0.9999, 0.99995, 1.0))
    r["case"] = f"{symA} -> {symB}"
    r.update(symA=symA, symB=symB, a0=a0, b0=b0)
    return r


SPLICES = [(a, b, off, off) for a, b in (("AAPL", "MSFT"), ("MSFT", "AAPL"), ("AMZN", "AAPL"))
           for off in (40_000, 120_000, 200_000)]


# ----------------------------------------------------------------------------- aggregation
def ci(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return None
    m = float(v.mean())
    half = float(1.96 * v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return {"mean": m, "ci95": half, "n": int(v.size)}


def summarise_synthetic(rows):
    summ = {}
    for sc in sorted({r["scenario"] for r in rows}):
        rs = [r for r in rows if r["scenario"] == sc]
        summ[sc] = {}
        for model in rs[0]["models"]:
            ms = [r["models"][model] for r in rs]
            s = {
                "r2_oos": ci([m["overall"]["r2_oos"] for m in ms]),
                "coverage": ci([m["overall"].get("coverage") for m in ms]),
                "call_rate": ci([m["overall"].get("call_rate") for m in ms]),
                "hit_rate": ci([m["overall"].get("hit_rate") for m in ms]),
                "false_alarms_per_50k": ci([m["false_alarms"] / m["false_alarm_events"] * 50_000 for m in ms]),
            }
            if "false_alarms_f" in ms[0]:
                s["fa_f_per_50k"] = ci([m["false_alarms_f"] / m["false_alarm_events"] * 50_000 for m in ms])
                s["fa_r_per_50k"] = ci([m["false_alarms_r"] / m["false_alarm_events"] * 50_000 for m in ms])
            ch = [m["changes"][0] for m in ms if m["changes"]]
            if ch:
                s["post_r2"] = ci([c["post"]["r2_oos"] for c in ch])
                s["post_coverage"] = ci([c["post"].get("coverage") for c in ch])
                s["post_wrong_calls"] = ci([c["post"].get("wrong_calls") for c in ch])
                s["post_call_rate"] = ci([c["post"].get("call_rate") for c in ch])
                s["post_hit_rate"] = ci([c["post"].get("hit_rate") for c in ch])
                s["cod_500"] = ci([c["cod_500"] for c in ch])
                s["cod_3000"] = ci([c["cod_3000"] for c in ch])
                s["sft"] = ci([c["sft"] for c in ch])
                s["detect_rate_any"] = float(np.mean([c["delay_any"] is not None for c in ch]))
                s["delay_any"] = ci([c["delay_any"] for c in ch])
                if "delay_f" in ch[0]:
                    s["detect_rate_f"] = float(np.mean([c["delay_f"] is not None for c in ch]))
                    s["detect_rate_r"] = float(np.mean([c["delay_r"] is not None for c in ch]))
                    s["delay_f"] = ci([c["delay_f"] for c in ch])
                    s["delay_r"] = ci([c["delay_r"] for c in ch])
            summ[sc][model] = s
    return summ


def strip(r):
    r = dict(r)
    r.pop("series", None)
    r.pop("regime_id", None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--skip-real", action="store_true")
    a = ap.parse_args()
    seeds = 3 if a.quick else a.seeds
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data", "cache"), exist_ok=True)
    have_real = os.path.exists(os.path.join(ROOT, "data", "lobster", "AAPL_2012-06-21_34200000_57600000_orderbook_10.csv"))
    if not have_real and not a.skip_real:
        print("LOBSTER files missing -> run scripts/fetch_lobster.py (skipping real-data experiments)")
    real = have_real and not a.skip_real
    syms = ["AAPL", "AMZN", "MSFT"]
    results = {"config_synthetic": SYN_CFG.__dict__, "config_real": REAL_CFG.__dict__}
    t0 = time.time()
    with Pool(a.workers) as pool:
        if real:  # warm the feature caches in parallel
            pool.starmap(real_features, [(s, h) for s in syms for h in (10, 50)])
        print("[1/6] synthetic 2x2 benchmark", flush=True)
        jobs = [(sc, s, SYN_T) for sc in SCENARIOS for s in range(seeds)]
        rows = pool.map(syn_task, jobs, chunksize=1)
        results["synthetic"] = summarise_synthetic(rows)
        results["synthetic_raw"] = rows
        print("  done %.0fs" % (time.time() - t0), flush=True)

        print("[2/6] recurrence stream", flush=True)
        d = generate("recurrence", T=100_000, horizon=SYN_CFG.horizon, seed=7)
        X, mid, spr = extract_all(d["bid_p"], d["bid_s"], d["ask_p"], d["ask_s"], 0.01, SYN_CFG.horizon)
        rec = evaluate_stream(X, mid, spr, SYN_CFG, d["change_points"], ablations={"A5 pool only (no detectors)": ABLATIONS["A5 pool only (no detectors)"]}, keep_series=True)
        rec["change_kinds"] = d["change_kinds"]
        rec["change_points"] = d["change_points"]
        rec["regime_id"] = d["regime_id"]

        print("[3/6] false-alarm calibration", flush=True)
        arl_jobs = [("synthetic", s, 60_000 if a.quick else 250_000) for s in range(4 if a.quick else 8)]
        if real:
            arl_jobs += [(sym, s, 0) for sym in syms for s in range(1 if a.quick else 2)]
        arl = pool.map(arl_task, arl_jobs, chunksize=1)
        results["arl"] = arl
        print("  done %.0fs" % (time.time() - t0), flush=True)

        if real:
            print("[4/6] real LOBSTER forecasting", flush=True)
            fj = [(s, h, "mid") for s in syms for h in (10, 50)] + [(s, 50, "spread") for s in syms]
            if a.quick:
                fj = [("AAPL", 50, "mid"), ("MSFT", 50, "mid")]
            fr = pool.map(real_forecast_task, fj, chunksize=1)
            results["real_forecast"] = [strip(r) for r in fr]
            print("  done %.0fs" % (time.time() - t0), flush=True)

            print("[5/6] injected concept shifts on real data", flush=True)
            ij = []
            for sym in syms:
                n_all = len(real_features(sym, REAL_CFG.horizon)[1])
                offs = np.linspace(10_000, n_all - 120_000, 2 if a.quick else 4).astype(int)
                for off in offs:
                    for beta in ((1.0, -3.0) if a.quick else (1.0, -1.0, -3.0)):
                        ij.append((sym, int(off), beta, 120_000))
            inj = pool.map(inject_task, ij, chunksize=1)
            results["real_injected"] = inj
            print("  done %.0fs" % (time.time() - t0), flush=True)
            print("[6/6] cross-ticker splices", flush=True)
            spl = splice_task(None)
            results["real_splice"] = strip(spl)
            results["real_splices"] = [strip(r) for r in pool.map(splice_task, SPLICES, chunksize=1)]
        else:
            fr, spl = [], None
    results["recurrence"] = strip(rec)
    results["elapsed_s"] = time.time() - t0
    with open(os.path.join(RES, "benchmarks.json"), "w") as f:
        json.dump(results, f, default=_json_default, indent=1)
    from report import write_tables, make_figures, write_summary  # noqa: E402
    write_tables(results, os.path.join(RES, "tables.md"))
    write_summary(results, os.path.join(ROOT, "cockpit", "summary.json"))
    make_figures(results, rec, fr, spl, os.path.join(RES, "figures"))
    export_replays(rec, fr, spl)
    print("all done in %.0fs -> results/" % (time.time() - t0))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def export_replays(rec, fr, spl):
    """Down-sampled series for the cockpit's 'evidence' replays."""
    from report import replay_payload
    out = {"recurrence": replay_payload(rec["series"], "Synthetic recurrence: calm -> concept -> covariate -> concept -> calm",
                                        change_points=rec["change_points"], t_live=rec["t_live"], thresholds=SYN_CFG.thresholds())}
    for r in fr or []:
        if "series" in r:
            out[f"{r['symbol']}_h{r['h']}"] = replay_payload(r["series"], f"LOBSTER {r['symbol']} 21-Jun-2012, h={r['h']} events (real data, walk-forward)", t_live=r["t_live"], thresholds=replace(REAL_CFG, horizon=r["h"]).thresholds())
    if spl is not None and "series" in spl:
        out["splice"] = replay_payload(spl["series"], "Real AAPL then MSFT spliced at a known point (natural regime change)", change_points=[x["tau"] for x in spl["models"]["TRIPWIRE-LOB"]["changes"]], t_live=spl["t_live"], thresholds=REAL_CFG.thresholds())
    with open(os.path.join(ROOT, "cockpit", "replays.json"), "w") as f:
        json.dump(out, f, default=_json_default)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
