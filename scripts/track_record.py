"""Track record of the live paper desk (pre-registration H4 and H5).

    python scripts/track_record.py                 # -> results/paper/track_record.md + figures

For every scheduled release in data/events.csv and every offline volatility change point
(tripwire/changepoints.py, never from TRIPWIRE), it reports, per symbol, from 5 minutes before
to 30 minutes after: P&L of S0/S1/S2, money lost to toxic fills (30 s markout) for S1/S2, and
how long the guard took to leave TRUST. H5: DISTRUST alarms per placebo hour (no event, no
change point within 30 min) against the nominal rate. Every event is reported; none dropped.
"""

from __future__ import annotations

import csv
import datetime as dt
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = getattr(sys, "_MEIPASS", None) or os.path.join(HERE, "..")   # bundle root when frozen
ROOT = os.environ.get("TRIPWIRE_HOME") or CODE   # writable: data/, results/
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

from tripwire.changepoints import volatility_change_times  # noqa: E402

PRE, POST = 300.0, 1800.0
ARL_EVENTS = 20_000          # paper desk engine: one false alarm per 20k book events per monitor


def load_ticks(sym_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(sym_dir, "ticks_*.csv"))):
        a = np.genfromtxt(f, delimiter=",", skip_header=1, ndmin=2, invalid_raise=False)
        if a.size:
            rows.append(a)
    if not rows:
        return None
    a = np.vstack(rows)
    a = a[np.argsort(a[:, 0])]
    return dict(ts=a[:, 0], mid=a[:, 1], state=a[:, 2], trust=a[:, 3], pnl=a[:, 4:7], inv=a[:, 7:10])


def load_jsonl(sym_dir, kind):
    out = []
    for f in sorted(glob.glob(os.path.join(sym_dir, f"{kind}_*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def load_calendar(path):
    ev = []
    if not os.path.exists(path):
        return ev
    with open(path) as f:
        for r in csv.DictReader(f):
            t = dt.datetime.strptime(r["time_utc"], "%Y-%m-%d %H:%M").replace(tzinfo=dt.timezone.utc).timestamp()
            ev.append(dict(ts=t, name=r["name"], verified=r.get("verified", "no").strip().lower() == "yes", kind="scheduled"))
    return ev


def window(T, fills, t0):
    a, b = t0 - PRE, t0 + POST
    i, j = np.searchsorted(T["ts"], a), np.searchsorted(T["ts"], b) - 1
    if j <= i or T["ts"][i] > t0 or T["ts"][j] < t0 + 60:
        return None                                   # desk was not running around this event
    pnl = T["pnl"][j] - T["pnl"][max(i - 1, 0)]
    tox = {}
    for d in ("S1", "S2"):
        fs = [x for x in fills if x["desk"] == d and a <= x["ts"] < b]
        lost = 0.0
        for x in fs:
            k = min(np.searchsorted(T["ts"], x["ts"] + 30.0), len(T["ts"]) - 1)
            lost -= x["side"] * x["qty"] * (T["mid"][k] - x["mid"])
        tox[d] = lost
    k0 = np.searchsorted(T["ts"], t0)
    after = np.flatnonzero(T["state"][k0:j + 1] >= 2)
    react = float(T["ts"][k0 + after[0]] - t0) if after.size else None
    return dict(pnl_s0=float(pnl[0]), pnl_s1=float(pnl[1]), pnl_s2=float(pnl[2]), toxic_s1=tox["S1"], toxic_s2=tox["S2"],
                distrust_after_s=react)


def report(root=os.path.join(ROOT, "data", "paper"), calendar=os.path.join(ROOT, "data", "events.csv"),
           out_dir=os.path.join(ROOT, "results", "paper")):
    os.makedirs(out_dir, exist_ok=True)
    cal = load_calendar(calendar)
    L = ["# Live paper desk: track record", "",
         f"Generated {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC from `data/paper/`. Paper trading only: no order ever left this machine. "
         "S0 = neutral desk, S1 = always-on alpha, S2 = TRIPWIRE-guarded alpha; all three see identical market data and fill rules. "
         "P&L in USDT, gross of fees; markouts at 30 s.", ""]
    if any(not e["verified"] for e in cal):
        L += ["> **Some calendar times are marked unverified in `data/events.csv`.** Check them against the official "
              "Fed / BLS release calendars and set `verified=yes`.", ""]
    summary = {}
    for sym_dir in sorted(glob.glob(os.path.join(root, "*"))):
        sym = os.path.basename(sym_dir)
        T = load_ticks(sym_dir)
        if T is None:
            continue
        fills = load_jsonl(sym_dir, "fills")
        evs = load_jsonl(sym_dir, "events")
        hours = (T["ts"][-1] - T["ts"][0]) / 3600
        L += [f"## {sym}: {len(T['ts']):,} book events over {hours:.1f} h "
              f"({dt.datetime.fromtimestamp(T['ts'][0], dt.timezone.utc):%Y-%m-%d %H:%M} → {dt.datetime.fromtimestamp(T['ts'][-1], dt.timezone.utc):%Y-%m-%d %H:%M} UTC)", ""]
        tot = T["pnl"][-1] - T["pnl"][0]
        share = {s: float(np.mean(T["state"] == c)) for s, c in (("TRUST", 0), ("CAUTION", 1), ("DISTRUST", 2), ("RESTORE", 3))}
        nf = {d: sum(1 for x in fills if x["desk"] == d) for d in ("S0", "S1", "S2")}
        L += ["| desk | total P&L | fills |", "|---|---|---|"]
        for k, d in enumerate(("S0", "S1", "S2")):
            L.append(f"| {d} | {tot[k]:,.2f} | {nf[d]:,} |")
        L += ["", "Guard time share: " + ", ".join(f"{k} {100 * v:.1f} %" for k, v in share.items()), ""]
        cps = volatility_change_times(T["ts"], T["mid"])
        events = [dict(e) for e in cal] + [dict(ts=t, name="volatility change point (offline)", verified=True, kind="changepoint") for t in cps]
        rows = []
        for e in sorted(events, key=lambda z: z["ts"]):
            w = window(T, fills, e["ts"])
            if w is None:
                continue
            rows.append((e, w))
        L += [f"### Event study ({len(rows)} events with the desk running)", "",
              "| time (UTC) | event | P&L S0 | P&L S1 | P&L S2 | toxic loss S1 | toxic loss S2 | guard left TRUST after |",
              "|---|---|---|---|---|---|---|---|"]
        for e, w in rows:
            ver = "" if e["verified"] else " (unverified time)"
            react = f"{w['distrust_after_s']:.0f} s" if w["distrust_after_s"] is not None else "stayed TRUST/CAUTION"
            L.append(f"| {dt.datetime.fromtimestamp(e['ts'], dt.timezone.utc):%Y-%m-%d %H:%M} | {e['name']}{ver} | {w['pnl_s0']:,.2f} | {w['pnl_s1']:,.2f} | "
                     f"{w['pnl_s2']:,.2f} | {w['toxic_s1']:,.2f} | {w['toxic_s2']:,.2f} | {react} |")
        if rows:
            d12 = np.array([w["pnl_s2"] - w["pnl_s1"] for _, w in rows])
            rng = np.random.default_rng(0)
            boots = [rng.choice(d12, len(d12)).mean() for _ in range(2000)]
            L += ["", f"Mean (S2 − S1) P&L per event window: {d12.mean():,.2f} (95 % bootstrap interval {np.percentile(boots, 2.5):,.2f} … {np.percentile(boots, 97.5):,.2f}); "
                  f"S2 ahead in {int(np.sum(d12 > 0))}/{len(d12)} windows. Descriptive, not a pass/fail test (pre-registration H4).", ""]
        # H5: alarms on placebo hours
        r_alarms = [x["ts"] for x in evs if x.get("kind") == "alarm" and x.get("monitor") == "R"]
        busy = [e["ts"] for e in events]
        h0 = int(T["ts"][0] // 3600)
        plac_events = plac_alarms = 0
        for hh in range(h0, int(T["ts"][-1] // 3600) + 1):
            a, b = hh * 3600, (hh + 1) * 3600
            if any(a - 1800 <= t <= b + 1800 for t in busy):
                continue
            n_ev = int(np.sum((T["ts"] >= a) & (T["ts"] < b)))
            if n_ev < 1000:
                continue
            plac_events += n_ev
            plac_alarms += sum(1 for t in r_alarms if a <= t < b)
        nominal = plac_events / ARL_EVENTS
        ratio = plac_alarms / nominal if nominal > 0 else float("nan")
        L += [f"**H5, placebo hours:** {plac_alarms} Monitor R alarms over {plac_events:,} quiet-hour events; nominal ≤ {nominal:.1f} "
              f"→ ratio {ratio:.2f} (pass if ≤ 2).", ""]
        summary[sym] = dict(hours=hours, events=len(rows), total=tot.tolist(), h5_ratio=ratio)
        _figure(T, events, out_dir, sym)
        L += [f"![{sym}](track_{sym}.png)", ""]
    if not summary:
        L += ["No paper-desk data yet. Start it with `python scripts/paper_desk.py` (or from the desktop app)."]
    with open(os.path.join(out_dir, "track_record.md"), "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(out_dir, "track_record.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return "\n".join(L)


def _figure(T, events, out_dir, sym):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from report import _style, INK, ACCENT
    _style(plt)
    k = max(1, len(T["ts"]) // 4000)
    t = (T["ts"][::k] - T["ts"][0]) / 3600
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for i, (name, c) in enumerate((("S0 neutral", "#aaa"), ("S1 always-on", ACCENT), ("S2 guarded", INK))):
        ax[0].plot(t, T["pnl"][::k, i] - T["pnl"][0, i], color=c, label=name, lw=1.2)
    ax[0].legend(fontsize=8); ax[0].set_title(f"{sym} live paper desks: cumulative P&L (USDT)")
    ax[1].plot(t, T["state"][::k], color=INK, lw=0.8, drawstyle="steps-post")
    ax[1].set_yticks([0, 1, 2, 3]); ax[1].set_yticklabels(["TRUST", "CAUTION", "DISTRUST", "RESTORE"])
    ax[1].set_title("guard state"); ax[1].set_xlabel("hours since start")
    for e in events:
        x = (e["ts"] - T["ts"][0]) / 3600
        if 0 <= x <= t[-1]:
            for a_ in ax:
                a_.axvline(x, color="#c0392b" if e["kind"] == "scheduled" else "#d9a0ad", lw=1 if e["kind"] == "scheduled" else 0.6)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"track_{sym}.png"), dpi=120); plt.close(fig)


if __name__ == "__main__":
    print(report())
