"""Detector-variant study: can we catch more cross-stock splices without
breaking the false-alarm budget? Reproduces the table in docs/JUDGE_QA.md.

    python scripts/splice_study.py            # ~10 min on 4 cores

For each Monitor-F variant it reports splices detected (of 9), the observed /
nominal false-alarm ratio on block-shuffled real data (committed target <= 2x),
and synthetic detection on validation seeds 100-103 (never the reporting seeds).
"""

import json
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import run_benchmarks as rb  # noqa: E402
from tripwire import TripwireEngine  # noqa: E402
from tripwire.features import extract_all  # noqa: E402
from tripwire.synthetic import generate  # noqa: E402

VARIANTS = {
    "current (shipped)": {},
    "instant re-arm": {"instant_rearm": True},
    "two-sided Monitor F": {"f_two_sided": True},
    "dual Monitor F, second at 100-event blocks": {"f_dual_stride": 100},
    "dual Monitor F, second at 400-event blocks": {"f_dual_stride": 400},
}


def _splice(args):
    kw, sp = args
    rb.REAL_CFG = replace(rb.REAL_CFG, **kw)
    c = rb.splice_task(sp)["models"]["TRIPWIRE-LOB"]["changes"][0]
    return c["delay_any"]


def _null(args):
    kw, sym = args
    rb.REAL_CFG = replace(rb.REAL_CFG, **kw)
    r = rb.arl_task((sym, 0, 0))
    nom = r["events"] / r["arl_events"]
    return r["alarms_f"] / nom, r["alarms_r"] / nom


def _syn(args):
    kw, sc, seed = args
    d = generate(sc, T=40000, seed=seed)
    X, mid, spr = extract_all(d["bid_p"], d["bid_s"], d["ask_p"], d["ask_s"], 0.01, 10)
    o = TripwireEngine(replace(rb.SYN_CFG, seed=seed, **kw)).run(X, mid, spr, record_weights=False)
    al = sorted(list(o["alarms_f"]) + list(o["alarms_r"]))
    hit = [a - 20000 for a in al if 20000 <= a < 30000]
    return hit[0] if hit else None


def main():
    out = {}
    with Pool() as p:
        for name, kw in VARIANTS.items():
            det = p.map(_splice, [(kw, s) for s in rb.SPLICES])
            nul = p.map(_null, [(kw, s) for s in ("AAPL", "AMZN", "MSFT")])
            cov = p.map(_syn, [(kw, "covariate", s) for s in range(100, 104)])
            out[name] = dict(splices=sum(d is not None for d in det), splice_delays=det,
                             null_f=[round(a, 2) for a, _ in nul], null_r=[round(b, 2) for _, b in nul],
                             covariate_delay=float(np.median([c for c in cov if c is not None])))
            print(name, out[name], flush=True)
    with open(os.path.join(HERE, "..", "results", "splice_study.json"), "w") as f:
        json.dump(out, f, indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else str(o))


if __name__ == "__main__":
    main()
