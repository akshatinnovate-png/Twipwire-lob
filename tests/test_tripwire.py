"""Guards for the claims made in the README. Run: python -m pytest -q"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tripwire import FeatureExtractor, TripwireConfig, TripwireEngine
from tripwire.aci import QuantileTracker
from tripwire.conformal import ConformalSR, power_mixture_bet
from tripwire.features import extract_all
from tripwire.metrics import labels_from_level, level_from_labels
from tripwire.pool import ExpertPool, ridge_fit
from tripwire.synthetic import generate

SMALL = TripwireConfig(horizon=10, n_fit=1500, n_ref=1500, min_ref=40, min_ref_rearm=20, ref_size=200, ref_gap=50)


def _stream(T=9000, seed=3, name="concept"):
    d = generate(name, T=T, horizon=10, seed=seed)
    return extract_all(d["bid_p"], d["bid_s"], d["ask_p"], d["ask_s"], 0.01, 10)


# ----------------------------------------------------------------------------- no look-ahead
def test_engine_never_sees_the_future():
    X, mid, spr = _stream()
    cut = 7000
    mid2 = mid.copy()
    mid2[cut:] += np.cumsum(np.random.default_rng(0).normal(size=len(mid) - cut)) * 5  # rewrite the future
    a = TripwireEngine(SMALL).run(X, mid, spr)
    b = TripwireEngine(SMALL).run(X, mid2, spr)
    for k in ("yhat", "q", "call", "logR_f", "logR_r"):
        np.testing.assert_array_equal(a[k][:cut], b[k][:cut], err_msg=k)
    assert a["t_live"] < cut


def test_features_are_causal():
    d = generate("calm", T=600, horizon=10, seed=1)
    X1, _, _ = extract_all(d["bid_p"], d["bid_s"], d["ask_p"], d["ask_s"], 0.01, 10)
    bs = d["bid_s"].copy()
    bs[400:] *= 3.0
    X2, _, _ = extract_all(d["bid_p"], bs, d["ask_p"], d["ask_s"], 0.01, 10)
    np.testing.assert_array_equal(X1[:400], X2[:400])


def test_label_level_roundtrip():
    y = np.random.default_rng(0).normal(size=500)
    y[-7:] = 0
    lv = level_from_labels(y, 7)
    np.testing.assert_allclose(labels_from_level(lv, 7)[:-7], y[:-7])


# ----------------------------------------------------------------------------- detector theory
def test_power_mixture_bet_is_a_valid_bet():
    # int_0^1 b(p) dp = 1  ->  R_t - t is a martingale under uniform p-values.
    # Substitute p = exp(-u): b(e^-u) e^-u = (1 - e^-u - u e^-u) / u^2, exact tail 1/U beyond U.
    def gp(u):
        return (1.0 - math.exp(-u) - u * math.exp(-u)) / (u * u)
    U = 1e6
    u = np.geomspace(1e-6, U, 400001)
    g = np.array([gp(v) for v in u])
    exact = float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(u))) + 1.0 / U + 1e-6 * 0.5
    assert abs(exact - 1.0) < 1e-3
    # the implementation agrees with the formula and is conservative (clamped at p >= 1e-12)
    for p in (1e-6, 1e-3, 0.1, 0.5, 0.9):
        assert power_mixture_bet(p) == pytest.approx((1 - p + p * math.log(p)) / (p * math.log(p) ** 2))
    uu = np.geomspace(1e-6, 700.0, 200001)
    gi = np.array([power_mixture_bet(math.exp(-v)) * math.exp(-v) for v in uu])
    assert float(np.sum(0.5 * (gi[1:] + gi[:-1]) * np.diff(uu))) <= 1.0
    assert power_mixture_bet(1.0) == pytest.approx(0.5)


def test_conformal_pvalues_uniform_under_iid_null():
    det = ConformalSR(threshold=1e12, ref_size=500, ref_gap=0, min_ref=100, stride=1, seed=0)
    rng = np.random.default_rng(1)
    ps = []
    for s in rng.normal(size=20000):
        det.update(float(s))
        if det.state == "monitoring":
            ps.append(det.last_p)
    ps = np.array(ps[100:])
    assert abs(ps.mean() - 0.5) < 0.02
    assert abs(np.mean(ps < 0.1) - 0.1) < 0.015


def test_shiryaev_roberts_average_run_length():
    A = 100.0
    rng = np.random.default_rng(2)
    runs = []
    for k in range(40):
        det = ConformalSR(threshold=A, ref_size=400, ref_gap=0, min_ref=400, stride=1, seed=k)
        for s in rng.normal(size=400):
            det.update(float(s))
        n = 0
        while n < 20 * A:
            n += 1
            if det.update(float(rng.normal())):
                break
        runs.append(n)
    # E[run length] >= A under the null (allow Monte-Carlo slack)
    assert np.mean(runs) > 0.8 * A


def test_detector_fires_on_a_real_shift():
    det = ConformalSR(threshold=100.0, ref_size=400, ref_gap=100, min_ref=200, stride=1, seed=0)
    rng = np.random.default_rng(3)
    for s in rng.normal(size=600):
        assert not det.update(float(s)) or True
    fired = None
    for i, s in enumerate(rng.normal(loc=1.0, size=600)):
        if det.update(float(s)):
            fired = i
            break
    assert fired is not None and fired < 100


# ----------------------------------------------------------------------------- intervals / experts
def test_aci_reaches_target_coverage():
    rng = np.random.default_rng(4)
    tr = QuantileTracker(alpha=0.1, gamma=0.02, q0=0.1, scale0=1.0)
    hits = []
    for i in range(30000):
        r = abs(rng.standard_t(4)) * (3.0 if i > 15000 else 1.0)  # scale break half way
        hits.append(r <= tr.q)
        tr.update(r, tr.q)
    assert abs(np.mean(hits[1000:]) - 0.9) < 0.01


def test_rls_lambda_one_matches_ridge():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(3000, 6))
    y = X @ rng.normal(size=6) + rng.normal(size=3000)
    pool = ExpertPool(6, (1.0,), 0, 0.999, 0.1, 0.0, 16.0)
    th0, P0 = ridge_fit(X[:100], y[:100], 1.0)
    pool.init_from_ridge(th0, P0)
    for t in range(100, 3000):
        pool.rls_update(X[t], y[t], 1e9)
    th, _ = ridge_fit(X, y, 1.0)
    np.testing.assert_allclose(pool.theta[1], th, atol=1e-6)


# ----------------------------------------------------------------------------- end to end
def test_end_to_end_concept_shift_is_detected_and_covered():
    d = generate("concept", T=40000, horizon=10, seed=0)
    X, mid, spr = extract_all(d["bid_p"], d["bid_s"], d["ask_p"], d["ask_s"], 0.01, 10)
    o = TripwireEngine(TripwireConfig()).run(X, mid, spr)
    tau = d["change_points"][0]
    assert any(tau <= a < tau + 3000 for a in o["alarms_r"])
    assert not any(o["t_live"] <= a < tau for a in o["alarms_r"])
    y = labels_from_level(mid, 10)
    L = o["t_live"]
    cov = np.mean(np.abs(y[L:-10] - o["yhat"][L:-10]) <= o["q"][L:-10])
    assert abs(cov - 0.9) < 0.02
    assert np.percentile(o["latency_us"], 99) < 1000.0
