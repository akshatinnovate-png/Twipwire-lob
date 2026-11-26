"""Desk simulator and guard tests: queue position, latency, limits, accounting, state machine."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tripwire.desk_models import MLPDeskModel
from tripwire.guard import GuardPolicy, TrustGuard
from tripwire.mm import DeskParams, DeskSim, _Order


def _desk(**kw):
    return DeskSim(DeskParams(latency=0.0, maker_fee_per_unit=0.0, **kw))


def test_queue_ahead_is_consumed_before_we_fill():
    d = _desk()
    d.orders[+1] = _Order(+1, 10.00, 100, ahead=300, live_at=0.0)
    d.on_message(1, 1.0, 4, 200, 10.00, +1, 300, 10.005)      # 200 executed: still 100 ahead
    assert d.inv == 0 and d.orders[+1].ahead == pytest.approx(100)
    d.on_message(2, 2.0, 4, 150, 10.00, +1, 100, 10.005)      # 150 executed: 50 reach us
    assert d.inv == pytest.approx(50)
    d.on_message(3, 3.0, 4, 500, 10.00, +1, 0, 10.005)        # rest of our 100
    assert d.inv == pytest.approx(100) and d.orders[+1] is None


def test_trade_through_our_price_fills_first():
    d = _desk()
    d.orders[-1] = _Order(-1, 10.02, 100, ahead=0, live_at=0.0)
    d.on_message(1, 1.0, 4, 60, 10.03, -1, 500, 10.015)       # buyer lifted 10.03; our 10.02 ask was better
    assert d.inv == pytest.approx(-60)


def test_cancellations_shrink_queue_pro_rata_but_not_in_back_model():
    d = _desk()
    d.orders[+1] = _Order(+1, 10.00, 100, ahead=200, live_at=0.0)
    d.on_message(1, 1.0, 3, 100, 10.00, +1, 400, 10.005)      # 100 of 400 cancelled, we're 200 deep
    assert d.orders[+1].ahead == pytest.approx(150)
    b = _desk(cancel_model="back")
    b.orders[+1] = _Order(+1, 10.00, 100, ahead=200, live_at=0.0)
    b.on_message(1, 1.0, 3, 100, 10.00, +1, 400, 10.005)
    assert b.orders[+1].ahead == pytest.approx(200)


def test_latency_quote_not_live_until_delay():
    d = DeskSim(DeskParams(latency=0.010, maker_fee_per_unit=0.0))
    d.decide(0.0, 10.00, 10.02, 0.0)
    d.activate(0.005, lambda s, p: 0.0)
    assert d.orders[+1] is None                                 # still in flight
    d.activate(0.011, lambda s, p: 0.0)
    assert d.orders[+1] is not None and d.orders[+1].price == pytest.approx(10.00)


def test_skew_and_inventory_limit():
    d = _desk()
    d.decide(0.0, 10.00, 10.20, forecast_ticks=+3.0)           # expect up -> lean both quotes up
    assert d.pending[+1][-1].price == pytest.approx(10.03) and d.pending[-1][-1].price == pytest.approx(10.23)
    d.inv = 1000.0                                              # at the 10-lot long limit
    d.decide(0.0, 10.00, 10.20, 0.0)
    assert d.pending[+1][-1].qty == 0.0 and d.pending[-1][-1].qty > 0


def test_requoting_faster_than_latency_still_reaches_the_market():
    d = DeskSim(DeskParams(latency=0.25, maker_fee_per_unit=0.0))
    for k in range(100):                                        # a new quote every 10 ms, latency 250 ms
        ts = 0.01 * k
        d.activate(ts, lambda s, p: 0.0)
        d.decide(ts, 10.00 + 0.01 * (k % 3), 10.05 + 0.01 * (k % 3), 0.0)
    assert d.orders[+1] is not None and d.orders[-1] is not None


def test_accounting_marks_to_mid_and_rebates():
    d = DeskSim(DeskParams(latency=0.0, maker_fee_per_unit=-0.002))
    d.orders[+1] = _Order(+1, 10.00, 100, ahead=0, live_at=0.0)
    d.on_message(1, 1.0, 4, 100, 10.00, +1, 100, 10.01)
    assert d.mark(10.05) == pytest.approx(100 * 0.05 + 0.2)


def test_guard_state_machine_distrust_restore_and_failsafe():
    g = TrustGuard(GuardPolicy(restore_window=50, restore_ramp=20, caution_hold=10))
    assert g.update()["state"] == "TRUST"
    assert g.update(alarm_f=True)["state"] == "CAUTION"
    d = g.update(alarm_r=True)
    assert d["state"] == "DISTRUST" and d["skew"] == 0 and d["width"] == 2.0
    rng = np.random.default_rng(0)
    for _ in range(49):                                          # model useful again: R^2 > 0
        y = rng.normal()
        g.update(y_matured=y, f_matured=0.9 * y)
    assert g.state == "DISTRUST"
    y = rng.normal()
    assert g.update(y_matured=y, f_matured=0.9 * y)["state"] == "RESTORE"
    for _ in range(25):
        d = g.update()
    assert d["state"] == "TRUST" and d["skew"] == 1.0
    g.update(ts=100.0)
    assert g.update(ts=107.0)["state"] == "DISTRUST"             # 7 s feed gap -> fail-safe


def test_guard_stays_distrustful_while_model_is_wrong():
    g = TrustGuard(GuardPolicy(restore_window=50))
    g.update(alarm_r=True)
    rng = np.random.default_rng(1)
    for _ in range(500):
        y = rng.normal()
        g.update(y_matured=y, f_matured=-0.9 * y)                 # reversed model
    assert g.state == "DISTRUST"


def test_desk_mlp_learns_a_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4000, 16))
    y = np.tanh(X[:, 0]) * 2 + 0.5 * X[:, 3] + rng.normal(size=4000)
    m = MLPDeskModel(epochs=20).fit(X[:3000], y[:3000])
    e = y[3000:] - m.predict(X[3000:])
    assert 1 - np.mean(e ** 2) / np.var(y[3000:]) > 0.4
