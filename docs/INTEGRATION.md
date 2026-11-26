# Integrating TRIPWIRE into a trading desk

TRIPWIRE is a **model-risk guard**. It sits beside a desk's existing short-term model and tells the quoting engine, on every book event, how far to trust that model. It never sends orders and never changes the desk's model. Its only output is a small decision record.

```
 market data ──► feature step ──► desk model f(x) ─────────────┐
      │                                                         ▼
      │          ┌──────────── TRIPWIRE (same process) ───────┐ quoting engine
      └────────► │ Monitor F: are the inputs new?   (P(X))    │  reservation = mid + θ·skew·f − κ·inv
                 │ Monitor R: are f's errors lining up        │  half-spread × width
   matured       │            against f?            (P(Y|X))  │  size × size, inventory cap × inventory
   labels ─────► │ guard: TRUST/CAUTION/DISTRUST/RESTORE      │ ─► {state, skew, width, size, inventory, reason}
   (h events)    └────────────────────────────────────────────┘
```

## Interface

```python
from tripwire import TripwireConfig, TripwireEngine, FeatureExtractor
from tripwire.guard import TrustGuard, GuardPolicy

cfg = TripwireConfig(horizon=h, monitor_r_ref="external")   # Monitor R tests YOUR model's errors
eng, guard, fx = TripwireEngine(cfg), TrustGuard(GuardPolicy()), FeatureExtractor(tick, h)

def on_book(bid_p, bid_s, ask_p, ask_s, ts):
    x, mid, spread = fx.update(bid_p, bid_s, ask_p, ask_s)
    f = desk_model(x)                                      # your model, unchanged
    o = eng.step(x, mid, spread, f_ext=f)                  # ~80 µs
    d = guard.update(alarm_f=o[5], alarm_r=o[6], y_matured=..., f_matured=..., ts=ts)
    return d   # {"state", "skew", "width", "size", "inventory", "reason"}
```

`scripts/paper_desk.py` is a complete working example (live Binance books → engine → guard → three paper desks → logs). `scripts/desk_benchmarks.py` is the same wiring with queue-accurate LOBSTER fills.

## The decision record

| state | entered when | skew | width | size | inventory cap | leaves when |
|---|---|---|---|---|---|---|
| TRUST | start; ramp finished | 1 | 1 | 1 | 1 | an alarm fires |
| CAUTION | Monitor F alarm: the book looks unlike the reference | 1 | 1.5 | 0.5 | 1 | 3h + 200 events pass |
| DISTRUST | Monitor R alarm: the model's errors line up against its forecasts; or the feed stops for 5 s (fail-safe) | 0 | 2 | 0.5 | 0.5 | the model's matured R² over the last 1,000 labels is > 0 |
| RESTORE | the model has earned trust back | ramps 0 → 1 over 2,000 events | ramps back | ramps back | ramps back | ramp finished |

Every transition is logged with its event index, time and reason (`events_<date>.jsonl` in the paper desk). The reason is meant to be readable by a person, e.g. `Monitor F: the book distribution moved (top movers: L1 imbalance, spread)`.

## Latency and footprint
* About 80 µs per event at p50 and ≤ 200 µs at p99 on one laptop core, in pure Python and NumPy. That is enough for an equities book at event rate. A desk with a microsecond budget would port the two monitors (O(d²) each) and the guard (O(1)) into the strategy's own language. The expert pool is optional when TRIPWIRE guards an external model.
* State: a few MB (ring buffers of h matured labels plus the reference windows). No GPU, no network.

## Fail-safes
* **Feed gap:** no book event for 5 s → DISTRUST until 200 events after the feed resumes.
* **Backend crash:** the paper desk's supervisor restarts it and warm-starts from its own raw log. The quoting engine should treat a missing decision record as DISTRUST.
* **Data sanity:** crossed or empty books are dropped before the feature step.
* **The guard can only reduce risk:** skew ≤ 1, width ≥ 1, size ≤ 1, inventory cap ≤ 1. A bug in TRIPWIRE can make a desk too cautious but never more aggressive than it already is.

## Alerting and audit
* Each transition into DISTRUST is a page-worthy event: it means the desk's model is no longer behaving as it did when it was signed off.
* `scripts/track_record.py` turns the logs into a report covering every scheduled macro release and every offline change point (labels that never come from TRIPWIRE), plus the false-alarm rate on quiet hours.
* Calibration is stated in advance: one false alarm per ARL events per monitor (60,000 events ≈ one AAPL market hour; 20,000 for the crypto paper desk). Observed / nominal is reported, not assumed.

## What it does not do
* It does not generate alpha. It decides when to stop believing someone else's.
* It does not fix a broken model. RESTORE waits for the model to work again; retraining is the desk's decision.
* It cannot see a break before labels mature. Monitor R needs about h + 2 blocks of matured errors; at h = 50 that is ≥ 250 events. Monitor F reacts sooner, but only to changes in the inputs.
* Detection power is not 100 %. In the desk benchmark, Monitor R missed 4 of 12 injected breaks within 30 minutes (`docs/PREREGISTRATION.md`, H2b).
