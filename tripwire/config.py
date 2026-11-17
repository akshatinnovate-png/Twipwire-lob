"""All tunable constants in one place. Defaults are fixed before evaluation and
are the same for every dataset unless a field says otherwise."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TripwireConfig:
    horizon: int = 10                      # h: forecast horizon in book events (label matures h events later)

    # warm-up (walk-forward): first n_fit events fit the standardiser and frozen ridge,
    # next n_ref events run everything online and fill the detectors' reference sets.
    n_fit: int = 3000
    n_ref: int = 3000
    ridge_l2: float = 10.0
    clip_z: float = 6.0                     # standardised features are clipped to +-clip_z

    # B. expert pool
    rls_lambdas: Tuple[float, ...] = (0.998, 0.9995, 0.99995, 1.0)
    spawn_lambda: float = 0.999
    max_spawned: int = 4                    # K = 1 ridge + 4 RLS + 4 spawned = 9
    spawn_window: int = 400                 # matured pairs used to warm-start a spawned expert
    spawn_prior_weight: float = 0.25
    fs_eta0: float = 0.3                    # fixed-share rate; per-label eta = fs_eta0 / horizon (labels overlap h-1 events)
    fs_alpha: float = 1e-4                  # base switching rate per matured label
    fs_alpha_boost: float = 0.02            # switching rate for `boost_events` after an R alarm
    boost_events: int = 100
    loss_clip: float = 16.0                 # standardised squared loss is winsorised here

    # C. sequential detectors (conformal p-values -> Shiryaev-Roberts)
    arl_events: float = 50_000.0            # target average run length in events ("one market hour")
    ref_size: int = 500                     # reference block-scores per detector
    ref_gap: int = 200                      # newest scores wait this many tests before entering the reference
    min_ref: int = 120                      # block-scores needed before arming
    min_ref_rearm: int = 60                 # ... and before re-arming after an alarm
    stride_f: int = 200                     # block length for Monitor F (features carry slow components on real books)
    stride_r: int = 0                       # block length for Monitor R; 0 -> max(50, 2h) (labels overlap by h-1 events)
    f_ref_size: int = 150                   # Monitor F reference / gap / arming sizes, in F-blocks
    f_ref_gap: int = 50
    f_min_ref: int = 40
    f_min_ref_rearm: int = 20
    monitor_r_score: str = "orthogonality"  # "orthogonality": -(y-yhat)*yhat ; "abs": |y-yhat| (abstract v1)
    monitor_r_ref: str = "slow"             # forecast tested by Monitor R: "slow" = longest-memory RLS expert, "pool" = aggregate
    enable_f: bool = True
    enable_r: bool = True
    scale_halflife: int = 3000              # EWMA half-life for volatility normalisation of the R score

    # D. adaptive conformal interval and call gate
    alpha: float = 0.10                     # 90 % target coverage
    aci_gamma: float = 0.02                 # step, relative to running |residual| scale
    aci_gamma_boost: float = 3.0            # step multiplier after an R alarm
    f_inflation: float = 1.5                # interval multiplier after an F alarm ...
    f_hold_events: int = 0                  # ... held for this many events (0 -> 3h + 200)
    enable_aci: bool = True
    gate: str = "ratio"                     # "ratio": |yhat| > k* q_t (k* calibrated) ; "abstract": |yhat| - q_t > cost
    call_budget: float = 0.20               # calm call rate used to calibrate k*
    cost_mult: float = 0.0                  # require |yhat| > cost_mult * half spread (0 = direction skill only)

    seed: int = 0

    @property
    def fs_eta(self) -> float:
        return self.fs_eta0 / max(self.horizon, 1)

    def strides(self):
        sf = self.stride_f or self.horizon
        sr = self.stride_r or max(50, 2 * self.horizon)
        return sf, sr

    def thresholds(self):
        sf, sr = self.strides()
        return max(self.arl_events / sf, 20.0), max(self.arl_events / sr, 20.0)
