# TRIPWIRE-LOB: midway progress report (1 December 2026)

**Team TM-BBAC6C06A8C** (Saharsh Kumar, Akshat Sarkar) · Theme 3, Problem 4: Order Book Forecasting Under Regime Shifts

**Status: on plan, with the MVP and both stretch goals working.** Everything below is reproduced by `python scripts/run_benchmarks.py` (results in `results/tables.md`). The code, 10 unit tests and a live demo are in the repository.

**Built (all four layers of the Round-1 plan).**
* *Features*: 16 causal features from 10 book levels (multi-level OFI, imbalances, micro-price, depth slopes, spread, volatility ratio).
* *Expert pool*: K = 9 experts (frozen ridge, 4 RLS, 4 alarm-spawned regime heads) with fixed-share weights on matured labels.
* *Detectors*: Monitor F (label-free) and Monitor R (label-aware), each a conformal p-value feeding a Shiryaev–Roberts statistic.
* *Intervals and calls*: an adaptive conformal 90 % interval, plus an abstaining call rule.
* *Evaluation*: baselines B1–B5, five ablations, a 2×2 synthetic benchmark with 20 seeds, and three real-data protocols on LOBSTER AAPL / AMZN / MSFT (full-day walk-forward, injected concept breaks, and cross-stock splices), plus a false-alarm calibration study.
* *Demo*: a live cockpit with shock injection on synthetic and real data. It runs offline on a laptop CPU.

**MVP bullets.** (1) Direction and spread are forecast at h = 10 and 50 events on all three stocks. Out-of-sample R² on the mid is 0.031 (AAPL), 0.064 (AMZN) and 0.239 (MSFT) at h = 50, matching the best fixed-forgetting RLS even though that baseline picks its forgetting factor in hindsight. (2) Shifts are detected at known times and the model reacts: concept flips are caught in 100 % of runs (Monitor R, about 730 events), covariate shifts in 100 % (Monitor F, about 630 events), and Monitor R stays quiet on harmless covariate shifts (5 %).

**Committed targets: all met.**
* False alarms within 2× nominal: 1.01 / 1.02 / 1.56 (F) and 0.17 / 0.38 / 1.14 (R) on block-shuffled AAPL / AMZN / MSFT.
* Lower cost of delay than B3 and B4 on concept cells: 2827 vs 3941 / 3028, and coupled 4185 vs 7755 / 5083.
* Coverage within 5 points of 90 %: 89.7–90.0 % on every real stream, where static intervals cover 98–99.8 %.

**Stretch goals.** Frozen regime heads win weight back when their regime recurs, with no alarm needed. Latency is p50 ≈ 80 µs and p99 ≤ 226 µs per event on one core, against a 1 ms budget.

**Two evidence-driven changes to the plan.** (1) Monitor R now tests the orthogonality E[(y−f)·f] = 0 instead of the |residual|. At real-data R² ≈ 1–3 % the residual test has almost no power: it caught 17 % of injected real breaks, where orthogonality caught 83 %. (2) The call rule is now |ŷ| > k*·q instead of |ŷ| − q > half-spread, because 13–15-tick spreads on AAPL and AMZN mean the original rule never fires.

**Remaining (2–14 Dec).** Rehearse the demo; write up the time-ordered real-day alarms as case studies; optionally record BTC-USDT around the December FOMC with `scripts/record_binance.py`.

**Known limitations, stated up front.** Monitors need a re-arm window after an alarm: stock-to-stock splices were alarmed in 5 of 9 runs, and every miss came shortly after a prior alarm. A pure change in the scale of y is absorbed by the volatility normalisation rather than flagged. Gross edge per call is below half the spread, so this is a forecasting and monitoring system, not a trading claim.
