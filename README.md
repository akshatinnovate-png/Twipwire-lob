# TRIPWIRE-LOB

**Sequential shift alarms, expert-tracking adaptation and loud degradation for limit-order-book forecasting.**

Perceptron Hackathon 2026, IITB–Optiver AI Innovation Lab · Theme 3, Problem 4: *Order Book Forecasting Under Regime Shifts* · Team TM-BBAC6C06A8C (Saharsh Kumar, Akshat Sarkar)

Order-book models rarely crash when the market changes. More often they keep producing confident forecasts that are wrong. TRIPWIRE-LOB is a short-horizon LOB forecaster that

1. **detects** when the data distribution has shifted, using two sequential tests whose false-alarm rate is fixed in advance,
2. **tells the two kinds of shift apart**: a change in P(X) (the book looks different) versus a change in P(Y | X) (the book means something different),
3. **adapts** with a fixed-share pool of experts instead of hard resets, keeping old regimes as frozen heads so they can be reused, and
4. **degrades loudly**: its 90 % intervals widen and it stops making directional calls when it no longer knows, instead of failing silently.

It runs on one CPU core, about 80 µs per book event, in pure Python and NumPy.

### The use case: a model-risk guard for a market-making desk

A market maker leans its quotes on a short-term model. When the market changes, the model keeps quoting confidently, and the desk pays for it in toxic fills until a person notices. TRIPWIRE runs beside the desk's own model and, on every book event, outputs a trust state (**TRUST / CAUTION / DISTRUST / RESTORE**) that scales the model's skew, the quote width, the size and the inventory limit. It can only make a desk more cautious, never more aggressive. See [`docs/INTEGRATION.md`](docs/INTEGRATION.md).

We registered the test in advance ([`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md), committed before any desk run) and ran it on real LOBSTER order flow. The simulator models queue position, 1 ms latency and maker rebates:

| Pre-registered hypothesis | Result | |
|---|---|---|
| H1: the guard is cheap on normal days | guarded desk ≥ always-on desk on 3/3 real days | **pass** |
| H2a: ≥ 30 % less adverse selection after a break | **48 % less** (67.8k vs 130.5k $) | **pass** |
| H2b: guarded desk wins ≥ 9 of 12 break slices | **8 / 12**; all 3 losses with no Monitor R alarm were within 1.7k $ | **fail** |
| H3: beats the industry defaults | mean post-break P&L: guarded −97k, KS guard −138k, drawdown stop −139k $ | **pass** |

Tables: [`results/desk/desk_tables.md`](results/desk/desk_tables.md). The H2b miss, the exploratory findings and the caveats (the injected break is a severe stress test) are written up next to the hypotheses. Two more lines of evidence run outside this sandbox: historical crypto books with real volatility breaks (`scripts/crypto_benchmarks.py`), and a **live paper desk on BTC-USDT and ETH-USDT from 1 December**. The paper desk's track record around every scheduled FOMC / CPI / payrolls release is produced by `scripts/track_record.py`.

### Desktop app

Download it from the repository's **Releases** page once a version tag such as `v1.0.0` has been pushed (Windows installer or portable `.exe`, macOS `.dmg`, Linux AppImage or `.deb`), or build it yourself (`desktop/README.md`). The Python engine is bundled, so nothing else needs to be installed. The app contains the live cockpit with the **DESK** window (trust light and live P&L of the neutral, always-on and guarded desks), plus menus to start the live paper desk, open its track record, download data and re-run every benchmark.

---

## Quick start (Windows, macOS or Linux; Python ≥ 3.9)

```bash
git clone -b claude/sleepy-clarke-7g7s3l https://github.com/akshatinnovate-png/twipwire-lob.git
cd twipwire-lob
pip install -r requirements.txt
python demo.py                            # checks, warms up, opens the cockpit in your browser
```

That is enough for the live demo on synthetic data (`python scripts/cockpit.py` starts the server without the launcher). To also replay **real LOBSTER data** in the cockpit:

```bash
python scripts/fetch_lobster.py           # ~400 MB, once (AAPL, AMZN, MSFT; 21 June 2012)
python scripts/build_cache.py             # features for the cockpit, a few minutes
python scripts/preflight.py               # must print READY
python demo.py --source AAPL              # or AMZN / MSFT
```

Everything is also one command: `python tripwire_cli.py --help` (`serve`, `paper`, `report`, `record`, `fetch`, `cache`, `bench`, `preflight`).

**Live paper desk** (needs internet; run it from 1 December on a laptop or server that stays on): `python tripwire_cli.py paper`. It paper-trades BTC-USDT and ETH-USDT with three desks and logs everything to `data/paper/`. `python tripwire_cli.py report` writes the track record. Check the release times in `data/events.csv` against the official Fed and BLS calendars and set `verified=yes`.

**Live BTC-USDT** (needs internet on the laptop): record 30–45 minutes beforehand so the engine starts warm, then go live:

```bash
python scripts/record_binance.py --minutes 45
python demo.py --source BTC
```

If the feed drops, the cockpit replays the recording and labels it **REPLAY** on screen.

To reproduce **every number** in this README and in `results/`:

```bash
python scripts/run_benchmarks.py          # ~25 min on 4 cores; --quick for a ~5 min smoke run
python -m pytest -q                       # 31 tests: no look-ahead, valid bet, ARL, coverage, queue fills, latency, guard, paper desk
python scripts/desk_benchmarks.py         # ~5 min: the pre-registered desk test (H1-H3) -> results/desk/
python scripts/splice_study.py            # ~40 min: detector variants vs splices and the false-alarm null
```

## The live cockpit

`scripts/cockpit.py` runs the real engine in-process and streams to the browser. It uses only the standard library, so it works offline.

* **Shock buttons / keys**: CONCEPT FLIP `1` (P(Y|X) reverses, P(X) identical), COVARIATE SHOCK `2` (2.2× order flow, wider spread), LIQUIDITY SWEEP `3` (both), BACK TO CALM `0`; `space` pause, `s` next source, `r` restart.
* **Sources**: synthetic generator; real LOBSTER AAPL/AMZN/MSFT replayed event by event; live BTC-USDT from Binance's public book (warm-started from a recording, with replay fallback). On real data, CONCEPT FLIP reverses the real signal in the real label stream.
* **The standard baseline runs live beside us**: rolling two-sample KS on every feature (exactly B3). Its alarms are marked ▼KS on the detector chart, so viewers watch it fire on calm data.
* **Damage meter**: after each shock, the summed squared error of the frozen model vs TRIPWIRE, live.
* **Alarms explain themselves**: a Monitor F alarm names the features that moved most; the log notes when a frozen regime head wins its weight back (**REGIME RECURRENCE**).
* **Windows**: forecast with ŷ ± q and the realised move; both Shiryaev–Roberts statistics against their thresholds; fixed-share weights (learning / frozen / empty slots); the 10-level book; telemetry (coverage, call rate, hit rate, latency); and an event log.
* **Evidence**: replays of recorded benchmark runs plus the headline tables. **Method**: the architecture.

The minute-by-minute finals script is in `docs/DEMO_SCRIPT.md`. Run `scripts/preflight.py` before going on stage.

## How it works

| Layer | What it does | What is guaranteed, and under which assumption |
|---|---|---|
| **A. Features** (`tripwire/features.py`) | 16 causal features per event from 10 book levels: multi-level OFI over trailing depth (L1–L5, deep L6–10), micro-price − mid, 4 depth imbalances, bid/ask log-depth slopes, log spread, 20/200 volatility ratio, past h-event return. | Causal by construction (tested). |
| **B. Expert pool** (`tripwire/pool.py`) | K = 9: a frozen ridge; RLS with λ ∈ {0.998, 0.9995, 0.99995, 1}; and up to 4 heads spawned on Monitor-R alarms, each frozen when the next one spawns. Herbster–Warmuth fixed-share weights on matured labels, with η = 0.3/h because consecutive labels overlap by h−1 events. | Regret against the best switching sequence of experts, for bounded (winsorised) losses. The label delay loosens the bound, so we measure the gap to an oracle. |
| **C. Detectors** (`tripwire/conformal.py`) | **Monitor F** (label-free): Mahalanobis novelty of the features. **Monitor R** (label-aware): the orthogonality score −(y−f)·f / (σ_y σ_f) of the long-memory expert f. Scores are averaged over non-overlapping blocks (F: 200 events; R: max(30, 2h)), turned into conformal p-values against a rolling, gapped reference, and fed to a Shiryaev–Roberts statistic R_t = (1+R_{t−1})·b(p_t) with the power-mixture bet b(p) = ∫₀¹ ε p^(ε−1) dε. Alarm at R_t ≥ A. | Under independent uniform p-values, R_t − t is a martingale, so the average run length to a false alarm is ≥ A. A = ARL_events / block, with ARL = 50k events (synthetic) or 60k events ≈ one AAPL market hour (real). On real data the guarantee is approximate; we **measure** it (see the false-alarm table). |
| **D. Loud degradation** (`tripwire/aci.py`, `engine.py`) | An online quantile of \|y − ŷ\| (adaptive conformal inference) gives a 90 % interval. An F alarm inflates it ×1.5 for 3h+200 events; an R alarm boosts its step, raises the switching rate for 100 labels and spawns a head. Calls: up/down only if \|ŷ\| > k*·q_t, where k* is calibrated on the warm-up to a 20 % call budget, so calls drop automatically when q widens. | Long-run miscoverage → α for any sequence at a fixed step size. The delay and the boosts are outside the theorem, so coverage is measured. |

**Reactions.** A Monitor-F alarm means "the inputs look new": the interval widens and the model is left alone. A Monitor-R alarm means "the inputs mean something new": the model adapts. Neither overrides the weights directly.

**Causality.** Every supervised update (RLS, fixed-share weights, Monitor R, the interval) uses only labels that have matured, i.e. the event from h steps ago. `tests/test_tripwire.py::test_engine_never_sees_the_future` rewrites the future of a stream and asserts that every past output is bit-identical.

## Changes since the Round-1 abstract, and why

| Abstract | Now | Evidence that forced the change |
|---|---|---|
| Monitor R scores \|matured residual\| | Monitor R scores **orthogonality** −(y−f)·f of the long-memory expert | When R² ≈ 1 % (real LOBSTER), a sign flip raises E\|r\|² only by a factor of about 1+3R². On injected real-data breaks the \|residual\| monitor (ablation A4) detects far fewer breaks than the orthogonality monitor; see `results/tables.md`. |
| Call iff \|ŷ\| − q > half spread | Call iff \|ŷ\| > k*·q (k* calibrated); the abstract rule is still available as `gate="abstract"` | AAPL and AMZN spreads are 13–15 ticks, so the abstract rule never fires on real data. The ratio form keeps the same "abstain when unsure" logic. |
| K = 8, RLS λ ∈ {0.98, 0.995, 0.9995} | K = 9, λ ∈ {0.998, 0.9995, 0.99995, 1} | Real books favour long memory; the best fixed λ in hindsight is 0.99995–1. |
| Test every event | Test non-overlapping block means | h-event labels overlap and features are autocorrelated; block means are close to independent. |
| — | Synthetic SNR set to R² ≈ 7–12 % | The earlier toy had R² ≈ 0.99, which makes every method look perfect. |

## Results

All tables are in [`results/tables.md`](results/tables.md) and figures in `results/figures/`. The raw output is `results/benchmarks.json`.

### Commitments from the Round-1 abstract

| Committed target | Result | Met |
|---|---|---|
| False alarms within 2× nominal | Real data, block-shuffled null (observed / nominal, F; R): AAPL 1.01; 0.17 · AMZN 1.02; 0.38 · MSFT 1.56; 1.14. Synthetic calm: 0.15; 0.36 | yes |
| Lower cost of delay than B3 and B4 on concept-shift cells | Concept 2685 ± 198 vs B3 3941 ± 570, B4 3028 ± 337 · coupled 4049 ± 313 vs 7755 ± 1080, 5083 ± 481 | yes (vs B4 on the concept cell the CIs overlap) |
| Coverage within 5 points of 90 % | 89.7–90.0 % on every real stream; 89.8–90.0 % on every synthetic cell | yes |
| MVP 1: predict short-term price direction or spread | AAPL / AMZN / MSFT, h = 10 and 50, mid and spread targets (table below) | yes |
| MVP 2: detect a shift with a statistical test and show the model reacting | Two sequential tests; live cockpit; replays | yes |
| Stretch: multiple regime-adaptive heads | Spawned heads are kept frozen; in the recurrence run the frozen head and the frozen ridge win weight back when their regimes return (`results/figures/recurrence.png`) | yes |
| Stretch: latency under a budget | p50 ≈ 70 µs, p99 ≤ 200 µs per event on one core (budget 1 ms) | yes |

### Synthetic 2×2 benchmark (20 seeds, h = 10, change at event 20 000)

| Cell | Cost of delay @3000, TRIPWIRE | B3 rolling-KS | B4 best λ (hindsight) | B1 frozen | Monitor R fired / delay | Monitor F fired / delay |
|---|---|---|---|---|---|---|
| covariate (P(X) only) | 607 ± 105 | 3683 ± 492 | 567 ± 109 | 3881 | 15 % / – | 100 % / 629 |
| concept (P(Y\|X) only) | **2685 ± 198** | 3941 ± 570 | 3028 ± 337 | 9339 | 100 % / 577 | 20 % / – |
| coupled (both) | **4049 ± 313** | 7755 ± 1080 | 5083 ± 481 | 31 712 | 100 % / 162 | 100 % / 629 |

* **Shift-type selectivity.** Monitor R fires on every concept shift and on 15 % of harmless covariate shifts (3 of 20; the price of its faster 30-event blocks, which cut concept-flip delay from 727 to 577 events). Monitor F fires on every P(X) shift.
* **Calm data.** False alarms per 50k events: Monitor F 0.37, Monitor R 0.07, rolling KS 18.0.
* **Silent-failure time after a concept flip.** 60 events for TRIPWIRE, 678 for B3, and about 14 000 for models that never alarm.
* **Ablations.** Removing Monitor R raises concept cost of delay from 2685 to 3210. With the abstract's \|residual\| score, Monitor R fires on only 25 % of concept flips (orthogonality: 100 %).

### Real LOBSTER data (21 June 2012, full day, walk-forward, labels delayed h events)

| | AAPL h=50 | AMZN h=50 | MSFT h=50 |
|---|---|---|---|
| R² out-of-sample: TRIPWIRE / B4 (hindsight λ) / B1 frozen | 0.031 / 0.032 / −0.116 | 0.064 / 0.062 / −0.177 | 0.239 / 0.239 / 0.157 |
| 90 % interval coverage: TRIPWIRE / static interval | 89.8 / 99.3 | 89.8 / 99.3 | 89.9 / 99.8 |
| Hit rate on calls (moves ≠ 0): TRIPWIRE / B1 | 68.2 / 63.1 | 74.8 / 66.1 | 95.1 / 98.9 |

*Injected concept break on real data* (12 runs per β; the slice's real signal is reversed after τ, P(X) untouched):

| β | Monitor R caught | \|residual\| monitor caught | CoD@3000: TRIPWIRE / B3 / B4 | Placebo (β = 1) |
|---|---|---|---|---|
| −3 | 83 % (median 849 events) | 17 % | **13 065** / 21 263 / 18 573 | 0 R alarms in 12 runs |
| −1 | 75 % (median 1349 events) | 17 % | **4 800** / 15 186 / 4 859 | |

*Natural shifts* (one stock's book spliced onto another's, 3 pairs × 3 offsets): alarms in 5/9 splices; post-change coverage 86.8 % vs 55–58 % for the baselines. Cost of delay 3090 vs B3 42 360, but B4 (2409) is lower here. In all 4 misses Monitor F had alarmed on the first stock 500–9 100 events before the splice, and 3 of them were still inside its re-arm window.

## Repository map

```
tripwire/          the method (numpy only)
  features.py      Stage A, streaming 16-feature extractor
  pool.py          Stage B, ridge + RLS experts, fixed-share, spawning
  conformal.py     Stage C, conformal p-values, power-mixture bet, Shiryaev-Roberts
  aci.py           Stage D, online quantile tracking
  engine.py        the per-event loop and the causality contract
  guard.py         TRUST/CAUTION/DISTRUST/RESTORE policy + fail-safes
  mm.py            market-making simulator: queue-accurate LOBSTER fills, latency, rebates, markouts
  minidesk.py      lightweight desks for the cockpit's DESK window
  desk_models.py   the 'desk's own model' (small MLP) used in the desk test
  changepoints.py  offline volatility change points (event labels that never come from TRIPWIRE)
  crypto_data.py   Tardis / recorded crypto books and trades
  livefeed.py      Binance public book + trades poller
  baselines.py     B1-B5
  metrics.py       cost of delay, silent-failure time, coverage, MCC, ...
  synthetic.py     2x2 regime generator (book-level)
  lobster.py       LOBSTER loader
scripts/
  cockpit.py       live demo server (stdlib) -> cockpit/index.html
  run_benchmarks.py  reproduces everything in results/
  report.py        tables, figures, cockpit summary
  desk_benchmarks.py   pre-registered desk test H1-H3 -> results/desk/
  crypto_benchmarks.py desk on historical crypto books (evidence B) -> results/crypto/
  paper_desk.py        live paper desk, BTC/ETH (evidence C) -> data/paper/
  track_record.py      paper-desk report around macro releases (H4/H5) -> results/paper/
  fetch_lobster.py / fetch_crypto.py / build_cache.py / record_binance.py / preflight.py
tripwire_cli.py    one CLI for all of the above (also the desktop app's backend)
desktop/           Electron app + PyInstaller spec; .github/workflows/desktop.yml builds installers
tests/             31 tests
docs/              PREREGISTRATION.md, INTEGRATION.md, DEMO_SCRIPT.md, JUDGE_QA.md, MIDWAY_REPORT.md
```

## Data

* **LOBSTER samples** (AAPL, AMZN, MSFT; 21 June 2012; 10 levels). Free from lobsterdata.com. `fetch_lobster.py` falls back to byte-identical public mirrors. We drop hidden executions (type 5) and halts (type 7), so every event changes the visible book.
* **BTC-USDT / ETH-USDT**: `scripts/record_binance.py` and the paper desk record the 10-level book and trades into LOBSTER-style files, so they run through the same pipeline.
* **Crypto history**: Tardis.dev publishes the first day of every month for free (`scripts/fetch_crypto.py`).
* **Event calendar**: `data/events.csv` (FOMC, CPI, payrolls in December 2026). Times are marked unverified until they are checked against the official calendars.

## Limitations (stated before a judge asks)

* The ARL guarantee is exact only for independent uniform p-values. Real LOB data are neither, so we report observed / nominal false-alarm ratios instead of assuming them.
* After an alarm a monitor needs a short re-arm window (F: 20 blocks = 4000 events; R: 60 blocks) during which a second change is caught only by the pool's own adaptation.
* Monitor R normalises by running volatility, so a pure change of scale in y is absorbed rather than flagged (this is visible in the AAPL→MSFT splice).
* Event-time forecasts at h = 10–50 do not beat the spread on these stocks: the gross edge per call is below half the spread. TRIPWIRE is a monitoring and risk system; it does not claim alpha.
* Desk test: the desk's frozen MLP has negative out-of-sample R² on all three real days, so "always trust the model" loses even without a break. The guard's normal-day result (H1) partly reflects how bad that model is. The injected break (β = −3) is a stress test: even the neutral desk loses about 138k $ in 30 minutes.
* The cockpit's DESK window uses simple touch fills for speed. Every published desk number uses the queue-accurate simulator.
* The crypto backtest and the live track record are produced on the team's machines; this repository's sandbox cannot reach exchanges.
