# Pre-registration: TRIPWIRE as a model-risk guard for a market-making desk

**Committed before any desk experiment was run.** The git timestamp of this file is the proof. Changes after that commit are listed at the bottom with a reason; the original text stays.

## The claim under test
A market maker that leans its quotes on a short-term alpha model loses money to toxic fills after a regime break, until someone notices. Gating that alpha with TRIPWIRE's trust state (TRUST / CAUTION / DISTRUST / RESTORE) reduces those losses, at little cost on normal days.

## Desks compared (same simulator, same fills, same latency, same fees)
| id | desk | alpha used for skew |
|---|---|---|
| S0 | neutral | none (inventory skew only) |
| S1 | always-on | the desk model, always |
| **S2** | **guarded** | the desk model × TRIPWIRE trust |
| S3 | KS-guard (industry default) | the desk model, pulled for 3,000 events after a rolling-KS alarm |
| S4 | drawdown guard ("the human") | the desk model, pulled for 10 minutes after the desk's P&L falls 3× its own median 10-minute P&L range |
| S2-F, S2-R | ablations | guard driven by Monitor F only / Monitor R only |

**Desk models.** *Primary:* a small neural net (16-32-1 MLP) trained on the first 20,000 events of each stream and then frozen, standing for "the desk's own model". TRIPWIRE guards it from the outside (Monitor R tests the MLP's errors). *Secondary:* TRIPWIRE's own adaptive pool.

## Fixed settings
* **Quoting (Avellaneda–Stoikov style):** reservation price = mid + θ·trust·forecast − κ·inventory, with θ = 1 and κ = 0.25 tick per lot. Half-spread = max(half the quoted spread, 1 tick) × width multiplier. Quotes snap to ticks and never cross the opposite best. Inventory limit 10 lots (one lot: 100 shares; 0.01 BTC; 0.1 ETH).
* **Fills:**
  * LOBSTER: queue-position model on order-level messages. Our order joins the back of the queue; executions consume the queue ahead first; cancellations at our level reduce the queue ahead pro-rata.
  * Crypto: traded volume at our price must exceed the queue ahead; trades through our price fill us.
* **Latency:** 1 ms from decision to live quote (base case).
* **Fees:**
  * LOBSTER: maker rebate +0.20 ¢/share; the desk never takes liquidity.
  * Crypto: the primary metric is gross of fees; fees are reported as a sensitivity (maker +0.02 %, 0 %, −0.005 %).
* **Guard policy:**

  | State | Skew | Width | Size | Inventory limit |
  |---|---|---|---|---|
  | TRUST | ×1 | ×1 | ×1 | ×1 |
  | CAUTION (Monitor F alarm, held 3h+200 events) | ×1 | ×1.5 | ×0.5 | ×1 |
  | DISTRUST (Monitor R alarm) | 0 | ×2 | ×0.5 | ×0.5 |

  **RESTORE:** once the guarded model's matured R² over the last 1,000 labels is > 0, trust ramps linearly back to 1 over 2,000 events.

  **Fail-safe (live only):** no book event for 5 s → DISTRUST until 200 events after the feed resumes.
* **TRIPWIRE settings:** exactly those committed in `tripwire/config.py` at this commit (real data: h = 50, ARL = 60,000 events).

## Metrics
* **Primary:** desk P&L, gross of fees, marked to mid. It is split into:
  * spread captured
  * adverse selection: mid markout at 1 s, 5 s and 30 s after each fill
  * inventory P&L
* **Secondary:** fills, toxic-fill share (30 s markout against us), max drawdown, P&L per fill.

## Hypotheses and pass criteria
* **H1 (cheap on normal days).** On the three full real LOBSTER days with no injection, the guarded desk's P&L is at least the always-on desk's P&L minus 5 % of its absolute value, on at least 2 of 3 days.
* **H2 (protects after a break).** On the 12 real LOBSTER slices with a reversed-and-tripled signal (β = −3; the same protocol as the forecasting benchmark, with the book shifted so prices and fills stay consistent), over the 30 minutes after the break:
  * **H2a:** mean adverse-selection loss for S2 is ≥ 30 % lower than for S1.
  * **H2b:** S2's P&L beats S1's in ≥ 9 of 12 slices.
* **H3 (beats the status quo).** On the same 12 slices, S2's mean post-break P&L is ≥ that of both S3 (KS guard) and S4 (drawdown guard).
* **H4 (live, forward, descriptive).** Paper trading BTC-USDT and ETH-USDT live from 1 December to the finals:
  * Every scheduled US macro release in `data/events.csv` is reported, none excluded. For each, we report S1 vs S2 adverse selection and P&L from 5 minutes before to 30 minutes after.
  * The sample is small, so this is reported with bootstrap intervals and is **not** a pass/fail test.
* **H5 (live alarm calibration).** On placebo hours (no scheduled event and no offline change point within ±30 min), the DISTRUST rate is ≤ 2× nominal (one per ARL).

**Event labels never come from TRIPWIRE.** They come from official release calendars and from an offline change-point method on realised volatility (`tripwire/changepoints.py`, binary segmentation with a BIC penalty).

## What a failure means
If H1–H3 fail, we report the failure in the finals deck and the README, with the numbers. No setting above changes after results are seen, except under the amendment rule below.

## Amendment rule
Bugs may be fixed. Any change is logged here with its date, the reason, and whether results had already been seen.

## Amendments
1. **2026-11-26, bug fix to the break injection for the desk simulation (one slice's results had been seen).**
   The forecasting benchmark injects a break into the *labels*, then rebuilds a level series as h interleaved chains (`level[t+h] = level[t] + y[t]`). That is valid for scoring forecasts but is **not a price path**: neighbouring events sit on different chains, so the shifted book jumps by hundreds of ticks between events. On the first slice tested this produced ~1,700 spurious fills and a −$11k loss for *every* desk, including the neutral one.
   The fix injects the same reversed-and-tripled signal into the price path itself: from τ on, each event's mid move gets (β−1)/h times the signal of the last h events, so the h-event-ahead move carries (β−1) × signal, as in our synthetic generator. Prices are rounded cumulatively to whole ticks.
   Desks, settings, metrics and pass criteria are unchanged.
2. **2026-11-26, bug fix in the latency model (no full-benchmark results seen; the first run was stopped).**
   A quote re-decided while its previous version was still in flight restarted the in-flight timer. With latency longer than the gap between book events, quotes could never reach the market. Found by the live paper-desk test.
   In-flight quote messages are now queued per side and applied in order as each one's latency elapses, as an order gateway does.

## Results (added 2026-11-26, after the run; nothing above was changed)

Run: `python scripts/desk_benchmarks.py` (≈ 5 min on 4 cores). Full tables: `results/desk/desk_tables.md`; raw output: `results/desk/desk_results.json`.

| Hypothesis | Bar | Result | Verdict |
|---|---|---|---|
| H1 (cheap on normal days) | S2 ≥ S1 − 5 % on ≥ 2/3 days | S2 beat S1 on all 3 days (AAPL −5,301 vs −8,758; AMZN −399 vs −2,259; MSFT −22 vs −23 $) | **PASS** |
| H2a (less adverse selection) | ≥ 30 % lower than S1 | 67,813 vs 130,463 $ lost to adverse selection: **48 % lower** | **PASS** |
| H2b (wins slice by slice) | S2 > S1 in ≥ 9/12 slices | **8/12** | **FAIL** |
| H3 (beats the status quo) | S2 ≥ S3 and S4 | mean post-break P&L: S2 −97,214; S3 (KS guard) −137,723; S4 (drawdown guard) −139,095 $ | **PASS** |

**Why H2b failed.** Monitor R did not fire after the break in 4 of the 12 slices. S2 lost three of those (AAPL and AMZN at the first offset, MSFT at the second; by 158 to 1,691 $, the cost of CAUTION widening) and won the fourth anyway. The fourth loss is the slice where Monitor R fired late (AAPL, after 649 events; −2,866 $). In every slice where Monitor R fired within 450 events, S2 beat S1, by up to 404,713 $. So this is a detection-power failure, not a reaction failure: the guard helps when it fires, and costs little when it doesn't.

**Things this run also shows, which we did not pre-register and report as exploratory:**
* **The frozen desk model is bad out of sample on real days** (R² −0.81 AAPL, −1.50 AMZN, −1.80 MSFT), and S1 loses to the neutral S0 even on normal days. This is the situation the guard exists for: a model that stopped working without anyone noticing. Monitor R switched S2's alpha off 4 times on AAPL, 3 on AMZN and 12 on MSFT (each time until the model's matured R² was positive again), which is why S2 beat S1 on all three days.
* **The KS guard (S3) equals the neutral desk (S0) exactly**: rolling KS alarms so often that the alpha is always switched off. It never loses to S1, but it also never uses the model. That is the status quo's failure mode.
* **The drawdown guard (S4)** reacts after the losses and recovers little (−139,095 vs S1 −156,198 $).
* **Monitor F alone (S2-F) does nothing for P&L**; all of the benefit comes from Monitor R (S2-R ≈ S2). This is what the design predicts: a concept break leaves P(X) unchanged.
* **Sensitivity.** S2 − S1 stays positive under 0 ms latency (+24,582 $, 6/12), back-of-queue cancellations (+58,955 $, 8/12) and 10 ms latency (+71,165 $, 11/12).
* **Secondary alpha (TRIPWIRE's own adaptive pool).** The adaptive pool learns the reversed signal and *profits* after the break (S1 +137,904 $), and the guard then costs it 7 % (S2 +127,912 $). When the model adapts by itself, switching it off costs money. The injected break makes the signal three times stronger, so this result is optimistic.
* **Scale.** The injected break (β = −3) is a severe stress test: even the neutral desk loses ≈ 138k $ in 30 minutes. Read the dollar figures as relative, not as typical losses.
