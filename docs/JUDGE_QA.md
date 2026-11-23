# Judge Q&A: hard questions and our answers

Each answer gives a number and where it comes from (`results/tables.md`, from `python scripts/run_benchmarks.py`). The rule for both speakers: if we fell short somewhere, say so first, then give the number.

---

### Problem understanding

**Q1. What exactly is hard about this problem?**
Three things. (1) The harmful shift, a change in P(Y|X), is invisible in the features, so a feature-only test such as rolling KS cannot see it. (2) Labels arrive h events late, so any error-based detector lags by at least h. (3) Most feature shifts are harmless, and retraining on every one of them hurts. Our two monitors split exactly along (1) and (3), and Monitor F acts inside the label delay of (2).

**Q2. Why not just retrain on a schedule?**
That is baseline B2. Its cost of delay on concept flips is 3641 vs our 2685, and on coupled shifts 6776 vs 4049. On real AAPL at h = 50 its out-of-sample R² is −0.147 vs our 0.031, because frequent refits on noisy windows overfit.

**Q3. Why not rolling KS? It is the obvious detector.**
It has no false-alarm control: it tests overlapping windows of dependent data at every check. On calm synthetic data it fired 18.0 times per 50k events, against 0.37 (F) and 0.07 (R) for our monitors. The cockpit runs it live beside us, so judges can watch it fire on calm data. It is also blind to concept shifts by construction. It "detects" them only because it alarms all the time, with a delay of 1615 events.

### Statistics

**Q4. Is the average-run-length guarantee real?**
It is exact when the p-values are independent and uniform: R_t − t is then a martingale, and optional stopping gives E[τ] ≥ A. Conformal p-values on real LOB data are only approximately like that. So we (a) average scores over non-overlapping blocks (F: 200 events; R: max(30, 2h)), (b) keep a gap between new scores and the reference, and (c) **measure** the false-alarm rate. On block-shuffled real data, observed / nominal is 1.01 / 1.02 / 1.56 for F and 0.17 / 0.38 / 1.14 for R (AAPL / AMZN / MSFT). That is within the 2× we committed to. The validity of the bet is checked numerically in `tests/test_tripwire.py`.

**Q5. Why a block-shuffled null rather than the real day?**
One trading day contains genuine intraday shifts, so "alarms on the real day" mixes real changes with miscalibration. Shuffling 500-event blocks keeps short-range dependence and the real marginals but destroys regime ordering, which isolates miscalibration. On the time-ordered AAPL day we see about 10 F and 10 R alarms over 6.5 hours. We present those as candidate real shifts, not as false alarms.

**Q6. Why does Monitor R test −(y−f)·f instead of the residual size?**
A least-squares forecaster that is still right satisfies E[(y−f)·f] = 0. When the meaning of the book flips, the errors line up against the forecast and the mean becomes −2E[s²]. The squared residual grows only by a factor of about 1+3R², and on real data R² ≈ 1–3 %, so the residual monitor has almost no power. The measurements agree: on injected real breaks, orthogonality caught 83 % (β = −3) where |residual| caught 17 %; on synthetic concept flips, 100 % vs 25 %. This is the one place we changed the Round-1 design, and the data forced it.

**Q7. Why test the long-memory expert and not the pool's forecast?**
The pool adapts within a few hundred events and erases its own evidence. The λ = 1 expert keeps the old relationship long enough for the evidence to accumulate. The pool's job is to adapt; the monitor's job is to notice.

**Q8. The fixed-share bound assumes bounded losses and no delay. Does it still apply?**
Losses are winsorised (squared standardised loss capped at 16), so they are bounded. The delay loosens the bound, so we do not lean on it; we report the gap to the in-hindsight regime oracle (cost of delay) instead. We set η = 0.3/h because consecutive labels overlap by h−1 events, which would otherwise count the same information h times. That rule is fixed; it was not tuned per dataset.

**Q9. Is 90 % coverage meaningful when the intervals could just be wide?**
The intervals are the narrowest that hit 90 %. Static intervals on the same data cover 98–99.8 %, so they are too wide and therefore uninformative. The adaptive ones sit at 89.7–90.0 % on every real stream. After a coupled shift, our post-change coverage is 89.8 % while frozen and retrained baselines drop to 65–86 %.

### Engineering and honesty

**Q10. Is there look-ahead anywhere?**
No. `test_engine_never_sees_the_future` rewrites the future of a stream and asserts that every past output (forecast, interval, call, both detector statistics) is bit-identical. Features are tested the same way. Labels sit in an h+1 ring buffer and are consumed only at maturity. The injected-shift experiment uses an in-slice regression only to *construct* the perturbation; the model never sees it.

**Q11. How were hyperparameters chosen? Did you tune on the test set?**
They are fixed in `tripwire/config.py` and identical across datasets, except for settings derived from the data's event rate (h, the fit-block length, ARL = one market hour ≈ 60k AAPL events). Two design changes were made after looking at results, and both are documented: the orthogonality score, and the longer Monitor F block (200 events). The second was adopted because it brought real-data false alarms within the committed 2×. We also tested a third idea (refitting Monitor F's score after alarms). It caught one more splice but pushed MSFT false alarms to 2.27×, so we rejected it.

**Q12. B4 matches you on real-data R². So what does TRIPWIRE add?**
B4's λ is chosen **in hindsight on the test day**, which no live system can do, and B4 never raises an alarm or produces a calibrated interval. TRIPWIRE matches it without hindsight (AAPL 0.031 vs 0.032, MSFT 0.239 vs 0.239). It adds alarms, 90 % coverage instead of 99 %, and much lower cost of delay on real injected breaks (13 065 vs 18 573 at β = −3).

**Q13. MSFT hit rate of 95 % looks too good.**
MSFT in 2012 was a large-tick stock: the mid changed on only 0.6 % of events, and when it does move within 50 events, queue imbalance predicts the direction well. That is a known microstructure effect. We call on 28.5 % of events and score only events where the price actually moved. The gross edge per call (0.21 ticks) is below half the spread (0.63 ticks), so this is not a trading claim.

**Q14. Can you make money with this?**
We don't claim to. At h = 10–50 events the gross edge per call is below half the spread on every stock. The system's value is in forecasting and monitoring: it knows when it is wrong.

**Q15. Why 2012 data?**
These are the LOBSTER sample files the abstract committed to: free, public, 10 levels, three stocks with very different tick regimes. The pipeline is format-agnostic, and `scripts/record_binance.py` records BTC-USDT into the same format. We did not use crypto in the reported results.

**Q16. Where does it fail?**
Four known places. (1) After an alarm, Monitor F needs a re-arm window. Cross-stock splices were detected in 5 of 9 runs. Three of the misses are AMZN→AAPL, where the features move only 0.3–0.4 standard deviations and the old model is just 4–11 % worse afterwards, so this is close to a harmless shift. (2) Monitor R normalises by running volatility, so a pure change in the scale of y is absorbed rather than flagged. (3) With 30-event blocks, Monitor R fires on 15 % of harmless covariate shifts; that is the cost of cutting concept-flip delay from 727 to 577 events. (4) In the recurrence run there is one Monitor R false alarm in 100k events, within the budget of 2.

**Q16b. Why not just add a second Monitor F, or re-arm instantly?**
We tried four variants and measured each against the false-alarm null (`scripts/splice_study.py`, `results/splice_study.json`). Instant re-arm, two-sided novelty, and a second Monitor F at 100- or 400-event blocks. The best caught 7 of 9 splices, but its real-data false-alarm ratio reached 2.2–3.7×, breaking the 2× budget we committed to. We ship the design that keeps the budget.

**Q17. Why not a deep model (DeepLOB, transformers)?**
The bottleneck under regime shifts is knowing *when* the model is wrong and adapting cheaply, not raw capacity. Our forecasters are deliberately simple so that the detectors and intervals can be validated. Any forecaster, deep ones included, can sit in the pool as one more expert: the fixed-share guarantee does not depend on how an expert was produced.

**Q18. Latency?**
p50 ≈ 70 µs and p99 ≤ 200 µs per book event on one core, in pure Python and NumPy, including feature extraction, all updates and both monitors. The 1 ms budget is met by a factor of about 4. Occasional maxima of 20–35 ms are Python garbage-collection pauses, and we report them as measured.

**Q19. What is novel? These components all exist.**
The pieces are published (conformal martingales, Shiryaev–Roberts, fixed-share, ACI). The contributions are: the combination with explicit delayed-label handling; the orthogonality score for concept monitoring at low R², which works where the textbook residual monitor fails; the pairing of a label-free and a label-aware monitor that separates harmless from harmful shifts; and two evaluation measures, cost of delay and silent-failure time, plus an injected-shift protocol on real data with P(X) fixed by construction.

**Q20. Can everything be reproduced?**
Yes. Run `python scripts/run_benchmarks.py` (about 25 minutes on 4 cores) and `python -m pytest -q` (10 tests). Seeds are fixed. The tables, figures and cockpit numbers are all generated from `results/benchmarks.json`.
