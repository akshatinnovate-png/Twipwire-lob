# Desk use case: results

Settings and pass criteria were fixed in `docs/PREREGISTRATION.md` before any of this ran. P&L is in USD per desk, gross of fees except the LOBSTER rebate; one lot is 100 shares.

## Pre-registered hypotheses

| hypothesis | result | detail |
|---|---|---|
| H1 | PASS | 3/3 days within 5 % of always-on (or better) |
| H2a | PASS | adverse selection S1 130,463 vs S2 67,813 $ (reduction 48 %) |
| H2b | FAIL | S2 beats S1 in 8/12 slices |
| H3 | PASS | mean post-break P&L: S2 -97,214, S3 -137,723, S4 -139,095 $ |

## H1: full real days, no injected break (desk model = frozen MLP)

| symbol | desk | P&L $ | fills | spread captured $ | markout 30 s $ | toxic fills % | max drawdown $ |
|---|---|---|---|---|---|---|---|
| AAPL | S0 neutral | 693 | 2,754 | 5,909 | -6,676 | 61 | 397 |
| AAPL | S1 always-on | -8,758 | 6,041 | 3,623 | -14,504 | 59 | 9,452 |
| AAPL | S2 guarded | -5,301 | 3,909 | 2,402 | -8,997 | 60 | 6,135 |
| AAPL | S2-F only | -8,617 | 6,025 | 3,636 | -14,648 | 59 | 9,313 |
| AAPL | S2-R only | -5,455 | 3,930 | 2,373 | -8,844 | 59 | 6,283 |
| AAPL | S3 KS-guard | 693 | 2,754 | 5,909 | -6,676 | 61 | 397 |
| AAPL | S4 drawdown-guard | -8,191 | 5,902 | 3,645 | -14,533 | 60 | 8,798 |
| AAPL | *desk model out-of-sample R²* | -0.8129 | | | | | |
| AMZN | S0 neutral | -578 | 977 | 1,758 | -2,401 | 69 | 875 |
| AMZN | S1 always-on | -2,259 | 2,165 | 1,175 | -5,811 | 66 | 2,766 |
| AMZN | S2 guarded | -399 | 621 | 299 | -1,124 | 63 | 1,302 |
| AMZN | S2-F only | -2,187 | 2,133 | 1,144 | -5,829 | 66 | 2,694 |
| AMZN | S2-R only | -432 | 625 | 297 | -1,107 | 62 | 1,302 |
| AMZN | S3 KS-guard | -578 | 977 | 1,758 | -2,401 | 69 | 875 |
| AMZN | S4 drawdown-guard | -2,192 | 1,965 | 1,223 | -5,417 | 67 | 2,501 |
| AMZN | *desk model out-of-sample R²* | -1.4951 | | | | | |
| MSFT | S0 neutral | 65 | 17 | 13 | -12 | 65 | 14 |
| MSFT | S1 always-on | -23 | 59 | 26 | -22 | 56 | 40 |
| MSFT | S2 guarded | -22 | 37 | 19 | -29 | 68 | 41 |
| MSFT | S2-F only | -26 | 58 | 24 | -18 | 55 | 47 |
| MSFT | S2-R only | -21 | 40 | 20 | -33 | 70 | 41 |
| MSFT | S3 KS-guard | 65 | 17 | 13 | -12 | 65 | 14 |
| MSFT | S4 drawdown-guard | -33 | 47 | 25 | -21 | 55 | 54 |
| MSFT | *desk model out-of-sample R²* | -1.8001 | | | | | |

## H2/H3: 30 minutes after a real-data break (β = −3), desk model = frozen MLP

| desk | mean P&L $ | mean adverse selection $ (lost) | mean fills | wins vs S1 |
|---|---|---|---|---|
| S0 neutral | -137,723 ± 68,484 | 110,201 | 742 | 10/12 |
| S1 always-on | -156,198 ± 84,799 | 130,463 | 829 | 0/12 |
| S2 guarded | -97,214 ± 32,449 | 67,813 | 596 | 8/12 |
| S2-F only | -155,848 ± 84,708 | 129,236 | 829 | 7/12 |
| S2-R only | -97,326 ± 32,405 | 67,772 | 596 | 8/12 |
| S3 KS-guard | -137,723 ± 68,484 | 110,201 | 742 | 10/12 |
| S4 drawdown-guard | -139,095 ± 68,598 | 112,121 | 743 | 9/12 |

Monitor R fired after the break in 8/12 slices (median 249 events).

## H2/H3: 30 minutes after a real-data break (β = −3), desk model = TRIPWIRE adaptive pool

| desk | mean P&L $ | mean adverse selection $ (lost) | mean fills | wins vs S1 |
|---|---|---|---|---|
| S0 neutral | -137,723 ± 68,484 | 110,201 | 742 | 0/12 |
| S1 always-on | 137,904 ± 93,526 | -123,833 | 569 | 0/12 |
| S2 guarded | 127,912 ± 85,657 | -116,656 | 568 | 5/12 |

Monitor R fired after the break in 7/12 slices (median 249 events).

## Sensitivity: fill and latency assumptions (S2 − S1, mean post-break P&L $)

| latency | cancellations | S1 P&L | S2 P&L | S2 − S1 | S2 wins |
|---|---|---|---|---|---|
| 0 ms | pro_rata | -64,068 | -39,486 | 24,582 | 6/12 |
| 1 ms | back | -156,199 | -97,243 | 58,955 | 8/12 |
| 1 ms | pro_rata | -156,198 | -97,214 | 58,984 | 8/12 |
| 10 ms | pro_rata | -196,279 | -125,114 | 71,165 | 11/12 |