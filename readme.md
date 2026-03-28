# Cameron's Scalping Model

An ICT-based intraday scalping model for **NQ (Nasdaq Futures)**, backtested on 30-second bars derived from tick data.

## Current Status

The original **Aug 28 - Nov 21, 2025** in-sample slice made the higher R:R variant look strong. That did **not** hold up once the test was expanded to the larger **Jul 12, 2022 - Nov 21, 2025** merged tick dataset and bar construction was forced to respect correct intra-timestamp trade ordering.

Current conclusion: **the model is not profitable on the larger sequencing-corrected dataset**.

## Strategy Overview

The model uses a top-down, three-step approach based on Inner Circle Trader (ICT) concepts:

### 1. Draw on Liquidity (1H / 15M)
Identify a directional target — an old high or old low on the hourly chart. This is the level you anticipate price will reach. If no clear draw exists on the 1H, drop to the 15-minute chart.

### 2. Stop Raid (5M)
Wait for a stop raid in the **opposite direction** of the draw:
- **Bullish draw →** look for a swing low sweep on the 5M, followed by an up-close confirmation candle.
- **Bearish draw →** look for a swing high sweep on the 5M, followed by a down-close confirmation candle.

### 3. Entry on Fair Value Gap (30s)
After the 5M confirmation, drop to the **30-second** chart and enter when price retraces into a fair value gap (FVG) in the direction of the trade.

---

## Two Variants

### Base Model (1:1 R:R)
- **Stop:** 12 points (or less if structure allows)
- **Target:** Fixed 1:1 risk-to-reward
- **Confirmation:** Waits for the 5M candle to close before acting on the raid

### Higher R:R Variant
- **Stop:** At displacement swing point (structural stop)
- **Target:** Nearest structural level on the 5M chart, capped at 5R
- **Min R:R:** 1.5:1 (trades below this are skipped)
- **Confirmation:** Acts on the sweep itself (no candle close required)
- **Extra entry:** Displacement order blocks are preferred over FVGs when available

---

## Backtest Results

Transaction cost: **0.25 pts** round trip.

### In-Sample: Original Tick Data

Data period: **Aug 28 - Nov 21, 2025** using `data/nq_ticks.parquet`.

| Metric | Base Model | Higher R:R |
|---|---|---|
| **Trades** | 799 | 618 |
| **Win Rate** | 48.1% | 31.9% |
| **Total P&L** | -287.2 pts | **+3,319.8 pts** |
| **Profit Factor** | 0.82 | **1.68** |
| **Expectancy** | -0.04R | **+0.47R** |
| **Sharpe (ann.)** | -4.01 | **6.18** |
| **Avg Winner** | 1.00R (3.32 pts) | 3.60R (41.61 pts) |
| **Avg Loser** | -1.00R (-3.77 pts) | -1.00R (-11.58 pts) |
| **Max Drawdown** | -6.40% | -7.98% |
| **Largest Win** | 11.8 pts | 81.0 pts |
| **Largest Loss** | -12.2 pts | -12.2 pts |

### Out-of-Sample: Sequencing-Corrected Merged Tick Data

Data period: **Jul 12, 2022 - Nov 21, 2025** using `data/merged_nq_ticks.parquet`, truncated to the overlap with `data/nq_1m.parquet`.

The merged dataset preserves trade order within identical timestamps via `intra_ts_rank`, so 30-second bars are built from correctly sequenced trades rather than assuming same-timestamp rows are interchangeable.

| Metric | Base Model | Higher R:R |
|---|---|---|
| **Trades** | 11,391 | 7,138 |
| **Win Rate** | 48.7% | 22.9% |
| **Total P&L** | **-3,768.5 pts** | **-3,271.0 pts** |
| **Profit Factor** | 0.81 | 0.95 |
| **Expectancy** | -0.03R | -0.03R |
| **Avg Winner** | 1.00R (2.84 pts) | 3.24R (36.41 pts) |
| **Avg Loser** | -1.00R (-3.35 pts) | -1.00R (-11.42 pts) |
| **Max Drawdown** | -3,822.5 pts | -3,936.8 pts |

### What Changed

1. **The positive 2025 result did not generalize.**
   The higher R:R variant went from **+3,319.8 pts in-sample** to **-3,271.0 pts** on the larger out-of-sample run.
2. **More data hurt both variants.**
   The base model remained negative, and the higher R:R variant also dropped below breakeven despite still producing larger winners than losers on average.
3. **Correct trade sequencing is now accounted for.**
   The merged parquet stores `intra_ts_rank`, and the backtest uses the sanitized merged schema when building 30-second bars so that same-timestamp trades keep their intended order.
4. **The model should be treated as unvalidated / currently unprofitable.**
   The old README headline was too optimistic because it centered the short in-sample slice instead of the broader validation set.

---

## Project Structure

```
├── backtest.py          # Main backtesting engine
├── strategy.md          # Detailed strategy specification
├── report.html          # Legacy report from the older pandas backtest
├── report_in_sample.html
├── report_out_of_sample.html
├── test_backtest.py     # Unit tests for component functions
├── trades_base.csv      # Legacy trade log from the older pandas backtest
├── trades_higher_rr.csv # Legacy trade log from the older pandas backtest
├── trades_base_in_sample.csv
├── trades_higher_rr_in_sample.csv
├── trades_base_out_of_sample.csv
├── trades_higher_rr_out_of_sample.csv
├── data/
│   ├── merged_nq_ticks.parquet # Sanitized merged ticks (Jul 2022 – Nov 2025)
│   ├── merged_ticks_example.csv
│   ├── merged_ticks_schema.txt
│   ├── nq_1m.parquet          # 1-minute OHLCV bars (Sep 2020 – Nov 2025)
│   └── nq_ticks.parquet       # Original raw tick data (Aug – Nov 2025)
└── docs/
```

## Running the Backtest

```bash
.venv/bin/python backtest.py
```

This runs two passes:

1. `data/nq_ticks.parquet` as the original in-sample test
2. `data/merged_nq_ticks.parquet` as the larger out-of-sample validation set

Both runs build 30-second entry bars, derive higher-timeframe context from `data/nq_1m.parquet`, and cap the tick data to the overlapping 1-minute range. Results are saved to:

- `report_in_sample.html`
- `report_out_of_sample.html`
- `trades_base_in_sample.csv`
- `trades_higher_rr_in_sample.csv`
- `trades_base_out_of_sample.csv`
- `trades_higher_rr_out_of_sample.csv`
