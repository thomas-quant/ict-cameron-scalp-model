# Cameron's Scalping Model

An ICT-based intraday scalping model for **NQ (Nasdaq Futures)**, backtested on 30-second bars derived from tick data.

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

Data period: **Aug 28 – Nov 21, 2025** (NQ tick data resampled to 30-second bars).
Transaction cost: **0.25 pts** round trip.

### Head-to-Head Comparison

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

### Monthly Breakdown — Higher R:R

| Month | Trades | P&L | Avg R | Win% |
|---|---|---|---|---|
| 2025-08 | 21 | +240.0 pts | 0.96 | 47.6% |
| 2025-09 | 186 | +1,139.5 pts | 0.47 | 35.5% |
| 2025-10 | 227 | +463.0 pts | 0.23 | 26.0% |
| 2025-11 | 184 | +1,477.2 pts | 0.70 | 33.7% |

### Direction Breakdown

| | Base Long | Base Short | HRR Long | HRR Short |
|---|---|---|---|---|
| Trades | 384 | 415 | 303 | 315 |
| Win Rate | 48.7% | 47.5% | 58.7% | 6.0% |

> **Note:** The short-side win rate for Higher R:R is low because the test period (Aug–Nov 2025) was a strong bull market. Short structural targets rarely hit in a persistent uptrend. Long winners (+6,236 pts) more than compensated.

### Key Findings

1. **Higher R:R variant is strongly positive** — +0.47R expectancy, 1.68 profit factor, all four months profitable.
2. **Base model is roughly break-even** — 48% win rate at 1:1 isn't sufficient to overcome transaction costs and small structural disadvantages.
3. **R-multiple distribution** — Base model clusters at ±1R. Higher R:R shows a long right tail with winners at 2–5R+ compensating for the lower hit rate.

---

## Project Structure

```
├── backtest.py          # Main backtesting engine
├── strategy.md          # Detailed strategy specification
├── report.html          # Interactive HTML backtest report (Plotly charts)
├── test_backtest.py     # Unit tests for component functions
├── trades_base.csv      # Trade log — Base Model
├── trades_higher_rr.csv # Trade log — Higher R:R
├── data/
│   ├── nq_1m.parquet    # 1-minute OHLCV bars (Sep 2020 – Nov 2025)
│   └── nq_ticks.parquet # Raw tick data (Aug – Nov 2025)
└── docs/
```

## Running the Backtest

```bash
python backtest.py
```

This loads tick data from `data/nq_ticks.parquet`, resamples to 30-second bars for entry, and builds 5m/15m/1h bars from `data/nq_1m.parquet` for higher-timeframe analysis. Results are saved to `report.html` and CSV trade logs.
