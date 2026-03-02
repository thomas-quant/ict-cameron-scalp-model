# Cameron's Scalping Model — Backtest Design

**Date:** 2026-02-26
**Data:** `nq_ticks.parquet` — ~25M tick trades, Aug 28 – Nov 27, 2025

## Architecture

A single Python script (`backtest.py`) that:

1. Loads tick data from `nq_ticks.parquet`
2. Resamples ticks into 30-second, 5-minute, 15-minute, and 1-hour OHLCV bars
3. Runs both strategy variants (base + higher R:R) as separate passes
4. Generates an HTML report with equity curves, trade logs, and summary stats
5. Exports trade logs as CSV

## Data Pipeline

```
nq_ticks.parquet → resample to 30s/5m/15m/1h bars → align by timestamp
```

Timestamps kept in UTC (matching tick data `ts_event`). Session labels derived from ET conversion if needed.

## Strategy Logic

### Step 1 — Draw on Liquidity (1h / 15m)

- Scan for swing highs/lows on 1h bars using N-bar pivot lookback (e.g., 20-bar highest high / lowest low not yet reached)
- If no clear draw on 1h, fall back to 15m swings
- Classify: bullish draw (old high above) or bearish draw (old low below)

### Step 2 — Stop Raid Detection (5m)

- Track swing highs/lows on 5m chart using 5-bar pivot
- Detect when price sweeps past a swing low (bullish setup) or swing high (bearish setup)
- **Base model**: Wait for 5m candle to close as confirmation (up-close after sweeping low, down-close after sweeping high)
- **Higher R:R model**: Trigger immediately on the sweep, don't wait for close

### Step 3 — Entry (30s)

- After a valid raid signal, scan 30s bars for a Fair Value Gap:
  - Bullish FVG: candle_1.high < candle_3.low (gap up)
  - Bearish FVG: candle_1.low > candle_3.high (gap down)
- Enter when price returns to touch the FVG
- **Higher R:R model**: Also detect displacement (large impulsive candle relative to recent ATR) and order blocks (last opposing candle before displacement)

### Step 4 — Exit

- **Base model**: Stop = 12 points (or tighter if structure allows). TP = 1:1 R:R
- **Higher R:R model**: Stop = swing point from displacement. TP = structural target (previous swing high/low on 5m, or higher-TF FVG midpoint)
- Exits checked bar-by-bar on 30s data
- If stop and TP both hit in same bar, assume stop hit first (conservative)

## Risk Management

- Max 1 trade open at a time
- 5-minute cooldown after exit before re-entry

## Output — HTML Report (plotly)

- Equity curve for each variant (cumulative P&L in points)
- Trade log table (entry time, exit time, direction, entry price, exit price, stop, target, P&L points, R multiple)
- Summary stats: total trades, win rate, avg R, max drawdown, profit factor, Sharpe, largest win/loss
- Monthly breakdown table
- CSV export of trade logs

## Dependencies

- pandas, numpy
- plotly
