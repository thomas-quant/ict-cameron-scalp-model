# Future Research Plan

Research directions and questions to explore for improving Cameron's Scalping Model.

---

## 1. Volatility Regime Awareness (Stop Loss)

The current model uses a fixed 12pt stop (base) or displacement-based stop (HRR). In high-volatility environments, these stops may be too tight; in low-vol, they may be too wide relative to expected moves.

**Questions:**
- Does ATR-scaled stop placement improve win rate or expectancy?
- What ATR period best corresponds to NQ intraday volatility cycles (5m ATR-14? 1H ATR-5?)
- Is there a sweet spot for stop distance expressed in ATR multiples rather than fixed points?
- How does stop distance distribution look on winners vs losers? Are we getting stopped out by noise on trades that would've won?

---

## 2. Regime Filtering

The HRR short win rate is 6% because the test period was a strong uptrend. If we can detect regime, we can filter or adjust.

**Questions:**
- Can a simple trend filter (e.g., 1H EMA slope, daily close vs. 20-day MA) improve short win rate without killing long entries?
- Does the model perform better in ranging vs trending regimes? How do we classify these?
- Would a "long-only above X, short-only below X" overlay help, or does it destroy the model's edge on reversals?
- What's the ideal lookback for regime detection — daily, multi-day, weekly?

---

## 3. Session / Time-of-Day Filtering

The backtest trades 24/7, but ICT concepts are built around specific session times. The report shows P&L varies significantly by hour.

**Questions:**
- What happens if we restrict to NY session only (9:30–16:00 ET)? Or London + NY (3:00–16:00)?
- Are there specific hours that are consistently profitable vs consistently losing?
- Does the model edge concentrate around specific events (NY open, London close, etc.)?
- Should we have different parameters per session (wider stops in London, tighter in Asia)?

---

## 4. Raid Quality Scoring

Not all stop raids are equal. A sweep of a major daily low is more significant than a sweep of a minor 5m swing.

**Questions:**
- Does filtering raids by the significance of the swept level (higher swing_n, length of time the level held) improve results?
- Can we score raids by depth of sweep (how far below/above the level price traded)?
- Does volume at the sweep level matter? (Could detect this from tick data)
- Does confirmation candle strength (body size, wick ratio) predict trade outcome?

---

## 5. FVG Quality & Selection

Currently all FVGs are treated equally. Some may be higher quality signals.

**Questions:**
- Does FVG size (gap width) correlate with win rate? Are larger gaps better signals?
- Does the candle displacement before the FVG matter (bigger move = stronger FVG)?
- Should we prefer the first FVG after a raid vs later ones? Does FVG ordinal position affect outcome?
- What's the optimal max FVG age before it becomes stale?

---

## 6. Partial Profits & Trailing Stops

The base model uses fixed 1:1 TP. The HRR model targets structural levels. Neither uses trailing stops.

**Questions:**
- Does taking 50% off at 1R and trailing the rest to structure improve total P&L?
- What trailing method works best — ATR trail, swing-based trail, or time-based exit?
- Is there a time decay component — do setups that don't reach TP within N minutes tend to lose?

---

## 7. Draw on Liquidity Refinement

The draw identification uses simple swing highs/lows on 1H/15M. This could be more nuanced.

**Questions:**
- Does weighting draws by how many times a level has been tested improve directional accuracy?
- Should we incorporate daily/weekly levels as "higher conviction" draws?
- Does the internal-to-external vs external-to-internal distinction (mentioned in strategy) actually predict which setups win?
- Can we classify draw strength and only trade on stronger draws?

---

## 8. Multi-Instrument & Out-of-Sample Testing

Currently tested only on NQ for 3 months.

**Questions:**
- Does the model work on ES, YM, or RTY futures?
- Can we acquire more tick data to test across different market regimes (2022 bear, 2023 range, 2024 bull)?
- How sensitive are results to the specific 3-month window? Would rolling-window analysis show consistency?

---

## 9. Position Sizing

Currently flat 1-contract sizing.

**Questions:**
- What does Kelly criterion suggest for optimal sizing given the HRR distribution?
- Does volatility-adjusted sizing (smaller position in high-vol) improve risk-adjusted returns?
- What's the optimal fixed-fractional risk per trade for the HRR variant?

---

## Priority Order

| Priority | Research Area | Expected Impact | Effort |
|---|---|---|---|
| 🔴 High | Session filtering | High — quick filter, likely big WR improvement | Low |
| 🔴 High | Regime filtering | High — addresses short-side bleed | Medium |
| 🟡 Medium | Volatility-based stops | Medium — could improve both models | Medium |
| 🟡 Medium | Raid quality scoring | Medium — filter out weak setups | Medium |
| 🟢 Low | FVG quality | Moderate — refine entry selection | Medium |
| 🟢 Low | Trailing stops / partials | Moderate — optimize exits | Low |
| 🟢 Low | Draw refinement | Uncertain — hard to measure | High |
| ⚪ Backlog | Multi-instrument | Validation — doesn't improve model | High |
| ⚪ Backlog | Position sizing | Polish — only matters if edge is confirmed | Low |
