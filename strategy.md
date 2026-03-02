# Cameron's Scalping Model

**Source:** YouTube walkthrough of "Cameron's Model" (based on ICT concepts)
**Instrument:** NQ (Nasdaq Futures)
**Style:** Intraday scalping with multi-timeframe analysis

---

## Overview

A structured three-component scalping model built on ICT (Inner Circle Trader) principles. The model uses a top-down approach: identify a draw on liquidity, wait for a stop raid to confirm direction, then enter on a 30-second fair value gap. The video also covers personal adjustments for higher risk-to-reward trading.

---

## The Three Components

### 1. Draw on Liquidity

- Identify an **old high** or **old low** on the **1-hour** chart.
- If no clear draw exists on the hourly, drop to the **15-minute** chart.
- This is the level you anticipate price will reach — your directional target.
- Consider whether price is moving **external to internal** (from a high/low toward a FVG or consolidation) or **internal to external** (from a FVG/consolidation toward a high/low).

### 2. Stop Raid

- On the **5-minute** chart, look for a stop raid in the **opposite direction** of your draw on liquidity.
  - **Bullish draw (target is above):** Look for a **swing low to be taken out** on the 5-minute.
  - **Bearish draw (target is below):** Look for a **swing high to be taken out** on the 5-minute.
- After the sweep, wait for a **confirmation candle**:
  - Bullish: An **up-close candle** (closes back into range after sweeping the low).
  - Bearish: A **down-close candle** (closes back into range after sweeping the high).

### 3. Entry on 30-Second Fair Value Gap

- After the confirmation candle on the 5-minute, drop to the **30-second** chart.
- Look for a **fair value gap (FVG)** to form in the direction of the trade.
- Enter when price returns to fill or react to the FVG.
- If the first FVG is missed, subsequent FVGs on the 30-second chart can also be used.

---

## Risk Management (Base Model)

| Parameter       | Value                  |
|-----------------|------------------------|
| **Stop Loss**   | 12 points (or less if structure allows) |
| **Take Profit** | Fixed 1:1 risk-to-reward |
| **Entry TF**    | 30-second chart        |
| **Bias TF**     | 1-hour / 15-minute     |
| **Raid TF**     | 5-minute               |

- If the nearest structural level for a stop is less than 12 points (e.g., 10 points), use the tighter stop — no need to add extra buffer beyond the structure.

---

## Personal Adjustments (Higher R:R Variant)

The base model uses a fixed 1:1 R:R, but can be modified for higher reward:

### Modified Take Profit

Instead of a fixed 1:1, target **structural levels**:

- A previous swing low/high
- A fair value gap on a higher time frame
- A discount/premium level of the range

**Example results from the video:**
- Trade 1: 1.55 R (targeting a previous low / FVG)
- Trade 2: 1.83 R (targeting a previous high)
- Trade 3: 2.0+ R (targeting discount of range)
- Some trades offered 4.22 R when held to the full draw on liquidity

### Modified Entry (Skip 5-Minute Close)

Instead of waiting for the 5-minute candle to close:

1. Identify the 5-minute sweep as it happens.
2. **Immediately** drop to the 30-second chart.
3. Look for **displacement** (aggressive, impulsive price movement) on the 30-second.
4. Use an entry model on the lower time frame:
   - **Breaker block** with a fair value gap (unicorn model)
   - **Order block** after displacement
   - **Fair value gap** after displacement through a key level

---

## Concepts Referenced

| Concept                        | Description |
|--------------------------------|-------------|
| **Fair Value Gap (FVG)**       | A three-candle pattern where the wicks of candle 1 and candle 3 don't overlap, leaving an imbalance |
| **Stop Raid / Sweep**         | Price moves beyond a swing high/low to trigger stop losses, then reverses |
| **External to Internal**       | Price moving from an extreme (high/low) toward the middle of a range (FVG, consolidation) |
| **Internal to External**       | Price moving from the middle of a range toward an extreme (high/low) |
| **Order Block**                | The last opposing candle before an impulsive move |
| **Breaker Block**              | A failed order block that becomes support/resistance after being violated |
| **Displacement**               | Aggressive, impulsive candle(s) showing strong commitment in one direction |
| **Unicorn Model**              | A breaker block that overlaps with a fair value gap — considered a high-probability entry |

---

## Step-by-Step Checklist (Base Model)

1. [ ] On the 1-hour or 15-minute chart, identify an old high/low as the draw on liquidity
2. [ ] Determine if price is going external-to-internal or internal-to-external
3. [ ] On the 5-minute chart, wait for a stop raid in the opposite direction of the draw
4. [ ] Wait for a confirmation candle (up-close for bullish / down-close for bearish)
5. [ ] Drop to the 30-second chart
6. [ ] Identify a fair value gap forming in the direction of the trade
7. [ ] Enter when price returns to the FVG
8. [ ] Place stop loss at 12 points or less (based on structure)
9. [ ] Take profit at 1:1 (base) or at a structural target (adjusted)

## Step-by-Step Checklist (Modified / Higher R:R)

1. [ ] On the 1-hour or 15-minute chart, identify an old high/low as the draw on liquidity
2. [ ] On the 5-minute chart, identify the sweep as it happens (don't wait for the candle to close)
3. [ ] Immediately drop to the 30-second chart
4. [ ] Wait for displacement through a key level on the 30-second
5. [ ] Enter on a breaker block, order block, or fair value gap after displacement
6. [ ] Place stop at the swing point created by the displacement
7. [ ] Target a structural level: previous high/low, higher-TF FVG, or discount/premium of the range
