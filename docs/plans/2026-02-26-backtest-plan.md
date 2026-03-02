# Cameron's Scalping Model Backtest — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a backtest of Cameron's ICT scalping model (both base and higher R:R variants) on NQ tick data, producing an HTML report with equity curves and trade logs.

**Architecture:** Single-file backtest (`backtest.py`) with modular functions for each strategy component. Tick data is resampled into multi-timeframe OHLCV bars. Two strategy passes (base + higher R:R) run sequentially. Results rendered as an HTML report via plotly. A separate `test_backtest.py` validates core detection functions.

**Tech Stack:** Python 3, pandas, numpy, plotly

---

### Task 1: Data Loading & Resampling

**Files:**
- Create: `backtest.py`
- Create: `test_backtest.py`

**Step 1: Write test for resampling**

```python
# test_backtest.py
import pandas as pd
import numpy as np
from backtest import resample_ticks

def test_resample_ticks_30s():
    """Verify ticks resample into correct 30s OHLCV bars."""
    ticks = pd.DataFrame({
        'ts_event': pd.to_datetime([
            '2025-09-01 14:00:00.100', '2025-09-01 14:00:00.200',
            '2025-09-01 14:00:15.000', '2025-09-01 14:00:29.999',
            '2025-09-01 14:00:30.000', '2025-09-01 14:00:45.000',
        ]),
        'price': [100.0, 101.0, 99.0, 102.0, 103.0, 104.0],
        'size': [1, 2, 1, 3, 1, 2],
    })
    bars = resample_ticks(ticks, '30s')
    assert len(bars) == 2
    # First bar: 14:00:00 - 14:00:29
    assert bars.iloc[0]['Open'] == 100.0
    assert bars.iloc[0]['High'] == 102.0
    assert bars.iloc[0]['Low'] == 99.0
    assert bars.iloc[0]['Close'] == 102.0
    assert bars.iloc[0]['Volume'] == 7  # 1+2+1+3
    # Second bar: 14:00:30 - 14:00:59
    assert bars.iloc[1]['Open'] == 103.0
    assert bars.iloc[1]['High'] == 104.0
```

**Step 2: Run test to verify it fails**

Run: `cd "/mnt/e/backup/code/Finance/Models/Assorted youtube models/Camerons-Scalping-Model" && python -m pytest test_backtest.py::test_resample_ticks_30s -v`
Expected: FAIL (backtest module doesn't exist yet)

**Step 3: Implement data loading and resampling**

```python
# backtest.py — top of file
"""
Cameron's Scalping Model Backtest
Both base (1:1 R:R) and higher R:R variants.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = Path(__file__).parent / "nq_ticks.parquet"


def load_ticks(path: Path = DATA_FILE) -> pd.DataFrame:
    """Load tick data, keep only trade fields we need."""
    df = pd.read_parquet(path, columns=['ts_event', 'price', 'size'])
    df = df.sort_values('ts_event').reset_index(drop=True)
    return df


def resample_ticks(ticks: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample tick data into OHLCV bars at the given frequency."""
    ticks = ticks.set_index('ts_event')
    bars = ticks['price'].resample(freq).ohlc()
    bars.columns = ['Open', 'High', 'Low', 'Close']
    bars['Volume'] = ticks['size'].resample(freq).sum()
    bars = bars.dropna(subset=['Open'])
    return bars


def build_bars(ticks: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all timeframe bars from ticks."""
    return {
        '30s': resample_ticks(ticks, '30s'),
        '5m': resample_ticks(ticks, '5min'),
        '15m': resample_ticks(ticks, '15min'),
        '1h': resample_ticks(ticks, '1h'),
    }
```

**Step 4: Run test to verify it passes**

Run: `cd "/mnt/e/backup/code/Finance/Models/Assorted youtube models/Camerons-Scalping-Model" && python -m pytest test_backtest.py::test_resample_ticks_30s -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backtest.py test_backtest.py
git commit -m "feat: data loading and tick-to-OHLCV resampling"
```

---

### Task 2: Swing High/Low Detection

**Files:**
- Modify: `backtest.py`
- Modify: `test_backtest.py`

**Step 1: Write test for swing detection**

```python
# test_backtest.py (append)
from backtest import find_swing_highs, find_swing_lows

def test_find_swing_highs():
    """A bar whose High is higher than N bars on each side is a swing high."""
    bars = pd.DataFrame({
        'High': [10, 12, 15, 13, 11, 14, 18, 16, 12, 10],
        'Low':  [ 8,  9, 11, 10,  9, 11, 14, 13, 10,  8],
    })
    highs = find_swing_highs(bars, n=2)
    # Index 2 (15) is a swing high: higher than indices 0,1 and 3,4
    # Index 6 (18) is a swing high: higher than indices 4,5 and 7,8
    assert list(highs.index) == [2, 6]
    assert list(highs['High']) == [15, 18]

def test_find_swing_lows():
    """A bar whose Low is lower than N bars on each side is a swing low."""
    bars = pd.DataFrame({
        'High': [10, 12, 15, 13, 11, 14, 18, 16, 12, 10],
        'Low':  [ 8,  9, 11, 10,  7, 11, 14, 13, 10,  6],
    })
    lows = find_swing_lows(bars, n=2)
    # Index 4 (7) is a swing low: lower than indices 2,3 and 5,6
    assert 4 in lows.index
    assert lows.loc[4, 'Low'] == 7
```

**Step 2: Run tests to verify they fail**

Run: `python -m pytest test_backtest.py::test_find_swing_highs test_backtest.py::test_find_swing_lows -v`
Expected: FAIL

**Step 3: Implement swing detection**

```python
# backtest.py (append)

def find_swing_highs(bars: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Find swing highs: bars whose High exceeds N bars on each side."""
    highs = bars['High']
    is_swing = pd.Series(False, index=bars.index)
    for i in range(n, len(bars) - n):
        window_left = highs.iloc[i - n:i]
        window_right = highs.iloc[i + 1:i + n + 1]
        if highs.iloc[i] > window_left.max() and highs.iloc[i] > window_right.max():
            is_swing.iloc[i] = True
    return bars[is_swing].copy()


def find_swing_lows(bars: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Find swing lows: bars whose Low is below N bars on each side."""
    lows = bars['Low']
    is_swing = pd.Series(False, index=bars.index)
    for i in range(n, len(bars) - n):
        window_left = lows.iloc[i - n:i]
        window_right = lows.iloc[i + 1:i + n + 1]
        if lows.iloc[i] < window_left.min() and lows.iloc[i] < window_right.min():
            is_swing.iloc[i] = True
    return bars[is_swing].copy()
```

**Step 4: Run tests**

Run: `python -m pytest test_backtest.py::test_find_swing_highs test_backtest.py::test_find_swing_lows -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backtest.py test_backtest.py
git commit -m "feat: swing high/low detection with N-bar pivot"
```

---

### Task 3: Draw on Liquidity Identification

**Files:**
- Modify: `backtest.py`
- Modify: `test_backtest.py`

**Step 1: Write test**

```python
# test_backtest.py (append)
from backtest import find_draw_on_liquidity

def test_draw_on_liquidity_bullish():
    """When an untouched old high exists above current price, draw is bullish."""
    bars_1h = pd.DataFrame({
        'Open':  [100, 105, 110, 108, 106, 104, 102, 100, 98, 100, 101, 99],
        'High':  [103, 108, 115, 111, 109, 107, 105, 103, 101, 103, 104, 102],
        'Low':   [ 98, 103, 108, 106, 104, 102, 100,  98,  96,  98,  99,  97],
        'Close': [102, 107, 112, 109, 107, 105, 103, 101,  99, 101, 102, 100],
    })
    draw = find_draw_on_liquidity(bars_1h, current_idx=11, lookback=10, swing_n=2)
    # The old high at index 2 (115) is above current price (100), so draw is bullish
    assert draw is not None
    assert draw['direction'] == 'bullish'
    assert draw['level'] == 115

def test_draw_on_liquidity_bearish():
    """When an untouched old low exists below current price, draw is bearish."""
    bars_1h = pd.DataFrame({
        'Open':  [100,  95,  90,  92,  94,  96,  98, 100, 102, 100,  99, 101],
        'High':  [103,  98,  93,  95,  97,  99, 101, 103, 105, 103, 102, 104],
        'Low':   [ 98,  93,  85,  89,  91,  93,  95,  97,  99,  97,  96,  98],
        'Close': [ 97,  92,  88,  91,  93,  95,  97,  99, 101,  99,  98, 100],
    })
    draw = find_draw_on_liquidity(bars_1h, current_idx=11, lookback=10, swing_n=2)
    assert draw is not None
    assert draw['direction'] == 'bearish'
    assert draw['level'] == 85
```

**Step 2: Run tests to verify failure**

Run: `python -m pytest test_backtest.py::test_draw_on_liquidity_bullish test_backtest.py::test_draw_on_liquidity_bearish -v`
Expected: FAIL

**Step 3: Implement draw on liquidity**

```python
# backtest.py (append)

def find_draw_on_liquidity(
    bars: pd.DataFrame,
    current_idx: int,
    lookback: int = 20,
    swing_n: int = 3,
) -> dict | None:
    """
    Find the nearest untouched swing high or low as the draw on liquidity.
    Returns {'direction': 'bullish'|'bearish', 'level': float} or None.
    """
    start = max(0, current_idx - lookback)
    window = bars.iloc[start:current_idx + 1].copy()
    window = window.reset_index(drop=True)
    current_price = window.iloc[-1]['Close']

    swing_highs = find_swing_highs(window, n=min(swing_n, len(window) // 3))
    swing_lows = find_swing_lows(window, n=min(swing_n, len(window) // 3))

    # Find old highs above current price (untouched = price hasn't traded through them since)
    old_highs = swing_highs[swing_highs['High'] > current_price]
    # Find old lows below current price
    old_lows = swing_lows[swing_lows['Low'] < current_price]

    best_high = old_highs['High'].min() if len(old_highs) > 0 else None  # nearest above
    best_low = old_lows['Low'].max() if len(old_lows) > 0 else None  # nearest below

    if best_high is not None and best_low is not None:
        dist_high = best_high - current_price
        dist_low = current_price - best_low
        if dist_high <= dist_low:
            return {'direction': 'bullish', 'level': best_high}
        else:
            return {'direction': 'bearish', 'level': best_low}
    elif best_high is not None:
        return {'direction': 'bullish', 'level': best_high}
    elif best_low is not None:
        return {'direction': 'bearish', 'level': best_low}
    return None
```

**Step 4: Run tests**

Run: `python -m pytest test_backtest.py::test_draw_on_liquidity_bullish test_backtest.py::test_draw_on_liquidity_bearish -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backtest.py test_backtest.py
git commit -m "feat: draw on liquidity identification from swing highs/lows"
```

---

### Task 4: Stop Raid Detection (5-Minute)

**Files:**
- Modify: `backtest.py`
- Modify: `test_backtest.py`

**Step 1: Write test**

```python
# test_backtest.py (append)
from backtest import detect_stop_raid

def test_detect_stop_raid_bullish():
    """Bullish: price sweeps below a swing low, then candle closes up."""
    bars_5m = pd.DataFrame({
        'Open':  [100, 102, 101, 99,  97, 98,  96,  93,  95, 97],
        'High':  [103, 104, 103, 101, 99, 100, 98,  96,  98, 100],
        'Low':   [ 99, 100,  99, 97,  95, 96,  94,  91,  93, 96],
        'Close': [102, 103, 100, 98,  96, 97,  95,  95,  97, 99],
    })
    # Swing low at index 7 (Low=91). Index 7 sweeps below prior swing lows.
    # But for detection: we need a swing low that gets swept.
    # Let's define it: swing low at index 4 (Low=95, lower than neighbors 2,3 and neighbors checking needed)
    # Then bar 7 sweeps below 95 (Low=91), and bar 8 closes up (95 > 93 open) = bullish raid
    raid = detect_stop_raid(bars_5m, direction='bullish', swing_n=2)
    # Should find a raid: sweep of a swing low followed by up-close candle
    assert len(raid) > 0
    # The raid bar should be one that swept below a prior swing low and closed up
    last_raid = raid[-1]
    assert last_raid['confirmation_close'] > last_raid['confirmation_open']  # up-close

def test_detect_stop_raid_bearish():
    """Bearish: price sweeps above a swing high, then candle closes down."""
    bars_5m = pd.DataFrame({
        'Open':  [100, 98, 99, 101, 103, 102, 104, 107, 105, 103],
        'High':  [102, 101, 102, 104, 106, 105, 107, 110, 108, 105],
        'Low':   [ 98, 96, 97, 99, 101, 100, 102, 105, 103, 101],
        'Close': [ 99, 99, 101, 103, 105, 104, 106, 106, 104, 102],
    })
    raid = detect_stop_raid(bars_5m, direction='bearish', swing_n=2)
    assert len(raid) > 0
    last_raid = raid[-1]
    assert last_raid['confirmation_close'] < last_raid['confirmation_open']  # down-close
```

**Step 2: Run tests to verify failure**

Run: `python -m pytest test_backtest.py::test_detect_stop_raid_bullish test_backtest.py::test_detect_stop_raid_bearish -v`
Expected: FAIL

**Step 3: Implement stop raid detection**

```python
# backtest.py (append)

def detect_stop_raid(
    bars_5m: pd.DataFrame,
    direction: str,
    swing_n: int = 5,
    wait_for_close: bool = True,
) -> list[dict]:
    """
    Detect stop raids on 5-minute chart.

    For bullish: find bars that sweep below a prior swing low, then close up.
    For bearish: find bars that sweep above a prior swing high, then close down.

    Args:
        bars_5m: 5-minute OHLCV bars (integer-indexed or datetime-indexed)
        direction: 'bullish' or 'bearish'
        swing_n: pivot lookback for swing detection
        wait_for_close: if True (base model), wait for confirmation candle close.
                       if False (higher R:R), trigger on the sweep bar itself.

    Returns:
        List of raid dicts with keys:
            sweep_idx, sweep_level, confirmation_idx,
            confirmation_open, confirmation_close, direction
    """
    raids = []
    idx = bars_5m.index

    if direction == 'bullish':
        swing_lows = find_swing_lows(bars_5m, n=swing_n)
        active_lows = []  # list of (index, level) for swing lows found so far
        for i in range(len(bars_5m)):
            pos = idx[i]
            # Register new swing lows (they become active once confirmed, i.e. swing_n bars later)
            if pos in swing_lows.index:
                active_lows.append((pos, swing_lows.loc[pos, 'Low']))

            # Check if current bar sweeps below any active swing low
            if len(active_lows) == 0:
                continue

            current_low = bars_5m.loc[pos, 'Low']
            swept_level = None
            for sl_idx, sl_level in active_lows:
                if sl_idx == pos:
                    continue  # don't sweep yourself
                if current_low < sl_level:
                    swept_level = sl_level
                    break  # take the first (most recent) swept level

            if swept_level is not None:
                if wait_for_close:
                    # Confirmation: this candle or next candle closes up
                    conf_idx = pos
                    conf_open = bars_5m.loc[pos, 'Open']
                    conf_close = bars_5m.loc[pos, 'Close']
                    if conf_close > conf_open:  # up-close on sweep bar itself
                        raids.append({
                            'sweep_idx': pos,
                            'sweep_level': swept_level,
                            'confirmation_idx': conf_idx,
                            'confirmation_open': conf_open,
                            'confirmation_close': conf_close,
                            'direction': 'bullish',
                        })
                        # Remove swept level so we don't re-trigger
                        active_lows = [(si, sl) for si, sl in active_lows if sl != swept_level]
                    elif i + 1 < len(bars_5m):
                        next_pos = idx[i + 1]
                        next_open = bars_5m.loc[next_pos, 'Open']
                        next_close = bars_5m.loc[next_pos, 'Close']
                        if next_close > next_open:
                            raids.append({
                                'sweep_idx': pos,
                                'sweep_level': swept_level,
                                'confirmation_idx': next_pos,
                                'confirmation_open': next_open,
                                'confirmation_close': next_close,
                                'direction': 'bullish',
                            })
                            active_lows = [(si, sl) for si, sl in active_lows if sl != swept_level]
                else:
                    # Higher R:R: trigger immediately on sweep
                    raids.append({
                        'sweep_idx': pos,
                        'sweep_level': swept_level,
                        'confirmation_idx': pos,
                        'confirmation_open': bars_5m.loc[pos, 'Open'],
                        'confirmation_close': bars_5m.loc[pos, 'Close'],
                        'direction': 'bullish',
                    })
                    active_lows = [(si, sl) for si, sl in active_lows if sl != swept_level]

    elif direction == 'bearish':
        swing_highs = find_swing_highs(bars_5m, n=swing_n)
        active_highs = []
        for i in range(len(bars_5m)):
            pos = idx[i]
            if pos in swing_highs.index:
                active_highs.append((pos, swing_highs.loc[pos, 'High']))

            if len(active_highs) == 0:
                continue

            current_high = bars_5m.loc[pos, 'High']
            swept_level = None
            for sh_idx, sh_level in active_highs:
                if sh_idx == pos:
                    continue
                if current_high > sh_level:
                    swept_level = sh_level
                    break

            if swept_level is not None:
                if wait_for_close:
                    conf_idx = pos
                    conf_open = bars_5m.loc[pos, 'Open']
                    conf_close = bars_5m.loc[pos, 'Close']
                    if conf_close < conf_open:
                        raids.append({
                            'sweep_idx': pos,
                            'sweep_level': swept_level,
                            'confirmation_idx': conf_idx,
                            'confirmation_open': conf_open,
                            'confirmation_close': conf_close,
                            'direction': 'bearish',
                        })
                        active_highs = [(si, sl) for si, sl in active_highs if sl != swept_level]
                    elif i + 1 < len(bars_5m):
                        next_pos = idx[i + 1]
                        next_open = bars_5m.loc[next_pos, 'Open']
                        next_close = bars_5m.loc[next_pos, 'Close']
                        if next_close < next_open:
                            raids.append({
                                'sweep_idx': pos,
                                'sweep_level': swept_level,
                                'confirmation_idx': next_pos,
                                'confirmation_open': next_open,
                                'confirmation_close': next_close,
                                'direction': 'bearish',
                            })
                            active_highs = [(si, sl) for si, sl in active_highs if sl != swept_level]
                else:
                    raids.append({
                        'sweep_idx': pos,
                        'sweep_level': swept_level,
                        'confirmation_idx': pos,
                        'confirmation_open': bars_5m.loc[pos, 'Open'],
                        'confirmation_close': bars_5m.loc[pos, 'Close'],
                        'direction': 'bearish',
                    })
                    active_highs = [(si, sl) for si, sl in active_highs if sl != swept_level]

    return raids
```

**Step 4: Run tests**

Run: `python -m pytest test_backtest.py::test_detect_stop_raid_bullish test_backtest.py::test_detect_stop_raid_bearish -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backtest.py test_backtest.py
git commit -m "feat: stop raid detection on 5-minute chart"
```

---

### Task 5: Fair Value Gap Detection (30-Second)

**Files:**
- Modify: `backtest.py`
- Modify: `test_backtest.py`

**Step 1: Write test**

```python
# test_backtest.py (append)
from backtest import find_fvgs

def test_find_fvgs_bullish():
    """Bullish FVG: candle 1 high < candle 3 low (gap up)."""
    bars = pd.DataFrame({
        'Open':  [100, 102, 106],
        'High':  [101, 105, 108],
        'Low':   [ 99, 101, 103],
        'Close': [100, 104, 107],
    })
    fvgs = find_fvgs(bars, direction='bullish')
    assert len(fvgs) == 1
    assert fvgs[0]['top'] == 103    # candle 3 low
    assert fvgs[0]['bottom'] == 101  # candle 1 high
    assert fvgs[0]['idx'] == 1       # middle candle index

def test_find_fvgs_bearish():
    """Bearish FVG: candle 1 low > candle 3 high (gap down)."""
    bars = pd.DataFrame({
        'Open':  [108, 105, 100],
        'High':  [109, 106, 102],
        'Low':   [106, 103,  98],
        'Close': [107, 104,  99],
    })
    fvgs = find_fvgs(bars, direction='bearish')
    assert len(fvgs) == 1
    assert fvgs[0]['top'] == 106    # candle 1 low
    assert fvgs[0]['bottom'] == 102  # candle 3 high
    assert fvgs[0]['idx'] == 1

def test_find_fvgs_no_gap():
    """No FVG when wicks overlap."""
    bars = pd.DataFrame({
        'Open':  [100, 101, 102],
        'High':  [103, 104, 105],
        'Low':   [ 99, 100, 101],
        'Close': [101, 102, 103],
    })
    fvgs = find_fvgs(bars, direction='bullish')
    assert len(fvgs) == 0
```

**Step 2: Run tests to verify failure**

Run: `python -m pytest test_backtest.py -k "test_find_fvgs" -v`
Expected: FAIL

**Step 3: Implement FVG detection**

```python
# backtest.py (append)

def find_fvgs(bars: pd.DataFrame, direction: str) -> list[dict]:
    """
    Find Fair Value Gaps in a series of bars.

    Bullish FVG: candle[i-1].High < candle[i+1].Low (gap between wicks)
    Bearish FVG: candle[i-1].Low > candle[i+1].High

    Returns list of dicts: {idx, top, bottom, timestamp}
        idx = index of the middle candle
        top/bottom = the FVG zone boundaries
    """
    fvgs = []
    idx = bars.index

    for i in range(1, len(bars) - 1):
        c1 = bars.iloc[i - 1]  # candle before
        c3 = bars.iloc[i + 1]  # candle after

        if direction == 'bullish':
            if c1['High'] < c3['Low']:
                fvgs.append({
                    'idx': idx[i],
                    'top': c3['Low'],
                    'bottom': c1['High'],
                    'mid': (c3['Low'] + c1['High']) / 2,
                })
        elif direction == 'bearish':
            if c1['Low'] > c3['High']:
                fvgs.append({
                    'idx': idx[i],
                    'top': c1['Low'],
                    'bottom': c3['High'],
                    'mid': (c1['Low'] + c3['High']) / 2,
                })

    return fvgs
```

**Step 4: Run tests**

Run: `python -m pytest test_backtest.py -k "test_find_fvgs" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backtest.py test_backtest.py
git commit -m "feat: fair value gap detection for bullish and bearish setups"
```

---

### Task 6: Displacement & Order Block Detection (Higher R:R Variant)

**Files:**
- Modify: `backtest.py`
- Modify: `test_backtest.py`

**Step 1: Write test**

```python
# test_backtest.py (append)
from backtest import detect_displacement, find_order_blocks

def test_detect_displacement_bullish():
    """Bullish displacement: large up candle relative to recent ATR."""
    bars = pd.DataFrame({
        'Open':  [100, 101, 100, 101, 100, 100, 100, 110],
        'High':  [102, 103, 102, 103, 102, 102, 102, 115],
        'Low':   [ 99, 100,  99, 100,  99,  99,  99, 109],
        'Close': [101, 102, 101, 102, 101, 101, 101, 114],
    })
    displacements = detect_displacement(bars, direction='bullish', atr_mult=2.0, atr_period=5)
    assert len(displacements) > 0
    assert displacements[-1]['idx'] == 7  # the big candle

def test_find_order_blocks_bullish():
    """Bullish OB: last bearish candle before a bullish displacement."""
    bars = pd.DataFrame({
        'Open':  [102, 101, 100,  99, 100, 110],
        'High':  [103, 102, 101, 101, 102, 115],
        'Low':   [100,  99,  98,  98,  99, 109],
        'Close': [101, 100,  99, 100, 101, 114],
    })
    obs = find_order_blocks(bars, direction='bullish', atr_mult=2.0, atr_period=4)
    assert len(obs) > 0
    # OB should be the last bearish candle before the displacement at index 5
    assert obs[-1]['ob_high'] <= bars.iloc[obs[-1]['ob_idx']]['High']
```

**Step 2: Run tests to verify failure**

Run: `python -m pytest test_backtest.py -k "test_detect_displacement or test_find_order_blocks" -v`
Expected: FAIL

**Step 3: Implement displacement and order block detection**

```python
# backtest.py (append)

def detect_displacement(
    bars: pd.DataFrame,
    direction: str,
    atr_mult: float = 2.0,
    atr_period: int = 14,
) -> list[dict]:
    """
    Detect displacement candles (impulsive moves) relative to recent ATR.

    A displacement candle has body size > atr_mult * ATR.
    """
    body = (bars['Close'] - bars['Open']).abs()
    tr = pd.concat([
        bars['High'] - bars['Low'],
        (bars['High'] - bars['Close'].shift(1)).abs(),
        (bars['Low'] - bars['Close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=1).mean()

    displacements = []
    idx = bars.index
    for i in range(atr_period, len(bars)):
        pos = idx[i]
        if direction == 'bullish' and bars.iloc[i]['Close'] > bars.iloc[i]['Open']:
            if body.iloc[i] > atr_mult * atr.iloc[i - 1]:
                displacements.append({
                    'idx': pos,
                    'open': bars.iloc[i]['Open'],
                    'close': bars.iloc[i]['Close'],
                    'high': bars.iloc[i]['High'],
                    'low': bars.iloc[i]['Low'],
                    'body': body.iloc[i],
                })
        elif direction == 'bearish' and bars.iloc[i]['Close'] < bars.iloc[i]['Open']:
            if body.iloc[i] > atr_mult * atr.iloc[i - 1]:
                displacements.append({
                    'idx': pos,
                    'open': bars.iloc[i]['Open'],
                    'close': bars.iloc[i]['Close'],
                    'high': bars.iloc[i]['High'],
                    'low': bars.iloc[i]['Low'],
                    'body': body.iloc[i],
                })
    return displacements


def find_order_blocks(
    bars: pd.DataFrame,
    direction: str,
    atr_mult: float = 2.0,
    atr_period: int = 14,
) -> list[dict]:
    """
    Find order blocks: the last opposing candle before a displacement.

    Bullish OB: last bearish candle before a bullish displacement.
    Bearish OB: last bullish candle before a bearish displacement.
    """
    displacements = detect_displacement(bars, direction, atr_mult, atr_period)
    order_blocks = []
    idx_list = list(bars.index)

    for disp in displacements:
        disp_pos = idx_list.index(disp['idx'])
        # Walk backwards to find last opposing candle
        for j in range(disp_pos - 1, max(disp_pos - 10, -1), -1):
            candle = bars.iloc[j]
            if direction == 'bullish' and candle['Close'] < candle['Open']:
                order_blocks.append({
                    'ob_idx': idx_list[j],
                    'ob_high': candle['High'],
                    'ob_low': candle['Low'],
                    'ob_open': candle['Open'],
                    'ob_close': candle['Close'],
                    'displacement_idx': disp['idx'],
                })
                break
            elif direction == 'bearish' and candle['Close'] > candle['Open']:
                order_blocks.append({
                    'ob_idx': idx_list[j],
                    'ob_high': candle['High'],
                    'ob_low': candle['Low'],
                    'ob_open': candle['Open'],
                    'ob_close': candle['Close'],
                    'displacement_idx': disp['idx'],
                })
                break

    return order_blocks
```

**Step 4: Run tests**

Run: `python -m pytest test_backtest.py -k "test_detect_displacement or test_find_order_blocks" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backtest.py test_backtest.py
git commit -m "feat: displacement and order block detection for higher R:R variant"
```

---

### Task 7: Base Model Strategy Engine

**Files:**
- Modify: `backtest.py`

**Step 1: Implement the base model runner**

This is the core integration that wires together all detection functions into a single-pass backtest loop. The logic iterates through time, maintaining state for draw on liquidity, active raid signals, and open positions.

```python
# backtest.py (append)

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp] = None
    direction: str = ''           # 'long' or 'short'
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    pnl_points: float = 0.0
    r_multiple: float = 0.0
    variant: str = ''             # 'base' or 'higher_rr'
    exit_reason: str = ''         # 'tp', 'sl', 'timeout'


def run_base_model(
    bars_30s: pd.DataFrame,
    bars_5m: pd.DataFrame,
    bars_15m: pd.DataFrame,
    bars_1h: pd.DataFrame,
    max_stop: float = 12.0,
    cooldown_seconds: int = 300,
) -> list[Trade]:
    """
    Run the base model backtest.

    Flow:
    1. Every new 1h bar: re-evaluate draw on liquidity
    2. Every new 5m bar: check for stop raids in direction opposite to draw
    3. After a valid raid: scan 30s bars for FVG entry
    4. Manage open position (SL/TP checked on each 30s bar)
    """
    trades: list[Trade] = []
    position: Optional[Trade] = None
    last_exit_time: Optional[pd.Timestamp] = None

    # Build time-aligned index mappings
    # For each 30s bar, know which 5m/15m/1h bar it belongs to
    bars_5m_times = bars_5m.index.sort_values()
    bars_15m_times = bars_15m.index.sort_values()
    bars_1h_times = bars_1h.index.sort_values()

    current_draw = None
    active_raid = None
    last_1h_check = None
    last_5m_check = None

    # Precompute 5m swing structures for raid detection
    # We'll use rolling windows as we go

    for ts in bars_30s.index:
        bar = bars_30s.loc[ts]

        # --- Check exits first ---
        if position is not None:
            hit_sl = False
            hit_tp = False

            if position.direction == 'long':
                if bar['Low'] <= position.stop_price:
                    hit_sl = True
                if bar['High'] >= position.target_price:
                    hit_tp = True
            else:  # short
                if bar['High'] >= position.stop_price:
                    hit_sl = True
                if bar['Low'] <= position.target_price:
                    hit_tp = True

            if hit_sl and hit_tp:
                # Conservative: assume stop hit first
                hit_tp = False

            if hit_sl:
                position.exit_time = ts
                position.exit_price = position.stop_price
                position.exit_reason = 'sl'
            elif hit_tp:
                position.exit_time = ts
                position.exit_price = position.target_price
                position.exit_reason = 'tp'

            if position.exit_time is not None:
                if position.direction == 'long':
                    position.pnl_points = position.exit_price - position.entry_price
                else:
                    position.pnl_points = position.entry_price - position.exit_price
                risk = abs(position.entry_price - position.stop_price)
                position.r_multiple = position.pnl_points / risk if risk > 0 else 0
                trades.append(position)
                last_exit_time = position.exit_time
                position = None
                active_raid = None
            continue  # don't look for new entries while in a position (already handled exit)

        # --- Cooldown check ---
        if last_exit_time is not None:
            if (ts - last_exit_time).total_seconds() < cooldown_seconds:
                continue

        # --- Step 1: Update draw on liquidity (every new 1h bar) ---
        current_1h = bars_1h_times[bars_1h_times <= ts]
        if len(current_1h) > 0:
            latest_1h = current_1h[-1]
            if last_1h_check is None or latest_1h > last_1h_check:
                last_1h_check = latest_1h
                h_idx = bars_1h_times.get_loc(latest_1h)
                if h_idx >= 10:
                    draw = find_draw_on_liquidity(bars_1h, current_idx=h_idx, lookback=20, swing_n=3)
                    if draw is None and len(bars_15m_times[bars_15m_times <= ts]) > 10:
                        # Fallback to 15m
                        m15_ts = bars_15m_times[bars_15m_times <= ts]
                        m15_idx = bars_15m_times.get_loc(m15_ts[-1])
                        draw = find_draw_on_liquidity(bars_15m, current_idx=m15_idx, lookback=40, swing_n=3)
                    current_draw = draw

        if current_draw is None:
            continue

        # --- Step 2: Detect stop raids (every new 5m bar) ---
        current_5m = bars_5m_times[bars_5m_times <= ts]
        if len(current_5m) > 0:
            latest_5m = current_5m[-1]
            if last_5m_check is None or latest_5m > last_5m_check:
                last_5m_check = latest_5m
                m5_idx = bars_5m_times.get_loc(latest_5m)
                if m5_idx >= 20:
                    window_start = max(0, m5_idx - 50)
                    window = bars_5m.iloc[window_start:m5_idx + 1]
                    raids = detect_stop_raid(
                        window,
                        direction=current_draw['direction'],
                        swing_n=5,
                        wait_for_close=True,
                    )
                    if raids:
                        active_raid = raids[-1]

        if active_raid is None:
            continue

        # Only look for entries after the raid confirmation
        if ts <= active_raid['confirmation_idx']:
            continue

        # --- Step 3: FVG entry on 30s ---
        # Look at the last 3 bars ending at current
        ts_loc = bars_30s.index.get_loc(ts)
        if ts_loc < 2:
            continue

        recent_3 = bars_30s.iloc[ts_loc - 2:ts_loc + 1]
        fvgs = find_fvgs(recent_3, direction=current_draw['direction'])

        if not fvgs:
            continue

        fvg = fvgs[-1]  # most recent FVG

        # Check if current bar returns to the FVG
        entered = False
        entry_price = None

        if current_draw['direction'] == 'bullish':
            # Price dips into the FVG zone (bar low touches FVG top)
            if bar['Low'] <= fvg['top']:
                entered = True
                entry_price = fvg['top']  # enter at top of FVG
        else:
            # Price rallies into the FVG zone (bar high touches FVG bottom)
            if bar['High'] >= fvg['bottom']:
                entered = True
                entry_price = fvg['bottom']

        if entered and entry_price is not None:
            # Calculate stop and target
            if current_draw['direction'] == 'bullish':
                stop = entry_price - max_stop
                # Tighter stop if structure is closer
                struct_stop = fvg['bottom'] - 1  # 1 pt below FVG bottom
                if entry_price - struct_stop < max_stop:
                    stop = struct_stop
                risk = entry_price - stop
                target = entry_price + risk  # 1:1

                position = Trade(
                    entry_time=ts,
                    direction='long',
                    entry_price=entry_price,
                    stop_price=stop,
                    target_price=target,
                    variant='base',
                )
            else:
                stop = entry_price + max_stop
                struct_stop = fvg['top'] + 1
                if struct_stop - entry_price < max_stop:
                    stop = struct_stop
                risk = stop - entry_price
                target = entry_price - risk  # 1:1

                position = Trade(
                    entry_time=ts,
                    direction='short',
                    entry_price=entry_price,
                    stop_price=stop,
                    target_price=target,
                    variant='base',
                )

            active_raid = None  # consumed

    # Close any open position at end
    if position is not None:
        last_bar = bars_30s.iloc[-1]
        position.exit_time = bars_30s.index[-1]
        position.exit_price = last_bar['Close']
        position.exit_reason = 'timeout'
        if position.direction == 'long':
            position.pnl_points = position.exit_price - position.entry_price
        else:
            position.pnl_points = position.entry_price - position.exit_price
        risk = abs(position.entry_price - position.stop_price)
        position.r_multiple = position.pnl_points / risk if risk > 0 else 0
        trades.append(position)

    return trades
```

**Step 2: Run a quick smoke test on real data (first day)**

Run: `cd "/mnt/e/backup/code/Finance/Models/Assorted youtube models/Camerons-Scalping-Model" && python -c "
from backtest import *
import pandas as pd
ticks = load_ticks()
# Take first day only for smoke test
day1 = ticks[ticks['ts_event'].dt.date == ticks['ts_event'].iloc[0].date()]
print(f'Ticks for day 1: {len(day1)}')
b = build_bars(day1)
for k, v in b.items():
    print(f'{k}: {len(v)} bars')
trades = run_base_model(b['30s'], b['5m'], b['15m'], b['1h'])
print(f'Base model trades on day 1: {len(trades)}')
for t in trades:
    print(f'  {t.direction} @ {t.entry_price:.2f}, exit {t.exit_price:.2f}, PnL {t.pnl_points:.2f} pts, {t.exit_reason}')
"`
Expected: Runs without error, prints trade count (may be 0-few for a single day)

**Step 3: Commit**

```bash
git add backtest.py
git commit -m "feat: base model strategy engine with full entry/exit loop"
```

---

### Task 8: Higher R:R Strategy Engine

**Files:**
- Modify: `backtest.py`

**Step 1: Implement the higher R:R runner**

The higher R:R variant differs from the base model in three ways:
1. Don't wait for 5m candle close — trigger on sweep
2. Look for displacement + order blocks on 30s (in addition to FVGs)
3. Target structural levels instead of fixed 1:1

```python
# backtest.py (append)

def find_structural_target(
    bars_5m: pd.DataFrame,
    direction: str,
    entry_price: float,
    current_5m_idx: int,
    swing_n: int = 5,
) -> float | None:
    """Find the nearest structural target (previous swing high/low on 5m)."""
    window = bars_5m.iloc[max(0, current_5m_idx - 50):current_5m_idx]
    if direction == 'bullish':
        highs = find_swing_highs(window, n=min(swing_n, max(1, len(window) // 3)))
        targets = highs[highs['High'] > entry_price]
        if len(targets) > 0:
            return targets['High'].min()  # nearest swing high above
    else:
        lows = find_swing_lows(window, n=min(swing_n, max(1, len(window) // 3)))
        targets = lows[lows['Low'] < entry_price]
        if len(targets) > 0:
            return targets['Low'].max()  # nearest swing low below
    return None


def run_higher_rr_model(
    bars_30s: pd.DataFrame,
    bars_5m: pd.DataFrame,
    bars_15m: pd.DataFrame,
    bars_1h: pd.DataFrame,
    cooldown_seconds: int = 300,
    atr_mult: float = 2.0,
    min_rr: float = 1.5,
) -> list[Trade]:
    """
    Run the higher R:R variant.

    Differences from base:
    - Don't wait for 5m close (wait_for_close=False)
    - Look for displacement + order blocks on 30s
    - Target structural levels, minimum 1.5 R:R
    """
    trades: list[Trade] = []
    position: Optional[Trade] = None
    last_exit_time: Optional[pd.Timestamp] = None

    bars_5m_times = bars_5m.index.sort_values()
    bars_15m_times = bars_15m.index.sort_values()
    bars_1h_times = bars_1h.index.sort_values()

    current_draw = None
    active_raid = None
    last_1h_check = None
    last_5m_check = None

    for ts in bars_30s.index:
        bar = bars_30s.loc[ts]

        # --- Exits ---
        if position is not None:
            hit_sl = False
            hit_tp = False

            if position.direction == 'long':
                if bar['Low'] <= position.stop_price:
                    hit_sl = True
                if bar['High'] >= position.target_price:
                    hit_tp = True
            else:
                if bar['High'] >= position.stop_price:
                    hit_sl = True
                if bar['Low'] <= position.target_price:
                    hit_tp = True

            if hit_sl and hit_tp:
                hit_tp = False

            if hit_sl:
                position.exit_time = ts
                position.exit_price = position.stop_price
                position.exit_reason = 'sl'
            elif hit_tp:
                position.exit_time = ts
                position.exit_price = position.target_price
                position.exit_reason = 'tp'

            if position.exit_time is not None:
                if position.direction == 'long':
                    position.pnl_points = position.exit_price - position.entry_price
                else:
                    position.pnl_points = position.entry_price - position.exit_price
                risk = abs(position.entry_price - position.stop_price)
                position.r_multiple = position.pnl_points / risk if risk > 0 else 0
                trades.append(position)
                last_exit_time = position.exit_time
                position = None
                active_raid = None
            continue

        # --- Cooldown ---
        if last_exit_time is not None:
            if (ts - last_exit_time).total_seconds() < cooldown_seconds:
                continue

        # --- Draw on liquidity ---
        current_1h = bars_1h_times[bars_1h_times <= ts]
        if len(current_1h) > 0:
            latest_1h = current_1h[-1]
            if last_1h_check is None or latest_1h > last_1h_check:
                last_1h_check = latest_1h
                h_idx = bars_1h_times.get_loc(latest_1h)
                if h_idx >= 10:
                    draw = find_draw_on_liquidity(bars_1h, current_idx=h_idx, lookback=20, swing_n=3)
                    if draw is None and len(bars_15m_times[bars_15m_times <= ts]) > 10:
                        m15_ts = bars_15m_times[bars_15m_times <= ts]
                        m15_idx = bars_15m_times.get_loc(m15_ts[-1])
                        draw = find_draw_on_liquidity(bars_15m, current_idx=m15_idx, lookback=40, swing_n=3)
                    current_draw = draw

        if current_draw is None:
            continue

        # --- Stop raid (don't wait for close) ---
        current_5m = bars_5m_times[bars_5m_times <= ts]
        if len(current_5m) > 0:
            latest_5m = current_5m[-1]
            if last_5m_check is None or latest_5m > last_5m_check:
                last_5m_check = latest_5m
                m5_idx = bars_5m_times.get_loc(latest_5m)
                if m5_idx >= 20:
                    window_start = max(0, m5_idx - 50)
                    window = bars_5m.iloc[window_start:m5_idx + 1]
                    raids = detect_stop_raid(
                        window,
                        direction=current_draw['direction'],
                        swing_n=5,
                        wait_for_close=False,  # KEY DIFFERENCE
                    )
                    if raids:
                        active_raid = raids[-1]

        if active_raid is None:
            continue

        if ts <= active_raid['confirmation_idx']:
            continue

        # --- Entry: displacement + OB/FVG on 30s ---
        ts_loc = bars_30s.index.get_loc(ts)
        if ts_loc < 14:
            continue

        # Look at recent 30s bars for displacement
        lookback_30s = bars_30s.iloc[max(0, ts_loc - 30):ts_loc + 1]

        # Check for FVGs (same as base)
        fvg_entry = None
        if ts_loc >= 2:
            recent_3 = bars_30s.iloc[ts_loc - 2:ts_loc + 1]
            fvgs = find_fvgs(recent_3, direction=current_draw['direction'])
            if fvgs:
                fvg = fvgs[-1]
                if current_draw['direction'] == 'bullish' and bar['Low'] <= fvg['top']:
                    fvg_entry = fvg['top']
                elif current_draw['direction'] == 'bearish' and bar['High'] >= fvg['bottom']:
                    fvg_entry = fvg['bottom']

        # Check for order blocks after displacement
        ob_entry = None
        obs = find_order_blocks(lookback_30s, direction=current_draw['direction'], atr_mult=atr_mult, atr_period=10)
        if obs:
            ob = obs[-1]
            if current_draw['direction'] == 'bullish':
                if bar['Low'] <= ob['ob_high']:
                    ob_entry = ob['ob_high']
            else:
                if bar['High'] >= ob['ob_low']:
                    ob_entry = ob['ob_low']

        # Prefer OB entry over FVG (higher probability per ICT)
        entry_price = ob_entry or fvg_entry
        if entry_price is None:
            continue

        # Stop at displacement swing point
        disps = detect_displacement(lookback_30s, direction=current_draw['direction'], atr_mult=atr_mult, atr_period=10)
        if current_draw['direction'] == 'bullish':
            if disps:
                stop = disps[-1]['low'] - 1
            else:
                stop = entry_price - 12
            risk = entry_price - stop
        else:
            if disps:
                stop = disps[-1]['high'] + 1
            else:
                stop = entry_price + 12
            risk = stop - entry_price

        if risk <= 0:
            continue

        # Structural target
        current_5m_slice = bars_5m_times[bars_5m_times <= ts]
        if len(current_5m_slice) > 0:
            m5_idx = bars_5m_times.get_loc(current_5m_slice[-1])
            target = find_structural_target(bars_5m, current_draw['direction'], entry_price, m5_idx)
        else:
            target = None

        # Fallback to draw level if no structural target
        if target is None:
            target = current_draw['level']

        # Check minimum R:R
        if current_draw['direction'] == 'bullish':
            potential_rr = (target - entry_price) / risk
        else:
            potential_rr = (entry_price - target) / risk

        if potential_rr < min_rr:
            continue

        if current_draw['direction'] == 'bullish':
            position = Trade(
                entry_time=ts,
                direction='long',
                entry_price=entry_price,
                stop_price=stop,
                target_price=target,
                variant='higher_rr',
            )
        else:
            position = Trade(
                entry_time=ts,
                direction='short',
                entry_price=entry_price,
                stop_price=stop,
                target_price=target,
                variant='higher_rr',
            )
        active_raid = None

    # Close open position at end
    if position is not None:
        last_bar = bars_30s.iloc[-1]
        position.exit_time = bars_30s.index[-1]
        position.exit_price = last_bar['Close']
        position.exit_reason = 'timeout'
        if position.direction == 'long':
            position.pnl_points = position.exit_price - position.entry_price
        else:
            position.pnl_points = position.entry_price - position.exit_price
        risk = abs(position.entry_price - position.stop_price)
        position.r_multiple = position.pnl_points / risk if risk > 0 else 0
        trades.append(position)

    return trades
```

**Step 2: Smoke test**

Run: same as Task 7 smoke test but calling `run_higher_rr_model`
Expected: Runs without error

**Step 3: Commit**

```bash
git add backtest.py
git commit -m "feat: higher R:R variant with displacement, order blocks, structural targets"
```

---

### Task 9: Reporting — HTML Report with Plotly

**Files:**
- Modify: `backtest.py`

**Step 1: Implement reporting functions**

```python
# backtest.py (append)
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def compute_stats(trades: list[Trade]) -> dict:
    """Compute summary statistics from a list of trades."""
    if not trades:
        return {'total_trades': 0}

    pnls = [t.pnl_points for t in trades]
    rs = [t.r_multiple for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_max

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0

    return {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100,
        'avg_pnl': np.mean(pnls),
        'avg_r': np.mean(rs),
        'total_pnl': sum(pnls),
        'max_drawdown': min(drawdowns) if len(drawdowns) > 0 else 0,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'largest_win': max(pnls),
        'largest_loss': min(pnls),
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
    }


def trades_to_df(trades: list[Trade]) -> pd.DataFrame:
    """Convert trade list to DataFrame."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([{
        'entry_time': t.entry_time,
        'exit_time': t.exit_time,
        'direction': t.direction,
        'entry_price': t.entry_price,
        'exit_price': t.exit_price,
        'stop_price': t.stop_price,
        'target_price': t.target_price,
        'pnl_points': t.pnl_points,
        'r_multiple': t.r_multiple,
        'exit_reason': t.exit_reason,
        'variant': t.variant,
    } for t in trades])


def monthly_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Group trades by month and compute stats."""
    if trades_df.empty:
        return pd.DataFrame()
    trades_df = trades_df.copy()
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    grouped = trades_df.groupby('month').agg(
        trades=('pnl_points', 'count'),
        total_pnl=('pnl_points', 'sum'),
        avg_r=('r_multiple', 'mean'),
        win_rate=('pnl_points', lambda x: (x > 0).mean() * 100),
        largest_win=('pnl_points', 'max'),
        largest_loss=('pnl_points', 'min'),
    ).reset_index()
    grouped['month'] = grouped['month'].astype(str)
    return grouped


def generate_report(
    base_trades: list[Trade],
    higher_rr_trades: list[Trade],
    output_path: Path,
):
    """Generate HTML report with equity curves and trade tables."""
    base_stats = compute_stats(base_trades)
    hrr_stats = compute_stats(higher_rr_trades)
    base_df = trades_to_df(base_trades)
    hrr_df = trades_to_df(higher_rr_trades)

    # Save CSVs
    if not base_df.empty:
        base_df.to_csv(output_path.parent / 'trades_base.csv', index=False)
    if not hrr_df.empty:
        hrr_df.to_csv(output_path.parent / 'trades_higher_rr.csv', index=False)

    # Build HTML
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Base Model — Equity Curve', 'Higher R:R — Equity Curve'),
        vertical_spacing=0.12,
    )

    for i, (trades_list, label) in enumerate([
        (base_trades, 'Base'),
        (higher_rr_trades, 'Higher R:R'),
    ], 1):
        if trades_list:
            cum = np.cumsum([t.pnl_points for t in trades_list])
            times = [t.exit_time for t in trades_list]
            fig.add_trace(
                go.Scatter(x=times, y=cum, mode='lines', name=f'{label} Equity',
                           line=dict(width=2)),
                row=i, col=1,
            )
            fig.update_yaxes(title_text='Cumulative P&L (pts)', row=i, col=1)

    fig.update_layout(height=800, title_text="Cameron's Scalping Model — Backtest Results")

    # Build stats HTML table
    def stats_table(stats: dict, label: str) -> str:
        if stats['total_trades'] == 0:
            return f'<h3>{label}</h3><p>No trades generated.</p>'
        return f'''
        <h3>{label}</h3>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; margin-bottom:20px;">
            <tr><td><b>Total Trades</b></td><td>{stats["total_trades"]}</td></tr>
            <tr><td><b>Wins / Losses</b></td><td>{stats["wins"]} / {stats["losses"]}</td></tr>
            <tr><td><b>Win Rate</b></td><td>{stats["win_rate"]:.1f}%</td></tr>
            <tr><td><b>Avg R</b></td><td>{stats["avg_r"]:.2f}</td></tr>
            <tr><td><b>Total P&L</b></td><td>{stats["total_pnl"]:.2f} pts</td></tr>
            <tr><td><b>Profit Factor</b></td><td>{stats["profit_factor"]:.2f}</td></tr>
            <tr><td><b>Max Drawdown</b></td><td>{stats["max_drawdown"]:.2f} pts</td></tr>
            <tr><td><b>Largest Win</b></td><td>{stats["largest_win"]:.2f} pts</td></tr>
            <tr><td><b>Largest Loss</b></td><td>{stats["largest_loss"]:.2f} pts</td></tr>
            <tr><td><b>Avg Win</b></td><td>{stats["avg_win"]:.2f} pts</td></tr>
            <tr><td><b>Avg Loss</b></td><td>{stats["avg_loss"]:.2f} pts</td></tr>
        </table>'''

    # Monthly breakdown tables
    def monthly_table(df: pd.DataFrame, label: str) -> str:
        mb = monthly_breakdown(df)
        if mb.empty:
            return ''
        rows = ''
        for _, r in mb.iterrows():
            rows += f'''<tr>
                <td>{r["month"]}</td><td>{r["trades"]}</td>
                <td>{r["total_pnl"]:.1f}</td><td>{r["avg_r"]:.2f}</td>
                <td>{r["win_rate"]:.1f}%</td></tr>'''
        return f'''
        <h4>{label} — Monthly Breakdown</h4>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; margin-bottom:20px;">
            <tr><th>Month</th><th>Trades</th><th>P&L (pts)</th><th>Avg R</th><th>Win Rate</th></tr>
            {rows}
        </table>'''

    # Trade log tables
    def trade_log_html(df: pd.DataFrame, label: str) -> str:
        if df.empty:
            return ''
        rows = ''
        for _, r in df.iterrows():
            color = '#d4edda' if r['pnl_points'] > 0 else '#f8d7da'
            rows += f'''<tr style="background:{color}">
                <td>{r["entry_time"]}</td><td>{r["exit_time"]}</td>
                <td>{r["direction"]}</td><td>{r["entry_price"]:.2f}</td>
                <td>{r["exit_price"]:.2f}</td><td>{r["stop_price"]:.2f}</td>
                <td>{r["target_price"]:.2f}</td><td>{r["pnl_points"]:.2f}</td>
                <td>{r["r_multiple"]:.2f}</td><td>{r["exit_reason"]}</td></tr>'''
        return f'''
        <h4>{label} — Trade Log</h4>
        <div style="max-height:400px; overflow-y:auto;">
        <table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse; font-size:12px;">
            <tr><th>Entry</th><th>Exit</th><th>Dir</th><th>Entry$</th><th>Exit$</th>
            <th>Stop</th><th>Target</th><th>P&L</th><th>R</th><th>Reason</th></tr>
            {rows}
        </table></div>'''

    html = f'''<!DOCTYPE html>
<html><head><title>Cameron's Scalping Model Backtest</title>
<style>body{{font-family:Arial,sans-serif;margin:40px;}} table{{font-size:14px;}}</style>
</head><body>
<h1>Cameron's Scalping Model — Backtest Report</h1>
<p>Data: NQ tick data, {base_df["entry_time"].min() if not base_df.empty else "N/A"} to {base_df["entry_time"].max() if not base_df.empty else "N/A"}</p>
<hr>
{stats_table(base_stats, "Base Model (1:1 R:R)")}
{stats_table(hrr_stats, "Higher R:R Variant")}
{fig.to_html(full_html=False, include_plotlyjs='cdn')}
{monthly_table(base_df, "Base Model")}
{monthly_table(hrr_df, "Higher R:R")}
{trade_log_html(base_df, "Base Model")}
{trade_log_html(hrr_df, "Higher R:R")}
</body></html>'''

    output_path.write_text(html)
    print(f'Report saved to {output_path}')
```

**Step 2: Commit**

```bash
git add backtest.py
git commit -m "feat: HTML reporting with plotly equity curves, stats, trade logs"
```

---

### Task 10: Main Entry Point & Full Backtest Run

**Files:**
- Modify: `backtest.py`

**Step 1: Add main block**

```python
# backtest.py (append at end of file)

def main():
    print("Loading tick data...")
    ticks = load_ticks()
    print(f"  {len(ticks):,} ticks loaded")

    print("Building multi-timeframe bars...")
    bars = build_bars(ticks)
    for tf, df in bars.items():
        print(f"  {tf}: {len(df):,} bars")

    print("\nRunning Base Model...")
    base_trades = run_base_model(bars['30s'], bars['5m'], bars['15m'], bars['1h'])
    print(f"  {len(base_trades)} trades")

    print("Running Higher R:R Model...")
    hrr_trades = run_higher_rr_model(bars['30s'], bars['5m'], bars['15m'], bars['1h'])
    print(f"  {len(hrr_trades)} trades")

    print("\nGenerating report...")
    report_path = Path(__file__).parent / 'report.html'
    generate_report(base_trades, hrr_trades, report_path)

    # Print summary
    for label, trades_list in [('Base', base_trades), ('Higher R:R', hrr_trades)]:
        stats = compute_stats(trades_list)
        if stats['total_trades'] > 0:
            print(f"\n--- {label} Model ---")
            print(f"  Trades: {stats['total_trades']}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Total P&L: {stats['total_pnl']:.1f} pts")
            print(f"  Avg R: {stats['avg_r']:.2f}")
            print(f"  Profit Factor: {stats['profit_factor']:.2f}")
            print(f"  Max Drawdown: {stats['max_drawdown']:.1f} pts")
        else:
            print(f"\n--- {label} Model ---")
            print(f"  No trades generated.")


if __name__ == '__main__':
    main()
```

**Step 2: Run all tests**

Run: `python -m pytest test_backtest.py -v`
Expected: All tests PASS

**Step 3: Run the full backtest**

Run: `cd "/mnt/e/backup/code/Finance/Models/Assorted youtube models/Camerons-Scalping-Model" && python backtest.py`
Expected: Loads data, runs both models, prints summary, generates `report.html` and CSV files

**Step 4: Commit**

```bash
git add backtest.py
git commit -m "feat: main entry point, full backtest pipeline"
```

---

### Task 11: Performance Optimization

The 30s loop over ~25M ticks resampled to ~500K+ bars will be slow if done naively. After the initial run, profile and optimize:

**Step 1: Profile the initial run**

Run: `python -c "import cProfile; cProfile.run('from backtest import main; main()', sort='cumtime')" 2>&1 | head -40`

**Step 2: Optimize hot paths**

Key optimizations to apply as needed:
- Vectorize swing detection using `pandas.rolling` instead of Python loop
- Pre-compute all FVGs on 30s bars in a single pass rather than per-bar
- Use `searchsorted` for time alignment instead of boolean indexing
- Cache draw-on-liquidity results (only changes every 1h bar)

**Step 3: Commit optimizations**

```bash
git add backtest.py
git commit -m "perf: optimize hot paths in strategy loop"
```

---

### Task 12: Final Validation & Cleanup

**Step 1: Run full test suite**

Run: `python -m pytest test_backtest.py -v`
Expected: All PASS

**Step 2: Run full backtest and verify report**

Run: `python backtest.py`
Expected: Completes successfully, `report.html` opens in browser with charts and data

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete Cameron's Scalping Model backtest with HTML report"
```
