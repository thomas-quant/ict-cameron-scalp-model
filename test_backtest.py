"""Tests for Cameron's Scalping Model backtest components."""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from backtest import (
    resample_ticks, find_swing_highs, find_swing_lows,
    find_draw_on_liquidity, detect_stop_raid, find_fvgs,
    find_fvgs_vectorized, detect_displacement, find_order_blocks,
)


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
    assert bars.iloc[0]['Open'] == 100.0
    assert bars.iloc[0]['High'] == 102.0
    assert bars.iloc[0]['Low'] == 99.0
    assert bars.iloc[0]['Close'] == 102.0
    assert bars.iloc[0]['Volume'] == 7
    assert bars.iloc[1]['Open'] == 103.0
    assert bars.iloc[1]['High'] == 104.0


def test_find_swing_highs():
    """A bar whose High is higher than N bars on each side is a swing high."""
    bars = pd.DataFrame({
        'High': [10, 12, 15, 13, 11, 14, 18, 16, 12, 10],
        'Low':  [ 8,  9, 11, 10,  9, 11, 14, 13, 10,  8],
    })
    highs = find_swing_highs(bars, n=2)
    assert 2 in highs.index
    assert 6 in highs.index
    assert highs.loc[2, 'High'] == 15
    assert highs.loc[6, 'High'] == 18


def test_find_swing_lows():
    """A bar whose Low is lower than N bars on each side is a swing low."""
    bars = pd.DataFrame({
        'High': [10, 12, 15, 13, 11, 14, 18, 16, 12, 10],
        'Low':  [ 8,  9, 11, 10,  7, 11, 14, 13, 10,  6],
    })
    lows = find_swing_lows(bars, n=2)
    assert 4 in lows.index
    assert lows.loc[4, 'Low'] == 7


def test_draw_on_liquidity_bullish():
    """When an untouched old high exists above current price, draw is bullish."""
    # Price rallies to 115, pulls back slightly to 108, no swing low forms below current price
    bars_1h = pd.DataFrame({
        'Open':  [105, 108, 112, 110, 108, 106, 108, 110, 112, 110, 109, 108],
        'High':  [108, 112, 115, 113, 111, 109, 111, 113, 114, 113, 112, 111],
        'Low':   [103, 106, 110, 108, 106, 104, 106, 108, 110, 108, 107, 106],
        'Close': [107, 111, 113, 111, 109, 108, 110, 112, 113, 111, 110, 109],
    })
    draw = find_draw_on_liquidity(bars_1h, current_idx=11, lookback=10, swing_n=2)
    assert draw is not None
    assert draw['direction'] == 'bullish'
    assert draw['level'] > 109  # above current close, any swing high above works


def test_draw_on_liquidity_bearish():
    """When an untouched old low exists below current price, draw is bearish."""
    # Price drops to 85, then rallies back to ~92 range. No swing high above current price.
    bars_1h = pd.DataFrame({
        'Open':  [ 95,  92,  88,  87,  88,  89,  90,  91,  92,  91,  90,  91],
        'High':  [ 96,  93,  90,  89,  90,  91,  92,  93,  93,  93,  92,  93],
        'Low':   [ 92,  89,  86,  85,  86,  87,  88,  89,  90,  89,  88,  89],
        'Close': [ 93,  90,  87,  88,  89,  90,  91,  92,  91,  90,  91,  92],
    })
    draw = find_draw_on_liquidity(bars_1h, current_idx=11, lookback=10, swing_n=2)
    assert draw is not None
    assert draw['direction'] == 'bearish'
    assert draw['level'] == 85


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
    assert fvgs[0]['top'] == 103
    assert fvgs[0]['bottom'] == 101
    assert fvgs[0]['idx'] == 1


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
    assert fvgs[0]['top'] == 106
    assert fvgs[0]['bottom'] == 102
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


def test_find_fvgs_vectorized_matches():
    """Vectorized FVG detection matches non-vectorized."""
    np.random.seed(42)
    n = 100
    prices = np.cumsum(np.random.randn(n) * 2) + 100
    bars = pd.DataFrame({
        'Open':  prices,
        'High':  prices + np.abs(np.random.randn(n)),
        'Low':   prices - np.abs(np.random.randn(n)),
        'Close': prices + np.random.randn(n) * 0.5,
    })
    for direction in ['bullish', 'bearish']:
        scalar = find_fvgs(bars, direction)
        vector = find_fvgs_vectorized(bars, direction)
        assert len(scalar) == len(vector), f"{direction}: {len(scalar)} vs {len(vector)}"


def test_detect_displacement_bullish():
    """Bullish displacement: large up candle relative to recent ATR."""
    # Normal bars have body ~1pt, TR ~3pt. Displacement bar has body=20pt, TR=21pt.
    # ATR ~3, threshold = 2*3=6, body=20 >> 6. Should detect.
    bars = pd.DataFrame({
        'Open':  [100, 101, 100, 101, 100, 100, 100, 100],
        'High':  [102, 103, 102, 103, 102, 102, 102, 121],
        'Low':   [ 99, 100,  99, 100,  99,  99,  99,  99],
        'Close': [101, 102, 101, 102, 101, 101, 101, 120],
    })
    displacements = detect_displacement(bars, direction='bullish', atr_mult=2.0, atr_period=5)
    assert len(displacements) > 0
    assert displacements[-1]['idx'] == 7


def test_find_order_blocks_bullish():
    """Bullish OB: last bearish candle before a bullish displacement."""
    # Bars 0-4: small, with some bearish. Bar 5: massive bullish displacement.
    bars = pd.DataFrame({
        'Open':  [102, 101, 100,  99, 101, 100],
        'High':  [103, 102, 101, 101, 103, 125],
        'Low':   [100,  99,  98,  98,  99,  99],
        'Close': [101, 100,  99, 100, 102, 124],
    })
    obs = find_order_blocks(bars, direction='bullish', atr_mult=2.0, atr_period=4)
    assert len(obs) > 0


def test_detect_stop_raid_bullish():
    """Bullish raid: sweep below swing low followed by up-close confirmation."""
    # Build clear structure: swing low forms, then gets swept, then up-close
    bars_5m = pd.DataFrame({
        'Open':  [110, 108, 106, 104, 102, 105, 108, 106, 104, 102, 100, 97, 93, 96, 99],
        'High':  [112, 110, 108, 106, 105, 108, 110, 109, 107, 105, 103, 100, 96, 99, 102],
        'Low':   [108, 106, 104, 102, 100, 103, 106, 104, 102, 100, 98, 95, 90, 94, 97],
        'Close': [109, 107, 105, 103, 103, 107, 109, 107, 105, 103, 99, 96, 95, 98, 101],
    })
    raids = detect_stop_raid(bars_5m, direction='bullish', swing_n=2)
    # There should be at least one raid found
    # Swing low at idx 4 (Low=100) gets swept by idx 12 (Low=90), then idx 14 closes up
    assert len(raids) > 0
    last_raid = raids[-1]
    assert last_raid['direction'] == 'bullish'


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
