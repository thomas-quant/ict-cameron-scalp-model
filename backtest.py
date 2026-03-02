"""
Cameron's Scalping Model Backtest
Both base (1:1 R:R) and higher R:R variants.

Based on ICT concepts: draw on liquidity → stop raid → 1m FVG entry.
Data: NQ 1-minute bars resampled to 5m/15m/1h.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

DATA_FILE = Path(__file__).parent / "data" / "nq_1m.parquet"
TICK_FILE = Path(__file__).parent / "data" / "nq_ticks.parquet"


# =============================================================================
# Data Loading & Resampling
# =============================================================================

def load_1m_bars(path: Path = DATA_FILE) -> pd.DataFrame:
    """Load pre-built 1-minute bars from parquet."""
    df = pd.read_parquet(path)
    df['DateTime_UTC'] = pd.to_datetime(df['DateTime_UTC'])
    df = df.set_index('DateTime_UTC').sort_index()
    return df[['DateTime_ET', 'Open', 'High', 'Low', 'Close', 'Volume']]


def resample_ticks(ticks: pd.DataFrame, freq: str = '30s') -> pd.DataFrame:
    """Resample raw tick data into OHLCV bars at the given frequency."""
    df = ticks.copy()
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    df = df.set_index('ts_event').sort_index()
    bars = df['price'].resample(freq).ohlc()
    bars.columns = ['Open', 'High', 'Low', 'Close']
    bars['Volume'] = df['size'].resample(freq).sum()
    bars = bars.dropna(subset=['Open'])
    # Convert UTC timestamps to Eastern Time for time-of-day analysis
    et_idx = bars.index.tz_localize('UTC').tz_convert('US/Eastern')
    bars['DateTime_ET'] = et_idx.tz_localize(None)
    return bars


def resample_bars(bars_1m: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Resample 1m OHLCV bars to a higher timeframe."""
    ohlcv = bars_1m[['Open', 'High', 'Low', 'Close', 'Volume']]
    resampled = ohlcv.resample(freq).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    })
    return resampled.dropna(subset=['Open'])


def build_bars(bars_1m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all timeframe bars from 1m bars."""
    return {
        '1m': bars_1m,
        '5m': resample_bars(bars_1m, '5min'),
        '15m': resample_bars(bars_1m, '15min'),
        '1h': resample_bars(bars_1m, '1h'),
    }


# =============================================================================
# Swing Detection
# =============================================================================

def find_swing_highs(bars: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Find swing highs: bars whose High exceeds N bars on each side."""
    highs = bars['High'].values
    length = len(highs)
    mask = np.zeros(length, dtype=bool)
    for i in range(n, length - n):
        left_max = highs[i - n:i].max()
        right_max = highs[i + 1:i + n + 1].max()
        if highs[i] > left_max and highs[i] > right_max:
            mask[i] = True
    return bars[mask].copy()


def find_swing_lows(bars: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Find swing lows: bars whose Low is below N bars on each side."""
    lows = bars['Low'].values
    length = len(lows)
    mask = np.zeros(length, dtype=bool)
    for i in range(n, length - n):
        left_min = lows[i - n:i].min()
        right_min = lows[i + 1:i + n + 1].min()
        if lows[i] < left_min and lows[i] < right_min:
            mask[i] = True
    return bars[mask].copy()


# =============================================================================
# Draw on Liquidity
# =============================================================================

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

    if len(window) < swing_n * 3:
        return None

    current_price = window.iloc[-1]['Close']
    effective_n = min(swing_n, max(1, len(window) // 3))

    swing_highs = find_swing_highs(window, n=effective_n)
    swing_lows = find_swing_lows(window, n=effective_n)

    old_highs = swing_highs[swing_highs['High'] > current_price]
    old_lows = swing_lows[swing_lows['Low'] < current_price]

    best_high = old_highs['High'].min() if len(old_highs) > 0 else None
    best_low = old_lows['Low'].max() if len(old_lows) > 0 else None

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


# =============================================================================
# Stop Raid Detection
# =============================================================================

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
    """
    raids = []
    idx = bars_5m.index
    length = len(bars_5m)

    if direction == 'bullish':
        swing_lows = find_swing_lows(bars_5m, n=swing_n)
        active_lows = []

        for i in range(length):
            pos = idx[i]
            if pos in swing_lows.index:
                active_lows.append((pos, swing_lows.loc[pos, 'Low']))

            if not active_lows:
                continue

            current_low = bars_5m.iloc[i]['Low']
            swept_level = None
            for sl_idx, sl_level in active_lows:
                if sl_idx == pos:
                    continue
                if current_low < sl_level:
                    swept_level = sl_level
                    break

            if swept_level is None:
                continue

            if wait_for_close:
                conf_open = bars_5m.iloc[i]['Open']
                conf_close = bars_5m.iloc[i]['Close']
                if conf_close > conf_open:
                    raids.append({
                        'sweep_idx': pos,
                        'sweep_level': swept_level,
                        'confirmation_idx': pos,
                        'confirmation_open': conf_open,
                        'confirmation_close': conf_close,
                        'direction': 'bullish',
                    })
                    active_lows = [(si, sl) for si, sl in active_lows if sl != swept_level]
                elif i + 1 < length:
                    next_pos = idx[i + 1]
                    next_open = bars_5m.iloc[i + 1]['Open']
                    next_close = bars_5m.iloc[i + 1]['Close']
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
                raids.append({
                    'sweep_idx': pos,
                    'sweep_level': swept_level,
                    'confirmation_idx': pos,
                    'confirmation_open': bars_5m.iloc[i]['Open'],
                    'confirmation_close': bars_5m.iloc[i]['Close'],
                    'direction': 'bullish',
                })
                active_lows = [(si, sl) for si, sl in active_lows if sl != swept_level]

    elif direction == 'bearish':
        swing_highs = find_swing_highs(bars_5m, n=swing_n)
        active_highs = []

        for i in range(length):
            pos = idx[i]
            if pos in swing_highs.index:
                active_highs.append((pos, swing_highs.loc[pos, 'High']))

            if not active_highs:
                continue

            current_high = bars_5m.iloc[i]['High']
            swept_level = None
            for sh_idx, sh_level in active_highs:
                if sh_idx == pos:
                    continue
                if current_high > sh_level:
                    swept_level = sh_level
                    break

            if swept_level is None:
                continue

            if wait_for_close:
                conf_open = bars_5m.iloc[i]['Open']
                conf_close = bars_5m.iloc[i]['Close']
                if conf_close < conf_open:
                    raids.append({
                        'sweep_idx': pos,
                        'sweep_level': swept_level,
                        'confirmation_idx': pos,
                        'confirmation_open': conf_open,
                        'confirmation_close': conf_close,
                        'direction': 'bearish',
                    })
                    active_highs = [(si, sl) for si, sl in active_highs if sl != swept_level]
                elif i + 1 < length:
                    next_pos = idx[i + 1]
                    next_open = bars_5m.iloc[i + 1]['Open']
                    next_close = bars_5m.iloc[i + 1]['Close']
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
                    'confirmation_open': bars_5m.iloc[i]['Open'],
                    'confirmation_close': bars_5m.iloc[i]['Close'],
                    'direction': 'bearish',
                })
                active_highs = [(si, sl) for si, sl in active_highs if sl != swept_level]

    return raids


# =============================================================================
# Fair Value Gap Detection
# =============================================================================

def find_fvgs(bars: pd.DataFrame, direction: str) -> list[dict]:
    """
    Find Fair Value Gaps in a series of bars.

    Bullish FVG: candle[i-1].High < candle[i+1].Low (gap between wicks)
    Bearish FVG: candle[i-1].Low > candle[i+1].High
    """
    fvgs = []
    idx = bars.index
    highs = bars['High'].values
    lows = bars['Low'].values

    for i in range(1, len(bars) - 1):
        if direction == 'bullish':
            if highs[i - 1] < lows[i + 1]:
                fvgs.append({
                    'idx': idx[i],
                    'top': lows[i + 1],
                    'bottom': highs[i - 1],
                    'mid': (lows[i + 1] + highs[i - 1]) / 2,
                })
        elif direction == 'bearish':
            if lows[i - 1] > highs[i + 1]:
                fvgs.append({
                    'idx': idx[i],
                    'top': lows[i - 1],
                    'bottom': highs[i + 1],
                    'mid': (lows[i - 1] + highs[i + 1]) / 2,
                })
    return fvgs


def find_fvgs_vectorized(bars: pd.DataFrame, direction: str) -> pd.DataFrame:
    """
    Vectorized FVG detection across all bars. Returns DataFrame with columns:
    idx (timestamp of middle candle), top, bottom, mid.
    """
    highs = bars['High'].values
    lows = bars['Low'].values
    idx = bars.index

    if direction == 'bullish':
        mask = highs[:-2] < lows[2:]
    else:
        mask = lows[:-2] > highs[2:]

    positions = np.where(mask)[0] + 1  # middle candle positions

    if len(positions) == 0:
        return pd.DataFrame(columns=['idx', 'top', 'bottom', 'mid'])

    if direction == 'bullish':
        tops = lows[positions + 1]
        bottoms = highs[positions - 1]
    else:
        tops = lows[positions - 1]
        bottoms = highs[positions + 1]

    return pd.DataFrame({
        'idx': idx[positions],
        'top': tops,
        'bottom': bottoms,
        'mid': (tops + bottoms) / 2,
    })


# =============================================================================
# Displacement & Order Block Detection (Higher R:R)
# =============================================================================

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
    opens = bars['Open'].values
    closes = bars['Close'].values
    highs = bars['High'].values
    lows = bars['Low'].values

    body = np.abs(closes - opens)
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))

    # Rolling ATR
    atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values

    displacements = []
    idx = bars.index
    start = max(atr_period, 1)

    for i in range(start, len(bars)):
        if direction == 'bullish' and closes[i] > opens[i]:
            if body[i] > atr_mult * atr[i - 1]:
                displacements.append({
                    'idx': idx[i],
                    'open': opens[i],
                    'close': closes[i],
                    'high': highs[i],
                    'low': lows[i],
                    'body': body[i],
                })
        elif direction == 'bearish' and closes[i] < opens[i]:
            if body[i] > atr_mult * atr[i - 1]:
                displacements.append({
                    'idx': idx[i],
                    'open': opens[i],
                    'close': closes[i],
                    'high': highs[i],
                    'low': lows[i],
                    'body': body[i],
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
    opens = bars['Open'].values
    closes = bars['Close'].values
    highs = bars['High'].values
    lows = bars['Low'].values

    for disp in displacements:
        disp_pos = idx_list.index(disp['idx'])
        for j in range(disp_pos - 1, max(disp_pos - 10, -1), -1):
            if direction == 'bullish' and closes[j] < opens[j]:
                order_blocks.append({
                    'ob_idx': idx_list[j],
                    'ob_high': highs[j],
                    'ob_low': lows[j],
                    'ob_open': opens[j],
                    'ob_close': closes[j],
                    'displacement_idx': disp['idx'],
                })
                break
            elif direction == 'bearish' and closes[j] > opens[j]:
                order_blocks.append({
                    'ob_idx': idx_list[j],
                    'ob_high': highs[j],
                    'ob_low': lows[j],
                    'ob_open': opens[j],
                    'ob_close': closes[j],
                    'displacement_idx': disp['idx'],
                })
                break

    return order_blocks


# =============================================================================
# Trade Dataclass
# =============================================================================

@dataclass
class Trade:
    entry_time: pd.Timestamp = pd.NaT
    exit_time: Optional[pd.Timestamp] = None
    direction: str = ''
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    pnl_points: float = 0.0
    r_multiple: float = 0.0
    variant: str = ''
    exit_reason: str = ''
    entry_hour_et: int = -1


# =============================================================================
# Base Model Strategy Engine
# =============================================================================

def run_base_model(
    bars_1m: pd.DataFrame,
    bars_5m: pd.DataFrame,
    bars_15m: pd.DataFrame,
    bars_1h: pd.DataFrame,
    max_stop: float = 12.0,
    cooldown_seconds: int = 300,
    cost_per_trade: float = 0.25,
) -> list[Trade]:
    """
    Run the base model backtest.

    Flow:
    1. Every new 1h bar: re-evaluate draw on liquidity (using COMPLETED bars only)
    2. Every new 5m bar: check for stop raids opposite to draw (COMPLETED bars only)
    3. After valid raid: scan 1m bars for FVG entry (delayed by 1 bar)
    4. Manage open position (SL/TP on each 1m bar)
    """
    trades: list[Trade] = []
    position: Optional[Trade] = None
    last_exit_time: Optional[pd.Timestamp] = None

    bars_5m_idx = bars_5m.index
    bars_15m_idx = bars_15m.index
    bars_1h_idx = bars_1h.index
    bars_1m_idx = bars_1m.index

    # Precompute HTF index arrays for searchsorted
    bars_5m_ts = bars_5m_idx.values
    bars_15m_ts = bars_15m_idx.values
    bars_1h_ts = bars_1h_idx.values

    # Precompute all 1m FVGs for both directions
    fvgs_bull = find_fvgs_vectorized(bars_1m, 'bullish')
    fvgs_bear = find_fvgs_vectorized(bars_1m, 'bearish')

    # Index FVGs by int64 nanosecond key for fast lookup
    def _build_fvg_map(fvgs_df):
        if fvgs_df.empty:
            return {}
        m = {}
        for row in fvgs_df.itertuples():
            key = pd.Timestamp(row.idx).value
            m[key] = row
        return m

    fvgs_bull_map = _build_fvg_map(fvgs_bull)
    fvgs_bear_map = _build_fvg_map(fvgs_bear)

    current_draw = None
    active_raid = None
    last_1h_pos = -1
    last_5m_pos = -1

    # Track active (unfilled) FVGs after a raid signal
    active_fvgs: list[tuple] = []
    MAX_ACTIVE_FVGS = 20

    # Precompute arrays for fast access
    bars_1m_open = bars_1m['Open'].values
    bars_1m_high = bars_1m['High'].values
    bars_1m_low = bars_1m['Low'].values
    bars_1m_close = bars_1m['Close'].values
    bars_1m_timestamps = bars_1m_idx.values
    bars_1m_ts_ns = bars_1m_timestamps.astype('datetime64[ns]').astype(np.int64)

    # Precompute ET hours for TOD chart
    bars_1m_et_hours = bars_1m['DateTime_ET'].dt.hour.values

    total_1m = len(bars_1m)
    report_interval = total_1m // 20

    for i in range(total_1m):
        ts = bars_1m_timestamps[i]

        if report_interval > 0 and i % report_interval == 0:
            pct = i / total_1m * 100
            print(f"  Base model: {pct:.0f}% ({len(trades)} trades so far)")

        # --- Check exits ---
        if position is not None:
            hit_sl = False
            hit_tp = False
            low_i = bars_1m_low[i]
            high_i = bars_1m_high[i]

            if position.direction == 'long':
                if low_i <= position.stop_price:
                    hit_sl = True
                if high_i >= position.target_price:
                    hit_tp = True
            else:
                if high_i >= position.stop_price:
                    hit_sl = True
                if low_i <= position.target_price:
                    hit_tp = True

            if hit_sl and hit_tp:
                hit_tp = False

            if hit_sl or hit_tp:
                position.exit_time = pd.Timestamp(ts)
                if hit_sl:
                    position.exit_price = position.stop_price
                    position.exit_reason = 'sl'
                else:
                    position.exit_price = position.target_price
                    position.exit_reason = 'tp'

                if position.direction == 'long':
                    position.pnl_points = (position.exit_price - position.entry_price) - cost_per_trade
                else:
                    position.pnl_points = (position.entry_price - position.exit_price) - cost_per_trade
                risk = abs(position.entry_price - position.stop_price)
                raw_pnl = position.exit_price - position.entry_price if position.direction == 'long' else position.entry_price - position.exit_price
                position.r_multiple = raw_pnl / risk if risk > 0 else 0
                trades.append(position)
                last_exit_time = ts
                position = None
                active_raid = None
                active_fvgs = []
            continue

        # --- Cooldown ---
        if last_exit_time is not None:
            if (ts - last_exit_time) / np.timedelta64(1, 's') < cooldown_seconds:
                continue

        # --- Step 1: Update draw on liquidity (on new 1h bar, using COMPLETED bars) ---
        h_pos_raw = np.searchsorted(bars_1h_ts, ts, side='right') - 1
        if h_pos_raw >= 0 and h_pos_raw != last_1h_pos:
            last_1h_pos = h_pos_raw
            h_pos = h_pos_raw - 1  # last COMPLETED 1h bar
            if h_pos >= 10:
                draw = find_draw_on_liquidity(bars_1h, current_idx=h_pos, lookback=20, swing_n=3)
                if draw is None:
                    m15_pos_raw = np.searchsorted(bars_15m_ts, ts, side='right') - 1
                    m15_pos = m15_pos_raw - 1  # last COMPLETED 15m bar
                    if m15_pos >= 10:
                        draw = find_draw_on_liquidity(bars_15m, current_idx=m15_pos, lookback=40, swing_n=3)
                current_draw = draw

        if current_draw is None:
            continue

        # --- Step 2: Detect stop raids (on new 5m bar, using COMPLETED bars) ---
        m5_pos_raw = np.searchsorted(bars_5m_ts, ts, side='right') - 1
        if m5_pos_raw >= 0 and m5_pos_raw != last_5m_pos:
            last_5m_pos = m5_pos_raw
            m5_pos = m5_pos_raw - 1  # last COMPLETED 5m bar
            if m5_pos >= 20:
                window_start = max(0, m5_pos - 50)
                window = bars_5m.iloc[window_start:m5_pos + 1]
                raids = detect_stop_raid(
                    window,
                    direction=current_draw['direction'],
                    swing_n=5,
                    wait_for_close=True,
                )
                if raids:
                    # Only accept raids confirmed on the last 2 completed
                    # 5m bars (mirrors real-time: react to what just happened,
                    # not historical sweeps from hours ago)
                    recent_cutoff = bars_5m_ts[max(0, m5_pos - 1)]
                    recent_raids = [r for r in raids
                                    if r['confirmation_idx'] >= recent_cutoff]
                    if recent_raids:
                        active_raid = recent_raids[-1]
                        active_fvgs = []

        if active_raid is None:
            continue

        # Only look for entries after confirmation
        if ts <= active_raid['confirmation_idx']:
            continue

        # --- Step 3: FVG entry on 1m (delayed: middle candle at i-2, third candle at i-1) ---
        fvg_map = fvgs_bull_map if current_draw['direction'] == 'bullish' else fvgs_bear_map

        # FVG confirmed at bar i-1 (third candle), middle candle at i-2
        if i >= 3:
            prev_key = bars_1m_ts_ns[i - 2]
            new_fvg = fvg_map.get(prev_key)
            if new_fvg is not None:
                active_fvgs.append((new_fvg.top, new_fvg.bottom, new_fvg.mid))
                if len(active_fvgs) > MAX_ACTIVE_FVGS:
                    active_fvgs = active_fvgs[-MAX_ACTIVE_FVGS:]

        if not active_fvgs:
            continue

        # Check if current bar returns to any active FVG
        entered = False
        entry_price = None
        entry_fvg_top = None
        entry_fvg_bottom = None
        used_fvg_idx = None

        low_i = bars_1m_low[i]
        high_i = bars_1m_high[i]

        for fi, (fvg_top, fvg_bottom, fvg_mid) in enumerate(active_fvgs):
            if current_draw['direction'] == 'bullish':
                if low_i <= fvg_top:
                    entered = True
                    entry_price = fvg_top
                    entry_fvg_top = fvg_top
                    entry_fvg_bottom = fvg_bottom
                    used_fvg_idx = fi
                    break
            else:
                if high_i >= fvg_bottom:
                    entered = True
                    entry_price = fvg_bottom
                    entry_fvg_top = fvg_top
                    entry_fvg_bottom = fvg_bottom
                    used_fvg_idx = fi
                    break

        # Prune FVGs that have been completely filled or used for entry
        remaining_fvgs = []
        for fi, (fvg_top, fvg_bottom, fvg_mid) in enumerate(active_fvgs):
            if fi == used_fvg_idx:
                continue
            if current_draw['direction'] == 'bullish':
                if low_i > fvg_bottom:
                    remaining_fvgs.append((fvg_top, fvg_bottom, fvg_mid))
            else:
                if high_i < fvg_top:
                    remaining_fvgs.append((fvg_top, fvg_bottom, fvg_mid))
        active_fvgs = remaining_fvgs

        if not entered or entry_price is None:
            continue

        # Calculate stop and target
        et_hour = int(bars_1m_et_hours[i])

        if current_draw['direction'] == 'bullish':
            struct_stop = entry_fvg_bottom - 1.0
            if entry_price - struct_stop < max_stop and entry_price - struct_stop > 0:
                stop = struct_stop
            else:
                stop = entry_price - max_stop
            risk = entry_price - stop
            if risk <= 0:
                continue
            target = entry_price + risk  # 1:1

            position = Trade(
                entry_time=pd.Timestamp(ts),
                direction='long',
                entry_price=entry_price,
                stop_price=stop,
                target_price=target,
                variant='base',
                entry_hour_et=et_hour,
            )
        else:
            struct_stop = entry_fvg_top + 1.0
            if struct_stop - entry_price < max_stop and struct_stop - entry_price > 0:
                stop = struct_stop
            else:
                stop = entry_price + max_stop
            risk = stop - entry_price
            if risk <= 0:
                continue
            target = entry_price - risk  # 1:1

            position = Trade(
                entry_time=pd.Timestamp(ts),
                direction='short',
                entry_price=entry_price,
                stop_price=stop,
                target_price=target,
                variant='base',
                entry_hour_et=et_hour,
            )

        active_raid = None
        active_fvgs = []

    # Close open position at end
    if position is not None:
        position.exit_time = pd.Timestamp(bars_1m_timestamps[-1])
        position.exit_price = bars_1m_close[-1]
        position.exit_reason = 'timeout'
        if position.direction == 'long':
            position.pnl_points = (position.exit_price - position.entry_price) - cost_per_trade
        else:
            position.pnl_points = (position.entry_price - position.exit_price) - cost_per_trade
        risk = abs(position.entry_price - position.stop_price)
        raw_pnl = position.exit_price - position.entry_price if position.direction == 'long' else position.entry_price - position.exit_price
        position.r_multiple = raw_pnl / risk if risk > 0 else 0
        trades.append(position)

    return trades


# =============================================================================
# Higher R:R Strategy Engine
# =============================================================================

def find_structural_target(
    bars_5m: pd.DataFrame,
    direction: str,
    entry_price: float,
    current_5m_idx: int,
    swing_n: int = 5,
) -> float | None:
    """Find the nearest structural target (previous swing high/low on 5m)."""
    start = max(0, current_5m_idx - 50)
    window = bars_5m.iloc[start:current_5m_idx]
    if len(window) < swing_n * 3:
        return None

    effective_n = min(swing_n, max(1, len(window) // 3))

    if direction == 'bullish':
        highs = find_swing_highs(window, n=effective_n)
        targets = highs[highs['High'] > entry_price]
        if len(targets) > 0:
            return targets['High'].min()
    else:
        lows = find_swing_lows(window, n=effective_n)
        targets = lows[lows['Low'] < entry_price]
        if len(targets) > 0:
            return targets['Low'].max()
    return None


def run_higher_rr_model(
    bars_1m: pd.DataFrame,
    bars_5m: pd.DataFrame,
    bars_15m: pd.DataFrame,
    bars_1h: pd.DataFrame,
    cooldown_seconds: int = 300,
    atr_mult: float = 2.0,
    min_rr: float = 1.5,
    cost_per_trade: float = 0.25,
) -> list[Trade]:
    """
    Run the higher R:R variant.

    Differences from base:
    - Don't wait for 5m close (wait_for_close=False)
    - Look for displacement + order blocks on 1m
    - Target structural levels, minimum 1.5 R:R
    """
    trades: list[Trade] = []
    position: Optional[Trade] = None
    last_exit_time: Optional[pd.Timestamp] = None

    bars_5m_ts = bars_5m.index.values
    bars_15m_ts = bars_15m.index.values
    bars_1h_ts = bars_1h.index.values
    bars_1m_timestamps = bars_1m.index.values

    # Precompute FVGs with int64 ns keys
    def _build_fvg_map(fvgs_df):
        if fvgs_df.empty:
            return {}
        return {pd.Timestamp(row.idx).value: row for row in fvgs_df.itertuples()}

    fvgs_bull_map = _build_fvg_map(find_fvgs_vectorized(bars_1m, 'bullish')) if len(bars_1m) > 2 else {}
    fvgs_bear_map = _build_fvg_map(find_fvgs_vectorized(bars_1m, 'bearish')) if len(bars_1m) > 2 else {}

    bars_1m_open = bars_1m['Open'].values
    bars_1m_high = bars_1m['High'].values
    bars_1m_low = bars_1m['Low'].values
    bars_1m_close = bars_1m['Close'].values
    bars_1m_ts_ns = bars_1m_timestamps.astype('datetime64[ns]').astype(np.int64)

    # Precompute ET hours for TOD chart
    bars_1m_et_hours = bars_1m['DateTime_ET'].dt.hour.values

    current_draw = None
    active_raid = None
    active_raid_5m_pos = -1
    active_fvgs: list[tuple] = []
    last_1h_pos = -1
    last_5m_pos = -1

    MAX_ACTIVE_FVGS = 5
    RAID_EXPIRY_5M_BARS = 20
    MAX_RR = 5.0

    total_1m = len(bars_1m)
    report_interval = total_1m // 20

    for i in range(total_1m):
        ts = bars_1m_timestamps[i]

        if report_interval > 0 and i % report_interval == 0:
            pct = i / total_1m * 100
            print(f"  Higher R:R: {pct:.0f}% ({len(trades)} trades so far)")

        # --- Exits ---
        if position is not None:
            hit_sl = False
            hit_tp = False
            low_i = bars_1m_low[i]
            high_i = bars_1m_high[i]

            if position.direction == 'long':
                if low_i <= position.stop_price:
                    hit_sl = True
                if high_i >= position.target_price:
                    hit_tp = True
            else:
                if high_i >= position.stop_price:
                    hit_sl = True
                if low_i <= position.target_price:
                    hit_tp = True

            if hit_sl and hit_tp:
                hit_tp = False

            if hit_sl or hit_tp:
                position.exit_time = pd.Timestamp(ts)
                if hit_sl:
                    position.exit_price = position.stop_price
                    position.exit_reason = 'sl'
                else:
                    position.exit_price = position.target_price
                    position.exit_reason = 'tp'

                if position.direction == 'long':
                    position.pnl_points = (position.exit_price - position.entry_price) - cost_per_trade
                else:
                    position.pnl_points = (position.entry_price - position.exit_price) - cost_per_trade
                risk = abs(position.entry_price - position.stop_price)
                raw_pnl = position.exit_price - position.entry_price if position.direction == 'long' else position.entry_price - position.exit_price
                position.r_multiple = raw_pnl / risk if risk > 0 else 0
                trades.append(position)
                last_exit_time = ts
                position = None
                active_raid = None
                active_fvgs = []
            continue

        # --- Cooldown ---
        if last_exit_time is not None:
            if (ts - last_exit_time) / np.timedelta64(1, 's') < cooldown_seconds:
                continue

        # --- Draw on liquidity (using COMPLETED HTF bars) ---
        h_pos_raw = np.searchsorted(bars_1h_ts, ts, side='right') - 1
        if h_pos_raw >= 0 and h_pos_raw != last_1h_pos:
            last_1h_pos = h_pos_raw
            h_pos = h_pos_raw - 1  # last COMPLETED 1h bar
            if h_pos >= 10:
                draw = find_draw_on_liquidity(bars_1h, current_idx=h_pos, lookback=20, swing_n=3)
                if draw is None:
                    m15_pos_raw = np.searchsorted(bars_15m_ts, ts, side='right') - 1
                    m15_pos = m15_pos_raw - 1  # last COMPLETED 15m bar
                    if m15_pos >= 10:
                        draw = find_draw_on_liquidity(bars_15m, current_idx=m15_pos, lookback=40, swing_n=3)
                current_draw = draw

        if current_draw is None:
            continue

        # --- Stop raid (don't wait for close, using COMPLETED 5m bars) ---
        m5_pos_raw = np.searchsorted(bars_5m_ts, ts, side='right') - 1
        if m5_pos_raw >= 0 and m5_pos_raw != last_5m_pos:
            last_5m_pos = m5_pos_raw
            m5_pos = m5_pos_raw - 1  # last COMPLETED 5m bar
            if m5_pos >= 20:
                window_start = max(0, m5_pos - 50)
                window = bars_5m.iloc[window_start:m5_pos + 1]
                raids = detect_stop_raid(
                    window,
                    direction=current_draw['direction'],
                    swing_n=5,
                    wait_for_close=False,
                )
                if raids:
                    # Only accept raids confirmed on the last 2 completed
                    # 5m bars (real-time detection, not historical)
                    recent_cutoff = bars_5m_ts[max(0, m5_pos - 1)]
                    recent_raids = [r for r in raids
                                    if r['confirmation_idx'] >= recent_cutoff]
                    if recent_raids:
                        active_raid = recent_raids[-1]
                        active_raid_5m_pos = m5_pos
                        active_fvgs = []

        if active_raid is None:
            continue

        # Expire stale raids
        if m5_pos_raw - 1 - active_raid_5m_pos > RAID_EXPIRY_5M_BARS:
            active_raid = None
            active_fvgs = []
            continue

        if ts <= active_raid['confirmation_idx']:
            continue

        # --- Entry: FVG or OB after displacement ---
        if i < 14:
            continue

        # Accumulate post-raid FVGs (delayed: middle at i-2, third candle at i-1)
        fvg_map = fvgs_bull_map if current_draw['direction'] == 'bullish' else fvgs_bear_map
        if i >= 3:
            prev_key = bars_1m_ts_ns[i - 2]
            new_fvg = fvg_map.get(prev_key)
            if new_fvg is not None:
                active_fvgs.append((new_fvg.top, new_fvg.bottom, new_fvg.mid))
                if len(active_fvgs) > MAX_ACTIVE_FVGS:
                    active_fvgs = active_fvgs[-MAX_ACTIVE_FVGS:]

        # Check whether current bar retraces into any active (post-raid) FVG
        fvg_entry = None
        used_fvg_idx = None
        for fi, (fvg_top, fvg_bottom, fvg_mid) in enumerate(active_fvgs):
            if current_draw['direction'] == 'bullish' and bars_1m_low[i] <= fvg_top:
                fvg_entry = fvg_top
                used_fvg_idx = fi
                break
            elif current_draw['direction'] == 'bearish' and bars_1m_high[i] >= fvg_bottom:
                fvg_entry = fvg_bottom
                used_fvg_idx = fi
                break

        # Prune FVGs that have been fully filled
        remaining_fvgs = []
        for fi, (fvg_top, fvg_bottom, fvg_mid) in enumerate(active_fvgs):
            if fi == used_fvg_idx:
                continue
            if current_draw['direction'] == 'bullish':
                if bars_1m_low[i] > fvg_bottom:
                    remaining_fvgs.append((fvg_top, fvg_bottom, fvg_mid))
            else:
                if bars_1m_high[i] < fvg_top:
                    remaining_fvgs.append((fvg_top, fvg_bottom, fvg_mid))
        active_fvgs = remaining_fvgs

        # Check for OB after displacement
        ob_entry = None
        lookback_start = max(0, i - 30)
        lookback_window = bars_1m.iloc[lookback_start:i]
        if len(lookback_window) > 14:
            obs = find_order_blocks(lookback_window, direction=current_draw['direction'], atr_mult=atr_mult, atr_period=10)
            if obs:
                ob = obs[-1]
                if current_draw['direction'] == 'bullish' and bars_1m_low[i] <= ob['ob_high']:
                    ob_entry = ob['ob_high']
                elif current_draw['direction'] == 'bearish' and bars_1m_high[i] >= ob['ob_low']:
                    ob_entry = ob['ob_low']

        # Prefer OB entry; fall back to FVG entry
        entry_price = ob_entry if ob_entry is not None else fvg_entry
        if entry_price is None:
            continue

        # Stop at displacement swing point
        disps = detect_displacement(lookback_window, direction=current_draw['direction'], atr_mult=atr_mult, atr_period=10) if len(lookback_window) > 10 else []

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
        m5_pos_now = np.searchsorted(bars_5m_ts, ts, side='right') - 1
        target = None
        if m5_pos_now > 0:
            target = find_structural_target(bars_5m, current_draw['direction'], entry_price, m5_pos_now)
        if target is None:
            target = current_draw['level']

        # Cap target at MAX_RR
        if current_draw['direction'] == 'bullish':
            target = min(target, entry_price + risk * MAX_RR)
            potential_rr = (target - entry_price) / risk if risk > 0 else 0
        else:
            target = max(target, entry_price - risk * MAX_RR)
            potential_rr = (entry_price - target) / risk if risk > 0 else 0

        if potential_rr < min_rr:
            continue

        et_hour = int(bars_1m_et_hours[i])

        if current_draw['direction'] == 'bullish':
            position = Trade(
                entry_time=pd.Timestamp(ts),
                direction='long',
                entry_price=entry_price,
                stop_price=stop,
                target_price=target,
                variant='higher_rr',
                entry_hour_et=et_hour,
            )
        else:
            position = Trade(
                entry_time=pd.Timestamp(ts),
                direction='short',
                entry_price=entry_price,
                stop_price=stop,
                target_price=target,
                variant='higher_rr',
                entry_hour_et=et_hour,
            )
        active_raid = None
        active_fvgs = []

    # Close open position at end
    if position is not None:
        position.exit_time = pd.Timestamp(bars_1m_timestamps[-1])
        position.exit_price = bars_1m_close[-1]
        position.exit_reason = 'timeout'
        if position.direction == 'long':
            position.pnl_points = (position.exit_price - position.entry_price) - cost_per_trade
        else:
            position.pnl_points = (position.entry_price - position.exit_price) - cost_per_trade
        risk = abs(position.entry_price - position.stop_price)
        raw_pnl = position.exit_price - position.entry_price if position.direction == 'long' else position.entry_price - position.exit_price
        position.r_multiple = raw_pnl / risk if risk > 0 else 0
        trades.append(position)

    return trades


# =============================================================================
# Reporting
# =============================================================================

def compute_stats(trades: list[Trade]) -> dict:
    """Compute summary statistics from a list of trades."""
    if not trades:
        return {'total_trades': 0}

    pnls = [t.pnl_points for t in trades]
    win_trades = [t for t in trades if t.pnl_points > 0]
    loss_trades = [t for t in trades if t.pnl_points <= 0]
    win_pnls = [t.pnl_points for t in win_trades]
    loss_pnls = [t.pnl_points for t in loss_trades]
    win_rs = [t.r_multiple for t in win_trades]
    loss_rs = [t.r_multiple for t in loss_trades]

    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_max

    gross_profit = sum(win_pnls) if win_pnls else 0
    gross_loss = abs(sum(loss_pnls)) if loss_pnls else 0

    # Expectancy: expected R per trade across all trades
    win_rate = len(win_trades) / len(trades)
    loss_rate = 1.0 - win_rate
    avg_win_r = float(np.mean(win_rs)) if win_rs else 0.0
    avg_loss_r = abs(float(np.mean(loss_rs))) if loss_rs else 0.0
    expectancy = win_rate * avg_win_r - loss_rate * avg_loss_r

    # Sharpe: annualised from daily PnL
    from datetime import date, timedelta
    daily: dict[date, float] = {}
    for t in trades:
        if t.exit_time is not None and not pd.isnull(t.exit_time):
            d = pd.Timestamp(t.exit_time).date()
            daily[d] = daily.get(d, 0.0) + t.pnl_points
    if len(daily) > 1:
        d_min, d_max = min(daily), max(daily)
        series = []
        cur = d_min
        while cur <= d_max:
            series.append(daily.get(cur, 0.0))
            cur += timedelta(days=1)
        arr = np.array(series, dtype=float)
        sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(252)) if np.std(arr) > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        'total_trades': len(trades),
        'wins': len(win_trades),
        'losses': len(loss_trades),
        'win_rate': win_rate * 100,
        'avg_pnl': float(np.mean(pnls)),
        'avg_r': float(np.mean(win_rs)) if win_rs else 0.0,
        'median_r': float(np.median(win_rs)) if win_rs else 0.0,
        'expectancy': expectancy,
        'sharpe': sharpe,
        'total_pnl': sum(pnls),
        'max_drawdown': float(min(drawdowns)) if len(drawdowns) > 0 else 0.0,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'largest_win': max(pnls),
        'largest_loss': min(pnls),
        'avg_win': float(np.mean(win_pnls)) if win_pnls else 0.0,
        'avg_loss': float(np.mean(loss_pnls)) if loss_pnls else 0.0,
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
        'entry_hour_et': t.entry_hour_et,
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
    """Generate HTML report matching the PCX/ICT dark-theme styling."""
    base_stats = compute_stats(base_trades)
    hrr_stats = compute_stats(higher_rr_trades)
    base_df = trades_to_df(base_trades)
    hrr_df = trades_to_df(higher_rr_trades)

    csv_dir = output_path.parent
    if not base_df.empty:
        base_df.to_csv(csv_dir / 'trades_base.csv', index=False)
    if not hrr_df.empty:
        hrr_df.to_csv(csv_dir / 'trades_higher_rr.csv', index=False)

    all_trades = base_trades + higher_rr_trades
    if all_trades:
        date_min = min(t.entry_time for t in all_trades)
        date_max = max(t.entry_time for t in all_trades)
    else:
        date_min = date_max = 'N/A'

    # --- Build Plotly figures and serialize to JSON ---
    import json
    POINT_VALUE  = 2       # USD per NQ point (2 = MNQ micro, 20 = full NQ)
    INIT_CAPITAL = 10_000  # Starting capital (USD)

    _dark_base = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#7d8590', size=12),
    )

    def _to_js(fig):
        """Serialize a Plotly figure to (data_json, layout_json) strings."""
        d = json.loads(fig.to_json())
        return json.dumps(d.get('data', [])), json.dumps(d.get('layout', {}))

    def _equity_fig(trades_list, title, line_color, fill_color):
        if not trades_list:
            return go.Figure()
        cum_pts = np.cumsum([t.pnl_points for t in trades_list])
        cum_pct = [round(v * POINT_VALUE / INIT_CAPITAL * 100, 2) for v in cum_pts]
        times = [str(t.exit_time) for t in trades_list]
        fig = go.Figure(go.Scatter(
            x=times, y=cum_pct, mode='lines', name=title,
            line=dict(color=line_color, width=2),
            fill='tozeroy', fillcolor=fill_color,
        ))
        fig.update_layout(**_dark_base,
            title=dict(text=title, font=dict(color='#e6edf3', size=14)),
            margin=dict(t=30, r=20, b=50, l=70),
            xaxis=dict(gridcolor='#21262d'),
            yaxis=dict(title='Cumulative Return (%)', ticksuffix='%', tickformat='.1f', gridcolor='#21262d'),
            shapes=[dict(type='line', x0=0, x1=1, xref='paper', y0=0, y1=0,
                         line=dict(color='#7d8590', width=1, dash='dot'))],
        )
        return fig

    def _dd_fig(trades_list):
        if not trades_list:
            return go.Figure()
        pnls = [t.pnl_points for t in trades_list]
        cum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum)
        dd_pct = [round((v * POINT_VALUE / INIT_CAPITAL) * 100, 3) for v in (cum - running_max)]
        times = [str(t.exit_time) for t in trades_list]
        fig = go.Figure(go.Scatter(
            x=times, y=dd_pct, mode='lines',
            line=dict(color='#f85149', width=1.5),
            fill='tozeroy', fillcolor='rgba(248,81,73,0.15)',
        ))
        fig.update_layout(**_dark_base,
            margin=dict(t=10, r=20, b=40, l=60),
            xaxis=dict(gridcolor='#21262d'),
            yaxis=dict(title='Drawdown (%)', ticksuffix='%', tickformat='.2f', gridcolor='#21262d'),
            showlegend=False,
        )
        return fig

    def _rdist_fig(rs_values, title, bar_color, line_color, xbins):
        fig = go.Figure(go.Histogram(
            x=rs_values, xbins=xbins,
            marker=dict(color=bar_color, line=dict(color=line_color, width=1)),
        ))
        fig.update_layout(**_dark_base,
            title=dict(text=title, font=dict(color='#c9d1d9', size=13)),
            margin=dict(t=30, r=20, b=50, l=50),
            xaxis=dict(title='R-Multiple', gridcolor='#21262d'),
            yaxis=dict(title='Count', gridcolor='#21262d'),
            shapes=[dict(type='line', x0=0, x1=0, y0=0, y1=1, yref='paper',
                         line=dict(color='#7d8590', width=1, dash='dot'))],
        )
        return fig

    def _tod_fig(trades_list, title, pos_color, neg_color):
        """Build a time-of-day profitability bar chart."""
        if not trades_list:
            return go.Figure()
        hourly_pnl = {}
        for t in trades_list:
            h = t.entry_hour_et
            if h < 0:
                continue
            hourly_pnl[h] = hourly_pnl.get(h, 0.0) + t.pnl_points
        hours = list(range(24))
        pnls = [hourly_pnl.get(h, 0.0) for h in hours]
        colors = [pos_color if p >= 0 else neg_color for p in pnls]
        fig = go.Figure(go.Bar(
            x=[f"{h:02d}:00" for h in hours],
            y=pnls,
            marker_color=colors,
        ))
        fig.update_layout(**_dark_base,
            title=dict(text=title, font=dict(color='#c9d1d9', size=13)),
            margin=dict(t=30, r=20, b=50, l=50),
            xaxis=dict(title='Hour (ET)', gridcolor='#21262d'),
            yaxis=dict(title='Total PnL (pts)', gridcolor='#21262d'),
        )
        return fig

    base_eq_data, base_eq_layout = _to_js(_equity_fig(
        base_trades, 'Base Model (1:1 R:R)', '#58a6ff', 'rgba(88,166,255,0.08)'))
    hrr_eq_data, hrr_eq_layout = _to_js(_equity_fig(
        higher_rr_trades, 'Higher R:R Variant', '#d29922', 'rgba(210,153,34,0.08)'))
    base_dd_data, base_dd_layout = _to_js(_dd_fig(base_trades))
    hrr_dd_data, hrr_dd_layout = _to_js(_dd_fig(higher_rr_trades))

    base_rs = [round(t.r_multiple, 2) for t in base_trades] if base_trades else []
    hrr_rs  = [round(t.r_multiple, 2) for t in higher_rr_trades] if higher_rr_trades else []
    base_rd_data, base_rd_layout = _to_js(_rdist_fig(
        base_rs, 'Base Model', 'rgba(88,166,255,0.6)', '#58a6ff',
        dict(start=-2, end=2, size=0.25)))
    hrr_rd_data, hrr_rd_layout = _to_js(_rdist_fig(
        hrr_rs, 'Higher R:R', 'rgba(210,153,34,0.6)', '#d29922',
        dict(start=-5, end=max(hrr_rs) + 1 if hrr_rs else 30, size=1)))

    # Time-of-day charts
    base_tod_data, base_tod_layout = _to_js(_tod_fig(
        base_trades, 'Base Model', 'rgba(63,185,80,0.7)', 'rgba(248,81,73,0.7)'))
    hrr_tod_data, hrr_tod_layout = _to_js(_tod_fig(
        higher_rr_trades, 'Higher R:R', 'rgba(63,185,80,0.7)', 'rgba(248,81,73,0.7)'))

    def _pts_to_pct(pts):
        return pts * POINT_VALUE / INIT_CAPITAL * 100

    # Direction breakdown
    base_longs = sum(1 for t in base_trades if t.direction == 'long')
    base_shorts = sum(1 for t in base_trades if t.direction == 'short')
    hrr_longs = sum(1 for t in higher_rr_trades if t.direction == 'long')
    hrr_shorts = sum(1 for t in higher_rr_trades if t.direction == 'short')

    # Win rate by direction
    def _wr_by_dir(trades_list, direction):
        subset = [t for t in trades_list if t.direction == direction]
        if not subset:
            return 0.0
        return sum(1 for t in subset if t.pnl_points > 0) / len(subset) * 100

    base_wr_long = _wr_by_dir(base_trades, 'long')
    base_wr_short = _wr_by_dir(base_trades, 'short')
    hrr_wr_long = _wr_by_dir(higher_rr_trades, 'long')
    hrr_wr_short = _wr_by_dir(higher_rr_trades, 'short')

    # Exit reason breakdown
    base_tp = sum(1 for t in base_trades if t.exit_reason == 'tp')
    base_sl = sum(1 for t in base_trades if t.exit_reason == 'sl')
    hrr_tp = sum(1 for t in higher_rr_trades if t.exit_reason == 'tp')
    hrr_sl = sum(1 for t in higher_rr_trades if t.exit_reason == 'sl')
    hrr_to = sum(1 for t in higher_rr_trades if t.exit_reason == 'timeout')

    # Monthly breakdown tables
    def _monthly_rows(df):
        mb = monthly_breakdown(df)
        if mb.empty:
            return '<tr><td colspan="7" style="text-align:center;color:#7d8590">No trades</td></tr>'
        rows = ''
        for _, r in mb.iterrows():
            pnl_class = 'green' if r['total_pnl'] > 0 else 'red' if r['total_pnl'] < 0 else ''
            rows += f'''<tr>
                <td class="highlight">{r["month"]}</td>
                <td>{int(r["trades"])}</td>
                <td class="{pnl_class} highlight">{r["total_pnl"]:+.1f}</td>
                <td>{r["avg_r"]:.2f}</td>
                <td>{r["win_rate"]:.1f}%</td>
                <td class="green">{r["largest_win"]:.1f}</td>
                <td class="red">{r["largest_loss"]:.1f}</td></tr>'''
        return rows

    base_monthly_rows = _monthly_rows(base_df)
    hrr_monthly_rows = _monthly_rows(hrr_df)

    # Trade log HTML (limit to last 200 for readability)
    def _trade_rows(df, max_rows=200):
        if df.empty:
            return '<tr><td colspan="10" style="text-align:center;color:#7d8590">No trades</td></tr>'
        display_df = df.tail(max_rows)
        rows = ''
        for _, r in display_df.iterrows():
            dir_tag = 'long' if r['direction'] == 'long' else 'short'
            pnl_class = 'green' if r['pnl_points'] > 0 else 'red'
            entry_str = str(r['entry_time'])[5:19] if pd.notna(r['entry_time']) else ''
            exit_str = str(r['exit_time'])[5:19] if pd.notna(r['exit_time']) else ''
            rows += f'''<tr>
                <td>{entry_str}</td><td>{exit_str}</td>
                <td><span class="tag {dir_tag}">{r["direction"].upper()}</span></td>
                <td>{r["entry_price"]:.2f}</td><td>{r["exit_price"]:.2f}</td>
                <td>{r["stop_price"]:.2f}</td><td>{r["target_price"]:.2f}</td>
                <td class="{pnl_class} highlight">{r["pnl_points"]:+.2f}</td>
                <td class="{pnl_class}">{r["r_multiple"]:+.2f}R</td>
                <td>{r["exit_reason"].upper()}</td></tr>'''
        return rows

    base_trade_rows = _trade_rows(base_df)
    hrr_trade_rows = _trade_rows(hrr_df)
    base_showing = min(len(base_df), 200)
    hrr_showing = min(len(hrr_df), 200)

    # Stat formatting helpers
    def _pf(stats):
        if stats['total_trades'] == 0:
            return 'N/A'
        return f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else '&infin;'

    def _val_class(val, positive_good=True):
        if val > 0:
            return 'positive' if positive_good else 'negative'
        elif val < 0:
            return 'negative' if positive_good else 'positive'
        return ''

    # --- Build HTML ---
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cameron's Scalping Model &mdash; Backtest Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, sans-serif;
        background: #0a0e17;
        color: #c9d1d9;
        padding: 24px;
        line-height: 1.6;
    }}
    .header {{
        text-align: center;
        padding: 32px 0;
        border-bottom: 1px solid #21262d;
        margin-bottom: 32px;
    }}
    .header h1 {{
        font-size: 28px;
        color: #e6edf3;
        margin-bottom: 8px;
    }}
    .header .subtitle {{
        color: #7d8590;
        font-size: 14px;
    }}
    .stats-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 16px;
        margin-bottom: 32px;
    }}
    .stat-card {{
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }}
    .stat-card .label {{
        font-size: 12px;
        color: #7d8590;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }}
    .stat-card .value {{
        font-size: 24px;
        font-weight: 700;
        color: #e6edf3;
    }}
    .stat-card .value.positive {{ color: #3fb950; }}
    .stat-card .value.negative {{ color: #f85149; }}
    .stat-card .value.neutral {{ color: #d29922; }}
    .stat-card .detail {{
        font-size: 11px;
        color: #7d8590;
        margin-top: 4px;
    }}
    .section {{
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 24px;
    }}
    .section h2 {{
        font-size: 18px;
        color: #e6edf3;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #21262d;
    }}
    .section h3 {{
        font-size: 14px;
        color: #c9d1d9;
        margin-bottom: 12px;
        margin-top: 20px;
    }}
    .chart-container {{
        width: 100%;
        height: 350px;
        margin-bottom: 16px;
    }}
    .chart-container.tall {{
        height: 400px;
    }}
    .two-col {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 24px;
    }}
    @media (max-width: 800px) {{
        .two-col {{ grid-template-columns: 1fr; }}
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 16px;
        font-size: 14px;
    }}
    th, td {{
        padding: 10px 14px;
        text-align: left;
        border-bottom: 1px solid #21262d;
    }}
    th {{
        color: #7d8590;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }}
    td {{ color: #c9d1d9; }}
    tr:last-child td {{ border-bottom: none; }}
    .highlight {{ color: #e6edf3; font-weight: 600; }}
    .green {{ color: #3fb950; }}
    .red {{ color: #f85149; }}
    .yellow {{ color: #d29922; }}
    .finding-box {{
        background: #0d1117;
        border-left: 3px solid #3fb950;
        padding: 16px 20px;
        margin: 16px 0;
        border-radius: 0 6px 6px 0;
    }}
    .finding-box.warn {{
        border-left-color: #d29922;
    }}
    .finding-box.info {{
        border-left-color: #58a6ff;
    }}
    .finding-box p {{
        margin: 0;
        font-size: 14px;
    }}
    .finding-box .finding-label {{
        font-size: 11px;
        color: #7d8590;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }}
    .method-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 16px;
    }}
    @media (max-width: 800px) {{
        .method-grid {{ grid-template-columns: 1fr; }}
    }}
    .method-card {{
        background: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 20px;
    }}
    .method-card h4 {{
        color: #e6edf3;
        font-size: 15px;
        margin-bottom: 8px;
    }}
    .method-card p {{
        font-size: 13px;
        color: #7d8590;
        margin-bottom: 0;
    }}
    .method-card .acc {{
        font-size: 20px;
        font-weight: 700;
        margin-top: 8px;
    }}
    .tag {{
        display: inline-block;
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }}
    .tag.long {{ background: rgba(63, 185, 80, 0.15); color: #3fb950; }}
    .tag.short {{ background: rgba(248, 81, 73, 0.15); color: #f85149; }}
    .trade-log-wrap {{
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #21262d;
        border-radius: 6px;
    }}
    .trade-log-wrap table {{
        margin-bottom: 0;
    }}
    .trade-log-wrap thead th {{
        position: sticky;
        top: 0;
        background: #161b22;
        z-index: 1;
    }}
</style>
</head>
<body>

<div class="header">
    <h1>Cameron&rsquo;s Scalping Model</h1>
    <div class="subtitle">Backtest Report &mdash; NQ Futures &mdash; 1-minute bars &mdash; {str(date_min)[:10]} to {str(date_max)[:10]}</div>
</div>

<!-- ── Top-level stats ─────────────────────────────────────── -->

<div class="stats-grid">
    <div class="stat-card">
        <div class="label">Base Model Trades</div>
        <div class="value">{base_stats['total_trades'] if base_stats['total_trades'] else 0:,}</div>
        <div class="detail">{base_longs} Long / {base_shorts} Short</div>
    </div>
    <div class="stat-card">
        <div class="label">Base Win Rate</div>
        <div class="value positive">{base_stats.get('win_rate', 0):.1f}%</div>
        <div class="detail">{base_stats.get('wins', 0):,} / {base_stats['total_trades'] if base_stats['total_trades'] else 0:,}</div>
    </div>
    <div class="stat-card">
        <div class="label">Base Total P&amp;L</div>
        <div class="value {_val_class(base_stats.get('total_pnl', 0))}">{base_stats.get('total_pnl', 0):+,.1f} pts</div>
        <div class="detail">PF: {_pf(base_stats)}</div>
    </div>
    <div class="stat-card">
        <div class="label">Higher R:R Trades</div>
        <div class="value">{hrr_stats['total_trades'] if hrr_stats['total_trades'] else 0:,}</div>
        <div class="detail">{hrr_longs} Long / {hrr_shorts} Short</div>
    </div>
    <div class="stat-card">
        <div class="label">Higher R:R Win Rate</div>
        <div class="value neutral">{hrr_stats.get('win_rate', 0):.1f}%</div>
        <div class="detail">Avg R (winners): {hrr_stats.get('avg_r', 0):.2f}</div>
    </div>
    <div class="stat-card">
        <div class="label">Higher R:R Total P&amp;L</div>
        <div class="value {_val_class(hrr_stats.get('total_pnl', 0))}">{hrr_stats.get('total_pnl', 0):+,.1f} pts</div>
        <div class="detail">PF: {_pf(hrr_stats)}</div>
    </div>
</div>

<!-- ── Models Overview ─────────────────────────────────────── -->

<div class="section">
    <h2>Models Under Test</h2>
    <div class="method-grid">
        <div class="method-card">
            <h4>Base Model (1:1 R:R)</h4>
            <p>Wait for 5-minute confirmation candle close after a stop raid opposite the draw on liquidity. Enter on 1-minute fair value gap retracement. Fixed 12-point stop loss, 1:1 take profit.</p>
            <div class="acc green">{base_stats.get('win_rate', 0):.1f}% <span style="font-size:13px;font-weight:400;color:#7d8590">win rate</span></div>
        </div>
        <div class="method-card">
            <h4>Higher R:R Variant</h4>
            <p>Skip the 5-minute close &mdash; trigger on the sweep itself. Enter on displacement order blocks or FVGs on the 1-minute chart. Stop at displacement swing point, target structural levels (min 1.5R).</p>
            <div class="acc yellow">{hrr_stats.get('avg_r', 0):.2f}R <span style="font-size:13px;font-weight:400;color:#7d8590">avg R (winners)</span></div>
        </div>
    </div>
    <div class="finding-box info">
        <div class="finding-label">Model Comparison</div>
        <p>The base model trades at high frequency with a <strong>{base_stats.get('win_rate',0):.0f}% hit rate</strong> and fixed 1:1 payoff. The higher R:R variant trades less ({hrr_stats['total_trades'] if hrr_stats['total_trades'] else 0:,} vs {base_stats['total_trades'] if base_stats['total_trades'] else 0:,}) but captures <strong>{hrr_stats.get('avg_r',0):.1f}x risk on winning trades</strong> (expectancy: {hrr_stats.get('expectancy',0):+.2f}R). Higher R:R generated <strong>{hrr_stats.get('total_pnl',0)/base_stats.get('total_pnl',1) if base_stats.get('total_pnl',0)!=0 else 0:.1f}x</strong> the total P&amp;L of the base model.</p>
    </div>
</div>

<!-- ── Equity Curves ─────────────────────────────────────── -->

<div class="section">
    <h2>Equity Curves</h2>
    <div class="chart-container tall" id="equity-base"></div>
    <div class="chart-container tall" id="equity-hrr"></div>
</div>

<!-- ── Drawdown ─────────────────────────────────────────── -->

<div class="section">
    <h2>Drawdown Analysis</h2>
    <div class="two-col">
        <div>
            <h3>Base Model</h3>
            <div class="chart-container" id="dd-base" style="height:250px"></div>
            <div class="stats-grid" style="grid-template-columns: 1fr 1fr;">
                <div class="stat-card">
                    <div class="label">Max Drawdown</div>
                    <div class="value negative">{_pts_to_pct(base_stats.get('max_drawdown', 0)):.2f}%</div>
                </div>
                <div class="stat-card">
                    <div class="label">Largest Loss</div>
                    <div class="value negative">{_pts_to_pct(base_stats.get('largest_loss', 0)):.2f}%</div>
                </div>
            </div>
        </div>
        <div>
            <h3>Higher R:R</h3>
            <div class="chart-container" id="dd-hrr" style="height:250px"></div>
            <div class="stats-grid" style="grid-template-columns: 1fr 1fr;">
                <div class="stat-card">
                    <div class="label">Max Drawdown</div>
                    <div class="value negative">{_pts_to_pct(hrr_stats.get('max_drawdown', 0)):.2f}%</div>
                </div>
                <div class="stat-card">
                    <div class="label">Largest Loss</div>
                    <div class="value negative">{_pts_to_pct(hrr_stats.get('largest_loss', 0)):.2f}%</div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- ── R-Multiple Distribution ─────────────────────────────── -->

<div class="section">
    <h2>R-Multiple Distribution</h2>
    <div class="two-col">
        <div class="chart-container" id="rdist-base" style="height:300px"></div>
        <div class="chart-container" id="rdist-hrr" style="height:300px"></div>
    </div>
    <div class="finding-box">
        <div class="finding-label">Key Difference</div>
        <p>Base model is tightly clustered at &pm;1R (by design). Higher R:R shows a long right tail &mdash; winners at 2&ndash;5R+ compensate for the <strong>{hrr_stats.get('win_rate',0):.0f}%</strong> win rate, yielding a <strong>{_pf(hrr_stats)}</strong> profit factor.</p>
    </div>
</div>

<!-- ── Profitability by Time of Day ──────────────────────── -->

<div class="section">
    <h2>Profitability by Time of Day</h2>
    <div class="two-col">
        <div class="chart-container" id="tod-base" style="height:300px"></div>
        <div class="chart-container" id="tod-hrr" style="height:300px"></div>
    </div>
</div>

<!-- ── Monthly Breakdown ─────────────────────────────────── -->

<div class="section">
    <h2>Monthly Breakdown</h2>
    <div class="two-col">
        <div>
            <h3>Base Model</h3>
            <table>
                <thead>
                    <tr><th>Month</th><th>Trades</th><th>P&amp;L</th><th>Avg R</th><th>Win%</th><th>Best</th><th>Worst</th></tr>
                </thead>
                <tbody>{base_monthly_rows}</tbody>
            </table>
        </div>
        <div>
            <h3>Higher R:R</h3>
            <table>
                <thead>
                    <tr><th>Month</th><th>Trades</th><th>P&amp;L</th><th>Avg R</th><th>Win%</th><th>Best</th><th>Worst</th></tr>
                </thead>
                <tbody>{hrr_monthly_rows}</tbody>
            </table>
        </div>
    </div>
</div>

<!-- ── Direction Breakdown ─────────────────────────────────── -->

<div class="section">
    <h2>Direction Breakdown</h2>
    <div class="two-col">
        <div class="chart-container" id="dir-pie" style="height:300px"></div>
        <div>
            <table>
                <thead>
                    <tr><th>Metric</th><th>Base Long</th><th>Base Short</th><th>HRR Long</th><th>HRR Short</th></tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Trades</td>
                        <td>{base_longs}</td><td>{base_shorts}</td>
                        <td>{hrr_longs}</td><td>{hrr_shorts}</td>
                    </tr>
                    <tr>
                        <td>Win Rate</td>
                        <td class="{'green' if base_wr_long > 50 else 'red'}">{base_wr_long:.1f}%</td>
                        <td class="{'green' if base_wr_short > 50 else 'red'}">{base_wr_short:.1f}%</td>
                        <td class="{'green' if hrr_wr_long > 50 else 'yellow'}">{hrr_wr_long:.1f}%</td>
                        <td class="{'green' if hrr_wr_short > 50 else 'yellow'}">{hrr_wr_short:.1f}%</td>
                    </tr>
                    <tr>
                        <td>Exit: TP / SL</td>
                        <td colspan="2" style="text-align:center">{base_tp:,} / {base_sl:,}</td>
                        <td colspan="2" style="text-align:center">{hrr_tp:,} / {hrr_sl:,}</td>
                    </tr>
                </tbody>
            </table>
            <div class="finding-box warn">
                <div class="finding-label">Note</div>
                <p>Both models show short-side bias ({base_shorts / max(base_stats['total_trades'],1) * 100:.0f}% base, {hrr_shorts / max(hrr_stats['total_trades'],1) * 100:.0f}% HRR). This reflects the draw-on-liquidity function finding more nearby swing lows than highs in this data period.</p>
            </div>
        </div>
    </div>
</div>

<!-- ── Comparative Stats ─────────────────────────────────── -->

<div class="section">
    <h2>Head-to-Head Comparison</h2>
    <table>
        <thead>
            <tr><th>Metric</th><th>Base Model</th><th>Higher R:R</th></tr>
        </thead>
        <tbody>
            <tr><td>Total Trades</td><td class="highlight">{base_stats['total_trades'] if base_stats['total_trades'] else 0:,}</td><td class="highlight">{hrr_stats['total_trades'] if hrr_stats['total_trades'] else 0:,}</td></tr>
            <tr><td>Win Rate</td><td class="green">{base_stats.get('win_rate',0):.1f}%</td><td class="yellow">{hrr_stats.get('win_rate',0):.1f}%</td></tr>
            <tr><td>Avg R (Winners)</td><td>{base_stats.get('avg_r',0):.2f}</td><td class="green">{hrr_stats.get('avg_r',0):.2f}</td></tr>
            <tr><td>Median R (Winners)</td><td>{base_stats.get('median_r',0):.2f}</td><td class="yellow">{hrr_stats.get('median_r',0):.2f}</td></tr>
            <tr><td>Expectancy (R/trade)</td><td class="{_val_class(base_stats.get('expectancy',0))}">{base_stats.get('expectancy',0):+.3f}R</td><td class="{_val_class(hrr_stats.get('expectancy',0))}">{hrr_stats.get('expectancy',0):+.3f}R</td></tr>
            <tr><td>Sharpe (ann.)</td><td class="{_val_class(base_stats.get('sharpe',0))}">{base_stats.get('sharpe',0):.2f}</td><td class="{_val_class(hrr_stats.get('sharpe',0))}">{hrr_stats.get('sharpe',0):.2f}</td></tr>
            <tr><td>Total P&amp;L</td><td class="{_val_class(base_stats.get('total_pnl',0))}">{base_stats.get('total_pnl',0):+,.1f} pts</td><td class="{_val_class(hrr_stats.get('total_pnl',0))}">{hrr_stats.get('total_pnl',0):+,.1f} pts</td></tr>
            <tr><td>Profit Factor</td><td>{_pf(base_stats)}</td><td class="green">{_pf(hrr_stats)}</td></tr>
            <tr><td>Max Drawdown</td><td class="red">{_pts_to_pct(base_stats.get('max_drawdown',0)):.2f}%</td><td class="red">{_pts_to_pct(hrr_stats.get('max_drawdown',0)):.2f}%</td></tr>
            <tr><td>Largest Win</td><td class="green">{base_stats.get('largest_win',0):.1f} pts</td><td class="green">{hrr_stats.get('largest_win',0):.1f} pts</td></tr>
            <tr><td>Largest Loss</td><td class="red">{base_stats.get('largest_loss',0):.1f} pts</td><td class="red">{hrr_stats.get('largest_loss',0):.1f} pts</td></tr>
            <tr><td>Avg Win</td><td class="green">{base_stats.get('avg_win',0):.2f} pts</td><td class="green">{hrr_stats.get('avg_win',0):.2f} pts</td></tr>
            <tr><td>Avg Loss</td><td class="red">{base_stats.get('avg_loss',0):.2f} pts</td><td class="red">{hrr_stats.get('avg_loss',0):.2f} pts</td></tr>
        </tbody>
    </table>
</div>

<!-- ── Trade Logs ─────────────────────────────────────────── -->

<div class="section">
    <h2>Trade Log &mdash; Base Model (last {base_showing} of {len(base_df):,})</h2>
    <div class="trade-log-wrap">
        <table>
            <thead>
                <tr><th>Entry</th><th>Exit</th><th>Dir</th><th>Entry$</th><th>Exit$</th><th>Stop</th><th>Target</th><th>P&amp;L</th><th>R</th><th>Exit</th></tr>
            </thead>
            <tbody>{base_trade_rows}</tbody>
        </table>
    </div>
</div>

<div class="section">
    <h2>Trade Log &mdash; Higher R:R (last {hrr_showing} of {len(hrr_df):,})</h2>
    <div class="trade-log-wrap">
        <table>
            <thead>
                <tr><th>Entry</th><th>Exit</th><th>Dir</th><th>Entry$</th><th>Exit$</th><th>Stop</th><th>Target</th><th>P&amp;L</th><th>R</th><th>Exit</th></tr>
            </thead>
            <tbody>{hrr_trade_rows}</tbody>
        </table>
    </div>
</div>

<!-- ── Findings ─────────────────────────────────────────────── -->

<div class="section">
    <h2>Findings Summary</h2>

    <div class="finding-box">
        <div class="finding-label">1. Base model performance</div>
        <p>{base_stats.get('win_rate',0):.0f}% win rate at 1:1 R:R. Max drawdown {_pts_to_pct(base_stats.get('max_drawdown',0)):.2f}%. The model generates {base_stats['total_trades'] if base_stats['total_trades'] else 0:,} trades by entering on every FVG retracement after a confirmed raid.</p>
    </div>

    <div class="finding-box">
        <div class="finding-label">2. Higher R:R captures outsized moves</div>
        <p>Despite only {hrr_stats.get('win_rate',0):.0f}% wins, the higher R:R variant generates <strong>{hrr_stats.get('total_pnl',0):,.0f} pts</strong> total P&amp;L. Winning trades average <strong>{hrr_stats.get('avg_r',0):.2f}R</strong> and per-trade expectancy is <strong>{hrr_stats.get('expectancy',0):+.3f}R</strong>. Structural targets let winners run well beyond 1:1.</p>
    </div>

    <div class="finding-box warn">
        <div class="finding-label">3. Drawdown trade-off</div>
        <p>Higher R:R has a significantly deeper max drawdown ({_pts_to_pct(hrr_stats.get('max_drawdown',0)):.2f}% vs {_pts_to_pct(base_stats.get('max_drawdown',0)):.2f}%). The wider stops and lower win rate create longer losing streaks before large winners recover.</p>
    </div>

    <div class="finding-box warn">
        <div class="finding-label">4. Short-side directional bias</div>
        <p>Both models lean short (~{base_shorts / max(base_stats['total_trades'],1) * 100:.0f}%). This is a function of the draw-on-liquidity logic and the data period. Out-of-sample testing on different regimes would validate whether this persists.</p>
    </div>

    <div class="finding-box info">
        <div class="finding-label">5. Data &amp; methodology</div>
        <p>Backtest on NQ 30-second OHLCV bars (from tick data). Transaction cost: 0.25 pts round-trip. HTF bars resampled from 1m data (no lookahead). FVG entries delayed by one bar (no same-bar bias). All sessions included (no time filter). Conservative fill: when SL and TP hit on same bar, SL is assumed.</p>
    </div>
</div>

<!-- ── Plotly Charts ────────────────────────────────────────── -->

<script>
var C = {{ responsive: true, displayModeBar: false }};

Plotly.newPlot('equity-base', {base_eq_data}, {base_eq_layout}, C);
Plotly.newPlot('equity-hrr',  {hrr_eq_data},  {hrr_eq_layout},  C);
Plotly.newPlot('dd-base',     {base_dd_data}, {base_dd_layout}, C);
Plotly.newPlot('dd-hrr',      {hrr_dd_data},  {hrr_dd_layout},  C);
Plotly.newPlot('rdist-base',  {base_rd_data}, {base_rd_layout}, C);
Plotly.newPlot('rdist-hrr',   {hrr_rd_data},  {hrr_rd_layout},  C);
Plotly.newPlot('tod-base',    {base_tod_data}, {base_tod_layout}, C);
Plotly.newPlot('tod-hrr',     {hrr_tod_data},  {hrr_tod_layout},  C);

Plotly.newPlot('dir-pie', [{{
    labels: ['Base Long', 'Base Short', 'HRR Long', 'HRR Short'],
    values: [{base_longs}, {base_shorts}, {hrr_longs}, {hrr_shorts}],
    type: 'pie',
    marker: {{
        colors: ['rgba(63,185,80,0.7)', 'rgba(248,81,73,0.7)', 'rgba(63,185,80,0.35)', 'rgba(248,81,73,0.35)']
    }},
    textinfo: 'label+percent',
    textfont: {{ color: '#e6edf3', size: 12 }},
    hovertemplate: '%{{label}}: %{{value}} trades (%{{percent}})<extra></extra>',
    hole: 0.35
}}], {{
    paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
    font: {{ color: '#7d8590', size: 12 }},
    margin: {{ t: 10, r: 10, b: 10, l: 10 }},
    showlegend: false
}}, C);

</script>

</body>
</html>'''

    output_path.write_text(html, encoding='utf-8')
    print(f"Report saved to {output_path}")
    if not base_df.empty:
        print(f"Base trades CSV: {csv_dir / 'trades_base.csv'}")
    if not hrr_df.empty:
        print(f"Higher R:R trades CSV: {csv_dir / 'trades_higher_rr.csv'}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Cameron's Scalping Model — Backtest")
    print("=" * 60)

    # --- Load tick data and build 30s entry bars ---
    print("\nLoading tick data...")
    ticks = pd.read_parquet(TICK_FILE)
    print(f"  {len(ticks):,} ticks loaded")

    print("\nResampling ticks to 30-second bars...")
    bars_30s = resample_ticks(ticks)
    print(f"  {len(bars_30s):,} 30s bars")
    print(f"  Range: {bars_30s.index[0]} to {bars_30s.index[-1]}")
    del ticks  # free memory

    # --- Load 1m bars and resample to HTF (no tick data for HTF) ---
    print("\nLoading 1-minute bar data for HTF analysis...")
    bars_1m = load_1m_bars()
    # Filter to tick-data period to avoid lookahead
    bars_1m = bars_1m[(bars_1m.index >= bars_30s.index[0])
                      & (bars_1m.index <= bars_30s.index[-1])]
    print(f"  {len(bars_1m):,} bars (filtered to tick range)")

    print("\nBuilding HTF bars from 1m data...")
    bars = build_bars(bars_1m)
    for tf, df in bars.items():
        print(f"  {tf}: {len(df):,} bars")

    # --- Run models with 30s entry bars ---
    print("\n" + "-" * 60)
    print("Running Base Model (1:1 R:R, 12pt stop, 30s entry)...")
    print("-" * 60)
    base_trades = run_base_model(bars_30s, bars['5m'], bars['15m'], bars['1h'])
    print(f"  Completed: {len(base_trades)} trades")

    print("\n" + "-" * 60)
    print("Running Higher R:R Model (structural targets, 30s entry)...")
    print("-" * 60)
    hrr_trades = run_higher_rr_model(bars_30s, bars['5m'], bars['15m'], bars['1h'])
    print(f"  Completed: {len(hrr_trades)} trades")

    print("\n" + "-" * 60)
    print("Generating report...")
    print("-" * 60)
    report_path = Path(__file__).parent / 'report.html'
    generate_report(base_trades, hrr_trades, report_path)

    # Print summary
    for label, trades_list in [('Base Model', base_trades), ('Higher R:R', hrr_trades)]:
        stats = compute_stats(trades_list)
        print(f"\n{'=' * 40}")
        print(f"  {label}")
        print(f"{'=' * 40}")
        if stats['total_trades'] > 0:
            pf = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "inf"
            print(f"  Trades:        {stats['total_trades']}")
            print(f"  Win Rate:      {stats['win_rate']:.1f}%")
            print(f"  Total P&L:     {stats['total_pnl']:.1f} pts")
            print(f"  Avg R:         {stats['avg_r']:.2f}")
            print(f"  Median R:      {stats['median_r']:.2f}")
            print(f"  Profit Factor: {pf}")
            print(f"  Max Drawdown:  {stats['max_drawdown']:.1f} pts")
            print(f"  Largest Win:   {stats['largest_win']:.1f} pts")
            print(f"  Largest Loss:  {stats['largest_loss']:.1f} pts")
        else:
            print("  No trades generated.")


if __name__ == '__main__':
    main()
