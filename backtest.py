"""
Cameron's Scalping Model Backtest

Polars-only rewrite with separate explicit tick paths:
- in-sample: original Databento-style tick parquet
- out-of-sample: sanitized merged tick parquet

The engine keeps memory pressure down by:
- scanning parquet lazily
- selecting only required columns
- aggregating ticks to 30-second bars before collect
- dropping intermediate DataFrames aggressively
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import polars as pl


DATA_DIR = Path(__file__).parent / "data"
ONE_MINUTE_FILE = DATA_DIR / "nq_1m.parquet"
ORIGINAL_TICK_FILE = DATA_DIR / "nq_ticks.parquet"
MERGED_TICK_FILE = DATA_DIR / "merged_nq_ticks.parquet"

REPORT_DIR = Path(__file__).parent
ET_ZONE = "America/New_York"


@dataclass(slots=True)
class DatasetConfig:
    name: str
    label: str
    tick_file: Path
    tick_schema: str
    report_file: str
    csv_suffix: str


@dataclass(slots=True)
class Bars:
    ts_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    et_hour: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(self.ts_ns.size)

    @property
    def timestamps(self) -> np.ndarray:
        return self.ts_ns.view("datetime64[ns]")

    @property
    def start_ns(self) -> int:
        return int(self.ts_ns[0])

    @property
    def end_ns(self) -> int:
        return int(self.ts_ns[-1])

    def slice(self, start: int, stop: Optional[int] = None) -> "Bars":
        return Bars(
            ts_ns=self.ts_ns[start:stop],
            open=self.open[start:stop],
            high=self.high[start:stop],
            low=self.low[start:stop],
            close=self.close[start:stop],
            volume=self.volume[start:stop],
            et_hour=None if self.et_hour is None else self.et_hour[start:stop],
        )

    def range_slice(self, start_ns: int, end_ns: int) -> "Bars":
        left = int(np.searchsorted(self.ts_ns, start_ns, side="left"))
        right = int(np.searchsorted(self.ts_ns, end_ns, side="right"))
        return self.slice(left, right)


@dataclass(slots=True)
class Trade:
    entry_time_ns: int = 0
    exit_time_ns: Optional[int] = None
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    pnl_points: float = 0.0
    r_multiple: float = 0.0
    variant: str = ""
    exit_reason: str = ""
    entry_hour_et: int = -1


IN_SAMPLE_CONFIG = DatasetConfig(
    name="in_sample",
    label="In-Sample: Original Tick Data",
    tick_file=ORIGINAL_TICK_FILE,
    tick_schema="original",
    report_file="report_in_sample.html",
    csv_suffix="in_sample",
)

OUT_OF_SAMPLE_CONFIG = DatasetConfig(
    name="out_of_sample",
    label="Out-of-Sample: Sanitized Merged Ticks",
    tick_file=MERGED_TICK_FILE,
    tick_schema="merged",
    report_file="report_out_of_sample.html",
    csv_suffix="out_of_sample",
)


def aggressive_gc() -> None:
    gc.collect()


def ns_to_utc_datetime(ns_value: Optional[int]) -> Optional[datetime]:
    if ns_value is None:
        return None
    seconds, remainder = divmod(int(ns_value), 1_000_000_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc) + timedelta(microseconds=remainder / 1000.0)


def format_trade_time(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        dt = value
    else:
        return str(value)[5:19]
    return dt.astimezone(timezone.utc).strftime("%m-%d %H:%M:%S")


def bars_from_frame(df: pl.DataFrame, ts_col: str, include_et_hour: bool = False) -> Bars:
    df = df.rechunk()
    ts_ns = df.get_column(ts_col).cast(pl.Datetime("ns")).cast(pl.Int64).to_numpy()
    et_hour = None
    if include_et_hour and "DateTime_ET" in df.columns:
        et_hour = df.get_column("DateTime_ET").dt.hour().cast(pl.Int16).to_numpy()
    return Bars(
        ts_ns=ts_ns,
        open=df.get_column("Open").cast(pl.Float64).to_numpy(),
        high=df.get_column("High").cast(pl.Float64).to_numpy(),
        low=df.get_column("Low").cast(pl.Float64).to_numpy(),
        close=df.get_column("Close").cast(pl.Float64).to_numpy(),
        volume=df.get_column("Volume").cast(pl.Int64).to_numpy(),
        et_hour=et_hour,
    )


def collect_lazy_frame(lf: pl.LazyFrame, label: str) -> pl.DataFrame:
    print(f"  Collecting {label}...")
    return lf.collect(engine="streaming")


def load_original_entry_bars(path: Path, freq: str = "30s") -> Bars:
    lf = (
        pl.scan_parquet(path)
        .select(
            pl.col("ts_event").dt.replace_time_zone("UTC").set_sorted().alias("ts_event"),
            pl.col("price").cast(pl.Float64).alias("price"),
            pl.col("size").cast(pl.Int64).alias("size"),
        )
        .group_by_dynamic("ts_event", every=freq, period=freq, closed="left", label="left")
        .agg(
            pl.col("price").first().alias("Open"),
            pl.col("price").max().alias("High"),
            pl.col("price").min().alias("Low"),
            pl.col("price").last().alias("Close"),
            pl.col("size").sum().alias("Volume"),
        )
        .drop_nulls(["Open"])
        .with_columns(
            pl.col("ts_event").dt.convert_time_zone(ET_ZONE).dt.replace_time_zone(None).alias("DateTime_ET")
        )
    )
    df = collect_lazy_frame(lf, f"{path.name} -> {freq} bars")
    bars = bars_from_frame(df, "ts_event", include_et_hour=True)
    del df
    aggressive_gc()
    return bars


def load_merged_entry_bars(path: Path, freq: str = "30s") -> Bars:
    # This loader follows the schema described in data/merged_ticks_schema.txt:
    # ts_event, intra_ts_rank, side, price_ticks, size
    lf = (
        pl.scan_parquet(path)
        .select(
            pl.col("ts_event").set_sorted().alias("ts_event"),
            pl.col("intra_ts_rank").cast(pl.UInt8).alias("intra_ts_rank"),
            (pl.col("price_ticks").cast(pl.Float64) / 4.0).alias("price"),
            pl.col("size").cast(pl.Int64).alias("size"),
        )
        .group_by_dynamic("ts_event", every=freq, period=freq, closed="left", label="left")
        .agg(
            pl.col("price").first().alias("Open"),
            pl.col("price").max().alias("High"),
            pl.col("price").min().alias("Low"),
            pl.col("price").last().alias("Close"),
            pl.col("size").sum().alias("Volume"),
        )
        .drop_nulls(["Open"])
        .with_columns(
            pl.col("ts_event").dt.convert_time_zone(ET_ZONE).dt.replace_time_zone(None).alias("DateTime_ET")
        )
    )
    df = collect_lazy_frame(lf, f"{path.name} -> {freq} bars")
    bars = bars_from_frame(df, "ts_event", include_et_hour=True)
    del df
    aggressive_gc()
    return bars


def load_entry_bars(config: DatasetConfig) -> Bars:
    if config.tick_schema == "original":
        return load_original_entry_bars(config.tick_file)
    if config.tick_schema == "merged":
        return load_merged_entry_bars(config.tick_file)
    raise ValueError(f"Unknown tick schema: {config.tick_schema}")


def load_1m_bars(path: Path, start_ns: int, end_ns: int) -> pl.DataFrame:
    start_dt = ns_to_utc_datetime(start_ns)
    end_dt = ns_to_utc_datetime(end_ns)
    lf = (
        pl.scan_parquet(path)
        .select(
            pl.col("DateTime_UTC")
            .str.strptime(pl.Datetime, strict=False)
            .dt.replace_time_zone("UTC")
            .alias("DateTime_UTC"),
            pl.col("Open").cast(pl.Float64),
            pl.col("High").cast(pl.Float64),
            pl.col("Low").cast(pl.Float64),
            pl.col("Close").cast(pl.Float64),
            pl.col("Volume").cast(pl.Int64),
        )
        .filter((pl.col("DateTime_UTC") >= start_dt) & (pl.col("DateTime_UTC") <= end_dt))
        .with_columns(
            pl.col("DateTime_UTC").set_sorted(),
            pl.col("DateTime_UTC").dt.convert_time_zone(ET_ZONE).dt.replace_time_zone(None).alias("DateTime_ET"),
        )
    )
    return collect_lazy_frame(lf, f"{path.name} overlap")


def resample_bars(df_1m: pl.DataFrame, freq: str) -> pl.DataFrame:
    return (
        df_1m.lazy()
        .with_columns(pl.col("DateTime_UTC").set_sorted())
        .group_by_dynamic("DateTime_UTC", every=freq, period=freq, closed="left", label="left")
        .agg(
            pl.col("Open").first().alias("Open"),
            pl.col("High").max().alias("High"),
            pl.col("Low").min().alias("Low"),
            pl.col("Close").last().alias("Close"),
            pl.col("Volume").sum().alias("Volume"),
        )
        .drop_nulls(["Open"])
        .with_columns(
            pl.col("DateTime_UTC").dt.convert_time_zone(ET_ZONE).dt.replace_time_zone(None).alias("DateTime_ET")
        )
        .collect(engine="streaming")
    )


def build_bars(df_1m: pl.DataFrame) -> dict[str, Bars]:
    bars_5m_df = resample_bars(df_1m, "5m")
    bars_15m_df = resample_bars(df_1m, "15m")
    bars_1h_df = resample_bars(df_1m, "1h")

    bars = {
        "5m": bars_from_frame(bars_5m_df, "DateTime_UTC"),
        "15m": bars_from_frame(bars_15m_df, "DateTime_UTC"),
        "1h": bars_from_frame(bars_1h_df, "DateTime_UTC"),
    }

    del bars_5m_df, bars_15m_df, bars_1h_df
    aggressive_gc()
    return bars


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return np.empty(0, dtype=float)
    idx = np.arange(values.size)
    starts = np.maximum(0, idx - window + 1)
    padded = np.empty(values.size + 1, dtype=float)
    padded[0] = 0.0
    padded[1:] = np.cumsum(values, dtype=float)
    sums = padded[idx + 1] - padded[starts]
    counts = idx - starts + 1
    return sums / counts


def find_swing_highs(bars: Bars, n: int = 5) -> np.ndarray:
    highs = bars.high
    length = len(highs)
    mask = np.zeros(length, dtype=bool)
    for i in range(n, length - n):
        if highs[i] > highs[i - n : i].max() and highs[i] > highs[i + 1 : i + n + 1].max():
            mask[i] = True
    return np.flatnonzero(mask)


def find_swing_lows(bars: Bars, n: int = 5) -> np.ndarray:
    lows = bars.low
    length = len(lows)
    mask = np.zeros(length, dtype=bool)
    for i in range(n, length - n):
        if lows[i] < lows[i - n : i].min() and lows[i] < lows[i + 1 : i + n + 1].min():
            mask[i] = True
    return np.flatnonzero(mask)


def find_draw_on_liquidity(
    bars: Bars,
    current_idx: int,
    lookback: int = 20,
    swing_n: int = 3,
) -> Optional[dict[str, float | str]]:
    start = max(0, current_idx - lookback)
    window = bars.slice(start, current_idx + 1)

    if len(window) < swing_n * 3:
        return None

    current_price = float(window.close[-1])
    effective_n = min(swing_n, max(1, len(window) // 3))

    swing_high_idx = find_swing_highs(window, n=effective_n)
    swing_low_idx = find_swing_lows(window, n=effective_n)

    old_highs = window.high[swing_high_idx]
    old_highs = old_highs[old_highs > current_price]

    old_lows = window.low[swing_low_idx]
    old_lows = old_lows[old_lows < current_price]

    best_high = float(old_highs.min()) if old_highs.size else None
    best_low = float(old_lows.max()) if old_lows.size else None

    if best_high is not None and best_low is not None:
        dist_high = best_high - current_price
        dist_low = current_price - best_low
        if dist_high <= dist_low:
            return {"direction": "bullish", "level": best_high}
        return {"direction": "bearish", "level": best_low}
    if best_high is not None:
        return {"direction": "bullish", "level": best_high}
    if best_low is not None:
        return {"direction": "bearish", "level": best_low}
    return None


def detect_stop_raid(
    bars_5m: Bars,
    direction: str,
    swing_n: int = 5,
    wait_for_close: bool = True,
) -> list[dict[str, float | int | str]]:
    raids: list[dict[str, float | int | str]] = []
    length = len(bars_5m)

    if direction == "bullish":
        swing_low_idx = find_swing_lows(bars_5m, n=swing_n)
        swing_positions = set(int(i) for i in swing_low_idx.tolist())
        active_lows: list[tuple[int, float]] = []

        for i in range(length):
            if i in swing_positions:
                active_lows.append((i, float(bars_5m.low[i])))

            if not active_lows:
                continue

            current_low = float(bars_5m.low[i])
            swept_level = None
            for pos, level in active_lows:
                if pos == i:
                    continue
                if current_low < level:
                    swept_level = level
                    break

            if swept_level is None:
                continue

            if wait_for_close:
                conf_open = float(bars_5m.open[i])
                conf_close = float(bars_5m.close[i])
                if conf_close > conf_open:
                    raids.append(
                        {
                            "sweep_idx_ns": int(bars_5m.ts_ns[i]),
                            "sweep_level": swept_level,
                            "confirmation_idx_ns": int(bars_5m.ts_ns[i]),
                            "confirmation_open": conf_open,
                            "confirmation_close": conf_close,
                            "direction": "bullish",
                        }
                    )
                    active_lows = [(p, l) for p, l in active_lows if l != swept_level]
                elif i + 1 < length:
                    next_open = float(bars_5m.open[i + 1])
                    next_close = float(bars_5m.close[i + 1])
                    if next_close > next_open:
                        raids.append(
                            {
                                "sweep_idx_ns": int(bars_5m.ts_ns[i]),
                                "sweep_level": swept_level,
                                "confirmation_idx_ns": int(bars_5m.ts_ns[i + 1]),
                                "confirmation_open": next_open,
                                "confirmation_close": next_close,
                                "direction": "bullish",
                            }
                        )
                        active_lows = [(p, l) for p, l in active_lows if l != swept_level]
            else:
                raids.append(
                    {
                        "sweep_idx_ns": int(bars_5m.ts_ns[i]),
                        "sweep_level": swept_level,
                        "confirmation_idx_ns": int(bars_5m.ts_ns[i]),
                        "confirmation_open": float(bars_5m.open[i]),
                        "confirmation_close": float(bars_5m.close[i]),
                        "direction": "bullish",
                    }
                )
                active_lows = [(p, l) for p, l in active_lows if l != swept_level]

    elif direction == "bearish":
        swing_high_idx = find_swing_highs(bars_5m, n=swing_n)
        swing_positions = set(int(i) for i in swing_high_idx.tolist())
        active_highs: list[tuple[int, float]] = []

        for i in range(length):
            if i in swing_positions:
                active_highs.append((i, float(bars_5m.high[i])))

            if not active_highs:
                continue

            current_high = float(bars_5m.high[i])
            swept_level = None
            for pos, level in active_highs:
                if pos == i:
                    continue
                if current_high > level:
                    swept_level = level
                    break

            if swept_level is None:
                continue

            if wait_for_close:
                conf_open = float(bars_5m.open[i])
                conf_close = float(bars_5m.close[i])
                if conf_close < conf_open:
                    raids.append(
                        {
                            "sweep_idx_ns": int(bars_5m.ts_ns[i]),
                            "sweep_level": swept_level,
                            "confirmation_idx_ns": int(bars_5m.ts_ns[i]),
                            "confirmation_open": conf_open,
                            "confirmation_close": conf_close,
                            "direction": "bearish",
                        }
                    )
                    active_highs = [(p, l) for p, l in active_highs if l != swept_level]
                elif i + 1 < length:
                    next_open = float(bars_5m.open[i + 1])
                    next_close = float(bars_5m.close[i + 1])
                    if next_close < next_open:
                        raids.append(
                            {
                                "sweep_idx_ns": int(bars_5m.ts_ns[i]),
                                "sweep_level": swept_level,
                                "confirmation_idx_ns": int(bars_5m.ts_ns[i + 1]),
                                "confirmation_open": next_open,
                                "confirmation_close": next_close,
                                "direction": "bearish",
                            }
                        )
                        active_highs = [(p, l) for p, l in active_highs if l != swept_level]
            else:
                raids.append(
                    {
                        "sweep_idx_ns": int(bars_5m.ts_ns[i]),
                        "sweep_level": swept_level,
                        "confirmation_idx_ns": int(bars_5m.ts_ns[i]),
                        "confirmation_open": float(bars_5m.open[i]),
                        "confirmation_close": float(bars_5m.close[i]),
                        "direction": "bearish",
                    }
                )
                active_highs = [(p, l) for p, l in active_highs if l != swept_level]

    return raids


def find_fvgs(bars: Bars, direction: str) -> list[dict[str, float | int]]:
    fvgs = []
    if len(bars) < 3:
        return fvgs
    for i in range(1, len(bars) - 1):
        if direction == "bullish":
            if bars.high[i - 1] < bars.low[i + 1]:
                top = float(bars.low[i + 1])
                bottom = float(bars.high[i - 1])
                fvgs.append({"idx_ns": int(bars.ts_ns[i]), "top": top, "bottom": bottom, "mid": (top + bottom) / 2.0})
        else:
            if bars.low[i - 1] > bars.high[i + 1]:
                top = float(bars.low[i - 1])
                bottom = float(bars.high[i + 1])
                fvgs.append({"idx_ns": int(bars.ts_ns[i]), "top": top, "bottom": bottom, "mid": (top + bottom) / 2.0})
    return fvgs


def find_fvgs_vectorized(bars: Bars, direction: str) -> dict[int, tuple[float, float, float]]:
    if len(bars) < 3:
        return {}

    highs = bars.high
    lows = bars.low

    if direction == "bullish":
        mask = highs[:-2] < lows[2:]
    else:
        mask = lows[:-2] > highs[2:]

    positions = np.where(mask)[0] + 1
    if positions.size == 0:
        return {}

    fvg_map: dict[int, tuple[float, float, float]] = {}
    if direction == "bullish":
        tops = lows[positions + 1]
        bottoms = highs[positions - 1]
    else:
        tops = lows[positions - 1]
        bottoms = highs[positions + 1]

    mids = (tops + bottoms) / 2.0
    for pos, top, bottom, mid in zip(positions, tops, bottoms, mids, strict=False):
        fvg_map[int(bars.ts_ns[pos])] = (float(top), float(bottom), float(mid))
    return fvg_map


def detect_displacement(
    bars: Bars,
    direction: str,
    atr_mult: float = 2.0,
    atr_period: int = 14,
) -> list[dict[str, float | int]]:
    opens = bars.open
    closes = bars.close
    highs = bars.high
    lows = bars.low

    body = np.abs(closes - opens)
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = rolling_mean(tr, atr_period)

    displacements: list[dict[str, float | int]] = []
    start = max(atr_period, 1)

    for i in range(start, len(bars)):
        if direction == "bullish" and closes[i] > opens[i]:
            if body[i] > atr_mult * atr[i - 1]:
                displacements.append(
                    {
                        "pos": i,
                        "idx_ns": int(bars.ts_ns[i]),
                        "open": float(opens[i]),
                        "close": float(closes[i]),
                        "high": float(highs[i]),
                        "low": float(lows[i]),
                        "body": float(body[i]),
                    }
                )
        elif direction == "bearish" and closes[i] < opens[i]:
            if body[i] > atr_mult * atr[i - 1]:
                displacements.append(
                    {
                        "pos": i,
                        "idx_ns": int(bars.ts_ns[i]),
                        "open": float(opens[i]),
                        "close": float(closes[i]),
                        "high": float(highs[i]),
                        "low": float(lows[i]),
                        "body": float(body[i]),
                    }
                )
    return displacements


def find_order_blocks(
    bars: Bars,
    direction: str,
    atr_mult: float = 2.0,
    atr_period: int = 14,
) -> list[dict[str, float | int]]:
    displacements = detect_displacement(bars, direction, atr_mult, atr_period)
    order_blocks: list[dict[str, float | int]] = []

    for disp in displacements:
        disp_pos = int(disp["pos"])
        for j in range(disp_pos - 1, max(disp_pos - 10, -1), -1):
            if direction == "bullish" and bars.close[j] < bars.open[j]:
                order_blocks.append(
                    {
                        "ob_pos": j,
                        "ob_idx_ns": int(bars.ts_ns[j]),
                        "ob_high": float(bars.high[j]),
                        "ob_low": float(bars.low[j]),
                        "ob_open": float(bars.open[j]),
                        "ob_close": float(bars.close[j]),
                        "displacement_idx_ns": int(disp["idx_ns"]),
                    }
                )
                break
            if direction == "bearish" and bars.close[j] > bars.open[j]:
                order_blocks.append(
                    {
                        "ob_pos": j,
                        "ob_idx_ns": int(bars.ts_ns[j]),
                        "ob_high": float(bars.high[j]),
                        "ob_low": float(bars.low[j]),
                        "ob_open": float(bars.open[j]),
                        "ob_close": float(bars.close[j]),
                        "displacement_idx_ns": int(disp["idx_ns"]),
                    }
                )
                break
    return order_blocks


def finalize_trade(position: Trade, exit_time_ns: int, exit_price: float, exit_reason: str, cost_per_trade: float) -> Trade:
    position.exit_time_ns = exit_time_ns
    position.exit_price = exit_price
    position.exit_reason = exit_reason

    if position.direction == "long":
        position.pnl_points = (position.exit_price - position.entry_price) - cost_per_trade
        raw_pnl = position.exit_price - position.entry_price
    else:
        position.pnl_points = (position.entry_price - position.exit_price) - cost_per_trade
        raw_pnl = position.entry_price - position.exit_price

    risk = abs(position.entry_price - position.stop_price)
    position.r_multiple = raw_pnl / risk if risk > 0 else 0.0
    return position


def run_base_model(
    entry_bars: Bars,
    bars_5m: Bars,
    bars_15m: Bars,
    bars_1h: Bars,
    max_stop: float = 12.0,
    cooldown_seconds: int = 300,
    cost_per_trade: float = 0.25,
) -> list[Trade]:
    trades: list[Trade] = []
    position: Optional[Trade] = None
    last_exit_time_ns: Optional[int] = None

    fvgs_bull_map = find_fvgs_vectorized(entry_bars, "bullish")
    fvgs_bear_map = find_fvgs_vectorized(entry_bars, "bearish")

    current_draw: Optional[dict[str, float | str]] = None
    active_raid: Optional[dict[str, float | int | str]] = None
    last_1h_pos = -1
    last_5m_pos = -1
    active_fvgs: list[tuple[float, float, float]] = []

    total_entry = len(entry_bars)
    report_interval = max(total_entry // 20, 1) if total_entry else 0
    cooldown_ns = cooldown_seconds * 1_000_000_000

    for i in range(total_entry):
        ts_ns = int(entry_bars.ts_ns[i])

        if report_interval and i % report_interval == 0:
            pct = i / total_entry * 100
            print(f"  Base model: {pct:.0f}% ({len(trades)} trades so far)")

        if position is not None:
            hit_sl = False
            hit_tp = False
            low_i = float(entry_bars.low[i])
            high_i = float(entry_bars.high[i])

            if position.direction == "long":
                hit_sl = low_i <= position.stop_price
                hit_tp = high_i >= position.target_price
            else:
                hit_sl = high_i >= position.stop_price
                hit_tp = low_i <= position.target_price

            if hit_sl and hit_tp:
                hit_tp = False

            if hit_sl or hit_tp:
                exit_price = position.stop_price if hit_sl else position.target_price
                exit_reason = "sl" if hit_sl else "tp"
                trades.append(finalize_trade(position, ts_ns, exit_price, exit_reason, cost_per_trade))
                last_exit_time_ns = ts_ns
                position = None
                active_raid = None
                active_fvgs = []
            continue

        if last_exit_time_ns is not None and ts_ns - last_exit_time_ns < cooldown_ns:
            continue

        h_pos_raw = int(np.searchsorted(bars_1h.ts_ns, ts_ns, side="right") - 1)
        if h_pos_raw >= 0 and h_pos_raw != last_1h_pos:
            last_1h_pos = h_pos_raw
            h_pos = h_pos_raw - 1
            if h_pos >= 10:
                draw = find_draw_on_liquidity(bars_1h, current_idx=h_pos, lookback=20, swing_n=3)
                if draw is None:
                    m15_pos_raw = int(np.searchsorted(bars_15m.ts_ns, ts_ns, side="right") - 1)
                    m15_pos = m15_pos_raw - 1
                    if m15_pos >= 10:
                        draw = find_draw_on_liquidity(bars_15m, current_idx=m15_pos, lookback=40, swing_n=3)
                current_draw = draw

        if current_draw is None:
            continue

        m5_pos_raw = int(np.searchsorted(bars_5m.ts_ns, ts_ns, side="right") - 1)
        if m5_pos_raw >= 0 and m5_pos_raw != last_5m_pos:
            last_5m_pos = m5_pos_raw
            m5_pos = m5_pos_raw - 1
            if m5_pos >= 20:
                window_start = max(0, m5_pos - 50)
                window = bars_5m.slice(window_start, m5_pos + 1)
                raids = detect_stop_raid(window, direction=str(current_draw["direction"]), swing_n=5, wait_for_close=True)
                if raids:
                    recent_cutoff = int(bars_5m.ts_ns[max(0, m5_pos - 1)])
                    recent_raids = [r for r in raids if int(r["confirmation_idx_ns"]) >= recent_cutoff]
                    if recent_raids:
                        active_raid = recent_raids[-1]
                        active_fvgs = []

        if active_raid is None:
            continue

        if ts_ns <= int(active_raid["confirmation_idx_ns"]):
            continue

        fvg_map = fvgs_bull_map if current_draw["direction"] == "bullish" else fvgs_bear_map

        if i >= 3:
            new_fvg = fvg_map.get(int(entry_bars.ts_ns[i - 2]))
            if new_fvg is not None:
                active_fvgs.append(new_fvg)
                if len(active_fvgs) > 20:
                    active_fvgs = active_fvgs[-20:]

        if not active_fvgs:
            continue

        entered = False
        entry_price = None
        entry_fvg_top = None
        entry_fvg_bottom = None
        used_fvg_idx = None

        low_i = float(entry_bars.low[i])
        high_i = float(entry_bars.high[i])

        for fi, (fvg_top, fvg_bottom, _fvg_mid) in enumerate(active_fvgs):
            if current_draw["direction"] == "bullish":
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

        remaining_fvgs: list[tuple[float, float, float]] = []
        for fi, (fvg_top, fvg_bottom, fvg_mid) in enumerate(active_fvgs):
            if fi == used_fvg_idx:
                continue
            if current_draw["direction"] == "bullish":
                if low_i > fvg_bottom:
                    remaining_fvgs.append((fvg_top, fvg_bottom, fvg_mid))
            else:
                if high_i < fvg_top:
                    remaining_fvgs.append((fvg_top, fvg_bottom, fvg_mid))
        active_fvgs = remaining_fvgs

        if not entered or entry_price is None or entry_fvg_top is None or entry_fvg_bottom is None:
            continue

        et_hour = int(entry_bars.et_hour[i]) if entry_bars.et_hour is not None else -1

        if current_draw["direction"] == "bullish":
            struct_stop = entry_fvg_bottom - 1.0
            stop = struct_stop if 0 < entry_price - struct_stop < max_stop else entry_price - max_stop
            risk = entry_price - stop
            if risk <= 0:
                continue
            target = entry_price + risk
            position = Trade(
                entry_time_ns=ts_ns,
                direction="long",
                entry_price=entry_price,
                stop_price=stop,
                target_price=target,
                variant="base",
                entry_hour_et=et_hour,
            )
        else:
            struct_stop = entry_fvg_top + 1.0
            stop = struct_stop if 0 < struct_stop - entry_price < max_stop else entry_price + max_stop
            risk = stop - entry_price
            if risk <= 0:
                continue
            target = entry_price - risk
            position = Trade(
                entry_time_ns=ts_ns,
                direction="short",
                entry_price=entry_price,
                stop_price=stop,
                target_price=target,
                variant="base",
                entry_hour_et=et_hour,
            )

        active_raid = None
        active_fvgs = []

    if position is not None:
        trades.append(finalize_trade(position, int(entry_bars.ts_ns[-1]), float(entry_bars.close[-1]), "timeout", cost_per_trade))

    return trades


def find_structural_target(
    bars_5m: Bars,
    direction: str,
    entry_price: float,
    current_5m_idx: int,
    swing_n: int = 5,
) -> Optional[float]:
    start = max(0, current_5m_idx - 50)
    window = bars_5m.slice(start, current_5m_idx)
    if len(window) < swing_n * 3:
        return None

    effective_n = min(swing_n, max(1, len(window) // 3))
    if direction == "bullish":
        highs = find_swing_highs(window, n=effective_n)
        if highs.size:
            targets = window.high[highs]
            targets = targets[targets > entry_price]
            if targets.size:
                return float(targets.min())
    else:
        lows = find_swing_lows(window, n=effective_n)
        if lows.size:
            targets = window.low[lows]
            targets = targets[targets < entry_price]
            if targets.size:
                return float(targets.max())
    return None


def run_higher_rr_model(
    entry_bars: Bars,
    bars_5m: Bars,
    bars_15m: Bars,
    bars_1h: Bars,
    cooldown_seconds: int = 300,
    atr_mult: float = 2.0,
    min_rr: float = 1.5,
    cost_per_trade: float = 0.25,
) -> list[Trade]:
    trades: list[Trade] = []
    position: Optional[Trade] = None
    last_exit_time_ns: Optional[int] = None

    fvgs_bull_map = find_fvgs_vectorized(entry_bars, "bullish")
    fvgs_bear_map = find_fvgs_vectorized(entry_bars, "bearish")

    current_draw: Optional[dict[str, float | str]] = None
    active_raid: Optional[dict[str, float | int | str]] = None
    active_raid_5m_pos = -1
    active_fvgs: list[tuple[float, float, float]] = []
    last_1h_pos = -1
    last_5m_pos = -1

    total_entry = len(entry_bars)
    report_interval = max(total_entry // 20, 1) if total_entry else 0
    cooldown_ns = cooldown_seconds * 1_000_000_000

    for i in range(total_entry):
        ts_ns = int(entry_bars.ts_ns[i])

        if report_interval and i % report_interval == 0:
            pct = i / total_entry * 100
            print(f"  Higher R:R: {pct:.0f}% ({len(trades)} trades so far)")

        if position is not None:
            hit_sl = False
            hit_tp = False
            low_i = float(entry_bars.low[i])
            high_i = float(entry_bars.high[i])

            if position.direction == "long":
                hit_sl = low_i <= position.stop_price
                hit_tp = high_i >= position.target_price
            else:
                hit_sl = high_i >= position.stop_price
                hit_tp = low_i <= position.target_price

            if hit_sl and hit_tp:
                hit_tp = False

            if hit_sl or hit_tp:
                exit_price = position.stop_price if hit_sl else position.target_price
                exit_reason = "sl" if hit_sl else "tp"
                trades.append(finalize_trade(position, ts_ns, exit_price, exit_reason, cost_per_trade))
                last_exit_time_ns = ts_ns
                position = None
                active_raid = None
                active_fvgs = []
            continue

        if last_exit_time_ns is not None and ts_ns - last_exit_time_ns < cooldown_ns:
            continue

        h_pos_raw = int(np.searchsorted(bars_1h.ts_ns, ts_ns, side="right") - 1)
        if h_pos_raw >= 0 and h_pos_raw != last_1h_pos:
            last_1h_pos = h_pos_raw
            h_pos = h_pos_raw - 1
            if h_pos >= 10:
                draw = find_draw_on_liquidity(bars_1h, current_idx=h_pos, lookback=20, swing_n=3)
                if draw is None:
                    m15_pos_raw = int(np.searchsorted(bars_15m.ts_ns, ts_ns, side="right") - 1)
                    m15_pos = m15_pos_raw - 1
                    if m15_pos >= 10:
                        draw = find_draw_on_liquidity(bars_15m, current_idx=m15_pos, lookback=40, swing_n=3)
                current_draw = draw

        if current_draw is None:
            continue

        m5_pos_raw = int(np.searchsorted(bars_5m.ts_ns, ts_ns, side="right") - 1)
        if m5_pos_raw >= 0 and m5_pos_raw != last_5m_pos:
            last_5m_pos = m5_pos_raw
            m5_pos = m5_pos_raw - 1
            if m5_pos >= 20:
                window_start = max(0, m5_pos - 50)
                window = bars_5m.slice(window_start, m5_pos + 1)
                raids = detect_stop_raid(window, direction=str(current_draw["direction"]), swing_n=5, wait_for_close=False)
                if raids:
                    recent_cutoff = int(bars_5m.ts_ns[max(0, m5_pos - 1)])
                    recent_raids = [r for r in raids if int(r["confirmation_idx_ns"]) >= recent_cutoff]
                    if recent_raids:
                        active_raid = recent_raids[-1]
                        active_raid_5m_pos = m5_pos
                        active_fvgs = []

        if active_raid is None:
            continue

        if m5_pos_raw - 1 - active_raid_5m_pos > 20:
            active_raid = None
            active_fvgs = []
            continue

        if ts_ns <= int(active_raid["confirmation_idx_ns"]):
            continue

        if i < 14:
            continue

        fvg_map = fvgs_bull_map if current_draw["direction"] == "bullish" else fvgs_bear_map
        if i >= 3:
            new_fvg = fvg_map.get(int(entry_bars.ts_ns[i - 2]))
            if new_fvg is not None:
                active_fvgs.append(new_fvg)
                if len(active_fvgs) > 5:
                    active_fvgs = active_fvgs[-5:]

        fvg_entry = None
        used_fvg_idx = None
        for fi, (fvg_top, fvg_bottom, _fvg_mid) in enumerate(active_fvgs):
            if current_draw["direction"] == "bullish" and entry_bars.low[i] <= fvg_top:
                fvg_entry = fvg_top
                used_fvg_idx = fi
                break
            if current_draw["direction"] == "bearish" and entry_bars.high[i] >= fvg_bottom:
                fvg_entry = fvg_bottom
                used_fvg_idx = fi
                break

        remaining_fvgs: list[tuple[float, float, float]] = []
        for fi, (fvg_top, fvg_bottom, fvg_mid) in enumerate(active_fvgs):
            if fi == used_fvg_idx:
                continue
            if current_draw["direction"] == "bullish":
                if entry_bars.low[i] > fvg_bottom:
                    remaining_fvgs.append((fvg_top, fvg_bottom, fvg_mid))
            else:
                if entry_bars.high[i] < fvg_top:
                    remaining_fvgs.append((fvg_top, fvg_bottom, fvg_mid))
        active_fvgs = remaining_fvgs

        ob_entry = None
        lookback_start = max(0, i - 30)
        lookback_window = entry_bars.slice(lookback_start, i)
        if len(lookback_window) > 14:
            obs = find_order_blocks(lookback_window, direction=str(current_draw["direction"]), atr_mult=atr_mult, atr_period=10)
            if obs:
                ob = obs[-1]
                if current_draw["direction"] == "bullish" and entry_bars.low[i] <= float(ob["ob_high"]):
                    ob_entry = float(ob["ob_high"])
                elif current_draw["direction"] == "bearish" and entry_bars.high[i] >= float(ob["ob_low"]):
                    ob_entry = float(ob["ob_low"])

        entry_price = ob_entry if ob_entry is not None else fvg_entry
        if entry_price is None:
            continue

        disps = (
            detect_displacement(lookback_window, direction=str(current_draw["direction"]), atr_mult=atr_mult, atr_period=10)
            if len(lookback_window) > 10
            else []
        )

        if current_draw["direction"] == "bullish":
            stop = float(disps[-1]["low"] - 1) if disps else entry_price - 12.0
            risk = entry_price - stop
        else:
            stop = float(disps[-1]["high"] + 1) if disps else entry_price + 12.0
            risk = stop - entry_price

        if risk <= 0:
            continue

        m5_pos_now = int(np.searchsorted(bars_5m.ts_ns, ts_ns, side="right") - 1)
        target = find_structural_target(bars_5m, str(current_draw["direction"]), entry_price, m5_pos_now) if m5_pos_now > 0 else None
        if target is None:
            target = float(current_draw["level"])

        if current_draw["direction"] == "bullish":
            target = min(target, entry_price + risk * 5.0)
            potential_rr = (target - entry_price) / risk if risk > 0 else 0.0
        else:
            target = max(target, entry_price - risk * 5.0)
            potential_rr = (entry_price - target) / risk if risk > 0 else 0.0

        if potential_rr < min_rr:
            continue

        et_hour = int(entry_bars.et_hour[i]) if entry_bars.et_hour is not None else -1
        position = Trade(
            entry_time_ns=ts_ns,
            direction="long" if current_draw["direction"] == "bullish" else "short",
            entry_price=entry_price,
            stop_price=stop,
            target_price=target,
            variant="higher_rr",
            entry_hour_et=et_hour,
        )
        active_raid = None
        active_fvgs = []

    if position is not None:
        trades.append(finalize_trade(position, int(entry_bars.ts_ns[-1]), float(entry_bars.close[-1]), "timeout", cost_per_trade))

    return trades


def compute_stats(trades: list[Trade]) -> dict[str, float | int]:
    if not trades:
        return {"total_trades": 0}

    pnls = np.array([t.pnl_points for t in trades], dtype=float)
    win_trades = [t for t in trades if t.pnl_points > 0]
    loss_trades = [t for t in trades if t.pnl_points <= 0]
    win_pnls = np.array([t.pnl_points for t in win_trades], dtype=float)
    loss_pnls = np.array([t.pnl_points for t in loss_trades], dtype=float)
    win_rs = np.array([t.r_multiple for t in win_trades], dtype=float)
    loss_rs = np.array([t.r_multiple for t in loss_trades], dtype=float)

    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_max

    gross_profit = float(win_pnls.sum()) if win_pnls.size else 0.0
    gross_loss = abs(float(loss_pnls.sum())) if loss_pnls.size else 0.0

    win_rate = len(win_trades) / len(trades)
    loss_rate = 1.0 - win_rate
    avg_win_r = float(np.mean(win_rs)) if win_rs.size else 0.0
    avg_loss_r = abs(float(np.mean(loss_rs))) if loss_rs.size else 0.0
    expectancy = win_rate * avg_win_r - loss_rate * avg_loss_r

    daily: dict[date, float] = {}
    for trade in trades:
        if trade.exit_time_ns is None:
            continue
        d = ns_to_utc_datetime(trade.exit_time_ns)
        if d is None:
            continue
        key = d.date()
        daily[key] = daily.get(key, 0.0) + trade.pnl_points

    if len(daily) > 1:
        d_min = min(daily)
        d_max = max(daily)
        values = []
        current = d_min
        while current <= d_max:
            values.append(daily.get(current, 0.0))
            current += timedelta(days=1)
        arr = np.array(values, dtype=float)
        sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(252)) if np.std(arr) > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "total_trades": len(trades),
        "wins": len(win_trades),
        "losses": len(loss_trades),
        "win_rate": win_rate * 100.0,
        "avg_pnl": float(np.mean(pnls)),
        "avg_r": float(np.mean(win_rs)) if win_rs.size else 0.0,
        "median_r": float(np.median(win_rs)) if win_rs.size else 0.0,
        "expectancy": expectancy,
        "sharpe": sharpe,
        "total_pnl": float(pnls.sum()),
        "max_drawdown": float(drawdowns.min()) if drawdowns.size else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "largest_win": float(pnls.max()),
        "largest_loss": float(pnls.min()),
        "avg_win": float(np.mean(win_pnls)) if win_pnls.size else 0.0,
        "avg_loss": float(np.mean(loss_pnls)) if loss_pnls.size else 0.0,
    }


def trades_to_df(trades: list[Trade]) -> pl.DataFrame:
    if not trades:
        return pl.DataFrame(
            schema={
                "entry_time": pl.Datetime("ns", "UTC"),
                "exit_time": pl.Datetime("ns", "UTC"),
                "direction": pl.String,
                "entry_price": pl.Float64,
                "exit_price": pl.Float64,
                "stop_price": pl.Float64,
                "target_price": pl.Float64,
                "pnl_points": pl.Float64,
                "r_multiple": pl.Float64,
                "exit_reason": pl.String,
                "variant": pl.String,
                "entry_hour_et": pl.Int16,
            }
        )

    raw = pl.DataFrame(
        {
            "entry_time_ns": [t.entry_time_ns for t in trades],
            "exit_time_ns": [t.exit_time_ns for t in trades],
            "direction": [t.direction for t in trades],
            "entry_price": [t.entry_price for t in trades],
            "exit_price": [t.exit_price for t in trades],
            "stop_price": [t.stop_price for t in trades],
            "target_price": [t.target_price for t in trades],
            "pnl_points": [t.pnl_points for t in trades],
            "r_multiple": [t.r_multiple for t in trades],
            "exit_reason": [t.exit_reason for t in trades],
            "variant": [t.variant for t in trades],
            "entry_hour_et": [t.entry_hour_et for t in trades],
        }
    )

    return (
        raw.with_columns(
            pl.col("entry_time_ns").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC").alias("entry_time"),
            pl.col("exit_time_ns").cast(pl.Datetime("ns")).dt.replace_time_zone("UTC").alias("exit_time"),
            pl.col("entry_hour_et").cast(pl.Int16),
        )
        .drop("entry_time_ns", "exit_time_ns")
        .select(
            "entry_time",
            "exit_time",
            "direction",
            "entry_price",
            "exit_price",
            "stop_price",
            "target_price",
            "pnl_points",
            "r_multiple",
            "exit_reason",
            "variant",
            "entry_hour_et",
        )
    )


def monthly_breakdown(trades_df: pl.DataFrame) -> pl.DataFrame:
    if trades_df.is_empty():
        return pl.DataFrame()

    return (
        trades_df.with_columns(pl.col("entry_time").dt.strftime("%Y-%m").alias("month"))
        .group_by("month")
        .agg(
            pl.len().alias("trades"),
            pl.col("pnl_points").sum().alias("total_pnl"),
            pl.col("r_multiple").mean().alias("avg_r"),
            ((pl.col("pnl_points") > 0).cast(pl.Float64).mean() * 100.0).alias("win_rate"),
            pl.col("pnl_points").max().alias("largest_win"),
            pl.col("pnl_points").min().alias("largest_loss"),
        )
        .sort("month")
    )


def generate_report(
    dataset: DatasetConfig,
    base_trades: list[Trade],
    higher_rr_trades: list[Trade],
    output_path: Path,
) -> None:
    base_stats = compute_stats(base_trades)
    hrr_stats = compute_stats(higher_rr_trades)
    base_df = trades_to_df(base_trades)
    hrr_df = trades_to_df(higher_rr_trades)

    csv_dir = output_path.parent
    if not base_df.is_empty():
        base_df.write_csv(csv_dir / f"trades_base_{dataset.csv_suffix}.csv")
    if not hrr_df.is_empty():
        hrr_df.write_csv(csv_dir / f"trades_higher_rr_{dataset.csv_suffix}.csv")

    all_trades = base_trades + higher_rr_trades
    if all_trades:
        date_min = ns_to_utc_datetime(min(t.entry_time_ns for t in all_trades))
        date_max = ns_to_utc_datetime(max(t.entry_time_ns for t in all_trades))
    else:
        date_min = None
        date_max = None

    point_value = 2
    init_capital = 10_000

    dark_base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7d8590", size=12),
    )

    def to_js(fig: go.Figure) -> tuple[str, str]:
        payload = json.loads(fig.to_json())
        return json.dumps(payload.get("data", [])), json.dumps(payload.get("layout", {}))

    def equity_fig(trades_list: list[Trade], title: str, line_color: str, fill_color: str) -> go.Figure:
        if not trades_list:
            return go.Figure()
        cum_pts = np.cumsum([t.pnl_points for t in trades_list])
        cum_pct = [round(v * point_value / init_capital * 100.0, 2) for v in cum_pts]
        times = [str(ns_to_utc_datetime(t.exit_time_ns)) for t in trades_list]
        fig = go.Figure(
            go.Scatter(
                x=times,
                y=cum_pct,
                mode="lines",
                name=title,
                line=dict(color=line_color, width=2),
                fill="tozeroy",
                fillcolor=fill_color,
            )
        )
        fig.update_layout(
            **dark_base,
            title=dict(text=title, font=dict(color="#e6edf3", size=14)),
            margin=dict(t=30, r=20, b=50, l=70),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(title="Cumulative Return (%)", ticksuffix="%", tickformat=".1f", gridcolor="#21262d"),
            shapes=[
                dict(
                    type="line",
                    x0=0,
                    x1=1,
                    xref="paper",
                    y0=0,
                    y1=0,
                    line=dict(color="#7d8590", width=1, dash="dot"),
                )
            ],
        )
        return fig

    def dd_fig(trades_list: list[Trade]) -> go.Figure:
        if not trades_list:
            return go.Figure()
        pnls = np.array([t.pnl_points for t in trades_list], dtype=float)
        cum = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cum)
        dd_pct = [round(v * point_value / init_capital * 100.0, 3) for v in (cum - running_max)]
        times = [str(ns_to_utc_datetime(t.exit_time_ns)) for t in trades_list]
        fig = go.Figure(
            go.Scatter(
                x=times,
                y=dd_pct,
                mode="lines",
                line=dict(color="#f85149", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(248,81,73,0.15)",
            )
        )
        fig.update_layout(
            **dark_base,
            margin=dict(t=10, r=20, b=40, l=60),
            xaxis=dict(gridcolor="#21262d"),
            yaxis=dict(title="Drawdown (%)", ticksuffix="%", tickformat=".2f", gridcolor="#21262d"),
            showlegend=False,
        )
        return fig

    def rdist_fig(rs_values: list[float], title: str, bar_color: str, line_color: str, xbins: dict[str, float]) -> go.Figure:
        fig = go.Figure(
            go.Histogram(
                x=rs_values,
                xbins=xbins,
                marker=dict(color=bar_color, line=dict(color=line_color, width=1)),
            )
        )
        fig.update_layout(
            **dark_base,
            title=dict(text=title, font=dict(color="#c9d1d9", size=13)),
            margin=dict(t=30, r=20, b=50, l=50),
            xaxis=dict(title="R-Multiple", gridcolor="#21262d"),
            yaxis=dict(title="Count", gridcolor="#21262d"),
            shapes=[
                dict(
                    type="line",
                    x0=0,
                    x1=0,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="#7d8590", width=1, dash="dot"),
                )
            ],
        )
        return fig

    def tod_fig(trades_list: list[Trade], title: str, pos_color: str, neg_color: str) -> go.Figure:
        if not trades_list:
            return go.Figure()
        hourly_pnl: dict[int, float] = {}
        for trade in trades_list:
            h = trade.entry_hour_et
            if h < 0:
                continue
            hourly_pnl[h] = hourly_pnl.get(h, 0.0) + trade.pnl_points
        hours = list(range(24))
        pnls = [hourly_pnl.get(h, 0.0) for h in hours]
        colors = [pos_color if pnl >= 0 else neg_color for pnl in pnls]
        fig = go.Figure(go.Bar(x=[f"{h:02d}:00" for h in hours], y=pnls, marker_color=colors))
        fig.update_layout(
            **dark_base,
            title=dict(text=title, font=dict(color="#c9d1d9", size=13)),
            margin=dict(t=30, r=20, b=50, l=50),
            xaxis=dict(title="Hour (ET)", gridcolor="#21262d"),
            yaxis=dict(title="Total PnL (pts)", gridcolor="#21262d"),
        )
        return fig

    base_eq_data, base_eq_layout = to_js(equity_fig(base_trades, "Base Model (1:1 R:R)", "#58a6ff", "rgba(88,166,255,0.08)"))
    hrr_eq_data, hrr_eq_layout = to_js(equity_fig(higher_rr_trades, "Higher R:R Variant", "#d29922", "rgba(210,153,34,0.08)"))
    base_dd_data, base_dd_layout = to_js(dd_fig(base_trades))
    hrr_dd_data, hrr_dd_layout = to_js(dd_fig(higher_rr_trades))

    base_rs = [round(t.r_multiple, 2) for t in base_trades] if base_trades else []
    hrr_rs = [round(t.r_multiple, 2) for t in higher_rr_trades] if higher_rr_trades else []
    base_rd_data, base_rd_layout = to_js(
        rdist_fig(base_rs, "Base Model", "rgba(88,166,255,0.6)", "#58a6ff", dict(start=-2, end=2, size=0.25))
    )
    hrr_rd_data, hrr_rd_layout = to_js(
        rdist_fig(
            hrr_rs,
            "Higher R:R",
            "rgba(210,153,34,0.6)",
            "#d29922",
            dict(start=-5, end=max(hrr_rs) + 1 if hrr_rs else 30, size=1),
        )
    )
    base_tod_data, base_tod_layout = to_js(tod_fig(base_trades, "Base Model", "rgba(63,185,80,0.7)", "rgba(248,81,73,0.7)"))
    hrr_tod_data, hrr_tod_layout = to_js(
        tod_fig(higher_rr_trades, "Higher R:R", "rgba(63,185,80,0.7)", "rgba(248,81,73,0.7)")
    )

    def pts_to_pct(pts: float) -> float:
        return pts * point_value / init_capital * 100.0

    def wr_by_dir(trades_list: list[Trade], direction: str) -> float:
        subset = [t for t in trades_list if t.direction == direction]
        if not subset:
            return 0.0
        return sum(1 for t in subset if t.pnl_points > 0) / len(subset) * 100.0

    base_longs = sum(1 for t in base_trades if t.direction == "long")
    base_shorts = sum(1 for t in base_trades if t.direction == "short")
    hrr_longs = sum(1 for t in higher_rr_trades if t.direction == "long")
    hrr_shorts = sum(1 for t in higher_rr_trades if t.direction == "short")

    base_wr_long = wr_by_dir(base_trades, "long")
    base_wr_short = wr_by_dir(base_trades, "short")
    hrr_wr_long = wr_by_dir(higher_rr_trades, "long")
    hrr_wr_short = wr_by_dir(higher_rr_trades, "short")

    base_tp = sum(1 for t in base_trades if t.exit_reason == "tp")
    base_sl = sum(1 for t in base_trades if t.exit_reason == "sl")
    hrr_tp = sum(1 for t in higher_rr_trades if t.exit_reason == "tp")
    hrr_sl = sum(1 for t in higher_rr_trades if t.exit_reason == "sl")

    def monthly_rows(df: pl.DataFrame) -> str:
        mb = monthly_breakdown(df)
        if mb.is_empty():
            return '<tr><td colspan="7" style="text-align:center;color:#7d8590">No trades</td></tr>'
        rows = []
        for row in mb.iter_rows(named=True):
            pnl_class = "green" if row["total_pnl"] > 0 else "red" if row["total_pnl"] < 0 else ""
            rows.append(
                f"""<tr>
                <td class="highlight">{row["month"]}</td>
                <td>{int(row["trades"])}</td>
                <td class="{pnl_class} highlight">{row["total_pnl"]:+.1f}</td>
                <td>{row["avg_r"]:.2f}</td>
                <td>{row["win_rate"]:.1f}%</td>
                <td class="green">{row["largest_win"]:.1f}</td>
                <td class="red">{row["largest_loss"]:.1f}</td></tr>"""
            )
        return "".join(rows)

    def trade_rows(df: pl.DataFrame, max_rows: int = 200) -> str:
        if df.is_empty():
            return '<tr><td colspan="10" style="text-align:center;color:#7d8590">No trades</td></tr>'
        rows = []
        for row in df.tail(max_rows).iter_rows(named=True):
            dir_tag = "long" if row["direction"] == "long" else "short"
            pnl_class = "green" if row["pnl_points"] > 0 else "red"
            entry_str = format_trade_time(row["entry_time"])
            exit_str = format_trade_time(row["exit_time"])
            rows.append(
                f"""<tr>
                <td>{entry_str}</td><td>{exit_str}</td>
                <td><span class="tag {dir_tag}">{row["direction"].upper()}</span></td>
                <td>{row["entry_price"]:.2f}</td><td>{row["exit_price"]:.2f}</td>
                <td>{row["stop_price"]:.2f}</td><td>{row["target_price"]:.2f}</td>
                <td class="{pnl_class} highlight">{row["pnl_points"]:+.2f}</td>
                <td class="{pnl_class}">{row["r_multiple"]:+.2f}R</td>
                <td>{row["exit_reason"].upper()}</td></tr>"""
            )
        return "".join(rows)

    def pf(stats: dict[str, float | int]) -> str:
        total_trades = int(stats.get("total_trades", 0))
        if total_trades == 0:
            return "N/A"
        pf_value = float(stats["profit_factor"])
        return f"{pf_value:.2f}" if pf_value != float("inf") else "&infin;"

    def val_class(val: float, positive_good: bool = True) -> str:
        if val > 0:
            return "positive" if positive_good else "negative"
        if val < 0:
            return "negative" if positive_good else "positive"
        return ""

    base_monthly_rows = monthly_rows(base_df)
    hrr_monthly_rows = monthly_rows(hrr_df)
    base_trade_rows = trade_rows(base_df)
    hrr_trade_rows = trade_rows(hrr_df)
    base_showing = min(base_df.height, 200)
    hrr_showing = min(hrr_df.height, 200)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cameron's Scalping Model &mdash; {dataset.label}</title>
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
    .finding-box.warn {{ border-left-color: #d29922; }}
    .finding-box.info {{ border-left-color: #58a6ff; }}
    .finding-box p {{ margin: 0; font-size: 14px; }}
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
    .method-card h4 {{ color: #e6edf3; font-size: 15px; margin-bottom: 8px; }}
    .method-card p {{ font-size: 13px; color: #7d8590; margin-bottom: 0; }}
    .method-card .acc {{ font-size: 20px; font-weight: 700; margin-top: 8px; }}
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
    .trade-log-wrap table {{ margin-bottom: 0; }}
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
    <div class="subtitle">{dataset.label} &mdash; {str(date_min)[:10] if date_min else "N/A"} to {str(date_max)[:10] if date_max else "N/A"}</div>
</div>

<div class="stats-grid">
    <div class="stat-card">
        <div class="label">Base Model Trades</div>
        <div class="value">{int(base_stats.get("total_trades", 0)):,}</div>
        <div class="detail">{base_longs} Long / {base_shorts} Short</div>
    </div>
    <div class="stat-card">
        <div class="label">Base Win Rate</div>
        <div class="value positive">{float(base_stats.get("win_rate", 0)):.1f}%</div>
        <div class="detail">{int(base_stats.get("wins", 0)):,} / {int(base_stats.get("total_trades", 0)):,}</div>
    </div>
    <div class="stat-card">
        <div class="label">Base Total P&amp;L</div>
        <div class="value {val_class(float(base_stats.get("total_pnl", 0)))}">{float(base_stats.get("total_pnl", 0)):+,.1f} pts</div>
        <div class="detail">PF: {pf(base_stats)}</div>
    </div>
    <div class="stat-card">
        <div class="label">Higher R:R Trades</div>
        <div class="value">{int(hrr_stats.get("total_trades", 0)):,}</div>
        <div class="detail">{hrr_longs} Long / {hrr_shorts} Short</div>
    </div>
    <div class="stat-card">
        <div class="label">Higher R:R Win Rate</div>
        <div class="value neutral">{float(hrr_stats.get("win_rate", 0)):.1f}%</div>
        <div class="detail">Avg R (winners): {float(hrr_stats.get("avg_r", 0)):.2f}</div>
    </div>
    <div class="stat-card">
        <div class="label">Higher R:R Total P&amp;L</div>
        <div class="value {val_class(float(hrr_stats.get("total_pnl", 0)))}">{float(hrr_stats.get("total_pnl", 0)):+,.1f} pts</div>
        <div class="detail">PF: {pf(hrr_stats)}</div>
    </div>
</div>

<div class="section">
    <h2>Models Under Test</h2>
    <div class="method-grid">
        <div class="method-card">
            <h4>Base Model (1:1 R:R)</h4>
            <p>Wait for 5-minute confirmation candle close after a stop raid opposite the draw on liquidity. Enter on 30-second fair value gap retracement. Fixed 12-point stop loss, 1:1 take profit.</p>
            <div class="acc green">{float(base_stats.get("win_rate", 0)):.1f}% <span style="font-size:13px;font-weight:400;color:#7d8590">win rate</span></div>
        </div>
        <div class="method-card">
            <h4>Higher R:R Variant</h4>
            <p>Skip the 5-minute close and trigger on the sweep itself. Enter on displacement order blocks or FVGs on the 30-second chart. Stop at displacement swing point, target structural levels (min 1.5R).</p>
            <div class="acc yellow">{float(hrr_stats.get("avg_r", 0)):.2f}R <span style="font-size:13px;font-weight:400;color:#7d8590">avg R (winners)</span></div>
        </div>
    </div>
</div>

<div class="section">
    <h2>Equity Curves</h2>
    <div class="chart-container tall" id="equity-base"></div>
    <div class="chart-container tall" id="equity-hrr"></div>
</div>

<div class="section">
    <h2>Drawdown Analysis</h2>
    <div class="two-col">
        <div>
            <h3>Base Model</h3>
            <div class="chart-container" id="dd-base" style="height:250px"></div>
        </div>
        <div>
            <h3>Higher R:R</h3>
            <div class="chart-container" id="dd-hrr" style="height:250px"></div>
        </div>
    </div>
</div>

<div class="section">
    <h2>R-Multiple Distribution</h2>
    <div class="two-col">
        <div class="chart-container" id="rdist-base" style="height:300px"></div>
        <div class="chart-container" id="rdist-hrr" style="height:300px"></div>
    </div>
</div>

<div class="section">
    <h2>Profitability by Time of Day</h2>
    <div class="two-col">
        <div class="chart-container" id="tod-base" style="height:300px"></div>
        <div class="chart-container" id="tod-hrr" style="height:300px"></div>
    </div>
</div>

<div class="section">
    <h2>Monthly Breakdown</h2>
    <div class="two-col">
        <div>
            <h3>Base Model</h3>
            <table>
                <thead><tr><th>Month</th><th>Trades</th><th>P&amp;L</th><th>Avg R</th><th>Win%</th><th>Best</th><th>Worst</th></tr></thead>
                <tbody>{base_monthly_rows}</tbody>
            </table>
        </div>
        <div>
            <h3>Higher R:R</h3>
            <table>
                <thead><tr><th>Month</th><th>Trades</th><th>P&amp;L</th><th>Avg R</th><th>Win%</th><th>Best</th><th>Worst</th></tr></thead>
                <tbody>{hrr_monthly_rows}</tbody>
            </table>
        </div>
    </div>
</div>

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
                    <tr><td>Trades</td><td>{base_longs}</td><td>{base_shorts}</td><td>{hrr_longs}</td><td>{hrr_shorts}</td></tr>
                    <tr><td>Win Rate</td><td class="{'green' if base_wr_long > 50 else 'red'}">{base_wr_long:.1f}%</td><td class="{'green' if base_wr_short > 50 else 'red'}">{base_wr_short:.1f}%</td><td class="{'green' if hrr_wr_long > 50 else 'yellow'}">{hrr_wr_long:.1f}%</td><td class="{'green' if hrr_wr_short > 50 else 'yellow'}">{hrr_wr_short:.1f}%</td></tr>
                    <tr><td>Exit: TP / SL</td><td colspan="2" style="text-align:center">{base_tp:,} / {base_sl:,}</td><td colspan="2" style="text-align:center">{hrr_tp:,} / {hrr_sl:,}</td></tr>
                </tbody>
            </table>
        </div>
    </div>
</div>

<div class="section">
    <h2>Head-to-Head Comparison</h2>
    <table>
        <thead><tr><th>Metric</th><th>Base Model</th><th>Higher R:R</th></tr></thead>
        <tbody>
            <tr><td>Total Trades</td><td class="highlight">{int(base_stats.get("total_trades", 0)):,}</td><td class="highlight">{int(hrr_stats.get("total_trades", 0)):,}</td></tr>
            <tr><td>Win Rate</td><td class="green">{float(base_stats.get("win_rate", 0)):.1f}%</td><td class="yellow">{float(hrr_stats.get("win_rate", 0)):.1f}%</td></tr>
            <tr><td>Avg R (Winners)</td><td>{float(base_stats.get("avg_r", 0)):.2f}</td><td class="green">{float(hrr_stats.get("avg_r", 0)):.2f}</td></tr>
            <tr><td>Median R (Winners)</td><td>{float(base_stats.get("median_r", 0)):.2f}</td><td class="yellow">{float(hrr_stats.get("median_r", 0)):.2f}</td></tr>
            <tr><td>Expectancy (R/trade)</td><td class="{val_class(float(base_stats.get("expectancy", 0)))}">{float(base_stats.get("expectancy", 0)):+.3f}R</td><td class="{val_class(float(hrr_stats.get("expectancy", 0)))}">{float(hrr_stats.get("expectancy", 0)):+.3f}R</td></tr>
            <tr><td>Sharpe (ann.)</td><td class="{val_class(float(base_stats.get("sharpe", 0)))}">{float(base_stats.get("sharpe", 0)):.2f}</td><td class="{val_class(float(hrr_stats.get("sharpe", 0)))}">{float(hrr_stats.get("sharpe", 0)):.2f}</td></tr>
            <tr><td>Total P&amp;L</td><td class="{val_class(float(base_stats.get("total_pnl", 0)))}">{float(base_stats.get("total_pnl", 0)):+,.1f} pts</td><td class="{val_class(float(hrr_stats.get("total_pnl", 0)))}">{float(hrr_stats.get("total_pnl", 0)):+,.1f} pts</td></tr>
            <tr><td>Profit Factor</td><td>{pf(base_stats)}</td><td class="green">{pf(hrr_stats)}</td></tr>
            <tr><td>Max Drawdown</td><td class="red">{pts_to_pct(float(base_stats.get("max_drawdown", 0))):.2f}%</td><td class="red">{pts_to_pct(float(hrr_stats.get("max_drawdown", 0))):.2f}%</td></tr>
            <tr><td>Largest Win</td><td class="green">{float(base_stats.get("largest_win", 0)):.1f} pts</td><td class="green">{float(hrr_stats.get("largest_win", 0)):.1f} pts</td></tr>
            <tr><td>Largest Loss</td><td class="red">{float(base_stats.get("largest_loss", 0)):.1f} pts</td><td class="red">{float(hrr_stats.get("largest_loss", 0)):.1f} pts</td></tr>
            <tr><td>Avg Win</td><td class="green">{float(base_stats.get("avg_win", 0)):.2f} pts</td><td class="green">{float(hrr_stats.get("avg_win", 0)):.2f} pts</td></tr>
            <tr><td>Avg Loss</td><td class="red">{float(base_stats.get("avg_loss", 0)):.2f} pts</td><td class="red">{float(hrr_stats.get("avg_loss", 0)):.2f} pts</td></tr>
        </tbody>
    </table>
</div>

<div class="section">
    <h2>Trade Log &mdash; Base Model (last {base_showing} of {base_df.height:,})</h2>
    <div class="trade-log-wrap">
        <table>
            <thead><tr><th>Entry</th><th>Exit</th><th>Dir</th><th>Entry$</th><th>Exit$</th><th>Stop</th><th>Target</th><th>P&amp;L</th><th>R</th><th>Exit</th></tr></thead>
            <tbody>{base_trade_rows}</tbody>
        </table>
    </div>
</div>

<div class="section">
    <h2>Trade Log &mdash; Higher R:R (last {hrr_showing} of {hrr_df.height:,})</h2>
    <div class="trade-log-wrap">
        <table>
            <thead><tr><th>Entry</th><th>Exit</th><th>Dir</th><th>Entry$</th><th>Exit$</th><th>Stop</th><th>Target</th><th>P&amp;L</th><th>R</th><th>Exit</th></tr></thead>
            <tbody>{hrr_trade_rows}</tbody>
        </table>
    </div>
</div>

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
</html>"""

    output_path.write_text(html, encoding="utf-8")
    print(f"Report saved to {output_path}")
    if not base_df.is_empty():
        print(f"Base trades CSV: {csv_dir / f'trades_base_{dataset.csv_suffix}.csv'}")
    if not hrr_df.is_empty():
        print(f"Higher R:R trades CSV: {csv_dir / f'trades_higher_rr_{dataset.csv_suffix}.csv'}")


def print_summary(label: str, trades_list: list[Trade]) -> None:
    stats = compute_stats(trades_list)
    print(f"\n{'=' * 40}")
    print(f"  {label}")
    print(f"{'=' * 40}")
    if int(stats.get("total_trades", 0)) == 0:
        print("  No trades generated.")
        return
    pf_value = float(stats["profit_factor"])
    pf_text = f"{pf_value:.2f}" if pf_value != float("inf") else "inf"
    print(f"  Trades:        {int(stats['total_trades'])}")
    print(f"  Win Rate:      {float(stats['win_rate']):.1f}%")
    print(f"  Total P&L:     {float(stats['total_pnl']):.1f} pts")
    print(f"  Avg R:         {float(stats['avg_r']):.2f}")
    print(f"  Median R:      {float(stats['median_r']):.2f}")
    print(f"  Profit Factor: {pf_text}")
    print(f"  Max Drawdown:  {float(stats['max_drawdown']):.1f} pts")
    print(f"  Largest Win:   {float(stats['largest_win']):.1f} pts")
    print(f"  Largest Loss:  {float(stats['largest_loss']):.1f} pts")


def run_dataset(config: DatasetConfig) -> None:
    print("\n" + "=" * 60)
    print(config.label)
    print("=" * 60)

    if not config.tick_file.exists():
        print(f"  Skipping missing file: {config.tick_file}")
        return

    print("\nLoading entry bars...")
    entry_bars = load_entry_bars(config)
    if len(entry_bars) == 0:
        print("  No entry bars produced.")
        return

    print(f"  Entry bars: {len(entry_bars):,}")
    print(f"  Tick range: {ns_to_utc_datetime(entry_bars.start_ns)} to {ns_to_utc_datetime(entry_bars.end_ns)}")

    print("\nLoading overlapping 1-minute bars...")
    bars_1m_df = load_1m_bars(ONE_MINUTE_FILE, entry_bars.start_ns, entry_bars.end_ns)
    if bars_1m_df.is_empty():
        print("  No overlapping 1-minute bars.")
        return

    one_minute_ns = bars_1m_df.get_column("DateTime_UTC").cast(pl.Datetime("ns")).cast(pl.Int64).to_numpy()
    overlap_start_ns = max(entry_bars.start_ns, int(one_minute_ns[0]))
    overlap_end_ns = min(entry_bars.end_ns, int(one_minute_ns[-1]))

    entry_bars = entry_bars.range_slice(overlap_start_ns, overlap_end_ns)
    bars_1m_df = bars_1m_df.filter(
        (pl.col("DateTime_UTC") >= ns_to_utc_datetime(overlap_start_ns))
        & (pl.col("DateTime_UTC") <= ns_to_utc_datetime(overlap_end_ns))
    )

    print(f"  Overlap range: {ns_to_utc_datetime(overlap_start_ns)} to {ns_to_utc_datetime(overlap_end_ns)}")
    print(f"  Entry bars after overlap trim: {len(entry_bars):,}")
    print(f"  1-minute bars: {bars_1m_df.height:,}")

    print("\nBuilding HTF bars from 1-minute data...")
    bars = build_bars(bars_1m_df)
    print(f"  5m bars:  {len(bars['5m']):,}")
    print(f"  15m bars: {len(bars['15m']):,}")
    print(f"  1h bars:  {len(bars['1h']):,}")

    del bars_1m_df, one_minute_ns
    aggressive_gc()

    print("\n" + "-" * 60)
    print("Running Base Model...")
    print("-" * 60)
    base_trades = run_base_model(entry_bars, bars["5m"], bars["15m"], bars["1h"])
    print(f"  Completed: {len(base_trades)} trades")

    print("\n" + "-" * 60)
    print("Running Higher R:R Model...")
    print("-" * 60)
    hrr_trades = run_higher_rr_model(entry_bars, bars["5m"], bars["15m"], bars["1h"])
    print(f"  Completed: {len(hrr_trades)} trades")

    report_path = REPORT_DIR / config.report_file
    print("\nGenerating report...")
    generate_report(config, base_trades, hrr_trades, report_path)

    print_summary(f"{config.label} — Base Model", base_trades)
    print_summary(f"{config.label} — Higher R:R", hrr_trades)

    del entry_bars, bars, base_trades, hrr_trades
    aggressive_gc()


def main() -> None:
    print("=" * 60)
    print("Cameron's Scalping Model — Polars Backtest")
    print("=" * 60)
    run_dataset(IN_SAMPLE_CONFIG)
    run_dataset(OUT_OF_SAMPLE_CONFIG)


if __name__ == "__main__":
    main()
