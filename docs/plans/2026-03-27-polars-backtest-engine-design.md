# Polars Backtest Engine Design

**Date:** 2026-03-27

**Goal:** Replace the pandas-based backtest engine with a polars-native engine that can run the original tick dataset first and the sanitized merged tick dataset as out-of-sample data, while aggressively limiting peak memory usage.

## Scope

- Remove pandas from the backtest engine path.
- Support two explicit tick sources:
  - `data/nq_ticks.parquet` for the original in-sample run
  - `data/merged_nq_ticks.parquet` for the out-of-sample run
- Keep `data/nq_1m.parquet` as the higher-timeframe source.
- Truncate any out-of-sample tick run to the overlapping range covered by the 1-minute bars.

## Why Separate Paths

The original and merged tick files are not the same schema:

- Original ticks: `ts_event`, `price`, `size`
- Merged ticks: `ts_event`, `price_ticks`, `size`, `intra_ts_rank`, `side`

Trying to hide that behind auto-detection would add unnecessary ambiguity right now. Separate explicit loaders keep the data contract obvious and easier to test.

## Internal Data Contract

Both tick loaders must normalize into the same internal tick schema before bar-building:

- `ts_event`: UTC datetime
- `price`: float price
- `size`: integer quantity

Merged ticks compute `price = price_ticks / 4.0`.

All bar builders must emit the same bar schema:

- `DateTime_UTC`
- `DateTime_ET`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

## Architecture

Use polars for all tabular processing:

- load tick parquet files lazily where practical
- select only required columns
- normalize schema
- sort only when required for deterministic bar construction
- build 30-second entry bars
- load 1-minute bars
- derive 5-minute, 15-minute, and 1-hour bars
- produce trade logs and summary tables

Use primitive arrays for the sequential execution core:

- path-dependent trade management is not a good fit for pure DataFrame expressions
- the engine will extract only required columns into primitive arrays for the hot loop
- outputs return to polars DataFrames after execution

This keeps pandas out of the system without forcing the state machine into unreadable pseudo-vectorization.

## Memory Rules

The implementation must treat memory as a primary constraint.

- Never materialize full tick tables if a lazy scan or narrow projection is enough.
- Select only the columns needed for each stage.
- Drop intermediate frames as soon as the next stage is built.
- Avoid keeping both raw ticks and derived bars alive longer than necessary.
- Avoid duplicate sorted copies when one canonical ordering is enough.
- Truncate the merged tick run to the overlapping 1-minute date range before building bars.
- Build arrays for the execution loop only after bar tables are finalized.
- Free temporary arrays and frames aggressively between in-sample and out-of-sample runs.

## Execution Flow

1. Load and normalize original ticks.
2. Build 30-second entry bars from original ticks.
3. Load and filter 1-minute bars to the tick overlap.
4. Build higher-timeframe bars from 1-minute data.
5. Run the base and higher-R models.
6. Persist results.
7. Release no-longer-needed objects.
8. Load and normalize merged ticks.
9. Truncate merged ticks to the 1-minute bar overlap.
10. Build 30-second entry bars.
11. Reuse the same higher-timeframe pipeline.
12. Run the same models out-of-sample.
13. Persist and compare results.

## Testing

Tests must verify:

- original tick schema resamples correctly
- merged tick schema normalizes and resamples correctly
- out-of-sample runs stop at the 1-minute overlap limit
- key strategy helpers still behave the same after the port
- end-to-end runs on the original dataset still complete

## Non-Goals

- auto-detecting tick schemas
- supporting arbitrary input datasets
- rewriting the strategy logic into fully declarative polars expressions
