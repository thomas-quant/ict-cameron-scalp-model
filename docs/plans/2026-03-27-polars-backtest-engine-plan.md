# Polars Backtest Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the pandas-based backtest engine with a polars-native engine that runs the original tick dataset first and the merged sanitized tick dataset as out-of-sample data, while aggressively minimizing peak memory usage.

**Architecture:** Keep explicit loaders for the two tick schemas, normalize both to one internal bar contract, use polars for all table operations, and use primitive arrays only for the path-dependent trade execution loop. Truncate the merged out-of-sample run to the overlapping range available in the 1-minute bars.

**Tech Stack:** Python 3, polars, numpy, plotly, pytest

---

### Task 1: Add Failing Tests For Explicit Tick Source Handling

**Files:**
- Modify: `test_backtest.py`
- Modify: `backtest.py`

**Step 1: Write the failing test**

Add tests that expect:

- original ticks to resample from `ts_event`, `price`, `size`
- merged ticks to resample from `ts_event`, `price_ticks`, `size`
- merged ticks to normalize `price_ticks / 4.0`

```python
def test_resample_original_ticks_30s():
    ...

def test_resample_merged_ticks_30s():
    ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_backtest.py -k "original_ticks or merged_ticks" -v`
Expected: FAIL because the explicit polars loaders do not exist yet.

**Step 3: Write minimal implementation**

Add placeholder polars-oriented loader/resampler function names in `backtest.py` so the tests can target the new API.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_backtest.py -k "original_ticks or merged_ticks" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add test_backtest.py backtest.py
git commit -m "test: cover explicit original and merged tick loaders"
```

### Task 2: Port Tick Loading And 30-Second Bar Building To Polars

**Files:**
- Modify: `backtest.py`
- Modify: `test_backtest.py`

**Step 1: Write the failing test**

Add tests that assert the 30-second output bar schema is:

- `DateTime_UTC`
- `DateTime_ET`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`

and that the resampled values match the current pandas behavior.

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_backtest.py -k "resample_ticks or merged_ticks" -v`
Expected: FAIL on missing columns or mismatched values.

**Step 3: Write minimal implementation**

Implement:

- `load_original_ticks_pl(...)`
- `load_merged_ticks_pl(...)`
- `build_30s_bars_from_original_ticks(...)`
- `build_30s_bars_from_merged_ticks(...)`

Rules:

- use polars column projection
- normalize merged `price_ticks` to float price
- build ET timestamps only after UTC timestamps are finalized
- drop raw tick frames immediately after 30-second bars are materialized

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_backtest.py -k "resample_ticks or merged_ticks" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add test_backtest.py backtest.py
git commit -m "feat: build 30s entry bars with polars"
```

### Task 3: Port 1-Minute And Higher-Timeframe Bar Building To Polars

**Files:**
- Modify: `backtest.py`
- Modify: `test_backtest.py`

**Step 1: Write the failing test**

Add tests for:

- loading 1-minute bars without pandas
- resampling 1-minute bars into 5-minute, 15-minute, and 1-hour bars
- preserving OHLCV semantics

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_backtest.py -k "1m or htf or resample_bars" -v`
Expected: FAIL because the polars resampling path is missing.

**Step 3: Write minimal implementation**

Implement:

- `load_1m_bars_pl(...)`
- `resample_bars_pl(...)`
- `build_bars_pl(...)`

Ensure:

- only required columns are loaded
- bar outputs follow one shared schema
- intermediate frames are released once derived bars are built

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_backtest.py -k "1m or htf or resample_bars" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add test_backtest.py backtest.py
git commit -m "feat: port higher timeframe bar pipeline to polars"
```

### Task 4: Add Overlap Truncation For Out-Of-Sample Runs

**Files:**
- Modify: `backtest.py`
- Modify: `test_backtest.py`

**Step 1: Write the failing test**

Add a test that feeds merged ticks extending past the final 1-minute timestamp and expects the tick input to be truncated before bar construction.

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_backtest.py -k "overlap or out_of_sample or oos" -v`
Expected: FAIL because the truncation logic does not exist yet.

**Step 3: Write minimal implementation**

Add a helper like:

```python
def truncate_ticks_to_1m_overlap(...)
```

The helper must:

- compute the inclusive overlap range from the 1-minute bars
- filter merged ticks before 30-second aggregation
- avoid keeping both pre- and post-truncation frames alive

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_backtest.py -k "overlap or out_of_sample or oos" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add test_backtest.py backtest.py
git commit -m "feat: truncate out-of-sample ticks to 1m overlap"
```

### Task 5: Port Strategy Helper Functions Off Pandas

**Files:**
- Modify: `backtest.py`
- Modify: `test_backtest.py`

**Step 1: Write the failing test**

Convert existing helper tests to construct polars-oriented inputs and keep the same expected behaviors for:

- swing highs
- swing lows
- draw on liquidity
- stop raids
- FVGs
- displacement
- order blocks

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_backtest.py -k "swing or draw or raid or fvg or displacement or order_blocks" -v`
Expected: FAIL because helpers still depend on pandas structures.

**Step 3: Write minimal implementation**

Port helper internals so they operate on:

- polars DataFrames where columnar access is natural
- numpy arrays where rolling or indexed logic is simpler

Preserve existing strategy semantics exactly.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_backtest.py -k "swing or draw or raid or fvg or displacement or order_blocks" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add test_backtest.py backtest.py
git commit -m "refactor: port strategy helpers to polars-native inputs"
```

### Task 6: Port Base Model Execution Off Pandas

**Files:**
- Modify: `backtest.py`
- Modify: `test_backtest.py`

**Step 1: Write the failing test**

Add a narrow base-model smoke test that runs the execution loop on a small deterministic dataset and asserts:

- trades open at the expected timestamps
- stop and target precedence remains unchanged
- trade list output shape remains stable

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_backtest.py -k "base_model" -v`
Expected: FAIL because the execution loop still assumes pandas structures.

**Step 3: Write minimal implementation**

Refactor `run_base_model(...)` to consume the finalized bar tables without pandas:

- extract only needed columns as arrays
- avoid repeated conversion inside loops
- free temporary per-model structures after use

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_backtest.py -k "base_model" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add test_backtest.py backtest.py
git commit -m "refactor: port base model execution off pandas"
```

### Task 7: Port Higher-R Model Execution Off Pandas

**Files:**
- Modify: `backtest.py`
- Modify: `test_backtest.py`

**Step 1: Write the failing test**

Add a higher-R model smoke test with deterministic bars and expected trade behavior.

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_backtest.py -k "higher_rr or structural_target" -v`
Expected: FAIL because the higher-R path still assumes pandas structures.

**Step 3: Write minimal implementation**

Port:

- `find_structural_target(...)`
- `run_higher_rr_model(...)`

Use the same array-first execution pattern as the base model and release temporary structures aggressively.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_backtest.py -k "higher_rr or structural_target" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add test_backtest.py backtest.py
git commit -m "refactor: port higher-r model execution off pandas"
```

### Task 8: Port Stats, Trade Logs, And Reporting Inputs

**Files:**
- Modify: `backtest.py`

**Step 1: Write the failing test**

Add tests covering:

- trade list to polars DataFrame conversion
- monthly breakdown output
- basic stats output stability

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_backtest.py -k "stats or trades_to_df or monthly_breakdown" -v`
Expected: FAIL because reporting helpers still return pandas objects.

**Step 3: Write minimal implementation**

Port:

- `trades_to_df(...)`
- `monthly_breakdown(...)`
- any report helper logic that still assumes pandas rows

The HTML report may still feed Plotly from lists or dictionaries, but pandas must be removed.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_backtest.py -k "stats or trades_to_df or monthly_breakdown" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add test_backtest.py backtest.py
git commit -m "refactor: remove pandas from trade stats and reporting inputs"
```

### Task 9: Add Explicit In-Sample And Out-Of-Sample Entry Points

**Files:**
- Modify: `backtest.py`
- Modify: `readme.md`

**Step 1: Write the failing test**

Add tests for explicit run orchestration:

- original dataset run
- merged out-of-sample run capped to the 1-minute overlap

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_backtest.py -k "main or in_sample or out_of_sample or orchestration" -v`
Expected: FAIL because the orchestration entry points do not exist yet.

**Step 3: Write minimal implementation**

Add explicit orchestration helpers such as:

- `run_in_sample_backtest(...)`
- `run_out_of_sample_backtest(...)`
- `main()`

Ensure in-sample objects are released before the out-of-sample run begins.

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_backtest.py -k "main or in_sample or out_of_sample or orchestration" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add test_backtest.py backtest.py readme.md
git commit -m "feat: add explicit in-sample and out-of-sample runs"
```

### Task 10: Full Verification And Memory-Safety Pass

**Files:**
- Modify: `backtest.py`
- Modify: `test_backtest.py`
- Modify: `readme.md`

**Step 1: Write the failing test**

Add any missing regression tests discovered during the port, especially for:

- timezone handling
- merged schema normalization
- same-bar SL/TP precedence

**Step 2: Run test to verify it fails**

Run: `python3 -m pytest test_backtest.py -v`
Expected: FAIL on any remaining gaps.

**Step 3: Write minimal implementation**

Finish cleanup:

- remove the pandas import entirely
- remove dead compatibility code
- add explicit `del` statements where they materially reduce peak memory
- keep the code readable; do not scatter meaningless cleanup calls

**Step 4: Run test to verify it passes**

Run: `python3 -m pytest test_backtest.py -v`
Expected: PASS

Then run:

`python3 backtest.py`

Expected:

- in-sample run completes on `data/nq_ticks.parquet`
- out-of-sample run completes on the truncated overlap from `data/merged_nq_ticks.parquet`
- reports and CSV outputs are generated

**Step 5: Commit**

```bash
git add backtest.py test_backtest.py readme.md
git commit -m "refactor: migrate backtest engine from pandas to polars"
```
