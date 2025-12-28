# Hybrid Table Segmentation Implementation Summary

**Date:** 2025-12-27  
**Commit:** `708c16b2622fa68dc493383941f5cb2c1c9a4b5e`  
**Previous Baseline:** `a2f1bb0f51787d27aede052b91590029b71355d8`

---

## Changes Implemented

### A) Hybrid Per-Axis Separator Selection (`table_segmentation.py`)

**Problem:**
- Valid vertical line separators (9 lines = 10 columns) were being discarded when horizontal lines were missing
- Old logic was all-or-nothing: if BOTH axes didn't meet thresholds, it fell back to projection for BOTH

**Solution:**
Implemented independent per-axis fallback logic:

```python
# Y-axis (rows): Use projection fallback only if horizontal lines not detected
if len(ys_lines) < min_rows:
    logger.info("Using projection fallback for ROW separators (no horizontal lines detected)")
    ys = fallback_projection_separators(binary, axis=0, min_separators=min_rows)
    y_method = "projection"
else:
    logger.info(f"Using line-based separators for ROWS ({len(ys_lines)} horizontal lines)")

# X-axis (cols): Use projection fallback only if vertical lines not detected
if len(xs_lines) < min_cols:
    logger.info("Using projection fallback for COL separators (no vertical lines detected)")
    xs = fallback_projection_separators(binary, axis=1, min_separators=min_cols)
    x_method = "projection"
else:
    logger.info(f"Using line-based separators for COLS ({len(xs_lines)} vertical lines)")
```

**Expected Logs:**
```
Line-based detection: 0 horizontal, 9 vertical lines
Using projection fallback for ROW separators (no horizontal lines detected)
Using line-based separators for COLS (9 vertical lines)
Final separators: rows_seps=12 (via projection), cols_seps=9 (via line-based)
Grid dimensions: 13 rows × 10 columns
```

---

### B) Frame Counter Verification (`tools/stream_scanner.py`)

**Status:** ✅ Already correct in current code

The frame counters were already properly implemented:
- `captured_frames`: incremented after every capture (line 159)
- `processed_frames`: incremented after frame validation (line 165)
- `segmented_frames`: incremented after successful segmentation (line 185)
- `ocr_attempted_frames`: incremented when new rows detected (line 202)
- `ocr_gated_frames`: incremented when OCR skipped due to gating (line 287)
- `events_emitted`: incremented after DB insert (line 369)

**Session Summary Output:**
```
================================================================================
SCANNER SESSION SUMMARY
================================================================================
Captured frames:     120
Processed frames:    120
Segmented frames:    115
OCR attempted:       45
OCR gated (skipped): 42
Events emitted:      3
================================================================================
```

**Note:** No "Total frames: 0" issue exists in the current code. The user may have been referencing an older version.

---

### C) Grid Miscalibration Warnings (`scanner/grid_segmenter.py`)

**Problem:**
- User had a 4-column grid with gap mapped to column 3, but last column occupied >50% of ROI width
- This indicated multiple physical columns were merged, breaking semantic mapping

**Solution:**
Added `_validate_grid_calibration()` method that checks:

1. **Wide Last Column Detection:**
   - If `num_cols <= 4` AND `last_col_width_ratio > 0.50`
   - Warns that columns are likely merged

2. **Semantic Mapping Validation:**
   - If `gap_index >= num_cols`
   - Warns that semantic mapping points to non-existent column

**Expected Warning Output:**
```
================================================================================
⚠️  GRID MISCALIBRATION WARNING
================================================================================
Grid has only 4 columns, but last column occupies 58.3% of ROI width.
This suggests multiple physical columns are merged into column 3.

Semantic mapping expects 'gap' at physical column 3,
but merged columns will break gap color gating.

RECOMMENDED FIX:
  1) Re-run: python tools/calibrate_grid.py --gap-col 7
     to generate a proper 10-column grid with gap at column 7

  OR

  2) Copy config/grid.example.10col.json to config/grid.json
     as a starting point and adjust separators
================================================================================
```

---

## Verification Commands

### Windows PowerShell
```powershell
cd C:\screenbot\screenbot
git pull origin main

# Verify commit
git rev-parse HEAD
# Expected: 708c16b2622fa68dc493383941f5cb2c1c9a4b5e

# Run with debug segmentation
$env:DEBUG_SEGMENTATION="1"
python tools/stream_scanner.py --max-seconds 10
```

### Windows Git Bash / Linux
```bash
cd /path/to/screenbot
git pull origin main

# Verify commit
git rev-parse HEAD
# Expected: 708c16b2622fa68dc493383941f5cb2c1c9a4b5e

# Run with debug segmentation
DEBUG_SEGMENTATION=1 python tools/stream_scanner.py --max-seconds 10
```

---

## Expected Behavior After Changes

### 1. Segmentation Logs
You should see logs like:
```
Line-based detection: 0 horizontal, 9 vertical lines
Using projection fallback for ROW separators (no horizontal lines detected)
Using line-based separators for COLS (9 vertical lines)
Final separators: rows_seps=12 (via projection), cols_seps=9 (via line-based)
Grid dimensions: 13 rows × 10 columns
```

**Key Point:** Even with 0 horizontal lines, vertical lines are preserved for column detection.

### 2. Grid Calibration Warning
If using an old 4-column `grid.json`:
```
⚠️  GRID MISCALIBRATION WARNING
Grid has only 4 columns, but last column occupies 58.3% of ROI width.
...
RECOMMENDED FIX:
  1) Re-run: python tools/calibrate_grid.py --gap-col 7
```

### 3. Frame Counter Summary
```
Captured frames:     120
Processed frames:    120
Segmented frames:    115
OCR attempted:       45
OCR gated (skipped): 42
Events emitted:      3
```

**Key Point:** `Processed frames` will be > 0 even if `Events emitted` is 0.

### 4. Semantic Mapping
With a proper 10-column grid:
```
✓ Grid calibration loaded: 10 columns
  Using calibrated column separators: 11
  Semantic mapping: {'time_hits': 0, 'symbol': 1, 'price': 2, 'gap': 7}
```

**Key Point:** Gap is now at physical column 7, matching the actual table structure.

---

## File Hashes (SHA256)

For verification that files match commit `708c16b`:

```
table_segmentation.py:     406bfedb105668709b561505cd688fc50ced2c92cdf4fa7b24eea624970ee9d0
scanner/grid_segmenter.py: f9281b2af695d1cc29ac0428ad16609dd75dc79a290fca916fee13de43cda920
tools/stream_scanner.py:   07927fd17d43d612ad7ea2475955574c30fa652fae5df01b5d17a5de5e3e9fc4
```

**PowerShell verification:**
```powershell
Get-FileHash table_segmentation.py -Algorithm SHA256
Get-FileHash scanner/grid_segmenter.py -Algorithm SHA256
Get-FileHash tools/stream_scanner.py -Algorithm SHA256
```

**Git Bash / Linux verification:**
```bash
sha256sum table_segmentation.py scanner/grid_segmenter.py tools/stream_scanner.py
```

---

## Troubleshooting

### If still seeing "0 events":

1. **Check grid calibration:**
   ```bash
   cat config/grid.json | grep -E "(num_columns|semantic_to_index)"
   ```
   Expected: `"num_columns": 10` and `"gap": 7` in semantic_to_index

2. **Check segmentation output:**
   - Look in `debug/` folder for `overlay_indexed.png`
   - Should show 10 columns labeled col0-col9
   - Confirm Gap(%) visual column aligns with col7

3. **Check OCR gating:**
   - If `OCR_ONLY_IF_GAP_GREEN=True`, verify gap cells are actually green
   - Check `OCR gated (skipped)` counter in summary
   - If all rows are gated, set `FORCE_OCR=True` temporarily to test

4. **Re-run calibration:**
   ```bash
   python tools/calibrate_grid.py --gap-col 7
   ```

---

## Next Steps

1. **Test the changes:**
   - Run scanner for 10 seconds
   - Confirm logs show hybrid segmentation (rows=projection, cols=line-based)
   - Confirm frame counters are > 0

2. **Re-calibrate grid if needed:**
   - If you see the miscalibration warning, run:
     ```bash
     python tools/calibrate_grid.py --gap-col 7
     ```
   - This will create a proper 10-column grid

3. **Verify events:**
   - Query database:
     ```bash
     python tools/query_events.py 50 --green --minutes 10
     ```
   - Should show events with `parsed=1` and correct symbol/price

---

## Implementation Notes

### Why Hybrid Segmentation?

The scanner table has:
- **Vertical structure:** Strong vertical gray (#808080) lines between columns
- **Horizontal structure:** No consistent horizontal lines (row delimiters are subtle or color-based)

The old all-or-nothing approach discarded valid vertical separators when horizontal detection failed. The hybrid approach treats each axis independently, preserving good data.

### Why Grid Validation?

Miscalibrated grids cause silent failures:
- User thinks they have a valid grid (loads without error)
- But semantic mapping points to wrong physical columns
- Gap color gating samples the wrong cell (e.g., Volume instead of Gap%)
- Result: 0 events because gating filters everything

The validation warnings make miscalibration visible and actionable.

---

## Summary

✅ **Hybrid segmentation:** Preserves valid vertical separators even when horizontal lines are missing  
✅ **Frame counters:** Already correct, showing processed frames even when events=0  
✅ **Grid validation:** Warns loudly when grid is miscalibrated with actionable fix instructions  
✅ **Expected outcome:** 10-column grid detected, gap at column 7, events emitted when gap is green
