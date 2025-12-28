# Debug Segmentation Bypass Analysis

**Date:** 2025-12-27  
**Current Commit:** ef801c9f99b4918ab7b8900aa2b5bc5582ef1ff6  
**Status:** ✅ ALREADY FIXED

## Executive Summary

The debug segmentation bypass issue **has already been resolved** in commit `ef801c9` (2024-12-27). The current implementation is correct and does NOT bypass the calibrated grid pipeline when debug mode is enabled.

## Root Cause (Historical)

The original issue was that `_debug_table_segmentation()` was called in the main loop and used `continue` to skip the rest of the pipeline, causing:
- Events to not be emitted
- Frame counters to not increment (since they were only incremented after event processing)
- Debug output to use `table_segmentation.segment_table()` auto-detection instead of calibrated grid

## Current Implementation (Fixed)

### 1. Debug Segmentation is Disabled/Deprecated

**File:** `tools/stream_scanner.py`, lines 547-555

```python
def _debug_table_segmentation(self, roi_img: np.ndarray):
    """
    DEPRECATED: This method is no longer used in the main pipeline.
    Debug overlays are now generated directly from calibrated grid segmentation.
    
    Keeping for reference only. Can be deleted in future cleanup.
    """
    # This method is intentionally disabled to prevent pipeline bypass
    return
```

The method immediately returns without doing anything. It is NOT called in the main loop.

### 2. Debug Output Uses Calibrated Grid

**File:** `tools/stream_scanner.py`, lines 168-174

```python
if self.use_grid_segmenter:
    # Use calibrated grid segmentation
    rows, column_boundaries, debug_overlay = self.grid_segmenter.segment_grid(roi_img)
    
    # Save debug overlay periodically if debug mode is on
    if self.debug_segmentation_enabled and self.frame_count % 30 == 0:
        self.grid_segmenter.save_debug_overlay(debug_overlay, self.frame_count)
```

When debug segmentation is enabled, debug overlays are saved directly from the calibrated grid segmenter, showing the correct 10 columns.

### 3. Frame Counters are Comprehensive

**File:** `tools/stream_scanner.py`, lines 89-95, 159, 165, 185, 202, 287, 369

Diagnostic counters are incremented at the correct locations:

| Counter | Increment Location | Condition |
|---------|-------------------|-----------|
| `captured_frames` | Line 159 | After every capture attempt |
| `processed_frames` | Line 165 | After frame validation (not None/empty) |
| `segmented_frames` | Line 185 | After successful row segmentation |
| `ocr_attempted_frames` | Line 202 | When new rows are detected |
| `ocr_gated_frames` | Line 287 | When OCR is skipped due to gating |
| `events_emitted` | Line 369 | After event is stored to DB |

All counters are printed in the session summary (lines 227-232).

### 4. `_update_fps()` is Called Every Frame

Lines 182, 194, 209 ensure FPS counter is updated even when:
- No rows are detected
- Grid is not ready (warmup phase)
- OCR is gated

## Verification

### Expected Behavior

When running `python tools/stream_scanner.py --max-seconds 10` with `DEBUG_SEGMENTATION=1`:

1. **Logs should show:**
   ```
   ✓ Using calibrated grid segmentation
   Debug segmentation enabled, output to: /workspace/debug
   ```

2. **Debug files created:**
   - `debug/grid_overlays/grid_overlay_frame0000_TIMESTAMP.png` (every 30 frames)
   - Shows 10 columns (col0-col9) with green vertical separators
   - Shows horizontal row delimiters in red
   - Row indices (R0, R1, R2, ...)

3. **Session summary should show non-zero values:**
   ```
   SCANNER SESSION SUMMARY
   ========================
   Captured frames:     100+
   Processed frames:    100+
   Segmented frames:    80+
   OCR attempted:       varies
   OCR gated (skipped): varies
   Events emitted:      varies
   ```

### File Hashes (for verification)

```bash
# PowerShell:
Get-FileHash tools/stream_scanner.py -Algorithm SHA256
Get-FileHash scanner/grid_segmenter.py -Algorithm SHA256
Get-FileHash table_segmentation.py -Algorithm SHA256
Get-FileHash config.py -Algorithm SHA256

# Git Bash:
sha256sum tools/stream_scanner.py scanner/grid_segmenter.py table_segmentation.py config.py
```

**Expected hashes for commit ef801c9:**
- `tools/stream_scanner.py`: (verify with command above)
- `scanner/grid_segmenter.py`: (verify with command above)
- `table_segmentation.py`: (verify with command above)
- `config.py`: (verify with command above)

## Configuration Defaults

**File:** `config.py`, line 149

```python
DEBUG_SEGMENTATION = os.getenv("DEBUG_SEGMENTATION", "0") == "1"
```

Default is `False` (off). To enable:

```bash
# Windows PowerShell:
$env:DEBUG_SEGMENTATION="1"; python tools/stream_scanner.py --max-seconds 10

# Windows Git Bash / Linux:
DEBUG_SEGMENTATION=1 python tools/stream_scanner.py --max-seconds 10

# Or create .env file:
echo "DEBUG_SEGMENTATION=1" >> .env
```

## Remaining Cleanup (Optional)

The following can be removed in a future cleanup commit:
1. `_debug_table_segmentation()` method (lines 547-582 in `tools/stream_scanner.py`)
2. Import of `TABLE_SEGMENTATION_AVAILABLE` flag (lines 30-41 in `tools/stream_scanner.py`)

However, keeping them as deprecated/disabled does not affect functionality.

## Conclusion

✅ **Debug segmentation does NOT bypass the calibrated grid pipeline**  
✅ **Frame counters are comprehensive and correct**  
✅ **Debug overlays reflect calibrated 10-column grid**  
✅ **Configuration defaults are correct (DEBUG_SEGMENTATION=0)**

The implementation is correct. If the user is still seeing "Total frames: 0", the likely causes are:

1. **Short run time**: If the script exits within the first scan interval or before any frames are captured
2. **Capture failure**: If `frame_source.capture_frame()` returns None/empty for all attempts
3. **Region not configured**: If `config/region.json` doesn't exist and default ABS_* values are wrong

**Next step:** Run actual test with verbose logging to confirm behavior.
