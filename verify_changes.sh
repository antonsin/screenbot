#!/bin/bash
# Verification script for hybrid segmentation implementation
# Run this after pulling commit 708c16b

echo "=================================="
echo "Hybrid Segmentation Verification"
echo "=================================="
echo ""

# Check commit
CURRENT_COMMIT=$(git rev-parse HEAD)
EXPECTED_COMMIT="708c16b2622fa68dc493383941f5cb2c1c9a4b5e"

echo "1. Checking commit hash..."
if [[ "$CURRENT_COMMIT" == "$EXPECTED_COMMIT"* ]]; then
    echo "   ✓ Commit matches: $CURRENT_COMMIT"
else
    echo "   ✗ Commit mismatch!"
    echo "     Current:  $CURRENT_COMMIT"
    echo "     Expected: $EXPECTED_COMMIT"
    echo "   Run: git pull origin main"
    exit 1
fi
echo ""

# Check file hashes
echo "2. Verifying file hashes..."
HASH_TABLE=$(sha256sum table_segmentation.py 2>/dev/null | awk '{print $1}')
HASH_GRID=$(sha256sum scanner/grid_segmenter.py 2>/dev/null | awk '{print $1}')

EXPECTED_TABLE="406bfedb105668709b561505cd688fc50ced2c92cdf4fa7b24eea624970ee9d0"
EXPECTED_GRID="f9281b2af695d1cc29ac0428ad16609dd75dc79a290fca916fee13de43cda920"

if [[ "$HASH_TABLE" == "$EXPECTED_TABLE" ]]; then
    echo "   ✓ table_segmentation.py matches"
else
    echo "   ✗ table_segmentation.py hash mismatch"
    echo "     Got: $HASH_TABLE"
fi

if [[ "$HASH_GRID" == "$EXPECTED_GRID" ]]; then
    echo "   ✓ scanner/grid_segmenter.py matches"
else
    echo "   ✗ scanner/grid_segmenter.py hash mismatch"
    echo "     Got: $HASH_GRID"
fi
echo ""

# Check grid calibration
echo "3. Checking grid calibration..."
if [[ -f "config/grid.json" ]]; then
    NUM_COLS=$(grep -o '"num_columns":[[:space:]]*[0-9]*' config/grid.json | grep -o '[0-9]*')
    echo "   Found config/grid.json: $NUM_COLS columns"
    
    if [[ "$NUM_COLS" -le 4 ]]; then
        echo "   ⚠️  Grid has only $NUM_COLS columns"
        echo "   You should see a miscalibration warning when running scanner"
        echo "   Recommended: python tools/calibrate_grid.py --gap-col 7"
    elif [[ "$NUM_COLS" -ge 9 ]]; then
        echo "   ✓ Grid has $NUM_COLS columns (good for 10-column table)"
    fi
else
    echo "   ⚠️  No config/grid.json found"
    echo "   Run: python tools/calibrate_grid.py --gap-col 7"
fi
echo ""

# Test syntax
echo "4. Checking Python syntax..."
python3 -m py_compile table_segmentation.py scanner/grid_segmenter.py tools/stream_scanner.py 2>&1
if [[ $? -eq 0 ]]; then
    echo "   ✓ All Python files compile successfully"
else
    echo "   ✗ Syntax errors detected"
    exit 1
fi
echo ""

echo "=================================="
echo "Verification complete!"
echo ""
echo "Next steps:"
echo "1. Run scanner: python tools/stream_scanner.py --max-seconds 10"
echo "2. Check logs for:"
echo "   - 'Using line-based separators for COLS (N vertical lines)'"
echo "   - 'Final separators: rows_seps=X ... cols_seps=Y'"
echo "   - 'Grid dimensions: M rows × N columns'"
echo "3. Confirm frame summary shows 'Processed frames' > 0"
echo "4. If you see grid miscalibration warning, re-run calibration"
echo "=================================="
