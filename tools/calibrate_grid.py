#!/usr/bin/env python3
"""
Grid calibration tool - manually define column separators

Usage:
    python tools/calibrate_grid.py
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import config
from capture import FrameSource

# Global state for calibration
separators = []
roi_width = 0
drawing = False
temp_x = 0


def mouse_callback(event, x, y, flags, param):
    """Handle mouse events for drawing vertical separators"""
    global separators, drawing, temp_x
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing a vertical line
        drawing = True
        temp_x = x
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_x = x
    
    elif event == cv2.EVENT_LBUTTONUP:
        # Finish drawing - add separator
        drawing = False
        if 0 < x < roi_width:  # Don't add if at edges
            # Add separator and keep sorted
            separators.append(x)
            separators.sort()
            print(f"Added separator at x={x} (normalized: {x/roi_width:.3f})")


def draw_grid_overlay(img, show_temp=False):
    """Draw current calibration on image"""
    overlay = img.copy()
    h, w = overlay.shape[:2]
    
    # Draw ROI edges (implicit boundaries)
    cv2.line(overlay, (0, 0), (0, h), (0, 255, 0), 3)  # Left edge - green
    cv2.line(overlay, (w-1, 0), (w-1, h), (0, 255, 0), 3)  # Right edge - green
    cv2.putText(overlay, "ROI Left", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(overlay, "ROI Right", (w-60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw existing separators
    for i, sep_x in enumerate(separators):
        cv2.line(overlay, (sep_x, 0), (sep_x, h), (0, 0, 255), 2)  # Red
        cv2.putText(overlay, f"Sep{i+1}", (sep_x+5, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw temp separator while dragging
    if drawing and show_temp:
        cv2.line(overlay, (temp_x, 0), (temp_x, h), (255, 255, 0), 2)  # Yellow
    
    # Draw column labels
    boundaries = [0] + separators + [w]
    for i in range(len(boundaries) - 1):
        x1, x2 = boundaries[i], boundaries[i+1]
        cx = (x1 + x2) // 2
        cv2.putText(overlay, f"Col{i}", (cx-20, h//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return overlay


def auto_suggest_separators(img):
    """Auto-suggest vertical separators by detecting #808080 structures"""
    h, w = img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Look for vertical lines around gray value (128 ± 30)
    lower = 128 - 30
    upper = 128 + 30
    mask = cv2.inRange(gray, lower, upper)
    
    # Morphology to isolate vertical structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 4))
    vertical_lines = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find x-coordinates with strong vertical signals
    vertical_projection = np.sum(vertical_lines, axis=0)
    threshold = np.max(vertical_projection) * 0.3 if np.max(vertical_projection) > 0 else 1
    
    suggestions = []
    in_line = False
    line_start = 0
    
    for x, val in enumerate(vertical_projection):
        if val > threshold and not in_line:
            in_line = True
            line_start = x
        elif val <= threshold and in_line:
            in_line = False
            # Use middle of detected line
            sep_x = (line_start + x) // 2
            if 10 < sep_x < w - 10:  # Not too close to edges
                suggestions.append(sep_x)
    
    return suggestions


def save_calibration(roi_rect, gap_col_index=None):
    """Save calibration to config/grid.json"""
    config_dir = Path(config.BASE_DIR) / "config"
    config_dir.mkdir(exist_ok=True)
    
    grid_file = config_dir / "grid.json"
    
    # Normalize separator positions to 0..1
    normalized_separators = [x / roi_width for x in separators]
    
    # Define column semantics for ALL columns
    num_cols = len(separators) + 1
    column_semantics = [f"col{i}" for i in range(num_cols)]
    
    # Define semantic-to-index mapping for key columns
    # Default assumes first 4 columns for backward compatibility
    if gap_col_index is None:
        gap_col_index = min(3, num_cols - 1)  # Default to col 3 or last column
    
    semantic_to_index = {
        "time_hits": 0 if num_cols > 0 else None,
        "symbol": 1 if num_cols > 1 else None,
        "price": 2 if num_cols > 2 else None,
        "gap": gap_col_index
    }
    
    # Remove None entries
    semantic_to_index = {k: v for k, v in semantic_to_index.items() if v is not None}
    
    calibration_data = {
        "roi_rect": roi_rect,
        "roi_width": roi_width,
        "separators_normalized": normalized_separators,
        "separators_absolute": separators,
        "column_semantics": column_semantics,
        "semantic_to_index": semantic_to_index,
        "num_columns": num_cols,
        "created_at": datetime.now().isoformat(),
        "calibration_method": "manual"
    }
    
    with open(grid_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"\n✓ Calibration saved to: {grid_file}")
    print(f"  Columns: {num_cols}")
    print(f"  Separators: {len(separators)}")
    print(f"  Semantic mapping: {semantic_to_index}")


def main():
    """Main calibration UI"""
    global roi_width, separators
    
    import argparse
    parser = argparse.ArgumentParser(description='Grid calibration tool')
    parser.add_argument('--gap-col', type=int, help='Column index for gap (0-based, default: 3)')
    args = parser.parse_args()
    
    gap_col_index = args.gap_col
    
    print("=" * 60)
    print("Grid Calibration Tool")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Click and drag to add vertical separators")
    print("2. Separators will auto-sort left-to-right")
    print("3. ROI edges are implicit boundaries (don't draw them)")
    print("4. Press 's' to auto-suggest separators")
    print("5. Press 'r' to reset/clear all separators")
    print("6. Press 'Enter' to save and exit")
    print("7. Press 'Esc' to cancel")
    if gap_col_index is not None:
        print(f"\nGap column will be set to: {gap_col_index}")
    print("=" * 60)
    
    # Load ROI
    frame_source = FrameSource(config)
    roi_img = frame_source.capture_frame()
    h, w = roi_img.shape[:2]
    roi_width = w
    
    # Get ROI rect for reference
    roi_rect = {
        "x": config.ABS_X,
        "y": config.ABS_Y,
        "w": config.ABS_W,
        "h": config.ABS_H
    }
    
    # Check if region.json exists
    region_file = Path(config.BASE_DIR) / "config" / "region.json"
    if region_file.exists():
        with open(region_file, 'r') as f:
            region_data = json.load(f)
            roi_rect = {
                "x": region_data.get("abs_x", config.ABS_X),
                "y": region_data.get("abs_y", config.ABS_Y),
                "w": region_data.get("abs_w", config.ABS_W),
                "h": region_data.get("abs_h", config.ABS_H)
            }
    
    print(f"\nROI: {roi_rect['w']}x{roi_rect['h']} at ({roi_rect['x']}, {roi_rect['y']})")
    
    # Setup window
    window_name = "Grid Calibration - Draw Vertical Separators"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    print("\nReady! Start drawing vertical separators...")
    
    while True:
        # Draw overlay
        display = draw_grid_overlay(roi_img, show_temp=drawing)
        
        # Show instructions
        cv2.putText(display, "s=auto-suggest | r=reset | Enter=save | Esc=cancel", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter
            if len(separators) > 0:
                save_calibration(roi_rect, gap_col_index)
                break
            else:
                print("No separators defined! Add at least one separator or press 's' for auto-suggest.")
        
        elif key == 27:  # Esc
            print("Calibration cancelled")
            break
        
        elif key == ord('r'):
            separators = []
            print("Reset - all separators cleared")
        
        elif key == ord('s'):
            suggestions = auto_suggest_separators(roi_img)
            if suggestions:
                print(f"\nAuto-suggest found {len(suggestions)} potential separators:")
                for i, x in enumerate(suggestions):
                    print(f"  {i+1}. x={x} (normalized: {x/roi_width:.3f})")
                
                response = input("\nAccept suggestions? [y/N]: ").lower()
                if response == 'y':
                    separators = suggestions
                    print("Suggestions accepted")
                else:
                    print("Suggestions rejected - continue manual calibration")
            else:
                print("No separators auto-detected")
    
    cv2.destroyAllWindows()
    frame_source.close()


if __name__ == "__main__":
    main()
