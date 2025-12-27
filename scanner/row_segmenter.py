"""
Row segmenter - dynamic horizontal line detection for cascading scanner
"""
import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class RowSegmenter:
    """
    Dynamically segments rows by detecting horizontal separator lines
    """
    
    def __init__(self, config):
        """
        Initialize row segmenter
        
        Args:
            config: Configuration module
        """
        self.config = config
        self.min_line_length_pct = config.ROW_LINE_MIN_LENGTH_PCT
        self.line_thickness = config.ROW_LINE_THICKNESS_PX
        self.debug_enabled = config.ROW_SEGMENT_DEBUG
    
    def segment_rows(self, roi_img: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect horizontal lines and segment rows
        
        Args:
            roi_img: ROI image (BGR)
            
        Returns:
            List of (x, y, w, h) tuples for each row, top to bottom
        """
        h, w = roi_img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold (Otsu's method for automatic threshold)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Create horizontal kernel to isolate horizontal lines
        # Wide kernel detects horizontal structures
        kernel_length = int(w * self.min_line_length_pct)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find y-coordinates of horizontal lines
        y_positions = self._find_line_y_positions(horizontal_lines)
        
        if len(y_positions) < 2:
            logger.warning(f"Found only {len(y_positions)} horizontal lines, expected more for row separation")
            # Fallback: return single row covering entire ROI
            return [(0, 0, w, h)]
        
        # Build row bounding boxes between consecutive lines
        rows = self._build_row_boxes(y_positions, w, h)
        
        # Debug visualization
        if self.debug_enabled and len(rows) > 0:
            self._save_debug_image(roi_img, y_positions, rows)
        
        logger.debug(f"Segmented {len(rows)} rows from {len(y_positions)} horizontal lines")
        
        return rows
    
    def _find_line_y_positions(self, line_mask: np.ndarray) -> list[int]:
        """
        Find y-coordinates of horizontal lines
        
        Args:
            line_mask: Binary mask with horizontal lines
            
        Returns:
            Sorted list of y-coordinates
        """
        # Sum horizontally to find rows with high pixel counts
        horizontal_projection = np.sum(line_mask, axis=1)
        
        # Find peaks (lines)
        threshold = np.max(horizontal_projection) * 0.3  # 30% of max
        
        y_positions = []
        in_line = False
        line_start = 0
        
        for y, val in enumerate(horizontal_projection):
            if val > threshold and not in_line:
                in_line = True
                line_start = y
            elif val <= threshold and in_line:
                in_line = False
                # Use middle of line thickness
                y_positions.append((line_start + y) // 2)
        
        # Sort and remove duplicates
        y_positions = sorted(set(y_positions))
        
        return y_positions
    
    def _build_row_boxes(self, y_positions: list[int], width: int, height: int) -> list[tuple[int, int, int, int]]:
        """
        Build row bounding boxes between consecutive lines
        
        Args:
            y_positions: Y-coordinates of separator lines
            width: ROI width
            height: ROI height
            
        Returns:
            List of (x, y, w, h) for each row
        """
        rows = []
        
        # Add top boundary if first line is not at top
        if y_positions[0] > 5:
            y_positions = [0] + y_positions
        
        # Add bottom boundary if last line is not at bottom
        if y_positions[-1] < height - 5:
            y_positions = y_positions + [height]
        
        # Create boxes between consecutive lines
        for i in range(len(y_positions) - 1):
            y_top = y_positions[i]
            y_bottom = y_positions[i + 1]
            
            # Add small padding to avoid including separator line
            y_top += self.line_thickness
            y_bottom -= self.line_thickness
            
            row_height = y_bottom - y_top
            
            # Skip very thin rows (likely noise)
            if row_height < 10:
                continue
            
            rows.append((0, y_top, width, row_height))
        
        return rows
    
    def _save_debug_image(self, roi_img: np.ndarray, y_positions: list[int], rows: list[tuple[int, int, int, int]]):
        """
        Save debug visualization of segmentation
        
        Args:
            roi_img: Original ROI image
            y_positions: Y-coordinates of detected lines
            rows: Detected row boxes
        """
        debug_img = roi_img.copy()
        h, w = debug_img.shape[:2]
        
        # Draw detected horizontal lines in red
        for y in y_positions:
            cv2.line(debug_img, (0, y), (w, y), (0, 0, 255), 2)
        
        # Draw row boxes in green
        for idx, (x, y, box_w, box_h) in enumerate(rows):
            cv2.rectangle(debug_img, (x, y), (x + box_w, y + box_h), (0, 255, 0), 1)
            # Add row number
            cv2.putText(debug_img, f"Row {idx}", (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save to debug directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_dir = self.config.DEBUG_FRAMES_DIR / "segmentation"
        debug_dir.mkdir(exist_ok=True)
        
        filename = debug_dir / f"seg_{timestamp}.png"
        cv2.imwrite(str(filename), debug_img)
        logger.debug(f"Saved segmentation debug image: {filename}")
