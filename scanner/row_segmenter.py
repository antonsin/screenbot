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
        self.min_rows = getattr(config, 'MIN_ROWS', 3)  # Minimum expected rows
    
    def segment_rows(self, roi_img: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect horizontal lines and segment rows using multi-scale approach
        
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
        
        # Multi-scale horizontal line detection
        kernel_percentages = [self.min_line_length_pct, 0.60, 0.45, 0.30]
        combined_line_mask = np.zeros_like(binary)
        
        for pct in kernel_percentages:
            kernel_length = max(15, int(w * pct))  # Clamp to minimum 15px
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            # OR-combine with accumulated mask
            combined_line_mask = cv2.bitwise_or(combined_line_mask, horizontal_lines)
        
        # Find y-coordinates of horizontal lines from combined mask
        y_positions = self._find_line_y_positions(combined_line_mask)
        
        detection_method = "morph-multiscale"
        
        # Fallback to projection-based if insufficient lines detected
        if len(y_positions) < 2:
            logger.warning(f"Multi-scale morphology found only {len(y_positions)} lines, using projection fallback")
            y_positions = self._projection_fallback(binary, h)
            detection_method = "projection-fallback"
        
        # Ensure boundaries and strict sorting
        y_positions = self._ensure_boundaries(y_positions, h)
        
        if len(y_positions) < 2:
            logger.warning(f"Insufficient separators detected ({len(y_positions)}), method: {detection_method}")
            # Ultimate fallback: return single row
            return [(0, 0, w, h)]
        
        # Build row bounding boxes between consecutive lines
        rows = self._build_row_boxes(y_positions, w, h)
        
        # Debug visualization
        if self.debug_enabled and len(rows) > 0:
            self._save_debug_image(roi_img, binary, combined_line_mask, y_positions, rows, detection_method)
        
        logger.debug(f"Segmented {len(rows)} rows from {len(y_positions)} separators (method: {detection_method})")
        
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
    
    def _projection_fallback(self, binary: np.ndarray, height: int) -> list[int]:
        """
        Projection-based fallback for finding row separators
        
        Uses horizontal projection valleys to detect row boundaries
        
        Args:
            binary: Binary image
            height: Image height
            
        Returns:
            List of y-coordinates for separators
        """
        # Horizontal projection (sum along rows)
        projection = np.sum(binary, axis=1)
        
        # Smooth with small moving average to reduce noise
        window_size = 5
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(projection, kernel, mode='same')
        
        # Find valleys (local minima)
        threshold = np.mean(smoothed) * 0.4
        
        y_positions = []
        in_valley = False
        valley_start = 0
        
        for y, val in enumerate(smoothed):
            if val < threshold and not in_valley:
                in_valley = True
                valley_start = y
            elif val >= threshold and in_valley:
                in_valley = False
                # Use middle of valley
                y_positions.append((valley_start + y) // 2)
        
        # If not enough separators, do uniform splits
        if len(y_positions) < self.min_rows:
            logger.debug(f"Projection found {len(y_positions)} separators, creating uniform splits")
            num_splits = max(self.min_rows + 1, 5)
            y_positions = [int(height * i / num_splits) for i in range(1, num_splits)]
        
        return sorted(set(y_positions))
    
    def _ensure_boundaries(self, y_positions: list[int], height: int) -> list[int]:
        """
        Ensure y_positions includes boundaries and is strictly sorted unique
        
        Args:
            y_positions: List of y-coordinates
            height: Image height
            
        Returns:
            Sorted unique list with boundaries
        """
        # Convert to set to remove duplicates
        y_set = set(y_positions)
        
        # Add boundaries
        y_set.add(0)
        y_set.add(height)
        
        # Return sorted list
        return sorted(y_set)
    
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
    
    def _save_debug_image(self, roi_img: np.ndarray, binary: np.ndarray, combined_line_mask: np.ndarray, 
                          y_positions: list[int], rows: list[tuple[int, int, int, int]], method: str):
        """
        Save debug visualization of segmentation
        
        Args:
            roi_img: Original ROI image
            binary: Binarized image
            combined_line_mask: Combined line detection mask
            y_positions: Y-coordinates of detected lines
            rows: Detected row boxes
            method: Detection method used
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_dir = self.config.DEBUG_FRAMES_DIR / "segmentation"
        debug_dir.mkdir(exist_ok=True)
        
        h, w = roi_img.shape[:2]
        
        # Main debug image with annotations
        debug_img = roi_img.copy()
        
        # Draw detected horizontal lines in red
        for y in y_positions:
            if 0 < y < h:  # Skip boundaries
                cv2.line(debug_img, (0, y), (w, y), (0, 0, 255), 2)
        
        # Draw row boxes in green
        for idx, (x, y, box_w, box_h) in enumerate(rows):
            cv2.rectangle(debug_img, (x, y), (x + box_w, y + box_h), (0, 255, 0), 1)
            # Add row number
            cv2.putText(debug_img, f"Row {idx}", (x + 5, y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add method annotation
        cv2.putText(debug_img, f"Method: {method}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Save main debug image
        filename = debug_dir / f"seg_{timestamp}.png"
        cv2.imwrite(str(filename), debug_img)
        
        # Save combined lines mask
        combined_filename = debug_dir / f"combined_lines_{timestamp}.png"
        cv2.imwrite(str(combined_filename), combined_line_mask)
        
        # Save projection visualization
        if method == "projection-fallback":
            projection = np.sum(binary, axis=1)
            projection_viz = self._create_projection_visualization(projection, y_positions, h, w)
            projection_filename = debug_dir / f"projection_{timestamp}.png"
            cv2.imwrite(str(projection_filename), projection_viz)
        
        logger.debug(f"Saved segmentation debug images: {debug_dir}")
    
    def _create_projection_visualization(self, projection: np.ndarray, y_positions: list[int], 
                                        height: int, width: int) -> np.ndarray:
        """
        Create visualization of horizontal projection
        
        Args:
            projection: Horizontal projection values
            y_positions: Detected y-positions
            height: Image height
            width: Image width
            
        Returns:
            Visualization image
        """
        # Create canvas
        viz_width = 400
        viz = np.zeros((height, viz_width, 3), dtype=np.uint8)
        
        # Normalize projection to fit canvas
        max_val = np.max(projection) if np.max(projection) > 0 else 1
        normalized = (projection / max_val * (viz_width - 50)).astype(int)
        
        # Draw projection curve
        for y in range(height - 1):
            x1 = int(normalized[y])
            x2 = int(normalized[y + 1])
            cv2.line(viz, (x1, y), (x2, y + 1), (255, 255, 255), 1)
        
        # Draw detected separators
        for y in y_positions:
            if 0 < y < height:
                cv2.line(viz, (0, y), (viz_width, y), (0, 0, 255), 1)
        
        # Add grid
        for i in range(0, viz_width, 50):
            cv2.line(viz, (i, 0), (i, height), (64, 64, 64), 1)
        
        return viz
