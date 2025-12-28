"""
Grid segmenter - use calibrated column boundaries + dynamic horizontal delimiter detection
"""
import cv2
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class GridSegmenter:
    """
    Segments rows and columns using calibrated grid + color-based row detection
    """
    
    def __init__(self, config):
        """
        Initialize grid segmenter
        
        Args:
            config: Configuration module
        """
        self.config = config
        self.calibration = None
        self.column_boundaries = None
        self.column_semantics = None
        
        # Horizontal delimiter detection parameters
        self.delimiter_color = (128, 128, 128)  # #808080 RGB
        self.color_tolerance = 30
        self.min_line_length_pct = 0.5  # Minimum 50% of width
        
        # Load calibration
        self._load_calibration()
    
    def _load_calibration(self):
        """Load grid calibration from config/grid.json"""
        grid_file = self.config.BASE_DIR / "config" / "grid.json"
        
        if not grid_file.exists():
            logger.warning(f"Grid calibration not found: {grid_file}")
            logger.warning("Run: python tools/calibrate_grid.py")
            return
        
        try:
            with open(grid_file, 'r') as f:
                self.calibration = json.load(f)
            
            # Build absolute column boundaries for current ROI
            roi_width = self.calibration.get('roi_width')
            normalized_seps = self.calibration.get('separators_normalized', [])
            
            # Boundaries: [0, sep1, sep2, ..., width]
            self.column_boundaries = [0] + [int(x * roi_width) for x in normalized_seps] + [roi_width]
            self.column_semantics = self.calibration.get('column_semantics', [])
            
            logger.info(f"✓ Grid calibration loaded: {len(self.column_boundaries)-1} columns")
            logger.info(f"  Column order: {', '.join(self.column_semantics)}")
        
        except Exception as e:
            logger.error(f"Failed to load grid calibration: {e}")
            self.calibration = None
    
    def is_calibrated(self) -> bool:
        """Check if grid is calibrated"""
        return self.calibration is not None and self.column_boundaries is not None
    
    def segment_grid(self, roi_img: np.ndarray) -> tuple:
        """
        Segment ROI into grid using calibrated columns + detected horizontal delimiters
        
        Args:
            roi_img: ROI image (BGR)
            
        Returns:
            Tuple of (rows, column_boundaries, debug_overlay)
            rows: List of (x, y, w, h) row bounding boxes
            column_boundaries: List of x positions for column separators
            debug_overlay: Annotated image for debugging
        """
        if not self.is_calibrated():
            logger.error("Grid not calibrated - run calibrate_grid.py first")
            return [], [], roi_img
        
        h, w = roi_img.shape[:2]
        
        # Detect horizontal delimiters by color
        y_separators = self._detect_horizontal_delimiters(roi_img)
        
        # Build row bounding boxes between consecutive separators
        rows = []
        for i in range(len(y_separators) - 1):
            y1 = y_separators[i]
            y2 = y_separators[i + 1]
            
            # Add small padding to avoid including separator lines
            padding = 2
            row = (
                0,
                y1 + padding,
                w,
                max(1, y2 - y1 - 2 * padding)
            )
            rows.append(row)
        
        # Create debug overlay
        debug_overlay = self._create_debug_overlay(roi_img, y_separators)
        
        logger.debug(f"Grid segmentation: {len(rows)} rows, {len(self.column_boundaries)-1} columns")
        
        return rows, self.column_boundaries, debug_overlay
    
    def _detect_horizontal_delimiters(self, roi_img: np.ndarray) -> list:
        """
        Detect horizontal row delimiters by color (#808080)
        
        Args:
            roi_img: ROI image (BGR)
            
        Returns:
            List of y-coordinates for horizontal separators (sorted, includes 0 and height)
        """
        h, w = roi_img.shape[:2]
        
        # Create mask for delimiter color (#808080 ± tolerance)
        target_bgr = np.array([128, 128, 128], dtype=np.uint8)
        lower = np.array([max(0, 128 - self.color_tolerance)] * 3, dtype=np.uint8)
        upper = np.array([min(255, 128 + self.color_tolerance)] * 3, dtype=np.uint8)
        
        color_mask = cv2.inRange(roi_img, lower, upper)
        
        # Morphology to connect horizontal structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
        horizontal_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        # Dilate to make detection more robust
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 20, 3))
        horizontal_mask = cv2.dilate(horizontal_mask, kernel_dilate)
        
        # Find y-coordinates with strong horizontal signals
        horizontal_projection = np.sum(horizontal_mask, axis=1)
        min_line_pixels = int(w * self.min_line_length_pct)
        
        y_separators = []
        in_line = False
        line_start = 0
        
        for y, val in enumerate(horizontal_projection):
            if val > min_line_pixels and not in_line:
                in_line = True
                line_start = y
            elif val <= min_line_pixels and in_line:
                in_line = False
                # Use middle of detected line
                y_separators.append((line_start + y) // 2)
        
        # Always include boundaries
        if not y_separators or y_separators[0] != 0:
            y_separators.insert(0, 0)
        if not y_separators or y_separators[-1] != h:
            y_separators.append(h)
        
        # Remove duplicates and sort
        y_separators = sorted(set(y_separators))
        
        logger.debug(f"Detected {len(y_separators)} horizontal delimiters (color-based)")
        
        return y_separators
    
    def _create_debug_overlay(self, roi_img: np.ndarray, y_separators: list) -> np.ndarray:
        """
        Create debug overlay showing grid structure
        
        Args:
            roi_img: Original ROI image
            y_separators: List of y-coordinates for horizontal separators
            
        Returns:
            Annotated image
        """
        overlay = roi_img.copy()
        h, w = overlay.shape[:2]
        
        # Draw vertical column boundaries (green, thick)
        for i, x in enumerate(self.column_boundaries):
            color = (0, 255, 0)  # Green
            thickness = 3 if i == 0 or i == len(self.column_boundaries) - 1 else 2
            cv2.line(overlay, (x, 0), (x, h), color, thickness)
            
            # Label column
            if i < len(self.column_boundaries) - 1:
                col_name = self.column_semantics[i] if i < len(self.column_semantics) else f"col{i}"
                x_mid = (self.column_boundaries[i] + self.column_boundaries[i+1]) // 2
                cv2.putText(overlay, col_name, (x_mid-30, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw horizontal row delimiters (red, medium)
        for y in y_separators:
            cv2.line(overlay, (0, y), (w, y), (0, 0, 255), 2)
        
        # Draw row indices
        for i in range(len(y_separators) - 1):
            y1 = y_separators[i]
            y2 = y_separators[i + 1]
            y_mid = (y1 + y2) // 2
            cv2.putText(overlay, f"R{i}", (5, y_mid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return overlay
    
    def slice_row_by_columns(self, row_img: np.ndarray) -> dict:
        """
        Slice a row image into columns using calibrated boundaries
        
        Args:
            row_img: Row image
            
        Returns:
            Dict mapping column name to cropped image
        """
        if not self.is_calibrated():
            return {}
        
        columns = {}
        h, w = row_img.shape[:2]
        
        for i in range(len(self.column_boundaries) - 1):
            x1 = self.column_boundaries[i]
            x2 = self.column_boundaries[i + 1]
            
            # Ensure bounds are valid
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            
            if x2 > x1:
                col_name = self.column_semantics[i] if i < len(self.column_semantics) else f"col{i}"
                columns[col_name] = row_img[:, x1:x2].copy()
        
        return columns
    
    def save_debug_overlay(self, overlay: np.ndarray, frame_id: int = 0):
        """
        Save debug overlay image
        
        Args:
            overlay: Overlay image
            frame_id: Frame identifier
        """
        debug_dir = self.config.DEBUG_FRAMES_DIR / "grid_overlays"
        debug_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = debug_dir / f"grid_overlay_frame{frame_id:04d}_{timestamp}.png"
        cv2.imwrite(str(filename), overlay)
        
        logger.debug(f"Saved grid overlay: {filename}")
