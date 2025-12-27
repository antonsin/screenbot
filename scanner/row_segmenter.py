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
        
        # Projection-based parameters
        self.min_row_height_px = 20  # Minimum height for a row
        self.blank_threshold_pct = 0.08  # Percentage of max projection for blank detection
    
    def segment_rows(self, roi_img: np.ndarray) -> list[tuple[int, int, int, int]]:
        """
        Detect horizontal lines and segment rows using multi-scale approach with projection fallback
        
        Args:
            roi_img: ROI image (BGR)
            
        Returns:
            List of (x, y, w, h) tuples for each row, top to bottom
        """
        h, w = roi_img.shape[:2]
        
        # Convert to grayscale for line detection
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
        binary_for_projection = None
        projection_data = None
        
        # Fallback to projection-based gap detection if insufficient lines
        if len(y_positions) < 2:
            logger.info(f"Line detection found only {len(y_positions)} separators, using projection-gap segmentation")
            
            # Binarize specifically for projection analysis
            binary_for_projection = self._binarize_for_row_projection(roi_img)
            
            # Find row separators by detecting gaps
            y_positions, projection_data = self._find_row_separators_by_gaps(binary_for_projection)
            detection_method = "projection-gap"
        
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
            self._save_debug_image(roi_img, binary, combined_line_mask, binary_for_projection, 
                                  projection_data, y_positions, rows, detection_method)
        
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
    
    def _binarize_for_row_projection(self, roi_bgr: np.ndarray) -> np.ndarray:
        """
        Binarize image specifically for projection-based row detection
        
        Uses adaptive thresholding and horizontal dilation to merge text into solid row bands
        
        Args:
            roi_bgr: Original ROI image in BGR
            
        Returns:
            Binary image with text as white (1), optimized for projection
        """
        h, w = roi_bgr.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold - works better on UI screenshots
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,  # Invert so text is white
            blockSize=15,
            C=5
        )
        
        # Horizontal dilation to merge characters into solid row bands
        # Kernel width scales with image width
        kernel_width = max(15, w // 80)
        kernel_height = 3
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
        dilated = cv2.dilate(binary, h_kernel, iterations=1)
        
        return dilated
    
    def _find_row_separators_by_gaps(self, binary: np.ndarray) -> tuple[list[int], dict]:
        """
        Find row separators by detecting blank gaps in horizontal projection
        
        Args:
            binary: Binary image (text is white)
            
        Returns:
            Tuple of (separator_y_positions, projection_data_dict)
        """
        h, w = binary.shape
        
        # Compute horizontal projection
        projection = np.sum(binary, axis=1)
        
        # Smooth with moving average (window 9-21 px)
        window_size = min(21, max(9, h // 50))  # Adaptive window size
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(projection, kernel, mode='same')
        
        # Determine blank threshold (relative to max projection)
        max_proj = np.max(smoothed) if np.max(smoothed) > 0 else 1
        blank_threshold = max_proj * self.blank_threshold_pct
        
        # Find blank runs (gaps between rows)
        separators = []
        in_blank = False
        blank_start = 0
        
        for y, val in enumerate(smoothed):
            if val < blank_threshold and not in_blank:
                # Entering blank region
                in_blank = True
                blank_start = y
            elif val >= blank_threshold and in_blank:
                # Exiting blank region
                in_blank = False
                blank_end = y
                # Use midpoint of blank run as separator
                separator_y = (blank_start + blank_end) // 2
                separators.append(separator_y)
        
        # Cluster/merge separators closer than min_row_height_px
        separators = self._cluster_separators(separators, self.min_row_height_px)
        
        # If still too few separators, fall back to uniform splits
        if len(separators) < self.min_rows:
            logger.debug(f"Gap detection found {len(separators)} separators, creating uniform splits")
            num_splits = max(self.min_rows + 1, 5)
            separators = [int(h * i / num_splits) for i in range(1, num_splits)]
        
        # Store projection data for debug visualization
        projection_data = {
            'projection': projection,
            'smoothed': smoothed,
            'threshold': blank_threshold
        }
        
        return sorted(set(separators)), projection_data
    
    def _cluster_separators(self, separators: list[int], min_distance: int) -> list[int]:
        """
        Cluster/merge separators that are too close together
        
        Args:
            separators: List of separator y-coordinates
            min_distance: Minimum distance between separators
            
        Returns:
            Clustered list of separators
        """
        if len(separators) <= 1:
            return separators
        
        sorted_seps = sorted(separators)
        clustered = [sorted_seps[0]]
        
        for sep in sorted_seps[1:]:
            if sep - clustered[-1] >= min_distance:
                clustered.append(sep)
            # else: skip this separator (too close to previous)
        
        return clustered
    
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
                          binary_for_projection: np.ndarray, projection_data: dict,
                          y_positions: list[int], rows: list[tuple[int, int, int, int]], method: str):
        """
        Save debug visualization of segmentation
        
        Args:
            roi_img: Original ROI image
            binary: Binarized image (from line detection)
            combined_line_mask: Combined line detection mask
            binary_for_projection: Binary image used for projection (if applicable)
            projection_data: Dict with projection analysis data (if applicable)
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
        cv2.putText(debug_img, f"Rows: {len(rows)} | Separators: {len(y_positions)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Save main debug image
        filename = debug_dir / f"seg_{timestamp}.png"
        cv2.imwrite(str(filename), debug_img)
        
        # Save method-specific debug images
        if method == "projection-gap" and binary_for_projection is not None:
            # Save binarized image used for projection
            binary_filename = debug_dir / f"binary_projection_{timestamp}.png"
            cv2.imwrite(str(binary_filename), binary_for_projection)
            
            # Save projection plot
            if projection_data:
                projection_viz = self._create_projection_plot(
                    projection_data, y_positions, h, w
                )
                projection_filename = debug_dir / f"projection_plot_{timestamp}.png"
                cv2.imwrite(str(projection_filename), projection_viz)
        else:
            # Save combined lines mask for morphology-based detection
            combined_filename = debug_dir / f"combined_lines_{timestamp}.png"
            cv2.imwrite(str(combined_filename), combined_line_mask)
        
        logger.debug(f"Saved segmentation debug images: {debug_dir}")
    
    def _create_projection_plot(self, projection_data: dict, y_positions: list[int], 
                                height: int, width: int) -> np.ndarray:
        """
        Create visualization of horizontal projection with gaps highlighted
        
        Args:
            projection_data: Dict with 'projection', 'smoothed', 'threshold'
            y_positions: Detected y-positions
            height: Image height
            width: Image width
            
        Returns:
            Visualization image
        """
        # Create canvas
        viz_width = 500
        viz = np.zeros((height, viz_width, 3), dtype=np.uint8)
        
        projection = projection_data['projection']
        smoothed = projection_data['smoothed']
        threshold = projection_data['threshold']
        
        # Normalize projections to fit canvas
        max_val = np.max(smoothed) if np.max(smoothed) > 0 else 1
        margin = 50
        scale = (viz_width - margin) / max_val
        
        # Draw threshold line (green)
        threshold_x = int(threshold * scale)
        cv2.line(viz, (threshold_x, 0), (threshold_x, height), (0, 255, 0), 1)
        cv2.putText(viz, "threshold", (threshold_x + 5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw raw projection (gray)
        for y in range(height - 1):
            x1 = int(projection[y] * scale)
            x2 = int(projection[y + 1] * scale)
            cv2.line(viz, (x1, y), (x2, y + 1), (128, 128, 128), 1)
        
        # Draw smoothed projection (white/cyan)
        for y in range(height - 1):
            x1 = int(smoothed[y] * scale)
            x2 = int(smoothed[y + 1] * scale)
            cv2.line(viz, (x1, y), (x2, y + 1), (255, 255, 128), 2)
        
        # Draw detected separators (red)
        for y in y_positions:
            if 0 < y < height:
                cv2.line(viz, (0, y), (viz_width, y), (0, 0, 255), 2)
        
        # Add grid
        for i in range(0, viz_width, 50):
            cv2.line(viz, (i, 0), (i, height), (64, 64, 64), 1)
        
        return viz
