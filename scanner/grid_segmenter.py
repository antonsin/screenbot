"""
Grid segmenter - use calibrated column boundaries + dynamic horizontal delimiter detection
"""
import cv2
import numpy as np
import json
import logging
import hashlib
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
        self.semantic_to_index = {}  # NEW: Semantic-to-physical-index mapping
        
        # Horizontal delimiter detection parameters
        self.delimiter_color = (128, 128, 128)  # #808080 RGB
        self.color_tolerance = 30
        self.min_line_length_pct = 0.5  # Minimum 50% of width
        
        # Load calibration
        self._load_calibration()
    
    def _load_calibration(self, grid_path=None):
        """
        Load grid calibration from config/grid.json
        
        Args:
            grid_path: Optional explicit path to grid.json. If None, uses config/grid.json
        """
        # Resolve grid file path from repo root (using __file__ not CWD)
        if grid_path is None:
            # Default: config/grid.json relative to this module's parent directory
            module_dir = Path(__file__).resolve().parent  # scanner/
            repo_root = module_dir.parent  # workspace/
            grid_file = repo_root / "config" / "grid.json"
            
            # Check for fallback files if primary doesn't exist
            if not grid_file.exists():
                fallback_files = [
                    repo_root / "config" / "grid.example.10col.json",
                    repo_root / "config" / "grid.example.json"
                ]
                for fallback in fallback_files:
                    if fallback.exists():
                        logger.warning(f"Primary grid.json not found, falling back to: {fallback}")
                        grid_file = fallback
                        break
        else:
            grid_file = Path(grid_path).resolve()
        
        if not grid_file.exists():
            logger.warning(f"Grid calibration not found: {grid_file}")
            logger.warning("Run: python tools/calibrate_grid.py")
            return
        
        try:
            # Read file and compute SHA-256
            with open(grid_file, 'rb') as f:
                file_contents = f.read()
                file_hash = hashlib.sha256(file_contents).hexdigest()
            
            # Parse JSON
            self.calibration = json.loads(file_contents.decode('utf-8'))
            
            # Build absolute column boundaries for current ROI
            roi_width = self.calibration.get('roi_width')
            normalized_seps = self.calibration.get('separators_normalized', [])
            num_columns = self.calibration.get('num_columns', len(normalized_seps) + 1)
            
            # Boundaries: [0, sep1, sep2, ..., width]
            self.column_boundaries = [0] + [int(x * roi_width) for x in normalized_seps] + [roi_width]
            self.column_semantics = self.calibration.get('column_semantics', [])
            
            # Load semantic-to-index mapping or generate from column order
            self.semantic_to_index = self.calibration.get('semantic_to_index', {})
            
            # If semantic_to_index not provided, generate from column_semantics
            if not self.semantic_to_index and self.column_semantics:
                self.semantic_to_index = {col: idx for idx, col in enumerate(self.column_semantics)}
                logger.info(f"  Generated semantic_to_index from column_semantics")
            
            # Backward compatibility: if no semantic_to_index, build from first 4 semantics
            if not self.semantic_to_index and len(self.column_semantics) >= 4:
                self.semantic_to_index = {
                    "time_hits": 0,
                    "symbol": 1,
                    "price": 2,
                    "gap": 3
                }
            
            # Log absolute path, hash, and summary
            logger.info(f"✓ Loaded grid calibration from: {grid_file.absolute()}")
            logger.info(f"  SHA-256: {file_hash}")
            logger.info(f"  Grid calibration loaded: {len(self.column_boundaries)-1} columns")
            logger.info(f"  Column semantics: {self.column_semantics}")
            logger.info(f"  Semantic mapping: {self.semantic_to_index}")
            
            # Store for validation
            self.grid_file_path = grid_file.absolute()
            self.grid_file_hash = file_hash
            self.num_columns = len(self.column_boundaries) - 1
            
            # Validate column count consistency
            self._validate_column_counts(num_columns, normalized_seps)
            
            # Validate grid calibration for semantic mapping issues
            self._validate_grid_calibration(normalized_seps, roi_width)
        
        except Exception as e:
            logger.error(f"Failed to load grid calibration: {e}")
            self.calibration = None
    
    def _validate_column_counts(self, declared_num_columns: int, normalized_seps: list):
        """
        Validate that column counts are consistent
        
        Args:
            declared_num_columns: Value from num_columns field
            normalized_seps: List of separator positions
        """
        expected_sep_count = declared_num_columns - 1
        actual_sep_count = len(normalized_seps)
        actual_num_columns = actual_sep_count + 1
        
        if declared_num_columns != len(self.column_semantics):
            logger.error("=" * 80)
            logger.error("❌ COLUMN COUNT MISMATCH")
            logger.error("=" * 80)
            logger.error(f"Grid file declares num_columns={declared_num_columns}")
            logger.error(f"But column_semantics has {len(self.column_semantics)} entries")
            logger.error(f"Column semantics: {self.column_semantics}")
            logger.error("")
            logger.error("These must match! Fix config/grid.json")
            logger.error("=" * 80)
            raise ValueError(f"Column count mismatch: {declared_num_columns} vs {len(self.column_semantics)}")
        
        if actual_sep_count != expected_sep_count:
            logger.warning("=" * 80)
            logger.warning("⚠️  SEPARATOR COUNT MISMATCH")
            logger.warning("=" * 80)
            logger.warning(f"Grid declares {declared_num_columns} columns (expects {expected_sep_count} separators)")
            logger.warning(f"But separators_normalized has {actual_sep_count} entries")
            logger.warning(f"This will create {actual_num_columns} columns instead!")
            logger.warning("")
            logger.warning("Fix config/grid.json: separators should have exactly (num_columns - 1) entries")
            logger.warning("=" * 80)
    
    def _validate_grid_calibration(self, normalized_seps: list, roi_width: int):
        """
        Validate grid calibration and warn about potential miscalibration
        
        Args:
            normalized_seps: List of normalized separator positions (0..1)
            roi_width: ROI width in pixels
        """
        num_cols = len(normalized_seps) + 1  # Number of columns
        
        # Check if we have semantic mapping that relies on specific physical columns
        has_gap_semantic = 'gap' in self.semantic_to_index
        
        # If we have few columns but semantic mapping expects more specific columns
        if num_cols <= 4 and has_gap_semantic:
            gap_index = self.semantic_to_index['gap']
            
            # Check if last column is very wide (> 50% of ROI width)
            if normalized_seps:
                last_sep = normalized_seps[-1]
                last_col_width_ratio = 1.0 - last_sep
                
                if last_col_width_ratio > 0.50:
                    logger.warning("=" * 80)
                    logger.warning("⚠️  GRID MISCALIBRATION WARNING")
                    logger.warning("=" * 80)
                    logger.warning(f"Grid has only {num_cols} columns, but last column occupies {last_col_width_ratio*100:.1f}% of ROI width.")
                    logger.warning(f"This suggests multiple physical columns are merged into column {num_cols-1}.")
                    logger.warning("")
                    logger.warning(f"Semantic mapping expects 'gap' at physical column {gap_index},")
                    logger.warning("but merged columns will break gap color gating.")
                    logger.warning("")
                    logger.warning("RECOMMENDED FIX:")
                    logger.warning("  1) Re-run: python tools/calibrate_grid.py --gap-col 7")
                    logger.warning("     to generate a proper 10-column grid with gap at column 7")
                    logger.warning("")
                    logger.warning("  OR")
                    logger.warning("")
                    logger.warning("  2) Copy config/grid.example.10col.json to config/grid.json")
                    logger.warning("     as a starting point and adjust separators")
                    logger.warning("=" * 80)
            
            # Also warn if gap_index >= num_cols
            if gap_index >= num_cols:
                logger.warning("=" * 80)
                logger.warning("⚠️  SEMANTIC MAPPING ERROR")
                logger.warning("=" * 80)
                logger.warning(f"Semantic mapping expects 'gap' at column {gap_index},")
                logger.warning(f"but grid only has {num_cols} columns (0-{num_cols-1}).")
                logger.warning("")
                logger.warning("This will cause gap color gating to fail (no gap column found).")
                logger.warning("")
                logger.warning("RECOMMENDED FIX:")
                logger.warning("  Re-run: python tools/calibrate_grid.py --gap-col 7")
                logger.warning("  to generate a proper 10-column grid")
                logger.warning("=" * 80)
    
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
        Returns dict with both physical (col0, col1...) and semantic aliases (gap, symbol...)
        
        Args:
            row_img: Row image
            
        Returns:
            Dict mapping column name to cropped image
        """
        if not self.is_calibrated():
            return {}
        
        columns = {}
        h, w = row_img.shape[:2]
        
        # Create physical column crops (col0, col1, ...)
        for i in range(len(self.column_boundaries) - 1):
            x1 = self.column_boundaries[i]
            x2 = self.column_boundaries[i + 1]
            
            # Ensure bounds are valid
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            
            if x2 > x1:
                col_img = row_img[:, x1:x2].copy()
                
                # Always store with generic col{i} key
                columns[f"col{i}"] = col_img
                
                # Also store with semantic name if defined
                col_name = self.column_semantics[i] if i < len(self.column_semantics) else None
                if col_name and col_name != f"col{i}":
                    columns[col_name] = col_img
        
        # Add semantic aliases from semantic_to_index mapping
        for semantic_name, physical_index in self.semantic_to_index.items():
            if 0 <= physical_index < len(self.column_boundaries) - 1:
                col_key = f"col{physical_index}"
                if col_key in columns:
                    columns[semantic_name] = columns[col_key]
                    logger.debug(f"Mapped semantic '{semantic_name}' -> physical col{physical_index}")
        
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
