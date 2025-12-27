"""
Column slicer - extract columns from row images by percentage
"""
import cv2
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ColumnSlicer:
    """
    Slices row images into columns based on percentage boundaries
    """
    
    def __init__(self, config):
        """
        Initialize column slicer
        
        Args:
            config: Configuration module
        """
        self.config = config
        self.col_time_hits = config.COL_TIME_HITS
        self.col_symbol = config.COL_SYMBOL
        self.col_price = config.COL_PRICE
        self.col_gap = config.COL_GAP
    
    def crop_by_percent(self, row_img: np.ndarray, start_pct: float, end_pct: float) -> np.ndarray:
        """
        Crop a region from row image using percentage boundaries
        
        Args:
            row_img: Row image
            start_pct: Start position as percentage (0.0-1.0)
            end_pct: End position as percentage (0.0-1.0)
            
        Returns:
            Cropped image
        """
        h, w = row_img.shape[:2]
        
        start_x = int(w * start_pct)
        end_x = int(w * end_pct)
        
        # Ensure valid bounds
        start_x = max(0, start_x)
        end_x = min(w, end_x)
        
        if end_x <= start_x:
            logger.warning(f"Invalid crop bounds: start={start_x}, end={end_x}")
            return row_img
        
        return row_img[:, start_x:end_x].copy()
    
    def slice_all_columns(self, row_img: np.ndarray) -> dict:
        """
        Slice all configured columns from row
        
        Args:
            row_img: Row image
            
        Returns:
            Dict with column name -> cropped image
        """
        columns = {}
        
        columns['time_hits'] = self.crop_by_percent(row_img, *self.col_time_hits)
        columns['symbol'] = self.crop_by_percent(row_img, *self.col_symbol)
        columns['price'] = self.crop_by_percent(row_img, *self.col_price)
        columns['gap'] = self.crop_by_percent(row_img, *self.col_gap)
        
        return columns
    
    def save_debug_columns(self, row_img: np.ndarray, columns: dict, prefix: str = "cols"):
        """
        Save debug visualization with column boundaries
        
        Args:
            row_img: Row image
            columns: Dict of column crops
            prefix: Filename prefix
        """
        debug_img = row_img.copy()
        h, w = debug_img.shape[:2]
        
        # Draw column boundaries
        boundaries = [
            ("time_hits", self.col_time_hits, (255, 0, 0)),    # Blue
            ("symbol", self.col_symbol, (0, 255, 0)),          # Green
            ("price", self.col_price, (0, 255, 255)),          # Yellow
            ("gap", self.col_gap, (255, 0, 255))               # Magenta
        ]
        
        for name, (start_pct, end_pct), color in boundaries:
            start_x = int(w * start_pct)
            end_x = int(w * end_pct)
            
            # Draw vertical lines
            cv2.line(debug_img, (start_x, 0), (start_x, h), color, 2)
            cv2.line(debug_img, (end_x, 0), (end_x, h), color, 2)
            
            # Add label
            cv2.putText(debug_img, name, (start_x + 5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Save debug image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_dir = self.config.DEBUG_FRAMES_DIR / "columns"
        debug_dir.mkdir(exist_ok=True)
        
        filename = debug_dir / f"{prefix}_{timestamp}.png"
        cv2.imwrite(str(filename), debug_img)
        logger.debug(f"Saved column debug image: {filename}")
