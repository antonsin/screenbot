"""
Color features - fast gap color classification using HSV
"""
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ColorFeatureExtractor:
    """
    Extract color features from gap cells for fast gating
    """
    
    def __init__(self, config):
        """
        Initialize color feature extractor
        
        Args:
            config: Configuration module
        """
        self.config = config
        self.green_hue_range = config.GAP_GREEN_HUE_RANGE
        self.min_sat = config.GAP_MIN_SAT
        self.min_val = config.GAP_MIN_VAL
    
    def classify_gap_color(self, gap_cell_img: np.ndarray) -> dict:
        """
        Classify gap cell color as GREEN, WARM, NEUTRAL, or UNKNOWN
        
        Args:
            gap_cell_img: Gap cell image (BGR)
            
        Returns:
            Dict with gap_color_bucket and HSV stats
        """
        if gap_cell_img is None or gap_cell_img.size == 0:
            return {
                'gap_color_bucket': 'UNKNOWN',
                'h_mean': 0.0,
                's_mean': 0.0,
                'v_mean': 0.0
            }
        
        # Convert to HSV
        hsv = cv2.cvtColor(gap_cell_img, cv2.COLOR_BGR2HSV)
        
        # Create mask to ignore very dark pixels (likely text)
        # Only consider pixels with value > 40
        value_mask = hsv[:, :, 2] > 40
        
        if not value_mask.any():
            return {
                'gap_color_bucket': 'UNKNOWN',
                'h_mean': 0.0,
                's_mean': 0.0,
                'v_mean': 0.0
            }
        
        # Compute mean HSV values (excluding dark pixels)
        h_values = hsv[:, :, 0][value_mask]
        s_values = hsv[:, :, 1][value_mask]
        v_values = hsv[:, :, 2][value_mask]
        
        h_mean = float(np.mean(h_values))
        s_mean = float(np.mean(s_values))
        v_mean = float(np.mean(v_values))
        
        # Classify color bucket
        gap_color_bucket = self._classify_bucket(h_mean, s_mean, v_mean)
        
        return {
            'gap_color_bucket': gap_color_bucket,
            'h_mean': h_mean,
            's_mean': s_mean,
            'v_mean': v_mean
        }
    
    def _classify_bucket(self, h_mean: float, s_mean: float, v_mean: float) -> str:
        """
        Classify color into bucket
        
        Args:
            h_mean: Mean hue (0-180)
            s_mean: Mean saturation (0-255)
            v_mean: Mean value (0-255)
            
        Returns:
            Color bucket: GREEN, WARM, NEUTRAL, or UNKNOWN
        """
        # Check if color is saturated enough
        if s_mean < self.min_sat or v_mean < self.min_val:
            return 'NEUTRAL'
        
        # Check for green hues
        h_min, h_max = self.green_hue_range
        if h_min <= h_mean <= h_max:
            return 'GREEN'
        
        # Check for warm colors (red/orange/yellow)
        # Red wraps around in HSV: 0-10 or 170-180
        if h_mean < 10 or h_mean > 170:
            return 'WARM'  # Red
        elif 10 <= h_mean < 35:
            return 'WARM'  # Orange/Yellow
        
        # Everything else
        return 'NEUTRAL'
    
    def is_green(self, color_result: dict) -> bool:
        """
        Check if color classification is GREEN
        
        Args:
            color_result: Result from classify_gap_color()
            
        Returns:
            True if GREEN
        """
        return color_result.get('gap_color_bucket') == 'GREEN'
