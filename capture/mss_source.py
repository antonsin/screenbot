"""
MSS-based screen capture backend (cross-platform)
"""
import numpy as np
import mss
import logging

logger = logging.getLogger(__name__)


class MSSSource:
    """Cross-platform screen capture using mss library"""
    
    def __init__(self):
        self.sct = mss.mss()
        logger.info("MSS capture backend initialized")
    
    def capture(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """
        Capture a region of the screen
        
        Args:
            x: Left coordinate
            y: Top coordinate
            w: Width
            h: Height
            
        Returns:
            numpy array in BGR format (OpenCV compatible)
        """
        monitor = {
            "left": x,
            "top": y,
            "width": w,
            "height": h
        }
        
        # Capture screen
        sct_img = self.sct.grab(monitor)
        
        # Convert to numpy array (BGRA format)
        frame = np.array(sct_img)
        
        # Convert BGRA to BGR (drop alpha channel)
        frame = frame[:, :, :3]
        
        # MSS returns RGB, convert to BGR for OpenCV
        frame = frame[:, :, ::-1]
        
        return frame
    
    def get_screen_size(self) -> tuple[int, int]:
        """Get primary monitor size"""
        monitor = self.sct.monitors[1]  # Primary monitor
        return monitor["width"], monitor["height"]
    
    def close(self):
        """Cleanup resources"""
        self.sct.close()
        logger.info("MSS capture backend closed")
