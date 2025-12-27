"""
DXCam-based screen capture backend (Windows-only, high performance)
"""
import numpy as np
import logging
import platform

logger = logging.getLogger(__name__)


class DXCamSource:
    """Windows-only high-performance screen capture using DirectX"""
    
    def __init__(self):
        if platform.system() != "Windows":
            raise RuntimeError("DXCam is only available on Windows")
        
        try:
            import dxcam
            self.camera = dxcam.create()
            logger.info("DXCam capture backend initialized")
        except ImportError:
            raise ImportError("dxcam not installed. Install with: pip install dxcam")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DXCam: {e}")
    
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
        # DXCam region is (left, top, right, bottom)
        region = (x, y, x + w, y + h)
        
        # Capture frame
        frame = self.camera.grab(region=region)
        
        if frame is None:
            raise RuntimeError("DXCam failed to capture frame")
        
        # DXCam returns RGB, convert to BGR for OpenCV
        frame = frame[:, :, ::-1]
        
        return frame
    
    def get_screen_size(self) -> tuple[int, int]:
        """Get primary monitor size"""
        # Get screen dimensions from DXCam
        return self.camera.width, self.camera.height
    
    def close(self):
        """Cleanup resources"""
        if hasattr(self, 'camera') and self.camera:
            self.camera.release()
            logger.info("DXCam capture backend closed")
