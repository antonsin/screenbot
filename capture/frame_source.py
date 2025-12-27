"""
Frame source - unified interface for screen capture with mode selection
"""
import logging
import platform
import json
from typing import Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class FrameSource:
    """
    Unified frame capture interface supporting:
    - Multiple backends (MSS, DXCam)
    - Multiple capture modes (absolute, window-anchored)
    """
    
    def __init__(self, config):
        """
        Initialize frame source
        
        Args:
            config: Configuration module
        """
        self.config = config
        self.backend = None
        self.backend_name = None
        self.capture_mode = config.CAPTURE_MODE
        
        # Load region override if exists
        self._load_region_override()
        
        # Initialize capture backend
        self._init_backend()
        
        # Cache for window position (used in window mode)
        self._window_rect = None
    
    def _load_region_override(self):
        """
        Load region.json if it exists and override config values
        """
        region_file = self.config.BASE_DIR / "config" / "region.json"
        
        if region_file.exists():
            try:
                with open(region_file, 'r') as f:
                    region_data = json.load(f)
                
                # Override config values
                if "capture_mode" in region_data:
                    self.capture_mode = region_data["capture_mode"]
                
                if "abs_x" in region_data:
                    self.config.ABS_X = region_data["abs_x"]
                if "abs_y" in region_data:
                    self.config.ABS_Y = region_data["abs_y"]
                if "abs_w" in region_data:
                    self.config.ABS_W = region_data["abs_w"]
                if "abs_h" in region_data:
                    self.config.ABS_H = region_data["abs_h"]
                
                logger.info(f"✓ Loaded region override from: {region_file}")
                logger.info(f"  Mode: {self.capture_mode}")
                logger.info(f"  Region: x={self.config.ABS_X}, y={self.config.ABS_Y}, "
                           f"w={self.config.ABS_W}, h={self.config.ABS_H}")
            
            except Exception as e:
                logger.warning(f"Failed to load region.json: {e}, using config.py defaults")
        else:
            logger.debug(f"No region override found at {region_file}, using config.py defaults")
    
    def _init_backend(self):
        """Initialize the capture backend based on configuration"""
        backend_choice = self.config.CAPTURE_BACKEND.lower()
        
        # Auto mode: try DXCam on Windows, fallback to MSS
        if backend_choice == "auto":
            if platform.system() == "Windows":
                try:
                    from .dxcam_source import DXCamSource
                    self.backend = DXCamSource()
                    self.backend_name = "dxcam"
                    logger.info("Auto-selected DXCam backend (Windows)")
                    return
                except Exception as e:
                    logger.warning(f"DXCam not available, falling back to MSS: {e}")
            
            # Fallback to MSS
            from .mss_source import MSSSource
            self.backend = MSSSource()
            self.backend_name = "mss"
            logger.info("Auto-selected MSS backend")
        
        # Explicit DXCam
        elif backend_choice == "dxcam":
            from .dxcam_source import DXCamSource
            self.backend = DXCamSource()
            self.backend_name = "dxcam"
            logger.info("Using DXCam backend")
        
        # Explicit MSS
        elif backend_choice == "mss":
            from .mss_source import MSSSource
            self.backend = MSSSource()
            self.backend_name = "mss"
            logger.info("Using MSS backend")
        
        else:
            raise ValueError(f"Unknown backend: {backend_choice}")
    
    def get_capture_region(self) -> tuple[int, int, int, int]:
        """
        Get the current capture region based on mode
        
        Returns:
            Tuple of (x, y, width, height) in absolute screen coordinates
        """
        if self.capture_mode == "absolute":
            return (
                self.config.ABS_X,
                self.config.ABS_Y,
                self.config.ABS_W,
                self.config.ABS_H
            )
        
        elif self.capture_mode == "window":
            # Try to find window
            win_rect = self._find_window()
            
            if win_rect is None:
                logger.warning(
                    f"Window containing '{self.config.WINDOW_TITLE_CONTAINS}' not found, "
                    "falling back to absolute mode"
                )
                return (
                    self.config.ABS_X,
                    self.config.ABS_Y,
                    self.config.ABS_W,
                    self.config.ABS_H
                )
            
            # Calculate absolute coordinates from window-relative
            win_x, win_y, win_w, win_h = win_rect
            abs_x = win_x + self.config.REL_X
            abs_y = win_y + self.config.REL_Y
            
            return (abs_x, abs_y, self.config.REL_W, self.config.REL_H)
        
        else:
            raise ValueError(f"Unknown capture mode: {self.capture_mode}")
    
    def _find_window(self) -> Optional[tuple[int, int, int, int]]:
        """
        Find window by title substring
        
        Returns:
            Tuple of (x, y, width, height) or None if not found
        """
        search_title = self.config.WINDOW_TITLE_CONTAINS.lower()
        
        # Platform-specific window finding
        if platform.system() == "Windows":
            return self._find_window_windows(search_title)
        elif platform.system() == "Linux":
            return self._find_window_linux(search_title)
        else:
            logger.warning(f"Window finding not implemented for {platform.system()}")
            return None
    
    def _find_window_windows(self, search_title: str) -> Optional[tuple[int, int, int, int]]:
        """Find window on Windows using win32gui"""
        try:
            import win32gui
            import win32con
        except ImportError:
            logger.warning("pywin32 not installed, window mode unavailable on Windows")
            return None
        
        found_rect = None
        
        def callback(hwnd, extra):
            nonlocal found_rect
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if search_title in title.lower():
                    rect = win32gui.GetWindowRect(hwnd)
                    # rect is (left, top, right, bottom)
                    x, y, right, bottom = rect
                    w = right - x
                    h = bottom - y
                    found_rect = (x, y, w, h)
                    return False  # Stop enumeration
            return True
        
        win32gui.EnumWindows(callback, None)
        
        if found_rect:
            logger.debug(f"Found window: {found_rect}")
        
        return found_rect
    
    def _find_window_linux(self, search_title: str) -> Optional[tuple[int, int, int, int]]:
        """Find window on Linux using X11 (best-effort)"""
        try:
            import subprocess
            
            # Try using wmctrl (if available)
            result = subprocess.run(
                ["wmctrl", "-lG"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode != 0:
                logger.warning("wmctrl not available, window mode may not work on Linux")
                return None
            
            # Parse wmctrl output
            # Format: window_id desktop x y w h host window_title
            for line in result.stdout.splitlines():
                parts = line.split(None, 7)
                if len(parts) >= 8:
                    title = parts[7]
                    if search_title in title.lower():
                        x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
                        logger.debug(f"Found window: ({x}, {y}, {w}, {h})")
                        return (x, y, w, h)
        
        except FileNotFoundError:
            logger.warning("wmctrl not found, window mode unavailable on Linux. Install: sudo apt-get install wmctrl")
        except Exception as e:
            logger.warning(f"Error finding window on Linux: {e}")
        
        return None
    
    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame
        
        Returns:
            numpy array in BGR format (OpenCV compatible)
        """
        x, y, w, h = self.get_capture_region()
        return self.backend.capture(x, y, w, h)
    
    def get_backend_name(self) -> str:
        """Get the name of the active backend"""
        return self.backend_name
    
    def get_capture_mode(self) -> str:
        """Get the current capture mode"""
        return self.capture_mode
    
    def get_screen_size(self) -> tuple[int, int]:
        """Get screen size"""
        return self.backend.get_screen_size()
    
    def close(self):
        """Cleanup resources"""
        if self.backend:
            self.backend.close()
