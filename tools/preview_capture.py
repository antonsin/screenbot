#!/usr/bin/env python3
"""
Live preview tool for visual verification of capture region

Usage:
    python tools/preview_capture.py --view crop   # Show only captured region (default)
    python tools/preview_capture.py --view full   # Show full screen with overlay

Hotkeys:
    q - Quit
    p - Save screenshot to debug_frames/
"""
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import config
from capture import FrameSource
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CapturePreview:
    """Visual preview of capture region"""
    
    def __init__(self, view_mode: str = "crop"):
        """
        Initialize preview
        
        Args:
            view_mode: "crop" or "full"
        """
        self.view_mode = view_mode
        self.frame_source = FrameSource(config)
        self.fps = 0
        self.frame_count = 0
        self.last_fps_update = time.time()
        
        logger.info(f"Preview mode: {view_mode}")
        logger.info(f"Backend: {self.frame_source.get_backend_name()}")
        logger.info(f"Capture mode: {self.frame_source.get_capture_mode()}")
    
    def _calculate_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_update
        
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def _draw_overlay_text(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Draw information overlay on frame"""
        # Ensure frame is OpenCV-compatible (C-contiguous, writable, BGR format)
        frame = np.ascontiguousarray(frame)
        
        # Convert BGRA to BGR if needed (4 channels -> 3 channels)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Create writable copy
        frame = frame.copy()
        overlay = frame.copy()
        
        # Information to display
        info_lines = [
            f"Backend: {self.frame_source.get_backend_name()}",
            f"Mode: {self.frame_source.get_capture_mode()}",
            f"Region: x={x}, y={y}, w={w}, h={h}",
            f"FPS: {self.fps:.1f}",
            "",
            "Hotkeys: [Q]uit  [P]rint screenshot"
        ]
        
        # Draw semi-transparent background
        bg_height = len(info_lines) * 25 + 20
        cv2.rectangle(overlay, (10, 10), (400, bg_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw text
        y_offset = 35
        for line in info_lines:
            cv2.putText(
                frame, line, (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1, cv2.LINE_AA
            )
            y_offset += 25
        
        return frame
    
    def _draw_capture_rectangle(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Draw capture region rectangle on full screen"""
        # Ensure frame is OpenCV-compatible (C-contiguous, writable, BGR format)
        frame = np.ascontiguousarray(frame)
        
        # Convert BGRA to BGR if needed (4 channels -> 3 channels)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Create writable copy
        frame = frame.copy()
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw corner markers
        marker_size = 20
        thickness = 3
        color = (0, 255, 0)
        
        # Top-left
        cv2.line(frame, (x, y), (x + marker_size, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + marker_size), color, thickness)
        
        # Top-right
        cv2.line(frame, (x + w, y), (x + w - marker_size, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + marker_size), color, thickness)
        
        # Bottom-left
        cv2.line(frame, (x, y + h), (x + marker_size, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - marker_size), color, thickness)
        
        # Bottom-right
        cv2.line(frame, (x + w, y + h), (x + w - marker_size, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - marker_size), color, thickness)
        
        return frame
    
    def _save_screenshot(self, frame: np.ndarray):
        """Save screenshot to debug_frames/"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = config.DEBUG_FRAMES_DIR / f"preview_{timestamp}.png"
        cv2.imwrite(str(filename), frame)
        logger.info(f"Screenshot saved: {filename}")
    
    def run(self):
        """Run preview loop"""
        window_name = f"ScreenBot Preview - {self.view_mode.upper()} mode"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        logger.info("Preview started. Press 'q' to quit, 'p' to save screenshot")
        
        try:
            while True:
                # Get capture region
                x, y, w, h = self.frame_source.get_capture_region()
                
                # Update FPS
                self._calculate_fps()
                
                if self.view_mode == "crop":
                    # Capture and show cropped region
                    frame = self.frame_source.capture_frame()
                    frame = self._draw_overlay_text(frame, x, y, w, h)
                    display_frame = frame
                
                elif self.view_mode == "full":
                    # Capture full screen and draw overlay
                    screen_w, screen_h = self.frame_source.get_screen_size()
                    full_frame = self.frame_source.backend.capture(0, 0, screen_w, screen_h)
                    
                    # Draw capture rectangle
                    full_frame = self._draw_capture_rectangle(full_frame, x, y, w, h)
                    full_frame = self._draw_overlay_text(full_frame, x, y, w, h)
                    
                    display_frame = full_frame
                
                # Show frame
                cv2.imshow(window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('p'):
                    self._save_screenshot(display_frame)
                
                # Maintain target FPS
                time.sleep(1.0 / config.TARGET_FPS)
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            cv2.destroyAllWindows()
            self.frame_source.close()
            logger.info("Preview closed")


def main():
    parser = argparse.ArgumentParser(
        description="Live preview tool for capture verification"
    )
    parser.add_argument(
        "--view",
        choices=["crop", "full"],
        default="crop",
        help="View mode: 'crop' shows captured region only, 'full' shows entire screen with overlay"
    )
    
    args = parser.parse_args()
    
    preview = CapturePreview(view_mode=args.view)
    preview.run()


if __name__ == "__main__":
    main()
