#!/usr/bin/env python3
"""
Live preview tool for visual verification of capture region

Usage:
    python tools/preview_capture.py --view crop   # Show only captured region (default)
    python tools/preview_capture.py --view full   # Show full screen with overlay
    python tools/preview_capture.py --view full --select-roi  # Select and save ROI interactively

Hotkeys:
    q - Quit
    p - Save screenshot to debug_frames/
"""
import sys
import argparse
import time
import json
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
    
    def __init__(self, view_mode: str = "crop", select_roi: bool = False):
        """
        Initialize preview
        
        Args:
            view_mode: "crop" or "full"
            select_roi: If True, allow ROI selection
        """
        self.view_mode = view_mode
        self.select_roi = select_roi
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
    
    def _save_region_config(self, x: int, y: int, w: int, h: int):
        """
        Save selected ROI to config/region.json
        
        Args:
            x, y, w, h: Region coordinates
        """
        # Ensure config directory exists
        config_dir = config.BASE_DIR / "config"
        config_dir.mkdir(exist_ok=True)
        
        region_data = {
            "capture_mode": "absolute",
            "abs_x": int(x),
            "abs_y": int(y),
            "abs_w": int(w),
            "abs_h": int(h),
            "created_at": datetime.now().isoformat()
        }
        
        region_file = config_dir / "region.json"
        with open(region_file, 'w') as f:
            json.dump(region_data, f, indent=2)
        
        logger.info(f"✓ Region saved to: {region_file}")
        logger.info(f"✓ Coordinates: x={x}, y={y}, w={w}, h={h}")
        print(f"\n{'='*60}")
        print(f"ROI SAVED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"File: {region_file}")
        print(f"Coordinates:")
        print(f"  X: {x}")
        print(f"  Y: {y}")
        print(f"  Width: {w}")
        print(f"  Height: {h}")
        print(f"{'='*60}\n")
    
    def select_roi_interactive(self):
        """
        Interactive ROI selection using OpenCV
        
        Returns:
            Tuple of (x, y, w, h) or None if cancelled
        """
        logger.info("=" * 60)
        logger.info("INTERACTIVE ROI SELECTION")
        logger.info("=" * 60)
        logger.info("Instructions:")
        logger.info("  1. Drag to select the scan region")
        logger.info("  2. Press ENTER to confirm")
        logger.info("  3. Press C to cancel")
        logger.info("=" * 60)
        
        # Capture full screen
        screen_w, screen_h = self.frame_source.get_screen_size()
        frame = self.frame_source.backend.capture(0, 0, screen_w, screen_h)
        
        # Ensure frame is OpenCV-compatible
        frame = np.ascontiguousarray(frame)
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = frame.copy()
        
        # Use cv2.selectROI for interactive selection
        window_name = "Select ROI - Press ENTER to confirm, C to cancel"
        roi = cv2.selectROI(window_name, frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_name)
        
        x, y, w, h = roi
        
        # Check if selection was cancelled (all zeros)
        if w == 0 or h == 0:
            logger.warning("ROI selection cancelled")
            return None
        
        logger.info(f"ROI selected: x={x}, y={y}, w={w}, h={h}")
        return (int(x), int(y), int(w), int(h))
    
    def run(self):
        """Run preview loop"""
        # If ROI selection mode, run selection and exit
        if self.select_roi:
            roi = self.select_roi_interactive()
            if roi:
                x, y, w, h = roi
                self._save_region_config(x, y, w, h)
            self.frame_source.close()
            return
        
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
    parser.add_argument(
        "--select-roi",
        action="store_true",
        help="Interactive ROI selection mode - select and save capture region"
    )
    
    args = parser.parse_args()
    
    preview = CapturePreview(view_mode=args.view, select_roi=args.select_roi)
    preview.run()


if __name__ == "__main__":
    main()
