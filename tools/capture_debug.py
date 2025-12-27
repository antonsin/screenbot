#!/usr/bin/env python3
"""
Capture debug tool - capture and save frames for debugging

Usage:
    python tools/capture_debug.py [--count N]
"""
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import config
from capture import FrameSource
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def capture_debug_frames(count: int = 5):
    """
    Capture multiple frames and save them for debugging
    
    Args:
        count: Number of frames to capture
    """
    logger.info("=" * 60)
    logger.info("CAPTURE DEBUG")
    logger.info("=" * 60)
    
    # Initialize frame source
    frame_source = FrameSource(config)
    
    # Display configuration
    logger.info(f"Backend: {frame_source.get_backend_name()}")
    logger.info(f"Capture mode: {frame_source.get_capture_mode()}")
    
    x, y, w, h = frame_source.get_capture_region()
    logger.info(f"Capture region: x={x}, y={y}, w={w}, h={h}")
    
    logger.info(f"Saving {count} frames to: {config.DEBUG_FRAMES_DIR}")
    logger.info("-" * 60)
    
    try:
        for i in range(count):
            # Capture frame
            logger.info(f"Capturing frame {i+1}/{count}...")
            frame = frame_source.capture_frame()
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = config.DEBUG_FRAMES_DIR / f"debug_{timestamp}.png"
            
            # Add metadata overlay
            overlay_frame = frame.copy()
            info_text = [
                f"Frame {i+1}/{count}",
                f"Backend: {frame_source.get_backend_name()}",
                f"Mode: {frame_source.get_capture_mode()}",
                f"Region: ({x}, {y}, {w}, {h})",
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ]
            
            # Draw semi-transparent background
            bg_height = len(info_text) * 25 + 20
            bg = overlay_frame.copy()
            cv2.rectangle(bg, (10, 10), (500, bg_height), (0, 0, 0), -1)
            overlay_frame = cv2.addWeighted(overlay_frame, 0.7, bg, 0.3, 0)
            
            # Draw text
            y_offset = 35
            for line in info_text:
                cv2.putText(
                    overlay_frame, line, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA
                )
                y_offset += 25
            
            # Save frame
            cv2.imwrite(str(filename), overlay_frame)
            logger.info(f"  Saved: {filename.name}")
            
            # Small delay between captures
            if i < count - 1:
                time.sleep(0.5)
        
        logger.info("-" * 60)
        logger.info(f"✓ Successfully captured {count} frames")
        logger.info(f"✓ Saved to: {config.DEBUG_FRAMES_DIR}")
        logger.info("=" * 60)
    
    except Exception as e:
        logger.error(f"Error during capture: {e}", exc_info=True)
    
    finally:
        frame_source.close()


def main():
    parser = argparse.ArgumentParser(
        description="Capture debug frames for troubleshooting"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of frames to capture (default: 5)"
    )
    
    args = parser.parse_args()
    
    capture_debug_frames(count=args.count)


if __name__ == "__main__":
    main()
