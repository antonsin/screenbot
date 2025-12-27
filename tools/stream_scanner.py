#!/usr/bin/env python3
"""
Scanner event stream - detect new rows and emit structured events

Usage:
    python tools/stream_scanner.py
"""
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import config
from capture import FrameSource
from scanner.row_segmenter import RowSegmenter
from scanner.row_tracker import RowTracker
from scanner.slicer import ColumnSlicer
from scanner.color_features import ColorFeatureExtractor
from scanner.ocr_parse import OCRParser
from scanner.store import ScannerStore
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScannerStreamer:
    """
    Main scanner event streaming pipeline
    """
    
    def __init__(self, config_module):
        """
        Initialize scanner streamer
        
        Args:
            config_module: Configuration module
        """
        self.config = config_module
        
        # Initialize components
        self.frame_source = FrameSource(config_module)
        self.row_segmenter = RowSegmenter(config_module)
        self.row_tracker = RowTracker(config_module, max_top_rows=config_module.MAX_TOP_ROWS_TO_CHECK)
        self.column_slicer = ColumnSlicer(config_module)
        self.color_extractor = ColorFeatureExtractor(config_module)
        self.ocr_parser = OCRParser(config_module)
        self.store = ScannerStore(config_module)
        
        # Performance tracking
        self.frame_count = 0
        self.event_count = 0
        self.last_fps_update = time.time()
        self.current_fps = 0.0
        
        # Timing
        self.scan_interval = config_module.SCAN_INTERVAL_MS / 1000.0  # Convert to seconds
        
        logger.info("=" * 60)
        logger.info("Scanner Event Stream Initialized")
        logger.info("=" * 60)
        logger.info(f"Capture backend: {self.frame_source.get_backend_name()}")
        logger.info(f"Capture mode: {self.frame_source.get_capture_mode()}")
        logger.info(f"Scan interval: {config_module.SCAN_INTERVAL_MS}ms")
        logger.info(f"OCR enabled: {self.ocr_parser.enabled}")
        logger.info(f"OCR gating: {'GREEN only' if config_module.OCR_ONLY_IF_GAP_GREEN else 'All rows'}")
        logger.info(f"Database: {config_module.DB_PATH}")
        logger.info("=" * 60)
    
    def run(self):
        """
        Main event loop
        """
        logger.info("Starting scanner stream... Press Ctrl+C to stop")
        
        try:
            while True:
                loop_start = time.time()
                
                # Capture ROI
                roi_img = self.frame_source.capture_frame()
                
                # Segment rows
                rows = self.row_segmenter.segment_rows(roi_img)
                
                # Check for new top rows
                new_rows = self.row_tracker.check_new_rows(rows, roi_img)
                
                # Process each new row
                for new_row in new_rows:
                    self._process_new_row(new_row, roi_img)
                
                # Update FPS
                self._update_fps()
                
                # Maintain scan interval
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.scan_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            logger.info("\nStopping scanner stream...")
        
        finally:
            self.frame_source.close()
            logger.info(f"Total frames: {self.frame_count}")
            logger.info(f"Total events: {self.event_count}")
            logger.info("Scanner stream closed")
    
    def _process_new_row(self, new_row: dict, roi_img: np.ndarray):
        """
        Process a newly detected row
        
        Args:
            new_row: New row dict from tracker
            roi_img: Full ROI image
        """
        event_start = time.time()
        
        row_img = new_row['image']
        row_hash = new_row['hash']
        row_idx = new_row['index']
        
        # Slice columns
        columns = self.column_slicer.slice_all_columns(row_img)
        
        # Extract gap color features (fast gating)
        gap_cell = columns.get('gap')
        color_result = self.color_extractor.classify_gap_color(gap_cell)
        
        gap_color_bucket = color_result['gap_color_bucket']
        h_mean = color_result['h_mean']
        s_mean = color_result['s_mean']
        v_mean = color_result['v_mean']
        
        # Determine if OCR should run
        should_ocr = self.ocr_parser.should_run_ocr(gap_color_bucket)
        
        # Initialize event data
        event_data = {
            'row_hash': row_hash,
            'gap_color_bucket': gap_color_bucket,
            'h_mean': h_mean,
            's_mean': s_mean,
            'v_mean': v_mean,
            'parsed': False,
            'symbol': None,
            'price': None,
            'hits_5s': None,
            'row_time': None,
            'raw_text': {}
        }
        
        # Run OCR if gating allows
        if should_ocr:
            ocr_result = self.ocr_parser.parse_row(columns)
            event_data.update(ocr_result)
        else:
            event_data['notes'] = f"OCR skipped: gap_color={gap_color_bucket}"
        
        # Save event to database
        event_id = self.store.insert_event(event_data)
        
        # Update ticker state if we have a symbol
        if event_data.get('symbol'):
            self.store.update_ticker_state(event_data['symbol'], event_data)
            ticker_state = self.store.get_ticker_state(event_data['symbol'])
            appearances_60s = ticker_state.get('appearances_60s', 0) if ticker_state else 0
        else:
            appearances_60s = 0
        
        # Calculate event latency
        event_latency = (time.time() - event_start) * 1000  # ms
        
        # Print concise event line
        self._print_event(event_data, appearances_60s, event_latency, event_id)
        
        # Save debug images if enabled
        if self.config.EVENT_IMAGE_DEBUG:
            self._save_debug_images(event_id, roi_img, row_img, columns)
        
        self.event_count += 1
    
    def _print_event(self, event_data: dict, appearances_60s: int, latency_ms: float, event_id: int):
        """
        Print concise event line
        
        Args:
            event_data: Event data
            appearances_60s: Appearance count in last 60s
            latency_ms: Event processing latency in ms
            event_id: Event ID
        """
        timestamp = datetime.now().strftime("%I:%M:%S %p")
        
        symbol = event_data.get('symbol', 'UNKNOWN')
        price = event_data.get('price')
        gap_color = event_data['gap_color_bucket']
        hits = event_data.get('hits_5s')
        parsed = 1 if event_data.get('parsed') else 0
        
        # Format price
        price_str = f"${price:.2f}" if price else "N/A"
        
        # Format hits
        hits_str = f"{hits}" if hits else "N/A"
        
        # Color code based on gap color
        if gap_color == 'GREEN':
            color_indicator = "🟢"
        elif gap_color == 'WARM':
            color_indicator = "🔴"
        else:
            color_indicator = "⚪"
        
        print(f"{color_indicator} {timestamp} {symbol:6s} price={price_str:8s} gap={gap_color:7s} "
              f"hits={hits_str:3s} app60={appearances_60s:2d} parsed={parsed} "
              f"[{latency_ms:.1f}ms] [id={event_id}]")
    
    def _save_debug_images(self, event_id: int, roi_img: np.ndarray, row_img: np.ndarray, columns: dict):
        """
        Save debug images for event
        
        Args:
            event_id: Event ID
            roi_img: Full ROI
            row_img: Row image
            columns: Column images
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_dir = self.config.DEBUG_FRAMES_DIR / "changes"
        debug_dir.mkdir(exist_ok=True)
        
        # Save ROI snapshot
        roi_file = debug_dir / f"event_{event_id:06d}_{timestamp}_roi.png"
        cv2.imwrite(str(roi_file), roi_img)
        
        # Save row image
        row_file = debug_dir / f"event_{event_id:06d}_{timestamp}_row.png"
        cv2.imwrite(str(row_file), row_img)
        
        # Save gap cell
        if 'gap' in columns:
            gap_file = debug_dir / f"event_{event_id:06d}_{timestamp}_gap.png"
            cv2.imwrite(str(gap_file), columns['gap'])
        
        logger.debug(f"Saved debug images for event {event_id}")
    
    def _update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        
        elapsed = time.time() - self.last_fps_update
        if elapsed >= 5.0:  # Update every 5 seconds
            self.current_fps = self.frame_count / elapsed
            logger.info(f"FPS: {self.current_fps:.1f} | Events: {self.event_count}")
            self.frame_count = 0
            self.last_fps_update = time.time()


def main():
    streamer = ScannerStreamer(config)
    streamer.run()


if __name__ == "__main__":
    main()
