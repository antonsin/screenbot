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

# Import table segmentation for debug mode
if config.DEBUG_SEGMENTATION:
    try:
        from table_segmentation import segment_table
        TABLE_SEGMENTATION_AVAILABLE = True
        logger = logging.getLogger(__name__)
        logger.info("Table segmentation debug mode enabled")
    except ImportError as e:
        TABLE_SEGMENTATION_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning(f"table_segmentation import failed: {e}")
else:
    TABLE_SEGMENTATION_AVAILABLE = False

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
        
        # Segmentation warmup gating
        self.stable_grid_frames = config_module.STABLE_GRID_FRAMES
        self.min_expected_rows = config_module.MIN_EXPECTED_ROWS
        self.expected_num_cols = config_module.EXPECTED_NUM_COLS
        self.grid_stable_count = 0
        self.segmentation_ready = False
        self.last_grid_stats = None
        
        # Timing
        self.scan_interval = config_module.SCAN_INTERVAL_MS / 1000.0  # Convert to seconds
        
        # Debug segmentation setup
        self.debug_segmentation_enabled = config_module.DEBUG_SEGMENTATION and TABLE_SEGMENTATION_AVAILABLE
        self.debug_segmentation_counter = 0
        if self.debug_segmentation_enabled:
            self.debug_dir = config_module.BASE_DIR / "debug"
            self.debug_dir.mkdir(exist_ok=True)
            logger.info(f"Debug segmentation enabled, output to: {self.debug_dir}")
        
        logger.info("=" * 60)
        logger.info("Scanner Event Stream Initialized")
        logger.info("=" * 60)
        logger.info(f"Capture backend: {self.frame_source.get_backend_name()}")
        logger.info(f"Capture mode: {self.frame_source.get_capture_mode()}")
        logger.info(f"Scan interval: {config_module.SCAN_INTERVAL_MS}ms")
        logger.info(f"OCR enabled: {self.ocr_parser.enabled}")
        logger.info(f"OCR gating: {'GREEN only' if config_module.OCR_ONLY_IF_GAP_GREEN else 'All rows'}")
        logger.info(f"Database: {config_module.DB_PATH}")
        logger.info(f"Debug segmentation: {self.debug_segmentation_enabled}")
        logger.info("=" * 60)
    
    def run(self, max_seconds=None, max_frames=None):
        """
        Main event loop with optional time/frame limits
        
        Args:
            max_seconds: Maximum seconds to run (None = infinite)
            max_frames: Maximum frames to process (None = infinite)
        """
        logger.info("Starting scanner stream... Press Ctrl+C to stop")
        
        if max_seconds:
            logger.info(f"Will stop after {max_seconds} seconds")
        if max_frames:
            logger.info(f"Will stop after {max_frames} frames")
        
        start_time = time.time()
        
        try:
            while True:
                loop_start = time.time()
                
                # Check limits
                if max_seconds and (time.time() - start_time) >= max_seconds:
                    logger.info(f"Reached time limit: {max_seconds}s")
                    break
                if max_frames and self.frame_count >= max_frames:
                    logger.info(f"Reached frame limit: {max_frames}")
                    break
                
                # Capture ROI
                roi_img = self.frame_source.capture_frame()
                
                # Debug: Run table segmentation if enabled
                if self.debug_segmentation_enabled:
                    self._debug_table_segmentation(roi_img)
                
                # Segment rows
                rows = self.row_segmenter.segment_rows(roi_img)
                
                if not rows:
                    logger.debug("No rows detected in ROI")
                    continue
                
                # Check grid stability for warmup gating
                grid_ready = self._check_grid_stability(len(rows))
                
                if not grid_ready:
                    # Segmentation not ready yet, skip event processing
                    logger.debug(f"Grid warmup: {self.grid_stable_count}/{self.stable_grid_frames} stable frames")
                    continue
                
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
            try:
                # Save debug crops for every OCR attempt
                save_debug = self.config.EVENT_IMAGE_DEBUG
                debug_prefix = f"slot{row_idx}_evt{self.event_count + 1}"
                
                ocr_result = self.ocr_parser.parse_row(columns, 
                                                       save_debug=save_debug, 
                                                       debug_prefix=debug_prefix)
                
                # ALWAYS save crops that will be OCR'd to ocr_samples for verification
                self._save_ocr_samples(columns, row_idx, self.event_count + 1)
                
                # Update event data with OCR results (handle parsed as int not bool)
                if 'symbol' in ocr_result:
                    event_data['symbol'] = ocr_result['symbol']
                if 'price' in ocr_result:
                    event_data['price'] = ocr_result['price']
                if 'hits_5s' in ocr_result:
                    event_data['hits_5s'] = ocr_result['hits_5s']
                if 'row_time' in ocr_result:
                    event_data['row_time'] = ocr_result['row_time']
                if 'parsed' in ocr_result:
                    event_data['parsed'] = ocr_result['parsed']
                if 'raw_text' in ocr_result:
                    event_data['raw_text'] = ocr_result['raw_text']
                
                # Consolidate notes
                if 'notes' in ocr_result and ocr_result['notes']:
                    event_data['notes'] = '; '.join(ocr_result['notes']) if isinstance(ocr_result['notes'], list) else str(ocr_result['notes'])
                
            except Exception as e:
                logger.error(f"OCR parsing failed: {e}", exc_info=True)
                event_data['notes'] = f"OCR exception: {str(e)}"
                event_data['parsed'] = 0
        else:
            event_data['notes'] = f"OCR skipped: gap_color={gap_color_bucket}"
            event_data['parsed'] = 0
        
        # Save event to database (wrap in try-catch to prevent crash)
        try:
            event_id = self.store.insert_event(event_data)
        except Exception as e:
            logger.error(f"Failed to insert event: {e}", exc_info=True)
            event_id = 0
        
        # Update ticker state if we have a symbol (wrap in try-catch)
        appearances_60s = 0
        if event_data.get('symbol'):
            try:
                self.store.update_ticker_state(event_data['symbol'], event_data)
                ticker_state = self.store.get_ticker_state(event_data['symbol'])
                appearances_60s = ticker_state.get('appearances_60s', 0) if ticker_state else 0
            except Exception as e:
                logger.error(f"Failed to update ticker state: {e}", exc_info=True)
        
        # Calculate event latency
        event_latency = (time.time() - event_start) * 1000  # ms
        
        # Print concise event line
        self._print_event(event_data, appearances_60s, event_latency, event_id)
        
        # Save debug images if enabled (wrap in try-catch)
        if self.config.EVENT_IMAGE_DEBUG:
            try:
                self._save_debug_images(event_id, roi_img, row_img, columns)
            except Exception as e:
                logger.error(f"Failed to save debug images: {e}", exc_info=True)
        
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
        
        # Safe extraction with fallback to placeholders
        symbol = event_data.get('symbol') or 'UNKNOWN'
        price = event_data.get('price')
        gap_color = event_data.get('gap_color_bucket', 'UNKNOWN')
        hits = event_data.get('hits_5s')
        parsed = event_data.get('parsed', 0)  # 0 or 1 (not bool)
        
        # Format price with 4 decimal places
        try:
            price_str = f"${price:.2f}" if price is not None else "N/A"
        except (TypeError, ValueError):
            price_str = "N/A"
        
        # Format hits (handle None safely)
        try:
            hits_str = f"{hits}" if hits is not None else "N/A"
        except (TypeError, ValueError):
            hits_str = "N/A"
        
        # Color code based on gap color
        if gap_color == 'GREEN':
            color_indicator = "🟢"
        elif gap_color == 'WARM':
            color_indicator = "🔴"
        else:
            color_indicator = "⚪"
        
        # Ensure symbol is safe for printing
        symbol = str(symbol)[:6] if symbol else "UNKNOWN"
        
        print(f"{color_indicator} {timestamp} {symbol:6s} price={price_str:12s} gap={gap_color:7s} "
              f"hits={hits_str:3s} app60={appearances_60s:2d} parsed={parsed} "
              f"[{latency_ms:.1f}ms] [id={event_id}]")
        
        # If parsing failed, log notes if present
        if parsed == 0 and event_data.get('notes'):
            logger.info(f"  └─ Notes: {event_data['notes']}")
    
    def _save_ocr_samples(self, columns: dict, slot_idx: int, event_id: int):
        """
        Save exact crops that will be OCR'd for verification
        
        Args:
            columns: Column images dict
            slot_idx: Row slot index
            event_id: Event ID
        """
        debug_dir = self.config.DEBUG_FRAMES_DIR / "ocr_samples"
        debug_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save symbol and price crops (both raw and preprocessed)
        for field in ['symbol', 'price']:
            if field in columns:
                col_img = columns[field]
                if col_img is not None and col_img.size > 0:
                    # Save raw crop
                    raw_file = debug_dir / f"evt{event_id:04d}_slot{slot_idx}_{field}_raw_{timestamp}.png"
                    cv2.imwrite(str(raw_file), col_img)
                    
                    # Save preprocessed crop (what Tesseract sees)
                    try:
                        preprocessed = self.ocr_parser._preprocess_for_ocr(col_img, mode=field)
                        pre_file = debug_dir / f"evt{event_id:04d}_slot{slot_idx}_{field}_preprocessed_{timestamp}.png"
                        cv2.imwrite(str(pre_file), preprocessed)
                    except Exception as e:
                        logger.debug(f"Failed to save preprocessed {field}: {e}")
        
        logger.debug(f"Saved OCR sample crops for event {event_id}")
    
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
    
    def _check_grid_stability(self, num_rows: int) -> bool:
        """
        Check if segmentation grid is stable enough for OCR
        
        Args:
            num_rows: Number of rows detected
            
        Returns:
            True if grid is stable and ready
        """
        # Current grid stats
        current_stats = {
            'num_rows': num_rows,
            # We only check rows for now; could add column count if column_slicer exposes it
        }
        
        # Check if grid meets minimum criteria
        grid_valid = (num_rows >= self.min_expected_rows)
        
        # Check if grid is same as last frame
        grid_unchanged = (self.last_grid_stats and 
                         current_stats['num_rows'] == self.last_grid_stats['num_rows'])
        
        if grid_valid and grid_unchanged:
            self.grid_stable_count += 1
        else:
            self.grid_stable_count = 0
        
        self.last_grid_stats = current_stats
        
        # Mark as ready once we hit the threshold
        if not self.segmentation_ready and self.grid_stable_count >= self.stable_grid_frames:
            self.segmentation_ready = True
            # Get detailed segmentation info for logging
            logger.info("=" * 80)
            logger.info(f"✓ SEGMENTATION READY")
            logger.info(f"  Rows detected: {num_rows}")
            logger.info(f"  Stable frames: {self.grid_stable_count}/{self.stable_grid_frames}")
            logger.info(f"  Grid stats: {current_stats}")
            logger.info("=" * 80)
        
        return self.segmentation_ready
    
    def _update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        
        elapsed = time.time() - self.last_fps_update
        if elapsed >= 5.0:  # Update every 5 seconds
            self.current_fps = self.frame_count / elapsed
            logger.info(f"FPS: {self.current_fps:.1f} | Events: {self.event_count}")
            self.frame_count = 0
            self.last_fps_update = time.time()
    
    def _debug_table_segmentation(self, roi_img: np.ndarray):
        """
        Debug helper: Run table segmentation and log results
        
        Args:
            roi_img: ROI image to segment
        """
        try:
            # Only run periodically (every 10 frames) to reduce overhead
            self.debug_segmentation_counter += 1
            if self.debug_segmentation_counter % 10 != 0:
                return
            
            # Run table segmentation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_debug_dir = self.debug_dir / f"frame_{timestamp}"
            
            binary, ys, xs, cells = segment_table(
                roi_img,
                frame_debug_dir,
                prefer_lines=True,
                min_rows=2,
                min_cols=2
            )
            
            # Log results
            logger.info(f"[DEBUG SEGMENTATION] Frame {self.debug_segmentation_counter}:")
            logger.info(f"  Y-separators (rows): {len(ys)}")
            logger.info(f"  X-separators (cols): {len(xs)}")
            logger.info(f"  Total cells: {len(cells)}")
            logger.info(f"  Debug output: {frame_debug_dir}")
            
        except Exception as e:
            logger.error(f"Debug segmentation failed: {e}", exc_info=True)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Stream scanner events')
    parser.add_argument('--max-seconds', type=int, help='Stop after N seconds')
    parser.add_argument('--max-frames', type=int, help='Stop after N frames')
    
    args = parser.parse_args()
    
    streamer = ScannerStreamer(config)
    streamer.run(max_seconds=args.max_seconds, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
