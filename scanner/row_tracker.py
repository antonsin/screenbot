"""
Row tracker - stable fingerprint-based row tracking with duplicate suppression
"""
import cv2
import numpy as np
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RowSlotState:
    """State for a specific row slot/index"""
    last_fp: bytes
    last_emit_ts: float
    last_bbox: tuple[int, int, int, int]
    stable_count: int = 0


class RowTracker:
    """
    Tracks rows using stable fingerprinting with duplicate suppression
    """
    
    def __init__(self, config, max_top_rows: int = 3):
        """
        Initialize row tracker
        
        Args:
            config: Configuration module
            max_top_rows: Number of top rows to check
        """
        self.config = config
        self.max_top_rows = max_top_rows
        
        # Configuration
        self.cooldown_sec = config.ROW_EVENT_COOLDOWN_SEC
        self.fp_size = config.ROW_FP_SIZE
        self.fp_blur_k = config.ROW_FP_BLUR_K
        self.bbox_jitter_px = config.ROW_BBOX_JITTER_PX
        
        # Per-slot state tracking
        self.slot_states = {}  # slot_index -> RowSlotState
        
        # Statistics for rate-limited logging
        self.suppressed_count = 0
        self.emitted_count = 0
        self.last_stats_log = time.time()
        
        logger.info(f"Row tracker initialized: cooldown={self.cooldown_sec}s, "
                   f"fp_size={self.fp_size}, jitter_threshold={self.bbox_jitter_px}px")
    
    def check_new_rows(self, rows: list[tuple[int, int, int, int]], roi_img: np.ndarray) -> list[dict]:
        """
        Check top rows for new entries with duplicate suppression
        
        Args:
            rows: List of (x, y, w, h) row boxes
            roi_img: Full ROI image
            
        Returns:
            List of dicts with new row info: {index, bbox, hash, image, reason}
        """
        new_rows = []
        now = time.time()
        
        # Check only top N rows
        check_count = min(self.max_top_rows, len(rows))
        
        for idx in range(check_count):
            x, y, w, h = rows[idx]
            bbox = (x, y, w, h)
            
            # Extract row image
            row_img = roi_img[y:y+h, x:x+w].copy()
            
            # Compute stable fingerprint
            row_fp = self._fingerprint_row(row_img)
            
            # Check if we should emit event for this slot
            should_emit, reason = self._should_emit_event(idx, row_fp, bbox, now)
            
            if should_emit:
                # Create hash string for backward compatibility
                row_hash = row_fp.hex()[:32]
                
                logger.info(f"New row detected at slot {idx} (reason: {reason})")
                
                new_rows.append({
                    'index': idx,
                    'bbox': bbox,
                    'hash': row_hash,
                    'image': row_img,
                    'reason': reason
                })
                
                # Update slot state
                self._update_slot_state(idx, row_fp, bbox, now)
                
                self.emitted_count += 1
            else:
                self.suppressed_count += 1
        
        # Rate-limited stats logging (every 2 seconds)
        if now - self.last_stats_log >= 2.0:
            if self.suppressed_count > 0 or self.emitted_count > 0:
                logger.debug(f"Event stats: emitted={self.emitted_count}, suppressed={self.suppressed_count}")
                self.suppressed_count = 0
                self.emitted_count = 0
            self.last_stats_log = now
        
        return new_rows
    
    def _fingerprint_row(self, row_img: np.ndarray) -> bytes:
        """
        Compute stable fingerprint of row image
        
        Args:
            row_img: Row image (BGR)
            
        Returns:
            Fingerprint as bytes
        """
        # Convert to grayscale
        gray = cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY)
        
        # Downscale aggressively to target width
        h, w = gray.shape
        aspect_ratio = w / h if h > 0 else 1
        target_width = self.fp_size
        target_height = max(1, int(target_width / aspect_ratio))
        resized = cv2.resize(gray, (target_width, target_height))
        
        # Apply slight blur to reduce noise
        blur_k = self.fp_blur_k if self.fp_blur_k % 2 == 1 else self.fp_blur_k + 1  # Must be odd
        blurred = cv2.GaussianBlur(resized, (blur_k, blur_k), 0)
        
        # Apply threshold to get stable binary representation
        _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Pack into bytes (bitstring)
        bits = (thresholded > 127).flatten()
        packed = np.packbits(bits)
        
        return packed.tobytes()
    
    def _should_emit_event(self, slot_idx: int, fp: bytes, bbox: tuple[int, int, int, int], now: float) -> tuple[bool, str]:
        """
        Determine if event should be emitted for this slot
        
        Args:
            slot_idx: Row slot index
            fp: Fingerprint bytes
            bbox: Bounding box (x, y, w, h)
            now: Current timestamp
            
        Returns:
            Tuple of (should_emit, reason)
        """
        # First time seeing this slot
        if slot_idx not in self.slot_states:
            return True, "first_detection"
        
        state = self.slot_states[slot_idx]
        
        # Check fingerprint change
        if fp != state.last_fp:
            return True, "fp_changed"
        
        # Check bounding box movement
        x, y, w, h = bbox
        last_x, last_y, last_w, last_h = state.last_bbox
        
        dy = abs(y - last_y)
        dh = abs(h - last_h)
        
        if dy > self.bbox_jitter_px or dh > self.bbox_jitter_px:
            return True, f"bbox_moved(dy={dy},dh={dh})"
        
        # Check cooldown
        elapsed = now - state.last_emit_ts
        if elapsed > self.cooldown_sec:
            return True, f"cooldown_expired({elapsed:.1f}s)"
        
        # Suppress - same content, stable position, within cooldown
        return False, "suppressed"
    
    def _update_slot_state(self, slot_idx: int, fp: bytes, bbox: tuple[int, int, int, int], now: float):
        """
        Update state for a slot after emitting event
        
        Args:
            slot_idx: Row slot index
            fp: Fingerprint bytes
            bbox: Bounding box
            now: Current timestamp
        """
        if slot_idx in self.slot_states:
            state = self.slot_states[slot_idx]
            state.last_fp = fp
            state.last_emit_ts = now
            state.last_bbox = bbox
            state.stable_count += 1
        else:
            self.slot_states[slot_idx] = RowSlotState(
                last_fp=fp,
                last_emit_ts=now,
                last_bbox=bbox,
                stable_count=1
            )
