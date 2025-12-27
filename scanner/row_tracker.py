"""
Row tracker - dHash-based row tracking with Hamming distance and stability frames
"""
import cv2
import numpy as np
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RowSlotState:
    """State for a specific row slot/index"""
    last_fp: int  # dHash as uint64
    last_emit_ts: float
    last_bbox: tuple[int, int, int, int]
    
    # Stability tracking
    candidate_fp: Optional[int] = None
    candidate_bbox: Optional[tuple[int, int, int, int]] = None
    stable_frames: int = 0


class RowTracker:
    """
    Tracks rows using dHash with Hamming distance and stability frames
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
        self.hamming_thresh = config.ROW_FP_HAMMING_THRESH
        self.stability_frames = config.ROW_STABILITY_FRAMES
        self.bbox_jitter_px = config.ROW_BBOX_JITTER_PX
        
        # Per-slot state tracking
        self.slot_states = {}  # slot_index -> RowSlotState
        
        # Statistics for rate-limited logging
        self.suppressed_count = 0
        self.emitted_count = 0
        self.last_stats_log = time.time()
        
        logger.info(f"Row tracker initialized: hamming_thresh={self.hamming_thresh}, "
                   f"stability_frames={self.stability_frames}, jitter={self.bbox_jitter_px}px, "
                   f"cooldown={self.cooldown_sec}s")
    
    def check_new_rows(self, rows: list[tuple[int, int, int, int]], roi_img: np.ndarray) -> list[dict]:
        """
        Check top rows for new entries with stability-based duplicate suppression
        
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
            
            # Compute dHash fingerprint
            row_fp = self._dhash_64(row_img)
            
            # Check if we should emit event for this slot
            should_emit, reason = self._should_emit_event(idx, row_fp, bbox, now)
            
            if should_emit:
                # Create hash string for backward compatibility
                row_hash = format(row_fp, '016x')[:32]
                
                logger.info(f"New row detected at slot {idx} (reason: {reason})")
                
                new_rows.append({
                    'index': idx,
                    'bbox': bbox,
                    'hash': row_hash,
                    'image': row_img,
                    'reason': reason
                })
                
                # Update slot state - emit happened, reset stability tracking
                self._update_slot_state_after_emit(idx, row_fp, bbox, now)
                
                self.emitted_count += 1
            else:
                # Update stability tracking for this slot
                self._update_stability_tracking(idx, row_fp, bbox)
                self.suppressed_count += 1
        
        # Rate-limited stats logging (every 2 seconds)
        if now - self.last_stats_log >= 2.0:
            if self.suppressed_count > 0 or self.emitted_count > 0:
                logger.debug(f"Event stats (2s window): emitted={self.emitted_count}, suppressed={self.suppressed_count}")
                self.suppressed_count = 0
                self.emitted_count = 0
            self.last_stats_log = now
        
        return new_rows
    
    def _dhash_64(self, row_img: np.ndarray) -> int:
        """
        Compute 64-bit difference hash (dHash) of row image
        
        Args:
            row_img: Row image (BGR)
            
        Returns:
            dHash as uint64
        """
        # Convert to grayscale
        if len(row_img.shape) == 3:
            gray = cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = row_img
        
        # Resize to 9x8 for dHash (need 9 width for 8-bit horizontal gradient)
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        
        # Compute horizontal gradient (compare adjacent pixels)
        # Result: 8x8 boolean array
        diff = resized[:, 1:] > resized[:, :-1]
        
        # Pack into 64-bit integer
        bits = diff.flatten()
        hash_val = 0
        for i, bit in enumerate(bits):
            if bit:
                hash_val |= (1 << i)
        
        return hash_val
    
    def _hamming_distance(self, hash1: int, hash2: int) -> int:
        """
        Compute Hamming distance between two hashes
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            Number of differing bits
        """
        xor = hash1 ^ hash2
        return bin(xor).count('1')
    
    def _bbox_distance(self, bbox1: tuple[int, int, int, int], bbox2: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """
        Compute component-wise distance between bounding boxes
        
        Args:
            bbox1: First bbox (x, y, w, h)
            bbox2: Second bbox (x, y, w, h)
            
        Returns:
            Tuple of (dx, dy, dw, dh) - absolute differences
        """
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        dw = abs(w1 - w2)
        dh = abs(h1 - h2)
        
        return (dx, dy, dw, dh)
    
    def _is_material_change(self, fp1: int, fp2: int, bbox1: tuple[int, int, int, int], 
                           bbox2: tuple[int, int, int, int]) -> tuple[bool, str]:
        """
        Determine if there's a material change between two states
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            bbox1: First bounding box
            bbox2: Second bounding box
            
        Returns:
            Tuple of (is_material, reason)
        """
        # Check fingerprint change
        hamming_dist = self._hamming_distance(fp1, fp2)
        if hamming_dist > self.hamming_thresh:
            return True, f"fp_changed(hamming={hamming_dist})"
        
        # Check bounding box movement
        dx, dy, dw, dh = self._bbox_distance(bbox1, bbox2)
        
        if dx > self.bbox_jitter_px or dy > self.bbox_jitter_px or \
           dw > self.bbox_jitter_px or dh > self.bbox_jitter_px:
            return True, f"bbox_moved(dx={dx},dy={dy},dw={dw},dh={dh})"
        
        return False, "no_material_change"
    
    def _should_emit_event(self, slot_idx: int, fp: int, bbox: tuple[int, int, int, int], 
                          now: float) -> tuple[bool, str]:
        """
        Determine if event should be emitted for this slot
        
        Args:
            slot_idx: Row slot index
            fp: Fingerprint (dHash uint64)
            bbox: Bounding box (x, y, w, h)
            now: Current timestamp
            
        Returns:
            Tuple of (should_emit, reason)
        """
        # First time seeing this slot - emit immediately
        if slot_idx not in self.slot_states:
            return True, "first_detection"
        
        state = self.slot_states[slot_idx]
        
        # Check if there's a material change from last *emitted* state
        is_material, change_reason = self._is_material_change(state.last_fp, fp, state.last_bbox, bbox)
        
        if not is_material:
            # No material change - suppress indefinitely
            # Reset any in-progress stability tracking since we're back to stable state
            if state.candidate_fp is not None:
                state.candidate_fp = None
                state.candidate_bbox = None
                state.stable_frames = 0
            return False, "suppressed_stable"
        
        # Material change detected!
        # Check if this is a new candidate or continuation of existing candidate
        if state.candidate_fp is None:
            # First frame of new candidate state
            state.candidate_fp = fp
            state.candidate_bbox = bbox
            state.stable_frames = 1
            return False, f"new_candidate({change_reason})"
        else:
            # Check if current fp/bbox matches the candidate
            candidate_material, _ = self._is_material_change(state.candidate_fp, fp, 
                                                             state.candidate_bbox, bbox)
            
            if not candidate_material:
                # Same as candidate - increment stability counter
                state.stable_frames += 1
                
                if state.stable_frames >= self.stability_frames:
                    # Stability threshold reached!
                    # Check cooldown before emitting
                    elapsed = now - state.last_emit_ts
                    if elapsed < self.cooldown_sec:
                        return False, f"cooldown_active({elapsed:.1f}s<{self.cooldown_sec}s)"
                    
                    # Emit!
                    return True, f"stable_change({change_reason},frames={state.stable_frames})"
                else:
                    return False, f"stabilizing({state.stable_frames}/{self.stability_frames})"
            else:
                # Different from candidate - reset to new candidate
                state.candidate_fp = fp
                state.candidate_bbox = bbox
                state.stable_frames = 1
                return False, f"candidate_reset({change_reason})"
    
    def _update_stability_tracking(self, slot_idx: int, fp: int, bbox: tuple[int, int, int, int]):
        """
        Update stability tracking for a slot (called when NOT emitting)
        
        Args:
            slot_idx: Row slot index
            fp: Current fingerprint
            bbox: Current bounding box
        """
        # Stability state is already updated in _should_emit_event
        # This is just a placeholder for any additional tracking
        pass
    
    def _update_slot_state_after_emit(self, slot_idx: int, fp: int, bbox: tuple[int, int, int, int], 
                                     now: float):
        """
        Update state for a slot after emitting event
        
        Args:
            slot_idx: Row slot index
            fp: Emitted fingerprint
            bbox: Emitted bounding box
            now: Current timestamp
        """
        if slot_idx in self.slot_states:
            state = self.slot_states[slot_idx]
            state.last_fp = fp
            state.last_emit_ts = now
            state.last_bbox = bbox
            # Reset stability tracking after emit
            state.candidate_fp = None
            state.candidate_bbox = None
            state.stable_frames = 0
        else:
            self.slot_states[slot_idx] = RowSlotState(
                last_fp=fp,
                last_emit_ts=now,
                last_bbox=bbox,
                candidate_fp=None,
                candidate_bbox=None,
                stable_frames=0
            )
