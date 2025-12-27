"""
Row tracker - hash-based new row detection with time-bounded cache
"""
import cv2
import numpy as np
import logging
import hashlib
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RowTracker:
    """
    Tracks rows using perceptual hashing to detect new entries
    """
    
    def __init__(self, config, max_top_rows: int = 3, cache_ttl_seconds: int = 60, max_cache_size: int = 500):
        """
        Initialize row tracker
        
        Args:
            config: Configuration module
            max_top_rows: Number of top rows to check
            cache_ttl_seconds: Time to keep hashes in cache
            max_cache_size: Maximum number of hashes to cache
        """
        self.config = config
        self.max_top_rows = max_top_rows
        self.cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self.max_cache_size = max_cache_size
        
        # Cache: deque of (hash, timestamp) tuples
        self.seen_hashes = deque(maxlen=max_cache_size)
        self.hash_set = set()  # For fast lookup
    
    def check_new_rows(self, rows: list[tuple[int, int, int, int]], roi_img: np.ndarray) -> list[dict]:
        """
        Check top rows for new entries
        
        Args:
            rows: List of (x, y, w, h) row boxes
            roi_img: Full ROI image
            
        Returns:
            List of dicts with new row info: {index, bbox, hash, image}
        """
        # Clean old hashes from cache
        self._cleanup_cache()
        
        new_rows = []
        
        # Check only top N rows
        check_count = min(self.max_top_rows, len(rows))
        
        for idx in range(check_count):
            x, y, w, h = rows[idx]
            
            # Extract row image
            row_img = roi_img[y:y+h, x:x+w].copy()
            
            # Compute hash
            row_hash = self._compute_row_hash(row_img)
            
            # Check if this is a new hash
            if row_hash not in self.hash_set:
                logger.info(f"New row detected at index {idx}, hash: {row_hash[:12]}...")
                
                new_rows.append({
                    'index': idx,
                    'bbox': (x, y, w, h),
                    'hash': row_hash,
                    'image': row_img
                })
                
                # Add to cache
                self._add_to_cache(row_hash)
        
        return new_rows
    
    def _compute_row_hash(self, row_img: np.ndarray) -> str:
        """
        Compute perceptual hash of row image
        
        Args:
            row_img: Row image (BGR)
            
        Returns:
            Hash string
        """
        # Normalize image for consistent hashing
        # 1. Convert to grayscale
        gray = cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY)
        
        # 2. Resize to fixed width for consistency
        target_width = 256
        aspect_ratio = gray.shape[1] / gray.shape[0]
        target_height = int(target_width / aspect_ratio)
        resized = cv2.resize(gray, (target_width, target_height))
        
        # 3. Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        
        # 4. Compute difference hash (dHash)
        # This is robust to small changes but sensitive to new content
        dhash = self._dhash(blurred)
        
        return dhash
    
    def _dhash(self, img: np.ndarray, hash_size: int = 16) -> str:
        """
        Compute difference hash (dHash)
        
        Args:
            img: Grayscale image
            hash_size: Size of hash (16 = 256 bits)
            
        Returns:
            Hash as hex string
        """
        # Resize to hash_size + 1 to allow horizontal gradient comparison
        resized = cv2.resize(img, (hash_size + 1, hash_size))
        
        # Compute horizontal gradient (difference between adjacent pixels)
        diff = resized[:, 1:] > resized[:, :-1]
        
        # Convert boolean array to hash string
        hash_bytes = np.packbits(diff.flatten())
        hash_hex = hash_bytes.tobytes().hex()
        
        return hash_hex
    
    def _add_to_cache(self, row_hash: str):
        """
        Add hash to cache with timestamp
        
        Args:
            row_hash: Hash to add
        """
        now = datetime.now()
        self.seen_hashes.append((row_hash, now))
        self.hash_set.add(row_hash)
    
    def _cleanup_cache(self):
        """
        Remove expired hashes from cache
        """
        now = datetime.now()
        cutoff = now - self.cache_ttl
        
        # Remove old entries
        while self.seen_hashes and self.seen_hashes[0][1] < cutoff:
            old_hash, _ = self.seen_hashes.popleft()
            self.hash_set.discard(old_hash)
        
        logger.debug(f"Hash cache size: {len(self.hash_set)}")
