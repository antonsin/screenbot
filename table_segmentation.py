"""
Table segmentation utility for extracting grid structure from screenshots

This module provides robust table segmentation with both line-based and
projection-based fallback methods.
"""
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class Rect:
    """Rectangle representing a cell boundary"""
    x: int
    y: int
    w: int
    h: int
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, w, h) tuple"""
        return (self.x, self.y, self.w, self.h)
    
    def to_slice(self) -> Tuple[slice, slice]:
        """Convert to numpy slice (y_slice, x_slice)"""
        return (slice(self.y, self.y + self.h), slice(self.x, self.x + self.w))


def binarize_for_table(bgr: np.ndarray) -> np.ndarray:
    """
    Binarize image for table detection using adaptive thresholding
    
    Args:
        bgr: Input BGR image
        
    Returns:
        Binary image (inverted, lines are white)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply median blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)
    
    # Adaptive threshold with binary inversion (lines become white)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=5
    )
    
    return binary


def detect_grid_lines(binary: np.ndarray) -> Tuple[List[int], List[int]]:
    """
    Detect horizontal and vertical grid lines using morphology
    
    Args:
        binary: Binary image (lines are white)
        
    Returns:
        Tuple of (y_separators, x_separators) as sorted lists
    """
    h, w = binary.shape
    
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 4))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    
    # Extract y-coordinates of horizontal lines
    horizontal_projection = np.sum(horizontal_lines, axis=1)
    threshold_h = np.max(horizontal_projection) * 0.5 if np.max(horizontal_projection) > 0 else 1
    y_separators = []
    
    in_line = False
    line_start = 0
    for y, val in enumerate(horizontal_projection):
        if val > threshold_h and not in_line:
            in_line = True
            line_start = y
        elif val <= threshold_h and in_line:
            in_line = False
            # Use middle of line
            y_separators.append((line_start + y) // 2)
    
    # Extract x-coordinates of vertical lines
    vertical_projection = np.sum(vertical_lines, axis=0)
    threshold_v = np.max(vertical_projection) * 0.5 if np.max(vertical_projection) > 0 else 1
    x_separators = []
    
    in_line = False
    line_start = 0
    for x, val in enumerate(vertical_projection):
        if val > threshold_v and not in_line:
            in_line = True
            line_start = x
        elif val <= threshold_v and in_line:
            in_line = False
            # Use middle of line
            x_separators.append((line_start + x) // 2)
    
    # Sort and remove duplicates
    y_separators = sorted(set(y_separators))
    x_separators = sorted(set(x_separators))
    
    return y_separators, x_separators


def fallback_projection_separators(binary: np.ndarray, axis: int = 0, min_separators: int = 2) -> List[int]:
    """
    Fallback method using projection to find separators
    
    Uses valleys in projection histogram to detect row/column boundaries
    
    Args:
        binary: Binary image
        axis: 0 for horizontal (row separators), 1 for vertical (column separators)
        min_separators: Minimum number of separators to find
        
    Returns:
        List of separator positions
    """
    # Project along axis
    projection = np.sum(binary, axis=axis)
    
    # Find valleys (low points in projection)
    # These represent gaps between rows/columns
    threshold = np.mean(projection) * 0.3
    
    separators = []
    in_valley = False
    valley_start = 0
    
    for i, val in enumerate(projection):
        if val < threshold and not in_valley:
            in_valley = True
            valley_start = i
        elif val >= threshold and in_valley:
            in_valley = False
            # Use middle of valley
            separators.append((valley_start + i) // 2)
    
    # If not enough separators, try dividing uniformly
    if len(separators) < min_separators:
        length = len(projection)
        # Create uniform divisions
        num_divisions = max(min_separators + 1, 5)  # At least 5 sections
        separators = [int(length * i / num_divisions) for i in range(1, num_divisions)]
    
    return sorted(set(separators))


def build_cells_from_separators(h: int, w: int, ys: List[int], xs: List[int]) -> List[Rect]:
    """
    Build cell rectangles from separator positions
    
    Args:
        h: Image height
        w: Image width
        ys: Y-coordinate separators (horizontal lines)
        xs: X-coordinate separators (vertical lines)
        
    Returns:
        List of Rect objects representing cells
    """
    cells = []
    
    # Ensure boundaries are included
    ys_with_bounds = [0] + ys + [h]
    xs_with_bounds = [0] + xs + [w]
    
    # Build cells from grid
    for i in range(len(ys_with_bounds) - 1):
        for j in range(len(xs_with_bounds) - 1):
            y1 = ys_with_bounds[i]
            y2 = ys_with_bounds[i + 1]
            x1 = xs_with_bounds[j]
            x2 = xs_with_bounds[j + 1]
            
            # Add small padding to avoid including separator lines
            padding = 2
            cell = Rect(
                x=x1 + padding,
                y=y1 + padding,
                w=max(1, x2 - x1 - 2 * padding),
                h=max(1, y2 - y1 - 2 * padding)
            )
            cells.append(cell)
    
    return cells


def draw_overlay(raw_bgr: np.ndarray, ys: List[int], xs: List[int], cells: List[Rect]) -> np.ndarray:
    """
    Draw debug overlay showing detected separators and cells
    
    Args:
        raw_bgr: Original BGR image
        ys: Y-coordinate separators
        xs: X-coordinate separators
        cells: List of cell rectangles
        
    Returns:
        Annotated BGR image
    """
    overlay = raw_bgr.copy()
    h, w = overlay.shape[:2]
    
    # Draw horizontal separators in red
    for y in ys:
        cv2.line(overlay, (0, y), (w, y), (0, 0, 255), 2)
    
    # Draw vertical separators in blue
    for x in xs:
        cv2.line(overlay, (x, 0), (x, h), (255, 0, 0), 2)
    
    # Draw sample cells (first 20 to avoid clutter)
    for i, cell in enumerate(cells[:20]):
        color = (0, 255, 0) if i < 10 else (0, 255, 255)  # Green for first 10, yellow for next 10
        cv2.rectangle(
            overlay,
            (cell.x, cell.y),
            (cell.x + cell.w, cell.y + cell.h),
            color,
            1
        )
        # Add cell number
        cv2.putText(
            overlay,
            str(i),
            (cell.x + 2, cell.y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1
        )
    
    return overlay


def segment_table(
    raw_bgr: np.ndarray,
    debug_dir: Path,
    prefer_lines: bool = True,
    min_rows: int = 2,
    min_cols: int = 2
) -> Tuple[np.ndarray, List[int], List[int], List[Rect]]:
    """
    Segment table from image with debug output
    
    Args:
        raw_bgr: Input BGR image
        debug_dir: Directory to save debug images
        prefer_lines: If True, prefer line-based detection over projection
        min_rows: Minimum number of rows expected
        min_cols: Minimum number of columns expected
        
    Returns:
        Tuple of (binary_image, y_separators, x_separators, cells)
    """
    h, w = raw_bgr.shape[:2]
    
    # Create debug directory
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Binarize
    binary = binarize_for_table(raw_bgr)
    
    # Step 2: Detect grid lines
    ys_lines, xs_lines = detect_grid_lines(binary)
    
    logger.info(f"Line-based detection: {len(ys_lines)} horizontal, {len(xs_lines)} vertical lines")
    
    # Step 3: Fallback if needed
    ys = ys_lines
    xs = xs_lines
    
    if prefer_lines and (len(ys_lines) >= min_rows and len(xs_lines) >= min_cols):
        # Use line-based detection
        method = "line-based"
    else:
        # Use projection fallback
        logger.info("Using projection fallback for separators")
        ys = fallback_projection_separators(binary, axis=0, min_separators=min_rows)
        xs = fallback_projection_separators(binary, axis=1, min_separators=min_cols)
        method = "projection"
    
    logger.info(f"Final separators ({method}): {len(ys)} rows, {len(xs)} columns")
    
    # Step 4: Build cells
    cells = build_cells_from_separators(h, w, ys, xs)
    logger.info(f"Built {len(cells)} cells")
    
    # Step 5: Create debug visualizations
    # Save raw
    cv2.imwrite(str(debug_dir / "raw.png"), raw_bgr)
    
    # Save binary
    cv2.imwrite(str(debug_dir / "bin.png"), binary)
    
    # Save overlay
    overlay = draw_overlay(raw_bgr, ys, xs, cells)
    cv2.imwrite(str(debug_dir / "overlay.png"), overlay)
    
    # Save line detection results
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    cv2.imwrite(str(debug_dir / "horiz_lines.png"), horizontal_lines)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 4))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    cv2.imwrite(str(debug_dir / "vert_lines.png"), vertical_lines)
    
    logger.info(f"Debug images saved to: {debug_dir}")
    
    return binary, ys, xs, cells


# Example usage function
def demo_table_segmentation(image_path: str, output_dir: str = "debug_table"):
    """
    Demo function showing how to use the table segmentation
    
    Args:
        image_path: Path to input image
        output_dir: Directory for debug output
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Segment table
    binary, ys, xs, cells = segment_table(
        img,
        Path(output_dir),
        prefer_lines=True,
        min_rows=2,
        min_cols=2
    )
    
    print(f"Detected {len(ys)} row separators")
    print(f"Detected {len(xs)} column separators")
    print(f"Created {len(cells)} cells")
    print(f"Debug images saved to: {output_dir}/")
    
    return binary, ys, xs, cells


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    print("Table Segmentation Utility")
    print("=" * 50)
    print("Usage:")
    print("  from table_segmentation import segment_table")
    print("  binary, ys, xs, cells = segment_table(img, debug_dir)")
