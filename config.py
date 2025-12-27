"""
Configuration for ScreenBot - momentum trading scanner
"""
import os
from pathlib import Path

# Load .env file if available (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import logging
    logging.getLogger(__name__).warning("python-dotenv not installed, .env file will not be loaded automatically")

# ==============================
# PATHS
# ==============================
BASE_DIR = Path(__file__).parent
DEBUG_FRAMES_DIR = BASE_DIR / "debug_frames"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DEBUG_FRAMES_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ==============================
# LOGGING
# ==============================
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "screenbot.log"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# ==============================
# CAPTURE BACKEND
# ==============================
# Options: "auto", "mss", "dxcam"
# "auto" = try dxcam on Windows, fallback to mss
CAPTURE_BACKEND = "auto"

# ==============================
# CAPTURE MODE
# ==============================
# Options: "absolute", "window"
CAPTURE_MODE = "absolute"

# ==============================
# ABSOLUTE MODE SETTINGS
# ==============================
# Coordinates for absolute screen capture (pixels from top-left)
ABS_X = 100
ABS_Y = 100
ABS_W = 800
ABS_H = 600

# ==============================
# WINDOW MODE SETTINGS
# ==============================
# Substring to match window title (case-insensitive)
WINDOW_TITLE_CONTAINS = "Chrome"

# Relative coordinates from window's top-left corner
REL_X = 10
REL_Y = 50
REL_W = 800
REL_H = 600

# ==============================
# CAPTURE PERFORMANCE
# ==============================
# Target FPS for capture loop
TARGET_FPS = 10

# ==============================
# SCANNER TRACKING
# ==============================
# Scan interval (milliseconds between captures)
SCAN_INTERVAL_MS = 100  # 10 fps

# Maximum top rows to check for new entries
MAX_TOP_ROWS_TO_CHECK = 3

# Row segmentation (horizontal line detection)
ROW_LINE_MIN_LENGTH_PCT = 0.75  # Minimum line length as % of width
ROW_LINE_THICKNESS_PX = 1  # Expected line thickness
MIN_ROWS = 3  # Minimum expected rows for fallback logic

# Debug settings
ROW_SEGMENT_DEBUG = True  # Save segmentation debug images
EVENT_IMAGE_DEBUG = True  # Save event debug images

# ==============================
# COLUMN BOUNDARIES (% of ROI width)
# ==============================
COL_TIME_HITS = (0.00, 0.22)  # Time and hits column
COL_SYMBOL = (0.22, 0.42)     # Symbol column
COL_PRICE = (0.42, 0.52)      # Price column
COL_GAP = (0.70, 0.78)        # Gap % column

# ==============================
# GAP COLOR CLASSIFICATION (HSV)
# ==============================
# Green hue range (degrees in 0-180 scale used by OpenCV)
GAP_GREEN_HUE_RANGE = (35, 95)  # Green hues

# Minimum saturation and value for color detection
GAP_MIN_SAT = 40
GAP_MIN_VAL = 40

# ==============================
# OCR SETTINGS
# ==============================
ENABLE_OCR = True  # Enable OCR parsing
OCR_ONLY_IF_GAP_GREEN = True  # Only OCR when gap is green (speed optimization)
FORCE_OCR = False  # If True, override gating and always OCR

# Symbol validation regex
SYMBOL_REGEX = r"^[A-Z]{1,5}(\.[A-Z])?$"

# ==============================
# STORAGE
# ==============================
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "screenbot.db"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

# ==============================
# SECRETS (loaded from .env)
# ==============================
# Tesseract OCR path (optional, falls back to PATH)
# Windows example: C:\Program Files\Tesseract-OCR\tesseract.exe
# Linux example: /usr/bin/tesseract
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")

# Debug flags
DEBUG_SEGMENTATION = os.getenv("DEBUG_SEGMENTATION", "0") == "1"

# Alpaca API credentials (for future trading implementation)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # paper trading default
