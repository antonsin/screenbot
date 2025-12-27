"""
Configuration for ScreenBot - momentum trading scanner
"""
import os
from pathlib import Path

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
# SECRETS (loaded from .env)
# ==============================
# Alpaca API credentials (for future trading implementation)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # paper trading default
