# ScreenBot 🤖📈

Cross-platform momentum trading bot that monitors stock scanners via screen capture and executes trades through Alpaca.

## Status: Chunk 3 Complete ✓

This repository contains:
- ✅ Cross-platform screen capture (Windows & Linux)
- ✅ Multiple capture backends (MSS + DXCam)
- ✅ Interactive ROI selection with persistent storage
- ✅ **Dynamic row segmentation using horizontal line detection**
- ✅ **Hash-based new row tracking**
- ✅ **Fast gap color classification (GREEN/WARM gating)**
- ✅ **Gated OCR parsing (phase 1 fields)**
- ✅ **SQLite event storage with per-ticker state**
- ✅ **Real-time scanner event stream**

**Not yet implemented:** Trading logic, Alpaca integration

---

## Features

### Capture System
- **MSS** (cross-platform, default)
- **DXCam** (Windows-only, high performance, optional)
- **Interactive ROI selection** - visually select scan region
- Absolute and window-anchored capture modes

### Scanner Tracking (NEW in Chunk 3)
- **Dynamic row segmentation** - detects horizontal separator lines
- **New row detection** - perceptual hashing with time-bounded cache
- **Fast gap color gating** - GREEN/WARM classification using HSV
- **Gated OCR** - only parse when gap is green (configurable)
- **SQLite storage** - persistent event log and per-ticker state
- **Appearance tracking** - counts per ticker in 60s/300s windows

### Visual Tools
- **Live preview** with FPS and region overlay
- **Interactive ROI selector** - drag to select scan region
- **Scanner event stream** - real-time new row detection
- **Debug images** - segmentation, columns, events

---

## Quick Start

### 1. Initial Setup

#### Windows (PowerShell)

```powershell
cd C:\path\to\screenbot
git checkout main
git pull origin main

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (required for text parsing)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR
```

#### Windows (Git Bash)

```bash
cd /c/path/to/screenbot
git checkout main
git pull origin main

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (required for text parsing)
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Add to PATH: C:\Program Files\Tesseract-OCR
```

#### Linux (bash)

```bash
cd /path/to/screenbot
git checkout main
git pull origin main

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract OCR (required for text parsing)
sudo apt-get update
sudo apt-get install tesseract-ocr

# Optional: For window-anchored mode (X11 only)
sudo apt-get install wmctrl
```

### 2. Select Scan Region (First Time)

```bash
# Interactive ROI selection
python tools/preview_capture.py --view full --select-roi

# Instructions:
# 1. A full-screen capture will appear
# 2. Drag your mouse to select the scanner table
# 3. Press ENTER to save
# 4. Your selection is saved to config/region.json
```

### 3. Run Scanner Event Stream

```bash
# Start the scanner stream
python tools/stream_scanner.py

# You should see events like:
# 🟢 09:15:02 AM RPGL   price=$12.34   gap=GREEN   hits=4   app60= 3 parsed=1 [23.4ms] [id=1]
# 🔴 09:15:03 AM ABCD   price=$5.67    gap=WARM    hits=2   app60= 1 parsed=0 [12.1ms] [id=2]
```

---

## Configuration

Edit `config.py` to tune scanner behavior:

### Capture Backend
```python
CAPTURE_BACKEND = "auto"  # Options: "auto", "mss", "dxcam"
```

### Scanner Tracking
```python
SCAN_INTERVAL_MS = 100  # Capture every 100ms (10 fps)
MAX_TOP_ROWS_TO_CHECK = 3  # Check top 3 rows for new entries
ROW_LINE_MIN_LENGTH_PCT = 0.75  # Horizontal line detection threshold
```

### Column Boundaries (% of ROI width)
```python
COL_TIME_HITS = (0.00, 0.22)  # Time and hits column
COL_SYMBOL = (0.22, 0.42)     # Symbol column
COL_PRICE = (0.42, 0.52)      # Price column
COL_GAP = (0.70, 0.78)        # Gap % column
```

### Gap Color Gating (HSV)
```python
GAP_GREEN_HUE_RANGE = (35, 95)  # Green hue range
GAP_MIN_SAT = 40  # Minimum saturation
GAP_MIN_VAL = 40  # Minimum value
```

### OCR Settings
```python
ENABLE_OCR = True  # Enable OCR parsing
OCR_ONLY_IF_GAP_GREEN = True  # Only OCR green gaps (speed optimization)
FORCE_OCR = False  # Override gating
```

### Debug Settings
```python
ROW_SEGMENT_DEBUG = True  # Save segmentation debug images
EVENT_IMAGE_DEBUG = True  # Save event debug images
```

---

## Tools

### 1. Interactive ROI Selection

```bash
python tools/preview_capture.py --view full --select-roi
```

Select your scan region visually. Saved to `config/region.json`.

### 2. Scanner Event Stream (Main Tool)

```bash
python tools/stream_scanner.py
```

**What it does:**
- Captures ROI continuously
- Detects horizontal separator lines dynamically
- Identifies new top rows using perceptual hashing
- Classifies gap color (GREEN/WARM/NEUTRAL)
- Optionally runs OCR (gated by gap color)
- Stores events in SQLite database
- Tracks per-ticker appearances (60s/300s windows)
- Prints real-time events to console

**Example output:**
```
🟢 09:15:02 AM RPGL   price=$12.34   gap=GREEN   hits=4   app60= 3 parsed=1 [23.4ms] [id=1]
🟢 09:15:03 AM TSLA   price=$245.67  gap=GREEN   hits=5   app60= 1 parsed=1 [18.2ms] [id=2]
🔴 09:15:04 AM ABCD   price=$5.67    gap=WARM    hits=2   app60= 1 parsed=0 [12.1ms] [id=3]
```

**Legend:**
- 🟢 = GREEN gap (positive momentum candidate)
- 🔴 = WARM gap (red/orange)
- ⚪ = NEUTRAL/UNKNOWN

**No trades are placed in this chunk.** This is an event detection and logging system only.

### 3. Live Preview

```bash
# Show full screen with capture rectangle
python tools/preview_capture.py --view full

# Show only captured region
python tools/preview_capture.py --view crop
```

### 4. Window Probe

```bash
python tools/window_probe.py
```

### 5. Capture Debug

```bash
python tools/capture_debug.py --count 5
```

---

## Data Storage

### SQLite Database

Location: `data/screenbot.db`

**Tables:**

1. **events** - Append-only event log
   - Timestamp, symbol, price, hits
   - Gap color bucket and HSV values
   - Row hash, parsed flag
   - Raw OCR text

2. **ticker_state** - Per-ticker state (upserted)
   - Last seen timestamp
   - Appearances in 60s/300s windows
   - Last price, hits, gap color

### Debug Images

Saved to `debug_frames/`:
- `segmentation/` - Row detection debug images
- `columns/` - Column boundary visualization
- `changes/` - Per-event ROI, row, and gap cell images

---

## How It Works

### 1. Row Segmentation

- Converts ROI to grayscale
- Detects horizontal separator lines using morphological operations
- Builds row bounding boxes between consecutive lines
- **Handles variable row heights** (not fixed)

### 2. New Row Detection

- Computes perceptual hash (dHash) of each row
- Maintains time-bounded cache of seen hashes (60s, max 500)
- Emits event only if hash is new

### 3. Gap Color Gating (Fast Filter)

- Extracts gap cell from row
- Converts to HSV color space
- Classifies as GREEN/WARM/NEUTRAL based on hue
- **GREEN = positive gap candidate** (for long-only strategy)

### 4. Gated OCR (Phase 1)

OCR runs only when:
- `ENABLE_OCR = True` AND
- (`FORCE_OCR = True` OR gap is GREEN)

Parses:
- Row time (e.g., "09:15:02 am")
- Hits (e.g., "4 in 5sec")
- Symbol (validated with regex)
- Price (float)

### 5. Event Storage & Ticker Tracking

- Each new row creates an event in SQLite
- If symbol is parsed, update ticker_state table
- Track appearances in 60s/300s sliding windows
- Used for filtering "hot" tickers in future trading logic

---

## Tesseract OCR Setup

### Windows

1. Download installer: https://github.com/UB-Mannheim/tesseract/wiki
2. Install to: `C:\Program Files\Tesseract-OCR`
3. Add to PATH:
   - Search "Environment Variables"
   - Edit Path
   - Add: `C:\Program Files\Tesseract-OCR`
4. Verify: `tesseract --version`

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
tesseract --version
```

### macOS

```bash
brew install tesseract
tesseract --version
```

**Note:** If Tesseract is not installed, the tool will still run but OCR will be disabled.

---

## Tuning Tips

### If rows are not detected:

1. Check debug images in `debug_frames/segmentation/`
2. Adjust `ROW_LINE_MIN_LENGTH_PCT` in `config.py`
3. Adjust `ROW_LINE_THICKNESS_PX`

### If too many false positives:

1. Increase hash cache TTL in `row_tracker.py`
2. Adjust perceptual hash size

### If OCR is inaccurate:

1. Check column boundaries in `debug_frames/columns/`
2. Adjust `COL_*` percentages in `config.py`
3. Verify Tesseract is installed correctly

### If gap color is wrong:

1. Adjust `GAP_GREEN_HUE_RANGE` in `config.py`
2. Check `debug_frames/changes/` for gap cell images
3. Tune `GAP_MIN_SAT` and `GAP_MIN_VAL`

---
ABS_W = 800  # Width
ABS_H = 600  # Height
```

### Window Mode
```python
WINDOW_TITLE_CONTAINS = "Chrome"  # Window title substring
REL_X = 10   # X offset from window's top-left
REL_Y = 50   # Y offset from window's top-left
REL_W = 800  # Capture width
REL_H = 600  # Capture height
```

---

## Tools

### 1. Live Preview
Visual verification of capture region with real-time overlay.

```bash
# Show captured region only (default)
python tools/preview_capture.py --view crop

# Show full screen with capture rectangle overlay
python tools/preview_capture.py --view full
```

**Hotkeys:**
- `q` - Quit
- `p` - Save screenshot to debug_frames/

### 2. Window Probe
List all visible windows with titles and positions.

```bash
python tools/window_probe.py
```

### 3. Capture Debug
Capture and save frames for troubleshooting.

```bash
# Capture 5 frames (default)
python tools/capture_debug.py

# Capture custom number of frames
python tools/capture_debug.py --count 10
```

---

## Running in VS Code

### Open Integrated Terminal
1. Open VS Code
2. Press `` Ctrl+` `` (backtick) to open terminal
3. Activate virtual environment:
   - Windows: `source .venv/Scripts/activate`
   - Linux: `source .venv/bin/activate`

### Run Tools
```bash
python tools/preview_capture.py --view full
python tools/window_probe.py
python tools/capture_debug.py
```

---

## Platform Notes

### Windows
- ✅ Both absolute and window modes fully supported
- ✅ DXCam available for high-performance capture
- ✅ pywin32 for window management

### Linux
- ✅ Absolute mode fully supported (MSS)
- ⚠️ Window mode requires X11 + wmctrl
- ❌ Wayland: Window anchoring may not work (use absolute mode)
- 💡 DXCam not available (MSS fallback automatic)

---

## Project Structure

```
screenbot/
├── app.py                    # Main application (future trading logic)
├── config.py                 # Configuration with tunable parameters
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── capture/                  # Screen capture system
│   ├── __init__.py
│   ├── frame_source.py       # Unified capture interface
│   ├── mss_source.py         # MSS backend
│   └── dxcam_source.py       # DXCam backend (Windows)
├── scanner/                  # Scanner tracking (NEW in Chunk 3)
│   ├── __init__.py
│   ├── row_segmenter.py      # Dynamic row segmentation by lines
│   ├── row_tracker.py        # New row detection via hashing
│   ├── slicer.py             # Column extraction
│   ├── color_features.py     # Gap color classification
│   ├── ocr_parse.py          # Gated OCR parsing
│   └── store.py              # SQLite storage
├── tools/                    # Utilities
│   ├── preview_capture.py    # Live visual preview + ROI selector
│   ├── stream_scanner.py     # Scanner event stream (MAIN TOOL)
│   ├── capture_debug.py      # Debug frame capture
│   └── window_probe.py       # Window listing
├── config/                   # User configuration
│   ├── region.json           # Selected ROI (gitignored, user-specific)
│   └── region.example.json   # Example ROI format
├── data/                     # Generated data (gitignored)
│   └── screenbot.db          # SQLite database
├── debug_frames/             # Debug images
│   ├── segmentation/         # Row detection debug
│   ├── columns/              # Column boundary debug
│   └── changes/              # Per-event images
└── logs/                     # Application logs
```

---

## Next Steps (Future Chunks)

- [ ] Trading logic and risk management
- [ ] Alpaca API integration
- [ ] Position sizing and entry/exit rules
- [ ] Backtesting framework
- [ ] Performance monitoring

---

## Troubleshooting

### Issue: "pytesseract not installed"
- **Solution**: Install pytesseract: `pip install pytesseract`
- **Windows**: Also install Tesseract binary and add to PATH

### Issue: "No rows detected"
- **Solution**: Check `debug_frames/segmentation/` images
- Adjust `ROW_LINE_MIN_LENGTH_PCT` in `config.py`

### Issue: "Window mode not working on Linux"
- **Solution**: Install wmctrl: `sudo apt-get install wmctrl`
- **Wayland users**: Use absolute mode instead

### Issue: Events not appearing
- **Solution**: Check that scan region contains the scanner table
- Re-run ROI selection: `python tools/preview_capture.py --view full --select-roi`

### Issue: Database locked
- **Solution**: Only run one instance of `stream_scanner.py` at a time

---

## Development Workflow

1. Pull latest code: `git pull origin main`
2. Select/verify ROI: `python tools/preview_capture.py --view full --select-roi`
3. Run scanner stream: `python tools/stream_scanner.py`
4. Tune config.py as needed
5. Check debug images for calibration

---

## License

MIT

---

## Repository

👉 https://github.com/antonsin/screenbot

**Current Status: Chunk 3 Complete - Scanner event stream with gap color gating and SQLite storage**

**No trading yet. This chunk produces an event stream only.**
