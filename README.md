# ScreenBot 🤖📈

Cross-platform momentum trading bot that monitors stock scanners via screen capture and executes trades through Alpaca.

## Status: Chunk 1-2 Complete ✓

This repository contains the initial scaffold with:
- ✅ Cross-platform screen capture (Windows & Linux)
- ✅ Multiple capture backends (MSS + DXCam)
- ✅ Absolute and window-anchored capture modes
- ✅ Live visual preview tool
- ✅ Window probing utilities
- ✅ Debug frame capture

**Not yet implemented:** OCR, scanner change detection, trading logic

---

## Features

### Capture Backends
- **MSS** (cross-platform, default)
- **DXCam** (Windows-only, high performance, optional)
- Automatic backend selection with fallback

### Capture Modes
- **Absolute mode**: Capture fixed screen region by coordinates
- **Window mode**: Anchor capture to a specific window (e.g., browser)

### Visual Tools
- **Live preview** with FPS and region overlay
- **Window probe** to list all windows and positions
- **Debug capture** to save frames with metadata

---

## Setup

### Windows (Git Bash)

```bash
cd /c/path/to/screenbot
git checkout main
git pull origin main

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install DXCam for better performance
pip install dxcam

# Test the capture system
python tools/preview_capture.py --view full
```

### Linux (bash)

```bash
cd /path/to/screenbot
git checkout main
git pull origin main

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For window-anchored mode (X11 only):
sudo apt-get install wmctrl

# Test the capture system
python tools/preview_capture.py --view full
```

---

## Configuration

Edit `config.py` to customize capture settings:

### Backend Selection
```python
CAPTURE_BACKEND = "auto"  # Options: "auto", "mss", "dxcam"
```

### Capture Mode
```python
CAPTURE_MODE = "absolute"  # Options: "absolute", "window"
```

### Absolute Mode
```python
ABS_X = 100  # Left coordinate
ABS_Y = 100  # Top coordinate
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
├── config.py                 # Configuration
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── capture/                  # Capture system
│   ├── __init__.py
│   ├── frame_source.py       # Unified capture interface
│   ├── mss_source.py         # MSS backend
│   └── dxcam_source.py       # DXCam backend (Windows)
├── tools/                    # Utilities
│   ├── preview_capture.py    # Live visual preview
│   ├── capture_debug.py      # Debug frame capture
│   └── window_probe.py       # Window listing
├── debug_frames/             # Saved debug frames
└── logs/                     # Application logs
```

---

## Next Steps (Future Chunks)

- [ ] Scanner change detection (frame differencing)
- [ ] OCR ticker extraction (pytesseract)
- [ ] Alpaca API integration (paper trading)
- [ ] Trading logic and risk management
- [ ] Configuration UI

---

## Troubleshooting

### Issue: "DXCam not available"
- **Solution**: This is expected on Linux. MSS backend will be used automatically.
- **Windows**: Install DXCam with `pip install dxcam` for better performance.

### Issue: "Window mode not working on Linux"
- **Solution**: Install wmctrl: `sudo apt-get install wmctrl`
- **Wayland users**: Use absolute mode instead (`CAPTURE_MODE = "absolute"`)

### Issue: "No module named 'win32gui'"
- **Solution**: Windows only. Install pywin32: `pip install pywin32`

### Issue: Capture shows wrong region
1. Run `python tools/window_probe.py` to list windows
2. Update coordinates in `config.py`
3. Verify with `python tools/preview_capture.py --view full`

---

## Development Workflow

1. Make changes to code
2. Test with preview tool: `python tools/preview_capture.py --view full`
3. Commit changes: `git commit -m "description"`
4. Push to main: `git push origin main`

---

## License

MIT

---

## Repository

👉 https://github.com/antonsin/screenbot
