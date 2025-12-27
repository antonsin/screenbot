#!/usr/bin/env python3
"""
Window probe tool - list all visible windows with their positions

Usage:
    python tools/window_probe.py
"""
import sys
import platform
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def probe_windows_windows():
    """List windows on Windows using win32gui"""
    try:
        import win32gui
    except ImportError:
        logger.error("pywin32 not installed. Install with: pip install pywin32")
        return
    
    print("=" * 80)
    print("VISIBLE WINDOWS ON WINDOWS")
    print("=" * 80)
    print(f"{'Handle':<12} {'X':<6} {'Y':<6} {'Width':<7} {'Height':<7} Title")
    print("-" * 80)
    
    def callback(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:  # Only show windows with titles
                rect = win32gui.GetWindowRect(hwnd)
                x, y, right, bottom = rect
                w = right - x
                h = bottom - y
                print(f"{hwnd:<12} {x:<6} {y:<6} {w:<7} {h:<7} {title}")
        return True
    
    win32gui.EnumWindows(callback, None)
    print("=" * 80)


def probe_windows_linux():
    """List windows on Linux using wmctrl (best-effort)"""
    import subprocess
    
    print("=" * 80)
    print("VISIBLE WINDOWS ON LINUX (using wmctrl)")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            ["wmctrl", "-lG"],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode != 0:
            logger.error("wmctrl command failed")
            print("\n⚠️  wmctrl not found or failed")
            print("Install with: sudo apt-get install wmctrl")
            print("\nNote: wmctrl requires X11. Wayland users may have limited functionality.")
            return
        
        print(f"{'Window ID':<12} {'X':<6} {'Y':<6} {'Width':<7} {'Height':<7} Title")
        print("-" * 80)
        
        # Parse wmctrl output
        # Format: window_id desktop x y w h host window_title
        for line in result.stdout.splitlines():
            parts = line.split(None, 7)
            if len(parts) >= 8:
                win_id = parts[0]
                x, y, w, h = parts[2], parts[3], parts[4], parts[5]
                title = parts[7]
                print(f"{win_id:<12} {x:<6} {y:<6} {w:<7} {h:<7} {title}")
        
        print("=" * 80)
    
    except FileNotFoundError:
        logger.error("wmctrl not found")
        print("\n⚠️  wmctrl not installed")
        print("Install with: sudo apt-get install wmctrl")
        print("\nNote: wmctrl requires X11. Wayland users may have limited functionality.")
    
    except Exception as e:
        logger.error(f"Error probing windows: {e}")
        print(f"\n⚠️  Error: {e}")


def main():
    system = platform.system()
    
    print(f"\nPlatform: {system}")
    print(f"Python: {sys.version}\n")
    
    if system == "Windows":
        probe_windows_windows()
    
    elif system == "Linux":
        probe_windows_linux()
    
    else:
        print(f"⚠️  Window probing not implemented for {system}")
        logger.warning(f"Window probing not supported on {system}")


if __name__ == "__main__":
    main()
