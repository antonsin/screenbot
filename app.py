#!/usr/bin/env python3
"""
ScreenBot - Momentum trading bot with screen-based scanner detection

Main application entry point (future trading logic will be added here)
"""
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import config


def setup_logging():
    """Configure logging for the application"""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler (rotating)
    file_handler = RotatingFileHandler(
        config.LOG_FILE,
        maxBytes=config.LOG_MAX_BYTES,
        backupCount=config.LOG_BACKUP_COUNT
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def main():
    """Main application entry point"""
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("ScreenBot - Momentum Trading Bot")
    logger.info("=" * 60)
    logger.info(f"Logs directory: {config.LOGS_DIR}")
    logger.info(f"Debug frames directory: {config.DEBUG_FRAMES_DIR}")
    logger.info(f"Capture mode: {config.CAPTURE_MODE}")
    logger.info(f"Capture backend: {config.CAPTURE_BACKEND}")
    logger.info("=" * 60)
    
    # Future implementation will include:
    # - Frame capture loop
    # - Scanner change detection
    # - OCR ticker extraction
    # - Alpaca trading integration
    # - Risk management
    
    logger.info("✓ Application initialized successfully")
    logger.info("")
    logger.info("This is Chunk 1-2: Initial scaffold + capture system")
    logger.info("Trading logic will be implemented in future chunks")
    logger.info("")
    logger.info("To test capture system, use:")
    logger.info("  python tools/preview_capture.py --view full")
    logger.info("  python tools/window_probe.py")
    logger.info("  python tools/capture_debug.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
