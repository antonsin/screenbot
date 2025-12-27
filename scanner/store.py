"""
SQLite storage - event log and per-ticker state
"""
import sqlite3
import logging
from datetime import datetime, timedelta
from collections import deque, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class ScannerStore:
    """
    SQLite storage for scanner events and ticker state
    """
    
    def __init__(self, config):
        """
        Initialize store
        
        Args:
            config: Configuration module
        """
        self.config = config
        self.db_path = config.DB_PATH
        
        # Ensure data directory exists
        self.db_path.parent.mkdir(exist_ok=True)
        
        # In-memory tracking for appearance counts
        self.ticker_timestamps = defaultdict(lambda: deque(maxlen=1000))
        
        # Initialize database
        self._init_db()
        
        logger.info(f"Database initialized: {self.db_path}")
    
    def _init_db(self):
        """Create tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Events table (append-only)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_utc TEXT NOT NULL,
                    ts_local TEXT NOT NULL,
                    symbol TEXT,
                    price REAL,
                    hits_5s INTEGER,
                    gap_color_bucket TEXT,
                    h_mean REAL,
                    s_mean REAL,
                    v_mean REAL,
                    row_hash TEXT NOT NULL,
                    parsed INTEGER NOT NULL,
                    raw_text TEXT,
                    notes TEXT
                )
            """)
            
            # Create index on timestamp for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_ts_utc 
                ON events(ts_utc)
            """)
            
            # Create index on symbol for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_symbol 
                ON events(symbol)
            """)
            
            # Ticker state table (upsert by symbol)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ticker_state (
                    symbol TEXT PRIMARY KEY,
                    last_seen_utc TEXT NOT NULL,
                    appearances_60s INTEGER DEFAULT 0,
                    appearances_300s INTEGER DEFAULT 0,
                    last_price REAL,
                    last_hits_5s INTEGER,
                    last_gap_color_bucket TEXT,
                    last_h_mean REAL,
                    last_s_mean REAL,
                    last_v_mean REAL
                )
            """)
            
            conn.commit()
    
    def insert_event(self, event_data: dict) -> int:
        """
        Insert new scanner event
        
        Args:
            event_data: Event data dict
            
        Returns:
            Event ID
        """
        now_utc = datetime.utcnow()
        now_local = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO events (
                    ts_utc, ts_local, symbol, price, hits_5s,
                    gap_color_bucket, h_mean, s_mean, v_mean,
                    row_hash, parsed, raw_text, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now_utc.isoformat(),
                now_local.isoformat(),
                event_data.get('symbol'),
                event_data.get('price'),
                event_data.get('hits_5s'),
                event_data.get('gap_color_bucket'),
                event_data.get('h_mean'),
                event_data.get('s_mean'),
                event_data.get('v_mean'),
                event_data.get('row_hash'),
                1 if event_data.get('parsed') else 0,
                str(event_data.get('raw_text', {})),
                event_data.get('notes')
            ))
            
            event_id = cursor.lastrowid
            conn.commit()
        
        logger.debug(f"Inserted event {event_id} for {event_data.get('symbol', 'UNKNOWN')}")
        
        return event_id
    
    def update_ticker_state(self, symbol: str, event_data: dict):
        """
        Update ticker state with new event
        
        Args:
            symbol: Ticker symbol
            event_data: Event data
        """
        if not symbol:
            return
        
        now_utc = datetime.utcnow()
        
        # Add timestamp to in-memory tracker
        self.ticker_timestamps[symbol].append(now_utc)
        
        # Calculate appearance counts
        appearances_60s = self._count_appearances(symbol, seconds=60)
        appearances_300s = self._count_appearances(symbol, seconds=300)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ticker_state (
                    symbol, last_seen_utc, appearances_60s, appearances_300s,
                    last_price, last_hits_5s, last_gap_color_bucket,
                    last_h_mean, last_s_mean, last_v_mean
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    last_seen_utc = excluded.last_seen_utc,
                    appearances_60s = excluded.appearances_60s,
                    appearances_300s = excluded.appearances_300s,
                    last_price = excluded.last_price,
                    last_hits_5s = excluded.last_hits_5s,
                    last_gap_color_bucket = excluded.last_gap_color_bucket,
                    last_h_mean = excluded.last_h_mean,
                    last_s_mean = excluded.last_s_mean,
                    last_v_mean = excluded.last_v_mean
            """, (
                symbol,
                now_utc.isoformat(),
                appearances_60s,
                appearances_300s,
                event_data.get('price'),
                event_data.get('hits_5s'),
                event_data.get('gap_color_bucket'),
                event_data.get('h_mean'),
                event_data.get('s_mean'),
                event_data.get('v_mean')
            ))
            
            conn.commit()
        
        logger.debug(f"Updated ticker state: {symbol} (60s={appearances_60s}, 300s={appearances_300s})")
    
    def _count_appearances(self, symbol: str, seconds: int) -> int:
        """
        Count appearances in the last N seconds
        
        Args:
            symbol: Ticker symbol
            seconds: Time window in seconds
            
        Returns:
            Appearance count
        """
        if symbol not in self.ticker_timestamps:
            return 0
        
        cutoff = datetime.utcnow() - timedelta(seconds=seconds)
        
        count = sum(1 for ts in self.ticker_timestamps[symbol] if ts >= cutoff)
        
        return count
    
    def get_ticker_state(self, symbol: str) -> dict | None:
        """
        Get current state for a ticker
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Ticker state dict or None
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM ticker_state WHERE symbol = ?
            """, (symbol,))
            
            row = cursor.fetchone()
            
            if row:
                return dict(row)
        
        return None
    
    def get_recent_events(self, limit: int = 100) -> list[dict]:
        """
        Get recent events
        
        Args:
            limit: Maximum number of events
            
        Returns:
            List of event dicts
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM events
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            
            return [dict(row) for row in rows]
