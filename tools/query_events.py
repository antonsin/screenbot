#!/usr/bin/env python3
"""
Quick verification tool to query the last N events from the database
"""
import sqlite3
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

def query_last_events(n=10):
    """Query last N events from database"""
    db_path = config.DB_PATH
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = """
    SELECT 
        id,
        ts_local,
        symbol,
        price,
        gap_color_bucket,
        parsed,
        raw_text,
        notes
    FROM events
    ORDER BY id DESC
    LIMIT ?
    """
    
    cursor.execute(query, (n,))
    rows = cursor.fetchall()
    
    if not rows:
        print("No events found in database")
        return
    
    print(f"\n{'='*120}")
    print(f"Last {len(rows)} events from database:")
    print(f"{'='*120}")
    print(f"{'ID':>6} {'Timestamp':19} {'Symbol':8} {'Price':12} {'Gap':8} {'P':1} {'Raw Text':30} {'Notes':20}")
    print(f"{'-'*120}")
    
    for row in reversed(rows):
        row_id = row['id']
        ts = row['ts_local'][:19] if row['ts_local'] else 'N/A'
        symbol = row['symbol'] or 'NULL'
        price = f"${row['price']:.4f}" if row['price'] is not None else 'NULL'
        gap = row['gap_color_bucket'] or 'N/A'
        parsed = row['parsed']
        raw_text = str(row['raw_text'] or '')[:30]
        notes = str(row['notes'] or '')[:20]
        
        print(f"{row_id:6d} {ts:19s} {symbol:8s} {price:12s} {gap:8s} {parsed:1d} {raw_text:30s} {notes:20s}")
    
    print(f"{'='*120}\n")
    
    # Summary stats
    cursor.execute("SELECT COUNT(*) as total, SUM(parsed) as successful FROM events")
    stats = cursor.fetchone()
    print(f"Total events: {stats['total']}, Successful parses: {stats['successful']}, Failed: {stats['total'] - stats['successful']}")
    
    # Price statistics for parsed events
    cursor.execute("""
        SELECT 
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(price) as avg_price,
            COUNT(DISTINCT symbol) as unique_symbols
        FROM events
        WHERE parsed = 1 AND price IS NOT NULL
    """)
    price_stats = cursor.fetchone()
    if price_stats['avg_price']:
        print(f"Price range: ${price_stats['min_price']:.4f} - ${price_stats['max_price']:.4f}, "
              f"Avg: ${price_stats['avg_price']:.4f}, "
              f"Unique symbols: {price_stats['unique_symbols']}")
    
    conn.close()

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    query_last_events(n)
