#!/usr/bin/env python3
"""
Quick verification tool to query the last N events from the database

Usage:
    python tools/query_events.py [N] [--green] [--minutes M]
"""
import sqlite3
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config

def query_last_events(n=10, green_only=False, minutes=None):
    """Query last N events from database with optional filters"""
    db_path = config.DB_PATH
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Build query with filters
    where_clauses = []
    params = []
    
    if green_only:
        where_clauses.append("gap_color_bucket = 'GREEN'")
    
    if minutes:
        cutoff = datetime.now() - timedelta(minutes=minutes)
        where_clauses.append("ts_local >= ?")
        params.append(cutoff.isoformat())
    
    where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
    
    query = f"""
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
    WHERE {where_clause}
    ORDER BY id DESC
    LIMIT ?
    """
    
    params.append(n)
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    if not rows:
        print("No events found matching criteria")
        return
    
    print(f"\n{'='*120}")
    filter_desc = []
    if green_only:
        filter_desc.append("GREEN gaps only")
    if minutes:
        filter_desc.append(f"last {minutes} minutes")
    filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
    print(f"Last {len(rows)} events from database{filter_str}:")
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
    cursor.execute(f"SELECT COUNT(*) as total, SUM(parsed) as successful FROM events WHERE {where_clause}", params[:-1])
    stats = cursor.fetchone()
    print(f"Total events: {stats['total']}, Successful parses: {stats['successful']}, Failed: {stats['total'] - stats['successful']}")
    
    # Empty OCR rate
    cursor.execute("""
        SELECT COUNT(*) as empty_ocr
        FROM events
        WHERE (raw_text LIKE "%'symbol': ''%" OR raw_text LIKE "%'price': ''%")
    """)
    empty_stats = cursor.fetchone()
    if empty_stats and stats['total'] > 0:
        empty_rate = (empty_stats['empty_ocr'] / stats['total']) * 100
        print(f"Empty OCR rate: {empty_stats['empty_ocr']}/{stats['total']} ({empty_rate:.1f}%)")
    
    # Parse success rate
    if stats['total'] > 0:
        success_rate = (stats['successful'] / stats['total']) * 100
        print(f"Parse success rate: {success_rate:.1f}%")
    
    # Top 5 most common parse failure notes
    cursor.execute("""
        SELECT notes, COUNT(*) as count
        FROM events
        WHERE parsed = 0 AND notes IS NOT NULL AND notes != ''
        GROUP BY notes
        ORDER BY count DESC
        LIMIT 5
    """)
    failure_notes = cursor.fetchall()
    if failure_notes:
        print("\nTop 5 failure reasons:")
        for idx, row in enumerate(failure_notes, 1):
            print(f"  {idx}. [{row['count']}x] {row['notes'][:60]}")
    
    # Price statistics for parsed events
    cursor.execute(f"""
        SELECT 
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(price) as avg_price,
            COUNT(DISTINCT symbol) as unique_symbols
        FROM events
        WHERE parsed = 1 AND price IS NOT NULL AND {where_clause}
    """, params[:-1])
    price_stats = cursor.fetchone()
    if price_stats['avg_price']:
        print(f"Price range: ${price_stats['min_price']:.4f} - ${price_stats['max_price']:.4f}, "
              f"Avg: ${price_stats['avg_price']:.4f}, "
              f"Unique symbols: {price_stats['unique_symbols']}")
    
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Query scanner events from database')
    parser.add_argument('n', nargs='?', type=int, default=20, help='Number of events to show (default: 20)')
    parser.add_argument('--green', action='store_true', help='Show only GREEN gap events')
    parser.add_argument('--minutes', type=int, help='Show only events from last N minutes')
    
    args = parser.parse_args()
    query_last_events(args.n, green_only=args.green, minutes=args.minutes)
