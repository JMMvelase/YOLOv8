import sqlite3
from pathlib import Path
from typing import Dict, List
import json

DATABASE_PATH = Path("snapshots.db")

def init_db():
    """Initialize the database with required tables."""
    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()
    
    # Create snapshots table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            image_path TEXT NOT NULL,
            detections TEXT NOT NULL
        )
    """)
    
    conn.commit()
    conn.close()

def insert_snapshot(timestamp: str, image_path: str, detections: Dict):
    """Insert a new snapshot record into the database."""
    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT INTO snapshots (timestamp, image_path, detections) VALUES (?, ?, ?)",
        (timestamp, image_path, json.dumps(detections))
    )
    
    conn.commit()
    conn.close()

def get_all_snapshots(start_date: str = None, end_date: str = None) -> List[Dict]:
    """
    Retrieve snapshots from the database with optional date filtering.
    Args:
        start_date: ISO format date string (YYYY-MM-DD)
        end_date: ISO format date string (YYYY-MM-DD)
    """
    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()
    
    query = "SELECT timestamp, image_path, detections FROM snapshots"
    params = []
    
    if start_date or end_date:
        conditions = []
        if start_date:
            conditions.append("date(timestamp) >= date(?)")
            params.append(start_date)
        if end_date:
            conditions.append("date(timestamp) <= date(?)")
            params.append(end_date)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY timestamp DESC"
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    snapshots = []
    for row in rows:
        snapshots.append({
            'timestamp': row[0],
            'image_path': row[1],
            'detections': json.loads(row[2])
        })
    
    conn.close()
    return snapshots
