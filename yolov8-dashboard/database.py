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

def get_all_snapshots() -> List[Dict]:
    """Retrieve all snapshots from the database."""
    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()
    
    cursor.execute("SELECT timestamp, image_path, detections FROM snapshots ORDER BY timestamp DESC")
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
