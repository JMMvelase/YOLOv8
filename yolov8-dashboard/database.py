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
            detections TEXT NOT NULL,
            camera TEXT
        )
    """)
    
    # Ensure 'camera' column exists for older DBs (SQLite ALTER TABLE ADD COLUMN is safe)
    cursor.execute("PRAGMA table_info(snapshots)")
    cols = [r[1] for r in cursor.fetchall()]
    if 'camera' not in cols:
        try:
            cursor.execute("ALTER TABLE snapshots ADD COLUMN camera TEXT")
        except Exception:
            # If ALTER TABLE fails for some SQLite versions, ignore and rely on image_path parsing later
            pass

    conn.commit()
    conn.close()

def insert_snapshot(timestamp: str, image_path: str, detections: Dict, camera: str = None):
    """Insert a new snapshot record into the database."""
    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()
    # Try inserting camera column value if present in table schema
    cursor.execute("PRAGMA table_info(snapshots)")
    cols = [r[1] for r in cursor.fetchall()]
    if 'camera' in cols:
        cursor.execute(
            "INSERT INTO snapshots (timestamp, image_path, detections, camera) VALUES (?, ?, ?, ?)",
            (timestamp, image_path, json.dumps(detections), camera)
        )
    else:
        cursor.execute(
            "INSERT INTO snapshots (timestamp, image_path, detections) VALUES (?, ?, ?)",
            (timestamp, image_path, json.dumps(detections))
        )
    
    conn.commit()
    conn.close()

def get_all_snapshots(start_date: str = None, end_date: str = None, cam: str = None) -> List[Dict]:
    """
    Retrieve snapshots from the database with optional date and camera filtering.

    Args:
        start_date: ISO format date string (YYYY-MM-DD)
        end_date: ISO format date string (YYYY-MM-DD)
        cam: camera identifier as string (e.g., '0' or '1'). If provided, only snapshots
             whose `image_path` contains `/snapshots/<cam>/` will be returned.
    """
    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()

    # Prefer returning camera column when available
    cursor.execute("PRAGMA table_info(snapshots)")
    cols = [r[1] for r in cursor.fetchall()]
    if 'camera' in cols:
        query = "SELECT timestamp, image_path, detections, camera FROM snapshots"
    else:
        query = "SELECT timestamp, image_path, detections FROM snapshots"
    params = []

    conditions = []
    if start_date:
        conditions.append("date(timestamp) >= date(?)")
        params.append(start_date)
    if end_date:
        conditions.append("date(timestamp) <= date(?)")
        params.append(end_date)
    if cam:
        # Match image_path that contains the camera folder, e.g. '/snapshots/1/'
        conditions.append("image_path LIKE ?")
        params.append(f"%/snapshots/{cam}/%")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY timestamp DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()

    snapshots = []
    for row in rows:
        if 'camera' in cols:
            snapshots.append({
                'timestamp': row[0],
                'image_path': row[1],
                'detections': json.loads(row[2]),
                'camera': row[3]
            })
        else:
            snapshots.append({
                'timestamp': row[0],
                'image_path': row[1],
                'detections': json.loads(row[2]),
                'camera': None
            })

    conn.close()
    return snapshots


def get_available_cameras() -> List[str]:
    """Return a list of camera identifiers present in the snapshots table.

    This inspects the `camera` column if present, and also parses `image_path` values
    to include cameras from older records saved without the column.
    """
    conn = sqlite3.connect(str(DATABASE_PATH))
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(snapshots)")
    cols = [r[1] for r in cursor.fetchall()]
    cameras = set()

    if 'camera' in cols:
        cursor.execute("SELECT DISTINCT camera FROM snapshots WHERE camera IS NOT NULL AND camera != ''")
        for (c,) in cursor.fetchall():
            cameras.add(str(c))

    # Also parse image_path values for patterns like '/snapshots/<cam>/'
    cursor.execute("SELECT DISTINCT image_path FROM snapshots WHERE image_path IS NOT NULL")
    for (path,) in cursor.fetchall():
        if not path:
            continue
        try:
            m = None
            # Look for /snapshots/<num>/
            import re
            m = re.search(r"/snapshots/(\d+)/", path)
            if m:
                cameras.add(m.group(1))
        except Exception:
            continue

    conn.close()
    return sorted(list(cameras))
