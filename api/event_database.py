
import sqlite3
from pathlib import Path
import pandas as pd


DATABASE_PATH = Path(__file__).resolve().parent / "events.db"


def create_event_table():
    """Create the events table if it does not already exist."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )


def save_event(user_id, item_id, event_type, timestamp):
    """Save one user event to the SQLite database."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            """
            INSERT INTO events (
                user_id,
                item_id,
                event_type,
                timestamp
            )
            VALUES (?, ?, ?, ?)
            """,
            (str(user_id), str(item_id), event_type, timestamp),
        )


def get_events():
    """Return all saved events as a list of dictionaries."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT id, user_id, item_id, event_type, timestamp
            FROM events
            ORDER BY id
            """
        ).fetchall()

    return [dict(row) for row in rows]


def load_events_dataframe():
    """Load saved events into a DataFrame with recency ranks."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        df = pd.read_sql_query(
            """
            SELECT id, user_id, item_id, event_type, timestamp
            FROM events
            ORDER BY timestamp DESC, id DESC
            """,
            connection,
        )

    if df.empty:
        df["recency_rank"] = pd.Series(dtype="int64")
        return df

    # Clean the event data.
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["item_id"] = pd.to_numeric(df["item_id"], errors="coerce")
    df["event_type"] = (
        df["event_type"].astype(str).str.strip().str.lower()
    )
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce", utc=True
    )

    # Remove records that cannot be used.
    df = df.dropna(
        subset=["user_id", "item_id", "event_type", "timestamp"]
    ).copy()

    df = df[df["event_type"] != ""]
    df["item_id"] = df["item_id"].astype(int)

    # Rank events separately for each user.
    # Rank 0 means the most recent event.
    df = df.sort_values(
        ["timestamp", "id"],
        ascending=[False, False],
    )

    df["recency_rank"] = (
        df.groupby("user_id").cumcount()
    )

    return df.reset_index(drop=True)
