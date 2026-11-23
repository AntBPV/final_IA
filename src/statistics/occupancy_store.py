import sqlite3
from datetime import datetime
from pathlib import Path


class OccupancyStore:
    def __init__(self, database: str = "models/occupancy.db"):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        return sqlite3.connect(self.database)

    def _initialize(self):
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS occupancy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    entries INTEGER NOT NULL,
                    exits INTEGER NOT NULL,
                    occupancy INTEGER NOT NULL
                )
                """
            )

    def record(self, entries: int, exits: int, occupancy: int):
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO occupancy(timestamp, entries, exits, occupancy) VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), entries, exits, occupancy),
            )

    def summary(self):
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*), COALESCE(MAX(entries), 0), "
                "COALESCE(MAX(exits), 0), COALESCE(MAX(occupancy), 0) "
                "FROM occupancy"
            ).fetchone()
        return {
            "registros": row[0],
            "entradas": row[1],
            "salidas": row[2],
            "ocupacion_maxima": row[3],
        }

