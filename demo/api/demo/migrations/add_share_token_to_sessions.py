"""
One-off migration: add share_token to sessions table.
Uses raw SQL so it works with both SQLite and PostgreSQL.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from demo.database import engine


def run():
    with engine.connect() as conn:
        if engine.dialect.name == "sqlite":
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"))
            if result.fetchone() is None:
                return
            result = conn.execute(text("PRAGMA table_info(sessions)"))
            columns = [row[1] for row in result]
            if "share_token" in columns:
                return
            conn.execute(text("ALTER TABLE sessions ADD COLUMN share_token VARCHAR(36)"))
        else:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS share_token VARCHAR(36)"))
        conn.commit()


if __name__ == "__main__":
    run()
