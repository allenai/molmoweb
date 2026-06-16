"""
One-off migration: add user_id to sessions table.
Run once: python -m demo.migrations.add_user_id_to_sessions
Uses raw SQL so it works with both SQLite and PostgreSQL.
"""
import os
import sys

# Add api root so we can import demo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text
from demo.database import engine

DEFAULT_USER_ID = "demo.user@allenai.org"


def run():
    with engine.connect() as conn:
        if engine.dialect.name == "sqlite":
            # Check if table exists
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"))
            if result.fetchone() is None:
                print("sessions table does not exist yet (fresh install); run the app to create it with user_id")
                return
            result = conn.execute(text("PRAGMA table_info(sessions)"))
            columns = [row[1] for row in result]
            if "user_id" in columns:
                print("user_id column already exists, skipping")
                return
            conn.execute(text("ALTER TABLE sessions ADD COLUMN user_id VARCHAR(255)"))
            conn.execute(text("UPDATE sessions SET user_id = :uid WHERE user_id IS NULL"), {"uid": DEFAULT_USER_ID})
        else:
            # PostgreSQL
            conn.execute(text("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id VARCHAR(255)"))
            conn.execute(text("UPDATE sessions SET user_id = :uid WHERE user_id IS NULL"), {"uid": DEFAULT_USER_ID})
        conn.commit()
    print("Migration complete: user_id added to sessions")


if __name__ == "__main__":
    run()
