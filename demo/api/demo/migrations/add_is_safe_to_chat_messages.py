"""
One-off migration: add is_safe to chat_messages table.
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
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_messages'"))
            if result.fetchone() is None:
                print("chat_messages table does not exist yet (fresh install); skipping")
                return
            result = conn.execute(text("PRAGMA table_info(chat_messages)"))
            columns = [row[1] for row in result]
            if "is_safe" not in columns:
                conn.execute(text("ALTER TABLE chat_messages ADD COLUMN is_safe BOOLEAN DEFAULT 1"))
        else:
            conn.execute(text("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS is_safe BOOLEAN DEFAULT TRUE"))
        conn.commit()
    print("Migration complete: is_safe added to chat_messages")


if __name__ == "__main__":
    run()
