"""
One-off migration: add is_answer and is_final to chat_messages table.
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
            if "is_answer" not in columns:
                conn.execute(text("ALTER TABLE chat_messages ADD COLUMN is_answer BOOLEAN DEFAULT 0"))
            if "is_final" not in columns:
                conn.execute(text("ALTER TABLE chat_messages ADD COLUMN is_final BOOLEAN DEFAULT 0"))
        else:
            conn.execute(text("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS is_answer BOOLEAN DEFAULT FALSE"))
            conn.execute(text("ALTER TABLE chat_messages ADD COLUMN IF NOT EXISTS is_final BOOLEAN DEFAULT FALSE"))
        conn.commit()
    print("Migration complete: is_answer/is_final added to chat_messages")


if __name__ == "__main__":
    run()
