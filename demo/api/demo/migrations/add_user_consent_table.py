"""
One-off migration: create user_consent table.
Run once: python -m demo.migrations.add_user_consent_table
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
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='user_consent'"))
            if result.fetchone() is not None:
                print("user_consent table already exists, skipping")
                return
            conn.execute(text("""
                CREATE TABLE user_consent (
                    user_id VARCHAR(255) PRIMARY KEY,
                    terms_accepted BOOLEAN NOT NULL DEFAULT 0,
                    open_science_consent BOOLEAN NOT NULL DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
        else:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS user_consent (
                    user_id VARCHAR(255) PRIMARY KEY,
                    terms_accepted BOOLEAN NOT NULL DEFAULT FALSE,
                    open_science_consent BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                )
            """))
        conn.commit()
    print("Migration complete: user_consent table created")


if __name__ == "__main__":
    run()
