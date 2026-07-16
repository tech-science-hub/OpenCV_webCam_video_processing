import sqlite3
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash


class AuthStore:
    """SQLite-backed dashboard user store."""

    def __init__(self, db_path=None, default_user="admin", default_password="admin"):
        self.db_path = Path(db_path or Path(__file__).resolve().parent / "users.db")
        self.default_user = default_user
        self.default_password = default_password
        self.ensure_default_user()

    def ensure_default_user(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                )
            """)
            cur.execute("SELECT * FROM users WHERE username=?", (self.default_user,))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                    (self.default_user, generate_password_hash(self.default_password)),
                )

    def get_first_username(self):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT username FROM users ORDER BY id ASC LIMIT 1")
            row = cur.fetchone()
            return row[0] if row else ""

    def authenticate(self, username, password):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, password_hash FROM users WHERE username=?",
                (username,),
            )
            user = cur.fetchone()

        if not user:
            return None

        user_id, stored_hash = user
        if not check_password_hash(stored_hash, password):
            return None
        return user_id

    def change_credentials(self, user_id, current_password, new_login, new_password):
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT password_hash FROM users WHERE id=?", (user_id,))
            user = cur.fetchone()

            if not user:
                return False, "User not found"

            if not check_password_hash(user[0], current_password):
                return False, "Wrong current password"

            cur.execute(
                "UPDATE users SET username=?, password_hash=? WHERE id=?",
                (new_login, generate_password_hash(new_password), user_id),
            )

        return True, ""
