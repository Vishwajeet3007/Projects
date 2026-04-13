import sqlite3
from werkzeug.security import generate_password_hash

DATABASE = "expense_tracker.db"


def get_db():
    """
    Opens connection to the SQLite database.
    Sets row_factory for dict-like access and enables foreign keys.
    Returns the connection.
    """
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """
    Creates both users and expenses tables using CREATE TABLE IF NOT EXISTS.
    Safe to call multiple times.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Create users table
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "name TEXT NOT NULL, "
        "email TEXT UNIQUE NOT NULL, "
        "password_hash TEXT NOT NULL, "
        "created_at TEXT DEFAULT (datetime('now'))"
        ")"
    )

    # Create expenses table
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS expenses ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "user_id INTEGER NOT NULL, "
        "amount REAL NOT NULL, "
        "category TEXT NOT NULL, "
        "date TEXT NOT NULL, "
        "description TEXT, "
        "created_at TEXT DEFAULT (datetime('now')), "
        "FOREIGN KEY (user_id) REFERENCES users(id)"
        ")"
    )

    conn.commit()
    conn.close()


def create_user(name, email, password):
    """
    Creates a new user with the given name, email, and password.
    Hashes the password using werkzeug before storing.
    Returns the new user's id.
    Raises sqlite3.IntegrityError if email already exists.
    """
    password_hash = generate_password_hash(password)
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
            (name, email, password_hash)
        )

        user_id = cursor.lastrowid
        conn.commit()
        return user_id
    finally:
        conn.close()


def get_user_by_email(email):
    """
    Fetches a user row by email address for authentication.
    Returns the user dict with keys: id, name, email, password_hash, created_at
    Returns None if no user found.
    """
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT id, name, email, password_hash, created_at FROM users WHERE email = ?",
            (email,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def seed_db():
    """
    Inserts demo user and 8 sample expenses.
    Checks for existing data to prevent duplicates on repeated runs.
    """
    conn = get_db()
    cursor = conn.cursor()

    # Check if users table already has data
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    if count > 0:
        conn.close()
        return

    # Insert demo user
    password_hash = generate_password_hash("demo123")
    cursor.execute(
        "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
        ("Demo User", "demo@spendly.com", password_hash)
    )

    # Get the user_id for the demo user
    cursor.execute("SELECT id FROM users WHERE email = ?", ("demo@spendly.com",))
    user_id = cursor.fetchone()[0]

    # Insert 8 sample expenses across all categories
    # Using current month (2026-04) for dates
    expenses = [
        (user_id, 45.50, "Food", "2026-04-01", "Lunch at cafe"),
        (user_id, 25.00, "Transport", "2026-04-02", "Uber ride"),
        (user_id, 120.00, "Bills", "2026-04-03", "Electric bill"),
        (user_id, 35.00, "Health", "2026-04-05", "Pharmacy"),
        (user_id, 60.00, "Entertainment", "2026-04-07", "Movie tickets"),
        (user_id, 89.99, "Shopping", "2026-04-10", "New shirt"),
        (user_id, 50.00, "Other", "2026-04-12", "Miscellaneous"),
        (user_id, 32.50, "Food", "2026-04-15", "Dinner out"),
    ]

    cursor.executemany(
        "INSERT INTO expenses (user_id, amount, category, date, description) VALUES (?, ?, ?, ?, ?)",
        expenses
    )

    conn.commit()
    conn.close()
