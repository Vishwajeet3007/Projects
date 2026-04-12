import sqlite3
import random
from werkzeug.security import generate_password_hash

DATABASE = "expense_tracker.db"


def get_db():
    """Opens connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# Common Indian first names (mixed gender)
FIRST_NAMES = [
    "Aarav", "Vihaan", "Aditya", "Sai", "Arjun", "Karan", "Rahul", "Amit",
    "Priya", "Ananya", "Diya", "Ishani", "Meera", "Kavya", "Riya", "Sneha",
    "Rohan", "Vikram", "Arjun", "Nikhil", "Raj", "Ajay", "Sanjay", "Vijay",
    "Pooja", "Neha", "Anjali", "Deepika", "Shruti", "Aditi", "Malini", "Tara",
    "Aryan", "Ishaan", "Reyansh", "Ayaan", "Krishna", "Ram", "Shyam", "Dev",
    "Lakshmi", "Saraswati", "Gauri", "Shreya", "Nandini", "Divya", "Jyoti"
]

# Common Indian last names from various regions
LAST_NAMES = [
    "Sharma", "Verma", "Gupta", "Agarwal", "Singh", "Kumar", "Yadav", "Patel",
    "Desai", "Joshi", "Bhatt", "Shah", "Mehta", "Kapoor", "Malhotra", "Khanna",
    "Reddy", "Rao", "Nair", "Iyer", "Iyengar", "Menon", "Pillai", "Varma",
    "Chatterjee", "Banerjee", "Mukherjee", "Ganguly", "Das", "Bose", "Sen",
    "Joshi", "Deshmukh", "Patil", "Kulkarni", "Pawar", "Shinde", "Jadhav",
    "Gowda", "Hegde", "Bhat", "Kamath", "Rai", "Shetty", "Poojary"
]

DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "rediffmail.com"]


def generate_user():
    """Generates a realistic random Indian user."""
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    name = f"{first_name} {last_name}"

    # Create email from name with random 2-3 digit suffix
    email_prefix = f"{first_name.lower()}.{last_name.lower()}"
    email_suffix = random.randint(10, 999)
    domain = random.choice(DOMAINS)
    email = f"{email_prefix}{email_suffix}@{domain}"

    # Hash the password
    password_hash = generate_password_hash("password123")

    return name, email, password_hash


def email_exists(email):
    """Checks if an email already exists in the users table."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def seed_user():
    """Generates a unique user and inserts into the database."""
    max_attempts = 100
    attempts = 0

    while attempts < max_attempts:
        name, email, password_hash = generate_user()

        if not email_exists(email):
            break

        attempts += 1

    if attempts >= max_attempts:
        print("Failed to generate unique email after multiple attempts.")
        return

    # Insert the user
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
        (name, email, password_hash)
    )
    conn.commit()

    # Get the inserted user's id
    user_id = cursor.lastrowid
    conn.close()

    # Print confirmation
    print(f"id: {user_id}")
    print(f"name: {name}")
    print(f"email: {email}")


if __name__ == "__main__":
    seed_user()
