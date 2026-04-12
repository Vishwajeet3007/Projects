#!/usr/bin/env python3
"""Seed script to generate random expenses for a user."""

import random
import sys
from datetime import datetime, timedelta
from database.db import get_db


# Category definitions with Indian rupee ranges and realistic descriptions
CATEGORIES = {
    "Food": {
        "min": 50,
        "max": 800,
        "weight": 25,
        "descriptions": [
            "Lunch at local restaurant",
            "Street food snacks",
            "Dinner at family restaurant",
            "Morning breakfast",
            "Coffee and snacks",
            "Biryani order",
            "Thali meal",
            "Fast food combo",
            "Tea and samosa",
            "Pizza delivery",
        ],
    },
    "Transport": {
        "min": 20,
        "max": 500,
        "weight": 18,
        "descriptions": [
            "Auto rickshaw fare",
            "Metro card recharge",
            "Bus ticket",
            "Uber/Ola ride",
            "Train ticket",
            "Taxi fare",
            "Fuel refill",
            "Parking charges",
            "Bike service",
            "Cycle rickshaw",
        ],
    },
    "Bills": {
        "min": 200,
        "max": 3000,
        "weight": 15,
        "descriptions": [
            "Electricity bill payment",
            "Mobile recharge",
            "Internet bill",
            "Water bill",
            "Gas cylinder refill",
            "DTH subscription",
            "Credit card payment",
            "Insurance premium",
            "Maintenance charges",
            "Property tax",
        ],
    },
    "Health": {
        "min": 100,
        "max": 2000,
        "weight": 8,
        "descriptions": [
            "Doctor consultation fee",
            "Medicine purchase",
            "Health checkup",
            "Dental treatment",
            "Eye examination",
            "Physiotherapy session",
            "Lab test charges",
            "Vaccination",
            "Medical equipment",
            "Health supplements",
        ],
    },
    "Entertainment": {
        "min": 100,
        "max": 1500,
        "weight": 10,
        "descriptions": [
            "Movie tickets",
            "Streaming subscription",
            "Concert entry",
            "Gaming zone visit",
            "Bowling alley",
            "Theme park entry",
            "Sports event ticket",
            "Comedy show",
            "Music album purchase",
            "Book purchase",
        ],
    },
    "Shopping": {
        "min": 200,
        "max": 5000,
        "weight": 15,
        "descriptions": [
            "Clothing purchase",
            "Footwear shopping",
            "Electronics accessory",
            "Home decor item",
            "Kitchen utensils",
            "Personal care products",
            "Gift item",
            "Watch purchase",
            "Bag purchase",
            "Furniture item",
        ],
    },
    "Other": {
        "min": 50,
        "max": 1000,
        "weight": 9,
        "descriptions": [
            "Miscellaneous expense",
            "Donation",
            "Gift to friend",
            "Stationery purchase",
            "Pet supplies",
            "Gardening supplies",
            "Car wash",
            "Laundry service",
            "Salon visit",
            "Small repair",
        ],
    },
}


def parse_args(args):
    """Parse command line arguments."""
    if len(args) != 3:
        return None

    try:
        user_id = int(args[0])
        count = int(args[1])
        months = int(args[2])
        return {"user_id": user_id, "count": count, "months": months}
    except ValueError:
        return None


def verify_user(user_id):
    """Check if user exists in the database."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def generate_expenses(user_id, count, months):
    """Generate random expenses spread across the past months."""
    expenses = []

    # Calculate date range
    today = datetime.now().date()
    start_date = today - timedelta(days=months * 30)

    # Build weighted category list for proportional distribution
    weighted_categories = []
    for category, config in CATEGORIES.items():
        weighted_categories.extend([category] * config["weight"])

    for _ in range(count):
        # Random date within the range
        random_days = random.randint(0, months * 30 - 1)
        expense_date = start_date + timedelta(days=random_days)

        # Select category based on weights
        category = random.choice(weighted_categories)
        config = CATEGORIES[category]

        # Generate random amount within range
        amount = round(random.uniform(config["min"], config["max"]), 2)

        # Select random description
        description = random.choice(config["descriptions"])

        expenses.append(
            (user_id, amount, category, expense_date.isoformat(), description)
        )

    return expenses


def insert_expenses(expenses):
    """Insert all expenses in a single transaction."""
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.executemany(
            "INSERT INTO expenses (user_id, amount, category, date, description) VALUES (?, ?, ?, ?, ?)",
            expenses,
        )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Error inserting expenses: {e}")
        return False
    finally:
        conn.close()


def main():
    # Step 1: Parse arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    parsed = parse_args(args)

    if parsed is None:
        print("Usage: /seed-expenses <user_id> <count> <months>")
        print("Example: /seed-expenses 1 50 6")
        sys.exit(1)

    user_id = parsed["user_id"]
    count = parsed["count"]
    months = parsed["months"]

    # Step 2: Verify user exists
    if not verify_user(user_id):
        print(f"No user found with id {user_id}.")
        sys.exit(1)

    # Step 3: Generate and insert expenses
    expenses = generate_expenses(user_id, count, months)

    if not insert_expenses(expenses):
        sys.exit(1)

    # Step 4: Confirm
    today = datetime.now().date()
    start_date = today - timedelta(days=months * 30)

    print(f"\nSuccessfully inserted {len(expenses)} expenses.")
    print(f"Date range: {start_date.isoformat()} to {today.isoformat()}")
    print("\nSample of 5 inserted records:")
    print("-" * 80)

    # Show sample from the generated data
    sample = random.sample(expenses, min(5, len(expenses)))
    for expense in sample:
        user_id, amount, category, date, description = expense
        print(f"Rs {amount:>7.2f} | {category:<15} | {date} | {description}")

    print("-" * 80)


if __name__ == "__main__":
    main()
