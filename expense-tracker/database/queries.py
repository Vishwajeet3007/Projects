import sqlite3
from database.db import get_db


def get_user_by_id(user_id):
    """
    Fetches user info by ID.
    Returns dict with name, email, member_since (formatted as "Month YYYY").
    Returns None if user not found.
    """
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT name, email, created_at FROM users WHERE id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        if row:
            # Format created_at as "Month YYYY"
            created_at = row["created_at"]
            if created_at:
                from datetime import datetime
                try:
                    dt = datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S")
                    member_since = dt.strftime("%B %Y")
                except ValueError:
                    member_since = created_at[:10] if len(created_at) >= 10 else "Unknown"
            else:
                member_since = "Unknown"

            return {
                "name": row["name"],
                "email": row["email"],
                "member_since": member_since
            }
        return None
    finally:
        conn.close()


def get_summary_stats(user_id):
    """
    Calculates summary statistics for a user's expenses.
    Returns dict with total_spent, transaction_count, top_category.
    Returns zeros and "—" if user has no expenses.
    """
    conn = get_db()
    cursor = conn.cursor()

    try:
        # Get total spent and transaction count
        cursor.execute(
            "SELECT COALESCE(SUM(amount), 0) AS total, COUNT(*) AS count FROM expenses WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        total_spent = round(row["total"], 2)
        transaction_count = row["count"]

        # Get top category (category with highest total spending)
        cursor.execute(
            "SELECT category FROM expenses WHERE user_id = ? GROUP BY category ORDER BY SUM(amount) DESC LIMIT 1",
            (user_id,)
        )
        top_row = cursor.fetchone()
        top_category = top_row["category"] if top_row else "—"

        return {
            "total_spent": total_spent,
            "transaction_count": transaction_count,
            "top_category": top_category
        }
    finally:
        conn.close()


def get_recent_transactions(user_id, limit=10):
    """
    Fetches recent expenses for a user, ordered by date descending.
    Returns list of dicts with date, description, category, amount.
    Returns empty list if no expenses.
    """
    conn = get_db()
    cursor = conn.cursor()

    try:
        cursor.execute(
            "SELECT date, description, category, amount FROM expenses WHERE user_id = ? ORDER BY date DESC LIMIT ?",
            (user_id, limit)
        )
        rows = cursor.fetchall()
        return [
            {
                "date": row["date"],
                "description": row["description"],
                "category": row["category"],
                "amount": row["amount"]
            }
            for row in rows
        ]
    finally:
        conn.close()


def get_category_breakdown(user_id):
    """
    Calculates per-category spending breakdown.
    Returns list of dicts with name, amount, pct (percentage rounded to int).
    Percentages sum to 100, with largest category absorbing rounding remainder.
    Returns empty list if no expenses.
    """
    conn = get_db()
    cursor = conn.cursor()

    try:
        # Get per-category totals
        cursor.execute(
            "SELECT category, SUM(amount) AS total FROM expenses WHERE user_id = ? GROUP BY category ORDER BY total DESC",
            (user_id,)
        )
        rows = cursor.fetchall()

        if not rows:
            return []

        # Calculate grand total
        grand_total = sum(row["total"] for row in rows)

        # Build breakdown with raw percentages
        breakdown = []
        for row in rows:
            breakdown.append({
                "name": row["category"],
                "amount": round(row["total"], 2),
                "raw_pct": (row["total"] / grand_total) * 100 if grand_total > 0 else 0
            })

        # Round percentages and adjust for rounding error
        total_pct = 0
        max_idx = 0
        max_amount = -1

        for i, item in enumerate(breakdown):
            item["pct"] = round(item["raw_pct"])
            total_pct += item["pct"]
            if item["amount"] > max_amount:
                max_amount = item["amount"]
                max_idx = i

        # Adjust largest category to ensure sum = 100
        if breakdown:
            breakdown[max_idx]["pct"] += 100 - total_pct

        # Ensure no negative or empty percentages
        for item in breakdown:
            if item["pct"] < 0:
                item["pct"] = 0

        # Remove raw_pct from output
        for item in breakdown:
            del item["raw_pct"]

        return breakdown
    finally:
        conn.close()
