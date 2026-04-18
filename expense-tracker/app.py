import sqlite3
from flask import Flask, render_template, request, flash, redirect, url_for
from database.db import get_db, init_db, seed_db, create_user, get_user_by_email

app = Flask(__name__)
app.secret_key = "spendly-dev-secret-key-change-in-production"


# ------------------------------------------------------------------ #
# Routes                                                              #
# ------------------------------------------------------------------ #

@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    from flask import session
    if session.get('user_id'):
        return redirect(url_for('profile'))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()

        # Validate required fields
        if not name or not email or not password or not confirm_password:
            flash("All fields are required")
            return render_template("register.html")

        # Validate password length
        if len(password) < 6:
            flash("Password must be at least 6 characters")
            return render_template("register.html")

        # Validate passwords match
        if password != confirm_password:
            flash("Passwords do not match")
            return render_template("register.html")

        try:
            create_user(name, email, password)
            flash("Account created successfully! Please sign in.")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered")
            return render_template("register.html")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    from flask import session
    if session.get('user_id'):
        return redirect(url_for('profile'))

    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        # Validate required fields
        if not email or not password:
            flash("All fields are required")
            return render_template("login.html")

        # Check if user exists
        user = get_user_by_email(email)
        if user is None:
            flash("Invalid email or password")
            return render_template("login.html")

        # Verify password
        from werkzeug.security import check_password_hash
        if not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password")
            return render_template("login.html")

        # Create session
        from flask import session
        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        flash("Welcome back!")
        return redirect(url_for("profile"))

    return render_template("login.html")


@app.route("/terms")
def terms():
    return render_template("terms.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


# ------------------------------------------------------------------ #
# Placeholder routes — students will implement these                  #
# ------------------------------------------------------------------ #

@app.route("/logout")
def logout():
    from flask import session
    session.clear()
    flash("You have been logged out")
    return redirect(url_for("login"))


@app.route("/profile")
def profile():
    from flask import session
    if not session.get("user_id"):
        return redirect(url_for("login"))

    # Fetch user info from database
    conn = get_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT name, email, created_at FROM users WHERE id = ?", (session.get("user_id"),))
    user = cursor.fetchone()
    conn.close()

    user_info = {
        "name": user["name"],
        "email": user["email"],
        "member_since": user["created_at"][:10] if user["created_at"] else "Unknown"
    }

    # Hardcoded summary stats
    summary_stats = {
        "total_spent": 1250.50,
        "transaction_count": 24,
        "top_category": "Food & Dining"
    }

    # Hardcoded transactions
    transactions = [
        {"date": "2026-04-15", "description": "Grocery shopping", "category": "Food & Dining", "amount": 85.20},
        {"date": "2026-04-14", "description": "Netflix subscription", "category": "Entertainment", "amount": 15.99},
        {"date": "2026-04-12", "description": "Gas station", "category": "Transport", "amount": 45.00},
        {"date": "2026-04-10", "description": "Restaurant dinner", "category": "Food & Dining", "amount": 62.50},
        {"date": "2026-04-08", "description": "Uber ride", "category": "Transport", "amount": 18.75},
    ]

    # Hardcoded category breakdown
    category_breakdown = [
        {"name": "Food & Dining", "amount": 450.75, "percentage": 36},
        {"name": "Transport", "amount": 320.00, "percentage": 26},
        {"name": "Entertainment", "amount": 180.50, "percentage": 14},
        {"name": "Shopping", "amount": 150.25, "percentage": 12},
        {"name": "Utilities", "amount": 149.00, "percentage": 12},
    ]

    return render_template("profile.html",
                           user_info=user_info,
                           summary_stats=summary_stats,
                           transactions=transactions,
                           category_breakdown=category_breakdown)


@app.route("/expenses/add")
def add_expense():
    return "Add expense — coming in Step 7"


@app.route("/expenses/<int:id>/edit")
def edit_expense(id):
    return "Edit expense — coming in Step 8"


@app.route("/expenses/<int:id>/delete")
def delete_expense(id):
    return "Delete expense — coming in Step 9"


if __name__ == "__main__":
    with app.app_context():
        init_db()
        seed_db()
    app.run(debug=True, port=5000)
