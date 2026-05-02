import sqlite3
from flask import Flask, render_template, request, flash, redirect, url_for
from database.db import get_db, init_db, seed_db, create_user, get_user_by_email
from database.queries import get_user_by_id, get_summary_stats, get_recent_transactions, get_category_breakdown

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
    from flask import session, request, flash
    from datetime import datetime, timedelta
    if not session.get("user_id"):
        return redirect(url_for("login"))

    user_id = session.get("user_id")

    # Parse date filter parameters
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()
    preset = request.args.get("preset", "").strip()

    # Handle preset filters
    if preset and not (date_from or date_to):
        today = datetime.now().date()
        if preset == "this_month":
            date_from = today.replace(day=1).isoformat()
            date_to = today.isoformat()
        elif preset == "last_3_months":
            date_to = today.isoformat()
            date_from = (today - timedelta(days=90)).isoformat()
        elif preset == "last_6_months":
            date_to = today.isoformat()
            date_from = (today - timedelta(days=180)).isoformat()
        elif preset == "all_time":
            date_from = None
            date_to = None

    # Validate date formats
    validated_from = None
    validated_to = None

    if date_from or date_to:
        try:
            if date_from:
                validated_from = datetime.strptime(date_from, "%Y-%m-%d").date()
            if date_to:
                validated_to = datetime.strptime(date_to, "%Y-%m-%d").date()

            # Validate date range
            if validated_from and validated_to and validated_from > validated_to:
                flash("Start date must be before end date.")
                validated_from = None
                validated_to = None
            else:
                # Convert back to ISO format strings for queries
                if validated_from:
                    validated_from = validated_from.isoformat()
                if validated_to:
                    validated_to = validated_to.isoformat()
        except ValueError:
            # Invalid date format - silently fall back to no filter
            validated_from = None
            validated_to = None

    # Fetch real data from database with date filter
    user_info = get_user_by_id(user_id)
    summary_stats = get_summary_stats(user_id, date_from=validated_from, date_to=validated_to)
    transactions = get_recent_transactions(user_id, date_from=validated_from, date_to=validated_to)
    category_breakdown = get_category_breakdown(user_id, date_from=validated_from, date_to=validated_to)

    # Determine active filter label for template
    filter_label = "All Time"
    if validated_from and validated_to:
        filter_label = f"{validated_from} to {validated_to}"
    elif preset:
        filter_label = preset.replace("_", " ").title()

    return render_template("profile.html",
                           user_info=user_info,
                           summary_stats=summary_stats,
                           transactions=transactions,
                           category_breakdown=category_breakdown,
                           date_from=validated_from,
                           date_to=validated_to,
                           filter_label=filter_label)


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
