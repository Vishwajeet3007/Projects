import sqlite3
from flask import Flask, render_template, request, flash, redirect, url_for, session
from database.db import get_db, init_db, seed_db, create_user, get_user_by_email, insert_expense
from database.queries import get_user_by_id, get_summary_stats, get_recent_transactions, get_category_breakdown

app = Flask(__name__)
app.secret_key = "spendly-dev-secret-key-change-in-production"
app.config['DATABASE'] = "expense_tracker.db"

# Module-level constants
ALLOWED_CATEGORIES = ["Food", "Transport", "Bills", "Health", "Entertainment", "Shopping", "Other"]


def validate_expense_form(form_data):
    """Validate expense form data and return (is_valid, validated_data_or_error_message)."""
    amount = form_data.get("amount", "").strip()
    category = form_data.get("category", "").strip()
    date = form_data.get("date", "").strip()
    description = form_data.get("description", "").strip()

    # Validate required fields
    if not amount or not category or not date:
        return False, "Amount, category, and date are required"

    # Validate category is in the allowed list
    if category not in ALLOWED_CATEGORIES:
        return False, "Invalid category"

    # Validate amount is a positive number
    try:
        amount_float = float(amount)
        if amount_float <= 0:
            return False, "Amount must be greater than zero"
    except ValueError:
        return False, "Amount must be a valid number"

    # Validate date format and not in future
    from datetime import datetime
    try:
        expense_date = datetime.strptime(date, "%Y-%m-%d").date()
        today = datetime.now().date()
        if expense_date > today:
            return False, "Date cannot be in the future"
    except ValueError:
        return False, "Invalid date format"

    # All validations passed
    return True, {
        "amount": amount_float,
        "category": category,
        "date": date,
        "description": description if description else None
    }


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


@app.route("/analytics")
def analytics():
    from flask import session, redirect, url_for
    if not session.get("user_id"):
        return redirect(url_for("login"))
    return render_template("analytics.html")


@app.route("/expenses/add", methods=["GET", "POST"])
def add_expense():
    if not session.get("user_id"):
        return redirect(url_for("login"))
    if request.method == "POST":
        is_valid, result = validate_expense_form(request.form)
        if not is_valid:
            flash(result)
            return render_template("expenses/add.html")

        # Insert expense
        try:
            expense_id = insert_expense(
                user_id=session["user_id"],
                amount=result["amount"],
                category=result["category"],
                date=result["date"],
                description=result["description"]
            )
            flash("Expense added successfully!")
            return redirect(url_for("profile"))
        except Exception as e:
            flash("Failed to add expense. Please try again.")
            return render_template("expenses/add.html")
    # GET request - show form
    return render_template("expenses/add.html")

@app.route("/expenses/<int:id>/edit", methods=["GET", "POST"])
def edit_expense(id):
    if not session.get("user_id"):
        return redirect(url_for("login"))

    if request.method == "GET":
        # Get the expense to ensure it belongs to the current user
        conn = get_db()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT id, amount, category, date, description FROM expenses WHERE id = ? AND user_id = ?",
                (id, session["user_id"])
            )
            expense = cursor.fetchone()

            if expense is None:
                flash("Expense not found or access denied.")
                return redirect(url_for("profile"))

            # Convert to dict for easier access in template
            expense_dict = dict(expense)
            return render_template("expenses/edit.html", expense=expense_dict)
        finally:
            conn.close()

    elif request.method == "POST":
        # Validate form data
        is_valid, result = validate_expense_form(request.form)
        if not is_valid:
            flash(result)
            # Re-fetch the expense to re-populate form on validation error
            conn = get_db()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT id, amount, category, date, description FROM expenses WHERE id = ? AND user_id = ?",
                    (id, session["user_id"])
                )
                expense = cursor.fetchone()
                if expense is None:
                    flash("Expense not found or access denied.")
                    return redirect(url_for("profile"))
                expense_dict = dict(expense)
                return render_template("expenses/edit.html", expense=expense_dict)
            finally:
                conn.close()

        # Update the expense
        conn = get_db()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "UPDATE expenses SET amount = ?, category = ?, date = ?, description = ? WHERE id = ? AND user_id = ?",
                (result["amount"], result["category"], result["date"], result["description"], id, session["user_id"])
            )
            conn.commit()

            if cursor.rowcount == 0:
                flash("Expense not found or access denied.")
                return redirect(url_for("profile"))

            flash("Expense updated successfully!")
            return redirect(url_for("profile"))
        except Exception as e:
            flash("Failed to update expense. Please try again.")
            return redirect(url_for("profile"))
        finally:
            conn.close()


@app.route("/expenses/<int:id>/delete", methods=["POST"])
def delete_expense(id):
    if not session.get("user_id"):
        return redirect(url_for("login"))

    # Verify the expense exists and belongs to the current user
    from database.queries import get_expense_by_id
    expense = get_expense_by_id(id, session["user_id"])
    if expense is None:
        # Abort with 404 as per spec
        from flask import abort
        abort(404)

    # Delete the expense
    from database.queries import delete_expense
    delete_expense(session["user_id"], id)
    flash("Expense deleted successfully!")

    return redirect(url_for("profile"))


if __name__ == "__main__":
    with app.app_context():
        init_db()
        seed_db()
    app.run(debug=True, port=5000)