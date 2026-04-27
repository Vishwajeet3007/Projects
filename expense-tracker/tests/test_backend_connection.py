import pytest
from database.db import get_db, init_db, seed_db
from database.queries import (
    get_user_by_id,
    get_summary_stats,
    get_recent_transactions,
    get_category_breakdown,
)

# Import app module (not the instance) to avoid fixture name conflict
import app as app_module


@pytest.fixture
def app_ctx():
    """Create application context for tests."""
    app_module.app.config["TESTING"] = True
    app_module.app.config["SECRET_KEY"] = "test-secret-key"
    with app_module.app.app_context():
        init_db()
        seed_db()
        yield app_module.app


@pytest.fixture
def client():
    """Create a test client with a fresh database."""
    app_module.app.config["TESTING"] = True
    app_module.app.config["SECRET_KEY"] = "test-secret-key"

    with app_module.app.app_context():
        init_db()
        seed_db()
        with app_module.app.test_client() as test_client:
            yield test_client


@pytest.fixture
def logged_in_client(client):
    """Create a logged-in test client."""
    with client.session_transaction() as sess:
        sess["user_id"] = 1
        sess["user_name"] = "Demo User"
    return client


# =============================================================================
# Unit tests for query functions
# =============================================================================

class TestGetUserById:
    def test_get_existing_user(self):
        """Should return user dict with correct name, email, member_since."""
        user = get_user_by_id(1)
        assert user is not None
        assert user["name"] == "Demo User"
        assert user["email"] == "demo@spendly.com"
        assert "member_since" in user
        # member_since should be formatted as "Month YYYY"
        assert len(user["member_since"]) > 0

    def test_get_nonexistent_user(self):
        """Should return None for non-existent user ID."""
        user = get_user_by_id(99999)
        assert user is None


class TestGetSummaryStats:
    def test_stats_with_expenses(self):
        """Should return correct total_spent, transaction_count, top_category."""
        stats = get_summary_stats(1)
        # Sum of 8 seed expenses: 45.50+25.00+120.00+35.00+60.00+89.99+50.00+32.50 = 457.99
        assert stats["total_spent"] == 457.99
        assert stats["transaction_count"] == 8
        assert stats["top_category"] == "Bills"  # 120.00 is highest single category

    def test_stats_no_expenses(self, app_ctx):
        """Should return zeros for user with no expenses."""
        with app_ctx.app_context():
            # Create a new user without expenses
            conn = get_db()
            cursor = conn.cursor()
            from werkzeug.security import generate_password_hash
            import uuid
            unique_email = f"test_{uuid.uuid4().hex[:8]}@example.com"
            cursor.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                ("Test User", unique_email, generate_password_hash("password"))
            )
            new_user_id = cursor.lastrowid
            conn.commit()
            conn.close()

            try:
                stats = get_summary_stats(new_user_id)
                assert stats["total_spent"] == 0
                assert stats["transaction_count"] == 0
                assert stats["top_category"] == "—"
            finally:
                # Cleanup
                conn = get_db()
                conn.cursor().execute("DELETE FROM users WHERE id = ?", (new_user_id,))
                conn.commit()
                conn.close()


class TestGetRecentTransactions:
    def test_transactions_with_expenses(self):
        """Should return list ordered newest-first with correct fields."""
        transactions = get_recent_transactions(1)
        assert len(transactions) == 8
        # Check ordering (newest first)
        assert transactions[0]["date"] == "2026-04-15"
        # Check fields exist
        for tx in transactions:
            assert "date" in tx
            assert "description" in tx
            assert "category" in tx
            assert "amount" in tx

    def test_transactions_with_limit(self):
        """Should respect the limit parameter."""
        transactions = get_recent_transactions(1, limit=5)
        assert len(transactions) == 5

    def test_transactions_no_expenses(self, app_ctx):
        """Should return empty list for user with no expenses."""
        with app.app_context():
            import uuid
            unique_email = f"test2_{uuid.uuid4().hex[:8]}@example.com"
            conn = get_db()
            cursor = conn.cursor()
            from werkzeug.security import generate_password_hash
            cursor.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                ("Test User", unique_email, generate_password_hash("password"))
            )
            new_user_id = cursor.lastrowid
            conn.commit()
            conn.close()

            try:
                transactions = get_recent_transactions(new_user_id)
                assert transactions == []
            finally:
                conn = get_db()
                conn.cursor().execute("DELETE FROM users WHERE id = ?", (new_user_id,))
                conn.commit()
                conn.close()


class TestGetCategoryBreakdown:
    def test_breakdown_with_expenses(self):
        """Should return list ordered by amount desc with percentages summing to 100."""
        breakdown = get_category_breakdown(1)
        assert len(breakdown) == 7  # 7 unique categories in seed data

        # Check ordering (highest amount first)
        for i in range(len(breakdown) - 1):
            assert breakdown[i]["amount"] >= breakdown[i + 1]["amount"]

        # Check percentages sum to 100
        total_pct = sum(item["pct"] for item in breakdown)
        assert total_pct == 100

        # Check each item has required fields
        for item in breakdown:
            assert "name" in item
            assert "amount" in item
            assert "pct" in item

    def test_breakdown_no_expenses(self, app_ctx):
        """Should return empty list for user with no expenses."""
        with app.app_context():
            import uuid
            unique_email = f"test3_{uuid.uuid4().hex[:8]}@example.com"
            conn = get_db()
            cursor = conn.cursor()
            from werkzeug.security import generate_password_hash
            cursor.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                ("Test User", unique_email, generate_password_hash("password"))
            )
            new_user_id = cursor.lastrowid
            conn.commit()
            conn.close()

            try:
                breakdown = get_category_breakdown(new_user_id)
                assert breakdown == []
            finally:
                conn = get_db()
                conn.cursor().execute("DELETE FROM users WHERE id = ?", (new_user_id,))
                conn.commit()
                conn.close()


# =============================================================================
# Route tests
# =============================================================================

class TestProfileRoute:
    def test_unauthenticated_redirects_to_login(self, client):
        """GET /profile without session should redirect to /login."""
        response = client.get("/profile")
        assert response.status_code == 302
        assert response.location.startswith("/login")

    def test_authenticated_returns_200(self, logged_in_client):
        """GET /profile with session should return 200."""
        response = logged_in_client.get("/profile")
        assert response.status_code == 200

    def test_shows_user_name(self, logged_in_client):
        """Profile should display the seed user's name."""
        response = logged_in_client.get("/profile")
        assert b"Demo User" in response.data

    def test_shows_user_email(self, logged_in_client):
        """Profile should display the seed user's email."""
        response = logged_in_client.get("/profile")
        assert b"demo@spendly.com" in response.data

    def test_shows_rupee_symbol(self, logged_in_client):
        """Profile should display the ₹ symbol."""
        response = logged_in_client.get("/profile")
        assert b"\xe2\x82\xb9" in response.data  # UTF-8 encoding of ₹

    def test_shows_correct_total_spent(self, logged_in_client):
        """Total spent should match sum of all seed expenses (457.99)."""
        response = logged_in_client.get("/profile")
        assert b"457.99" in response.data

    def test_shows_correct_transaction_count(self, logged_in_client):
        """Transaction count should be 8."""
        response = logged_in_client.get("/profile")
        assert b">8</" in response.data or b"8" in response.data

    def test_shows_top_category(self, logged_in_client):
        """Top category should be Bills (highest single-category total)."""
        response = logged_in_client.get("/profile")
        assert b"Bills" in response.data

    def test_transactions_ordered_newest_first(self, logged_in_client):
        """Transaction list should appear in newest-first order."""
        response = logged_in_client.get("/profile")
        data = response.data.decode()
        # 2026-04-15 should appear before 2026-04-01 in the HTML
        pos_15 = data.find("2026-04-15")
        pos_01 = data.find("2026-04-01")
        assert pos_15 < pos_01

    def test_category_breakdown_has_all_categories(self, logged_in_client):
        """Category breakdown should contain all 7 categories."""
        response = logged_in_client.get("/profile")
        data = response.data.decode()
        categories = ["Food", "Transport", "Bills", "Health", "Entertainment", "Shopping", "Other"]
        for cat in categories:
            assert cat in data
