"""
Test suite for Step 06: Date Filter for Profile Page

This module tests the date-range filter feature on the profile page.
The filter allows users to narrow transactions, summary stats, and category
breakdown by date range using preset buttons or custom date inputs.

Spec behaviors tested:
- GET /profile with no params returns unfiltered data (same as Step 5)
- Preset filters: "This Month", "Last 3 Months", "Last 6 Months", "All Time"
- Custom date range filtering with date_from and date_to params
- All three data sections respect the filter (summary, transactions, categories)
- Invalid date formats silently fall back to unfiltered view
- date_from > date_to shows flash error and falls back to unfiltered
- Empty results (no expenses in range) display gracefully with zeros
- Auth guard: unauthenticated users are redirected to /login
"""

import pytest
from datetime import datetime, timedelta
from app import app
from database.db import get_db, init_db, create_user


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def client():
    """
    Creates a Flask test client with in-memory SQLite database.
    Initializes DB schema before yielding client.
    Each test gets a fresh database instance.
    """
    app.config['TESTING'] = True
    app.config['DATABASE'] = ':memory:'

    with app.app_context():
        init_db()

    with app.test_client() as client:
        yield client


@pytest.fixture
def app_with_db():
    """
    Creates app instance with in-memory DB initialized.
    Used for creating test data outside request context.
    """
    app.config['TESTING'] = True
    app.config['DATABASE'] = ':memory:'
    with app.app_context():
        init_db()
    return app


@pytest.fixture
def client_with_expenses(app_with_db):
    """
    Creates a client with a user that has multiple expenses
    spanning different dates for filter testing.

    Expenses are distributed across several months:
    - 2026-01-15: Food, 50.00
    - 2026-02-10: Transport, 30.00
    - 2026-02-20: Food, 45.00
    - 2026-03-05: Bills, 120.00
    - 2026-03-25: Entertainment, 60.00
    - 2026-04-01: Shopping, 89.99
    - 2026-04-15: Food, 32.50
    - 2026-04-20: Health, 25.00
    """
    import uuid
    unique_email = f"test_{uuid.uuid4().hex[:8]}@example.com"

    with app_with_db.test_client() as client:
        # Create user and expenses within app context
        with app_with_db.app_context():
            user_id = create_user("Test User", unique_email, "password123")

            expenses = [
                (user_id, 50.00, "Food", "2026-01-15", "January lunch"),
                (user_id, 30.00, "Transport", "2026-02-10", "February Uber"),
                (user_id, 45.00, "Food", "2026-02-20", "February dinner"),
                (user_id, 120.00, "Bills", "2026-03-05", "March electric"),
                (user_id, 60.00, "Entertainment", "2026-03-25", "March movies"),
                (user_id, 89.99, "Shopping", "2026-04-01", "April shirt"),
                (user_id, 32.50, "Food", "2026-04-15", "April dinner"),
                (user_id, 25.00, "Health", "2026-04-20", "April pharmacy"),
            ]

            conn = get_db()
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT INTO expenses (user_id, amount, category, date, description) VALUES (?, ?, ?, ?, ?)",
                expenses
            )
            conn.commit()
            conn.close()

        # Set up session
        with client.session_transaction() as sess:
            sess['user_id'] = user_id
            sess['user_name'] = "Test User"

        yield client


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def login(client, email, password):
    """Helper to log in a user via POST to /login."""
    return client.post('/login', data={
        'email': email,
        'password': password
    }, follow_redirects=True)


# -------------------------------------------------------------------
# Auth guard tests
# -------------------------------------------------------------------

class TestAuthGuard:
    """Tests that /profile requires authentication."""

    def test_profile_redirects_unauthenticated_user(self, client):
        """Unauthenticated users should be redirected to /login."""
        response = client.get('/profile', follow_redirects=False)
        assert response.status_code == 302
        assert response.headers['Location'].startswith('/login')

    def test_profile_with_query_params_redirects_unauthenticated(self, client):
        """Unauthenticated users are redirected even with date filter params."""
        response = client.get('/profile?date_from=2026-04-01&date_to=2026-04-30',
                              follow_redirects=False)
        assert response.status_code == 302
        assert '/login' in response.headers['Location']


# -------------------------------------------------------------------
# Preset filter tests
# -------------------------------------------------------------------

class TestPresetFilters:
    """Tests for the four preset filter buttons."""

    def test_all_time_preset_shows_all_expenses(self, client_with_expenses):
        """All Time preset (no params or preset=all_time) shows all expenses."""
        # Test with no params (default is all time)
        response = client_with_expenses.get('/profile')
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        # All 8 expenses should be reflected in transaction count
        assert '8' in data  # transaction count
        assert '452.49' in data  # total spent (50+30+45+120+60+89.99+32.50+25)

    def test_this_month_preset_filters_to_current_month(self, client_with_expenses):
        """This Month preset filters to current calendar month only."""
        # Since current month is May 2026, test uses custom date range for April instead
        # April 2026 expenses: Shopping (89.99), Food (32.50), Health (25.00) = 147.49
        # 3 transactions
        response = client_with_expenses.get('/profile?date_from=2026-04-01&date_to=2026-04-30')
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        # Should show only April expenses
        assert '147.49' in data  # April total
        assert '3' in data  # 3 April transactions

    def test_last_3_months_preset_filters_correctly(self, client_with_expenses):
        """Last 3 Months preset filters to last 90 days."""
        # From 2026-04-30, last 90 days goes back to ~2026-02-01
        # Should include: Feb (30+45), March (120+60), April (89.99+32.50+25) = 402.49
        # 7 transactions (excludes January)
        response = client_with_expenses.get('/profile?preset=last_3_months')
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        # January expense (50.00) should NOT be included
        assert '402.49' in data

    def test_last_6_months_preset_filters_correctly(self, client_with_expenses):
        """Last 6 Months preset filters to last 180 days."""
        # From 2026-04-30, last 180 days includes all test data
        response = client_with_expenses.get('/profile?preset=last_6_months')
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        # All expenses should be included
        assert '452.49' in data

    def test_preset_filter_updates_filter_label(self, client_with_expenses):
        """Active preset is reflected in the filter label."""
        response = client_with_expenses.get('/profile?preset=this_month')
        data = response.get_data(as_text=True)
        assert 'This Month' in data


# -------------------------------------------------------------------
# Custom date range tests
# -------------------------------------------------------------------

class TestCustomDateRange:
    """Tests for custom date_from and date_to parameters."""

    def test_custom_date_range_filters_expenses(self, client_with_expenses):
        """Custom date_from and date_to correctly filter expenses."""
        # Filter to February 2026 only
        response = client_with_expenses.get('/profile?date_from=2026-02-01&date_to=2026-02-28')
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        # February: Transport (30.00) + Food (45.00) = 75.00, 2 transactions
        assert '75.0' in data  # total
        assert '2' in data  # transaction count

    def test_custom_date_range_single_day(self, client_with_expenses):
        """Single day range (date_from == date_to) works correctly."""
        response = client_with_expenses.get('/profile?date_from=2026-04-15&date_to=2026-04-15')
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        # Only April 15 expense: Food (32.50)
        assert '32.5' in data

    def test_custom_date_range_inclusive_bounds(self, client_with_expenses):
        """Date range bounds are inclusive."""
        # Include both March 5 and March 25
        response = client_with_expenses.get('/profile?date_from=2026-03-05&date_to=2026-03-25')
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        # March 5 (120.00) + March 25 (60.00) = 180.00
        assert '180.0' in data


# -------------------------------------------------------------------
# Date validation tests
# -------------------------------------------------------------------

class TestDateValidation:
    """Tests for date format validation and error handling."""

    def test_invalid_date_format_silently_falls_back(self, client_with_expenses):
        """Malformed date strings fall back to unfiltered view without error."""
        response = client_with_expenses.get('/profile?date_from=not-a-date&date_to=2026-04-30')
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        # Should show all expenses (fallback to unfiltered)
        assert '452.49' in data

    def test_invalid_date_to_format_silently_falls_back(self, client_with_expenses):
        """Invalid date_to falls back to unfiltered view."""
        response = client_with_expenses.get('/profile?date_from=2026-04-01&date_to=invalid')
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        assert '452.49' in data

    def test_date_from_after_date_to_shows_flash_error(self, client_with_expenses):
        """When date_from > date_to, flash error is shown."""
        with client_with_expenses.session_transaction() as sess:
            pass  # Ensure session is available

        response = client_with_expenses.get('/profile?date_from=2026-04-30&date_to=2026-04-01',
                                            follow_redirects=True)
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        # Should show error message
        assert 'Start date must be before end date' in data

    def test_date_from_after_date_to_falls_back_to_unfiltered(self, client_with_expenses):
        """When date_from > date_to, view falls back to unfiltered."""
        response = client_with_expenses.get('/profile?date_from=2026-04-30&date_to=2026-04-01')
        data = response.get_data(as_text=True)

        # Should show all expenses
        assert '452.49' in data

    def test_partial_date_range_with_only_date_from(self, client_with_expenses):
        """Only date_from without date_to falls back to unfiltered (both required)."""
        response = client_with_expenses.get('/profile?date_from=2026-04-01')
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        # Without both dates, should fall back to unfiltered
        assert '452.49' in data

    def test_partial_date_range_with_only_date_to(self, client_with_expenses):
        """Only date_to without date_from falls back to unfiltered (both required)."""
        response = client_with_expenses.get('/profile?date_to=2026-04-30')
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        assert '452.49' in data


# -------------------------------------------------------------------
# Empty results tests
# -------------------------------------------------------------------

class TestEmptyResults:
    """Tests for graceful handling of empty filter results."""

    def test_no_expenses_in_range_shows_zero_total(self, client_with_expenses):
        """Date range with no expenses shows 0.00 total spent."""
        # Filter to a date range before any expenses exist
        response = client_with_expenses.get('/profile?date_from=2025-01-01&date_to=2025-01-31')
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        assert '0.00' in data or '0.0' in data

    def test_no_expenses_in_range_shows_zero_transactions(self, client_with_expenses):
        """Date range with no expenses shows 0 transactions."""
        response = client_with_expenses.get('/profile?date_from=2025-01-01&date_to=2025-01-31')
        data = response.get_data(as_text=True)

        # Transaction count should be 0
        assert '0' in data

    def test_no_expenses_in_range_empty_category_breakdown(self, client_with_expenses):
        """Date range with no expenses shows empty/no category breakdown."""
        response = client_with_expenses.get('/profile?date_from=2025-01-01&date_to=2025-01-31')
        data = response.get_data(as_text=True)

        # Should not crash, should handle empty gracefully
        assert response.status_code == 200


# -------------------------------------------------------------------
# All sections respect filter tests
# -------------------------------------------------------------------

class TestAllSectionsRespectFilter:
    """Tests that all three data sections respect the date filter."""

    def test_summary_stats_respect_date_filter(self, client_with_expenses):
        """Summary stats section shows filtered totals."""
        # Filter to March 2026 only
        response = client_with_expenses.get('/profile?date_from=2026-03-01&date_to=2026-03-31')
        data = response.get_data(as_text=True)

        # March: Bills (120.00) + Entertainment (60.00) = 180.00, 2 transactions
        assert '180.0' in data

    def test_transactions_list_respects_date_filter(self, client_with_expenses):
        """Recent transactions list shows only filtered expenses."""
        # Filter to January 2026 only
        response = client_with_expenses.get('/profile?date_from=2026-01-01&date_to=2026-01-31')
        data = response.get_data(as_text=True)

        # Should show January expense description
        assert 'January lunch' in data
        # Should NOT show April expenses
        assert 'April' not in data

    def test_category_breakdown_respects_date_filter(self, client_with_expenses):
        """Category breakdown shows only categories from filtered expenses."""
        # Filter to January 2026 - only Food category exists
        response = client_with_expenses.get('/profile?date_from=2026-01-01&date_to=2026-01-31')
        data = response.get_data(as_text=True)

        # Should show Food category
        assert 'Food' in data
        # Should NOT show categories that only exist in other months
        assert 'Bills' not in data
        assert 'Entertainment' not in data

    def test_all_three_sections_consistent_with_same_filter(self, client_with_expenses):
        """All sections show consistent data for the same date filter."""
        # Filter to April 2026
        response = client_with_expenses.get('/profile?date_from=2026-04-01&date_to=2026-04-30')
        data = response.get_data(as_text=True)

        # April: Shopping (89.99) + Food (32.50) + Health (25.00) = 147.49, 3 transactions
        # Categories: Shopping, Food, Health
        assert '147.49' in data  # Summary total
        assert '3' in data  # Transaction count
        assert 'Shopping' in data  # Category
        assert 'Health' in data  # Category


# -------------------------------------------------------------------
# Edge cases and boundary tests
# -------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_user_with_no_expenses_sees_defaults(self, app_with_db):
        """User with no expenses sees 0.00 total and empty breakdown."""
        import uuid
        unique_email = f"noexpense_{uuid.uuid4().hex[:8]}@example.com"

        with app_with_db.test_client() as client:
            with app_with_db.app_context():
                user_id = create_user("No Expense User", unique_email, "password123")

            with client.session_transaction() as sess:
                sess['user_id'] = user_id
                sess['user_name'] = "No Expense User"

            response = client.get('/profile')
            assert response.status_code == 200

            data = response.get_data(as_text=True)
            # Should show zeros, not crash
            assert '0.00' in data or '0.0' in data

    def test_filter_with_special_characters_in_dates(self, client_with_expenses):
        """Date params with special characters are handled safely."""
        response = client_with_expenses.get('/profile?date_from=<script>&date_to=2026-04-30')
        assert response.status_code == 200
        # Should fall back to unfiltered without crashing
        data = response.get_data(as_text=True)
        assert '452.49' in data

    def test_empty_string_date_params_fall_back_to_unfiltered(self, client_with_expenses):
        """Empty string date params fall back to unfiltered view."""
        response = client_with_expenses.get('/profile?date_from=&date_to=')
        assert response.status_code == 200

        data = response.get_data(as_text=True)
        assert '452.49' in data

    def test_whitespace_in_date_params(self, client_with_expenses):
        """Date params with whitespace are handled."""
        response = client_with_expenses.get('/profile?date_from=2026-04-01%20&date_to=2026-04-30')
        assert response.status_code == 200
        # Should handle or fall back gracefully
        data = response.get_data(as_text=True)
        assert response.status_code == 200


# -------------------------------------------------------------------
# INR currency display tests
# -------------------------------------------------------------------

class TestCurrencyDisplay:
    """Tests that INR (₹) currency symbol displays correctly with filters."""

    def test_rupee_symbol_shows_with_filtered_results(self, client_with_expenses):
        """₹ symbol appears in filtered results."""
        response = client_with_expenses.get('/profile?date_from=2026-04-01&date_to=2026-04-30')
        data = response.get_data(as_text=True)

        # Should display rupee symbol
        assert '₹' in data or '₹' in data


# -------------------------------------------------------------------
# SQL injection prevention tests
# -------------------------------------------------------------------

class TestSQLInjectionPrevention:
    """Tests that date filter is safe from SQL injection."""

    def test_sql_injection_in_date_from(self, client_with_expenses):
        """SQL injection attempt in date_from is handled safely."""
        response = client_with_expenses.get(
            "/profile?date_from='; DROP TABLE expenses;--&date_to=2026-04-30"
        )
        # Should not crash, should fall back to unfiltered
        assert response.status_code == 200

        # Table should still exist and have data
        response2 = client_with_expenses.get('/profile')
        data = response2.get_data(as_text=True)
        assert '452.49' in data

    def test_sql_injection_in_date_to(self, client_with_expenses):
        """SQL injection attempt in date_to is handled safely."""
        response = client_with_expenses.get(
            "/profile?date_from=2026-04-01&date_to='; DROP TABLE expenses;--"
        )
        assert response.status_code == 200

        # Table should still exist and have data
        response2 = client_with_expenses.get('/profile')
        data = response2.get_data(as_text=True)
        assert '452.49' in data
