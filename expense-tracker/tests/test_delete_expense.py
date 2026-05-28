import pytest
from app import app
from database.db import init_db, create_user, get_db, insert_expense


@pytest.fixture
def client():
    app.config['TESTING'] = True
    # Create a temporary file for the database
    import tempfile
    import os
    db_fd, db_path = tempfile.mkstemp()
    app.config['DATABASE'] = db_path
    with app.test_client() as client:
        with app.app_context():
            init_db()
            # Create a test user
            user_id = create_user('Test User', 'test@example.com', 'testpass')
            client.user_id = user_id
    yield client
    # Close and remove the temporary file
    import os
    os.close(db_fd)
    os.unlink(db_path)


def login(client, email='test@example.com', password='testpass'):
    """Helper function to log in a test user."""
    return client.post('/login', data=dict(
        email=email,
        password=password
    ), follow_redirects=True)


def test_get_delete_expense_returns_405(client):
    """GET requests to delete endpoint should return 405 Method Not Allowed."""
    # Login first
    login(client)

    # Create a test expense to delete
    with app.app_context():
        user_id = create_user('Test User 2', 'test2@example.com', 'testpass')
        expense_id = insert_expense(user_id, 10.0, 'Food', '2026-01-01', 'Test expense')

    # Try to access delete endpoint via GET
    response = client.get(f'/expenses/{expense_id}/delete')
    assert response.status_code == 405  # Method Not Allowed


def test_post_delete_expense_redirects_when_not_logged_in(client):
    """Unauthenticated POST requests to delete endpoint redirect to login."""
    # Create a test expense
    with app.app_context():
        user_id = create_user('Test User 2', 'test2@example.com', 'testpass')
        expense_id = insert_expense(user_id, 10.0, 'Food', '2026-01-01', 'Test expense')

    # Try to delete without logging in
    response = client.post(f'/expenses/{expense_id}/delete', follow_redirects=False)
    assert response.status_code == 302  # Redirect
    assert '/login' in response.location


def test_post_delete_expense_success(client):
    """Authenticated users can delete their own expenses."""
    # Login
    login(client)

    # Create a test expense
    with app.app_context():
        # Get the user ID from the login above
        from database.db import get_user_by_email
        user = get_user_by_email('test@example.com')
        user_id = user['id']
        expense_id = insert_expense(user_id, 25.50, 'Transport', '2026-01-05', 'Uber ride')

    # Verify expense exists before deletion
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT id FROM expenses WHERE id = ?", (expense_id,))
        assert cursor.fetchone() is not None
        db.close()

    # Delete the expense
    response = client.post(f'/expenses/{expense_id}/delete', follow_redirects=True)
    assert response.status_code == 200
    assert b'Expense deleted successfully!' in response.data

    # Verify expense is deleted
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT id FROM expenses WHERE id = ?", (expense_id,))
        assert cursor.fetchone() is None
        db.close()


def test_post_delete_expense_unauthenticated_other_users_expense(client):
    """Users cannot delete expenses belonging to other users."""
    # Login as user 1
    login(client)

    # Create expense for user 2
    with app.app_context():
        user_id_2 = create_user('Test User 2', 'test2@example.com', 'testpass')
        expense_id = insert_expense(user_id_2, 15.0, 'Food', '2026-01-03', 'Other user expense')

    # Try to delete user 2's expense while logged in as user 1
    response = client.post(f'/expenses/{expense_id}/delete')
    assert response.status_code == 404  # Not found (for security)

    # Verify expense still exists
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute("SELECT id FROM expenses WHERE id = ?", (expense_id,))
        assert cursor.fetchone() is not None
        db.close()


def test_post_delete_expense_nonexistent_id(client):
    """Deleting a non-existent expense returns 404."""
    # Login
    login(client)

    # Try to delete non-existent expense
    response = client.post('/expenses/99999/delete')
    assert response.status_code == 404  # Not found