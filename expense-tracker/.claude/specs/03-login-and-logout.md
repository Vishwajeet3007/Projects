# Spec: Login and Logout

## Overview

Implement user authentication so registered users can sign in and sign out of Spendly. This step upgrades the existing stub `GET /login` route to handle POST submissions, validates credentials against the `users` table, and establishes session-based authentication. The logout route clears the session. This feature unlocks all authenticated functionality (profile, expense tracking) that follows in subsequent steps.

## Depends on

- Step 01 — Database setup (`users` table, `get_db()`)
- Step 02 — Registration (user creation, password hashing)

## Routes

- `GET /login` — render login form — public (already exists as stub, upgrade it)
- `POST /login` — validate credentials, create session, redirect to `/profile` — public
- `GET /logout` — clear session, redirect to `/login` — logged-in users

## Database changes

No new tables or columns. The existing `users` table covers all requirements.

A new DB helper must be added to `database/db.py`:
- `get_user_by_email(email)` — fetches a user row by email for authentication

## Templates

- **Modify**: `templates/login.html`
  - Change form `action` to `url_for('login')` with `method="post"`
  - Add flash message display for errors (e.g., "Invalid email or password")
  - Add flash message display for success messages (e.g., from registration redirect)
  - Keep all existing visual design

- **Modify**: `templates/base.html`
  - Add session-aware navigation:
    - When logged in: show user's name and "Logout" link
    - When logged out: show "Sign in" and "Get started" links
  - Use Flask `session` to check authentication state

## Files to change

- `app.py` — upgrade `login()` to handle POST, add `logout()` implementation, set `app.secret_key`
- `database/db.py` — add `get_user_by_email()` helper
- `templates/login.html` — wire up form action/method and flash message display
- `templates/base.html` — add session-aware navigation

## Files to create

None.

## New dependencies

No new dependencies. Uses `werkzeug.security` (already installed) and Flask's built-in `session`.

## Rules for implementation

- No SQLAlchemy or ORMs
- Parameterised queries only — never use f-strings in SQL
- Passwords verified with `werkzeug.security.check_password_hash` — never compare plaintext
- `app.secret_key` must be set in `app.py` for `session` to work (use hardcoded dev string for now)
- Server-side validation must check:
  1. Email and password are non-empty
  2. Email exists in database
  3. Password matches stored hash via `check_password_hash`
- On validation failure, re-render the form with a flashed error message — do not redirect
- On success, store `user_id` in `session`, flash a welcome message, and redirect to `url_for('profile')`
- Logout clears the session and redirects to `/login` with a flashed message
- Use `session.get('user_id')` to check if user is logged in
- All templates extend `base.html`
- Use CSS variables — never hardcode hex values
- Use `url_for()` for every internal link — never hardcode URLs

## Definition of done

- [ ] `GET /login` renders the login form without errors
- [ ] Submitting with valid credentials creates a session and redirects to `/profile`
- [ ] Submitting with invalid email shows "Invalid email or password" error
- [ ] Submitting with correct email but wrong password shows "Invalid email or password" error
- [ ] Submitting with empty fields shows validation error
- [ ] `GET /logout` clears the session and redirects to `/login`
- [ ] Navigation in `base.html` shows user name + logout when logged in
- [ ] Navigation shows "Sign in" + "Get started" when logged out
- [ ] Session stores only `user_id` — no password or sensitive data
