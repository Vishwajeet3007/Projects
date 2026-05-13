Code Review Report — 07-add-expense

Security Findings
🎓 What I checked
- SQL injection risks in new database queries
- Authentication checks on new expense routes
- Authorization for expense access controls
- Sensitive data exposure in new endpoints

✅ Doing well
- The add_expense route properly uses parameterized queries through the insert_expense helper function, preventing SQL injection
- New routes check `session.get('user_id')` for authentication before processing
- The insert_expense function follows the same secure pattern as other database helpers in the file
- New templates/expenses/add.html properly escapes variables with {{ }} syntax (no |safe filters observed), preventing XSS vulnerabilities
- Authentication is checked at the beginning of the add_expense route

Quality Findings
🎓 What I checked
Reviewed changes in app.py (add_expense route implementation), database/db.py (get_db update and new insert_expense function), and templates/base.html (nav link for Add Expense).

💡 Worth improving

**app.py:208 - Redundant imports inside function**
The add_expense function re-imports `session`, `request`, `flash`, `redirect`, `url_for`, `render_template` that are already imported at the top of the file (except `session`). This reduces readability and follows an inconsistent pattern (other routes import session locally when needed).
Move `session` to the top-level imports and remove the redundant import line inside the function. Keep imports at the module level for consistency.

**app.py:206-260 - Long function with repetitive validation**
The add_expense function is 55 lines long and handles validation, database insertion, and response generation in a single block. While functional, extracting validation logic to a helper function would improve readability and make the route easier to follow.
Consider creating a validate_expense_form() function that returns validated data or error messages, keeping the route focused on request handling and flow control.

**database/db.py:13-17 - Application context check in get_db**
The current approach of checking `current_app is not None` works well for flexibility, but consider setting the database path via `app.config['DATABASE'] = DATABASE` in app.py's initialization. This would make the get_db function simpler while maintaining configurability.

**app.py:221 - Hardcoded category list**
The allowed categories list is defined inline. Moving this to a module-level constant (e.g., ALLOWED_CATEGORIES = [...]) would make it easier to reuse and modify.

✅ Doing well

**database/db.py:155-171 - insert_expense function**
Well-named, clear docstring, uses parameterized queries for safety, properly manages database connection with try/finally, and returns the new ID. This follows the pattern of other database helpers in the file.

**templates/base.html:25 - Nav link with active state**
The Add Expense nav link correctly uses `url_for()` and includes an active state check matching the pattern used for other navigation links (like Analytics). This maintains UI consistency.

**app.py:206-260 - Form handling flow**
The add_expense route properly separates GET and POST handling, validates all required fields, provides specific error messages, flashes success on insertion, and redirects after POST to prevent form resubmission. The validation covers data types, ranges, and business logic (future dates).

Combined Action Plan
Ordered checklist of everything that needs to be fixed,
prioritized by severity:

[Critical/High security findings first]
- No critical/high security findings in the changed code

[Quality CHANGES REQUESTED items second]
1. Remove redundant imports inside add_expense function and move session import to top-level (app.py:208)
2. Extract validation logic from add_expense function into a helper function for better readability (app.py:206-260)
3. Simplify get_db function by setting database path in app.py initialization (database/db.py:13-17)
4. Move hardcoded category list to module-level constant (app.py:221)

[Medium/Low security findings third]
- No medium/low security findings identified

[Quality APPROVED WITH SUGGESTIONS items last]
- All security implementations are correct (parameterized queries, authentication checks)
- Template properly escapes variables to prevent XSS
- Navigation link follows existing patterns

Overall Verdict
✅ APPROVED WITH SUGGESTIONS — can commit, address suggestions in future steps