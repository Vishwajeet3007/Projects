# Spec: Date Filter for Profile Page

## Overview
Step 6 adds a date range filter to the profile page, allowing users to filter their expense transactions by a custom date range. The filter will include preset options (Last 7 days, Last 30 days, This month, Last month) as well as custom date range pickers. This feature enhances the profile page by letting users analyze their spending patterns over specific time periods without needing to manually scan through all transactions.

## Depends on
- Step 1: Database setup (tables and `get_db()` exist)
- Step 2: Registration (users are stored in the database)
- Step 3: Login / Logout (`session["user_id"]` is set on login)
- Step 4: Profile page static UI (template already renders all four sections)
- Step 5: Backend connection (profile page displays real data from database)

## Routes
No new routes. Modify existing `GET /profile` to accept optional query parameters:
- `start_date` — ISO format date string (YYYY-MM-DD)
- `end_date` — ISO format date string (YYYY-MM-DD)
- `preset` — preset filter name: `7d`, `30d`, `this_month`, `last_month`

## Database changes
No database changes. The `expenses` table already has the `date` column which will be used for filtering.

## Templates
- **Modify**: `templates/profile.html`
  - Add a filter bar above the "Recent Transactions" section
  - Include preset filter buttons (7 days, 30 days, This month, Last month)
  - Add custom date range inputs (start date, end date)
  - Add an "Apply" button and "Clear" button
  - Display the current filter range as a label (e.g., "Showing: Last 7 days" or "Showing: Apr 1, 2026 - Apr 30, 2026")
  - Update transaction list and category breakdown to reflect filtered data
  - Update summary stats to reflect filtered totals

## Files to change
- `app.py` — modify `profile()` route to parse date filter query parameters and pass to query functions
- `templates/profile.html` — add date filter UI components
- `database/queries.py` — add optional `start_date` and `end_date` parameters to existing query functions

## Files to create
No new files.

## New dependencies
No new dependencies.

## Rules for implementation
- No SQLAlchemy or ORMs — raw `sqlite3` only via `get_db()`
- Parameterised queries only — never string-format values into SQL
- Foreign keys PRAGMA must be enabled on every connection (already done in `get_db()`)
- Use CSS variables — never hardcode hex values
- All templates extend `base.html`
- No inline styles
- Currency must always display as ₹ — never £ or $
- Date filter UI must be responsive and mobile-friendly
- Preset filters must calculate date ranges dynamically based on current date
- Custom date range must validate that start_date <= end_date
- If invalid dates are provided, fall back to showing all data (no filter)
- Query functions must handle optional date parameters gracefully — if None, return all data
- Filter state should persist in URL query params so page refresh maintains the filter
- When no transactions match the filter, display a friendly "No transactions found" message

## Definition of done
- [ ] Profile page displays a date filter bar above the transactions table
- [ ] Four preset filter buttons are visible: "Last 7 days", "Last 30 days", "This month", "Last month"
- [ ] Two date input fields for custom range selection (start and end date)
- [ ] "Apply" button filters transactions based on selected range
- [ ] "Clear" button resets to show all transactions
- [ ] Current filter range is displayed as a text label
- [ ] Clicking preset buttons updates the URL with `?preset=7d`, `?preset=30d`, etc.
- [ ] Custom date range updates URL with `?start_date=YYYY-MM-DD&end_date=YYYY-MM-DD`
- [ ] Summary stats (total spent, transaction count, top category) update to reflect filtered data
- [ ] Transaction list shows only expenses within the selected date range
- [ ] Category breakdown recalculates percentages based on filtered transactions
- [ ] When no transactions match the filter, a "No transactions found" message is shown
- [ ] Invalid date ranges fall back to showing all data without error
- [ ] Page refresh maintains the current filter state (URL params persist)
- [ ] Filter UI is styled consistently with the rest of the profile page
