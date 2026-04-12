# Spec: Registration

## Overview

Implement user registration functionality allowing new users to create an account in the Spendly expense tracker. This feature provides a registration form where users can enter their name, email, and password. Upon successful submission, the system creates a new user record in the database and redirects to the login page. This is the first step in the authentication flow, enabling personalized expense tracking.

## Depends on

- Step 1 (Database Setup) - The users table must exist with the correct schema

## Routes

- `POST /register` - Handle registration form submission - public

## Database Changes

No database changes - the users table from Step 1 already contains all required fields (name, email, password_hash)

## Templates

- **Create:** 
  - `templates/register.html` - Registration form with name, email, password fields
- **Modify:** 
  - None

## Files to Change

- `app.py` - Add POST route handler for `/register`

## Files to Create

- `templates/register.html` - Registration form template

## New Dependencies

No new dependencies

## Rules for Implementation

- No SQLAlchemy or ORMs
- Parameterised queries only - never string formatting in SQL
- Passwords hashed with werkzeug's `generate_password_hash`
- Use CSS variables — never hardcode hex values
- All templates extend `base.html`
- Email must be validated for uniqueness before insertion
- Form must validate required fields (name, email, password)
- Display appropriate error messages for:
  - Email already registered
  - Missing required fields
  - Password too short (minimum 6 characters)

## Definition of Done

- [ ] Registration form renders at `/register` with name, email, password fields
- [ ] Form validates that all required fields are provided
- [ ] Form validates that email is not already registered
- [ ] Form validates that password is at least 6 characters
- [ ] Successful registration creates user in database with hashed password
- [ ] Successful registration redirects to `/login` page
- [ ] Error messages display clearly to the user
- [ ] Template extends `base.html` and uses CSS variables
- [ ] SQL uses parameterized queries only
- [ ] No duplicate emails can be inserted
