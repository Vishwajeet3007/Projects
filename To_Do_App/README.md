# To-Do App

A clean and simple to-do list app built with plain HTML, CSS, and JavaScript.

This project helps you keep track of tasks, add optional due dates, search tasks instantly, mark tasks as completed, and delete tasks you no longer need. It also saves your tasks and theme preference in the browser using `localStorage`, so your data stays available after refreshing the page.

## Features

- Add new tasks with an optional due date
- View task status as `Pending` or `Completed`
- Mark tasks as complete using a checkbox
- Delete tasks from the list
- Search tasks in real time
- See live statistics for total, completed, and pending tasks
- Toggle between light mode and dark mode
- Keep tasks and theme preference saved with `localStorage`
- Responsive layout that works on desktop and mobile screens

## Built With

- HTML5
- CSS3
- Vanilla JavaScript
- Browser `localStorage`

## Project Structure

```text
To_Do_App/
|-- index.html
|-- README.md
```

This is a single-file app, which means the structure, styling, and logic are all contained inside `index.html`.

## How to Run

No installation or build step is required.

1. Download or clone this repository.
2. Open the project folder.
3. Double-click `index.html` or open it in any modern web browser.

You can also run it with a local development server if you prefer, but it is not required.

## How to Use

### Add a Task

1. Enter your task in the text field.
2. Optionally choose a due date.
3. Click `Add Task`.

### Complete a Task

- Click the checkbox next to a task to mark it as completed.
- Completed tasks are shown with a strikethrough style and updated status.

### Search Tasks

- Use the search box to filter tasks in real time.
- The list updates as you type.

### Delete a Task

- Click the `Delete` button next to a task to remove it from the list.

### Change Theme

- Click `Toggle Dark Mode` to switch between light and dark themes.

## Data Storage

This app uses the browser's `localStorage` to save:

- Tasks under the key `todo-app-tasks`
- Theme preference under the key `todo-app-theme`

Because of this:

- Your tasks remain available after refreshing the page
- Your selected theme is remembered
- Data is stored locally in your browser, not in a database

## App Behavior

- New tasks are added to the top of the list
- Empty tasks are not allowed
- Due dates are optional
- If there are no tasks, an empty-state message is shown
- If no tasks match the search text, a different empty-state message is shown
- Statistics update automatically whenever tasks are added, completed, or deleted

## Why This Project Is Useful

This project is a good example for beginners who want to learn how to build an interactive web app without frameworks. It demonstrates:

- DOM selection and event handling
- Form submission handling
- Dynamic rendering of lists
- Conditional UI updates
- Local data persistence with `localStorage`
- Responsive UI design with CSS

## Future Improvements

- Edit existing tasks
- Add task categories or priorities
- Filter by completed or pending status
- Sort by due date
- Add notifications or reminders
- Store data with a backend for cross-device sync

## License

This project is open for personal and educational use.
