# Contributing to Football Player Heatmap & Tactical Analytics Platform

First off, thank you for considering contributing to this project!

## Code of Conduct
This project expects all contributors to maintain a respectful, welcoming, and collaborative environment.

## How Can I Contribute?

### Reporting Bugs
- Check the issue tracker to see if the bug has already been reported.
- If not, open a new issue using the **Bug Report** template.
- Include a clear description, steps to reproduce, and any relevant error logs or screenshots.

### Suggesting Enhancements
- Open a new issue using the **Feature Request** template.
- Describe the feature, why it is useful, and how it might be implemented.

### Pull Requests
1. Fork the repository and create your branch from `main`.
2. Ensure you run and test your changes locally.
3. Keep your commits clean and follow standard semantic commit message guidelines (e.g., `feat: add ball possession heuristic`, `fix: correct Kalman filter indexing error`).
4. Open a Pull Request detailing the changes, referencing any relevant issues, and filling out the Pull Request Template.

## Technical Architecture Overview
- **Backend**: FastAPI + SQLAlchemy (PostgreSQL) + OpenCV + YOLO (Ultralytics) + ByteTrack tracking.
- **Frontend**: React + TypeScript + Vite + Tailwind CSS.
- **Docker**: Containerized deployment for DB, backend, and frontend.

### Style Guide
- **Python**: Follow PEP 8 style guidelines.
- **TypeScript/React**: Follow modern functional component styles with hooks, and format with Prettier.
