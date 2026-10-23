# Recall Closer

An initial project scaffold for the Build, Ship, Shape: Amazon Developer Hackathon 2026.

## Project overview

Recall Closer is a work in progress. The problem statement, solution details, and
features will be documented here as the project takes shape.

## Technology stack

- Python
- FastAPI
- Uvicorn

## Getting started

1. Create and activate a virtual environment:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and replace the placeholder values as needed.
   Do not commit `.env` or put real credentials in `.env.example`.

4. Start the starter API:

   ```powershell
   python -m uvicorn server:app --reload
   ```

   Visit `http://127.0.0.1:8000/` to see the scaffold response. The application
   features and integrations have not been implemented yet.

## Project structure

- `core.py` — starter location for core application logic
- `feeds.py` — starter location for feed integrations
- `tools.py` — starter location for application tools
- `server.py` — minimal FastAPI entry point
- `static/index.html` — static placeholder page
- `tests/` — location for tests

## Development

Run the scaffold checks with:

```powershell
python -m compileall core.py feeds.py tools.py server.py
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

## Demo video

To be added.

## Team

To be added.

## Acknowledgements

Built for the Build, Ship, Shape: Amazon Developer Hackathon 2026.
