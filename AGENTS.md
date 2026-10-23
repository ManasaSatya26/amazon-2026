# Project instructions

## Project status

Recall Closer is an early-stage project scaffold. The Python modules and static
page are starter placeholders, not a complete application.

## Working in this repository

- Keep changes focused on the requested feature or fix; preserve unrelated work.
- Read the relevant existing files before changing them and follow their style.
- Keep modules small and use clear names and type hints where useful.
- Do not add real credentials, tokens, or other secrets to source control.
- Keep `.env` and local database files out of version control. Use `.env.example`
  only for descriptive placeholder values.
- Do not implement external feed integrations, tools, or full application
  behavior without an explicit feature request.
- Add or update tests when implementing application behavior.

## Running locally

- Install dependencies with `python -m pip install -r requirements.txt`.
- Start the API with `python -m uvicorn server:app --reload`.
- Run a syntax check with
  `python -m compileall core.py feeds.py tools.py server.py`.

## Before handing off changes

- Run the relevant tests and checks, and report any that were not run.
- Do not commit or push unless explicitly requested.
