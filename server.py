"""Minimal API entry point for the Recall Closer starter scaffold."""

from fastapi import FastAPI

app = FastAPI(
    title="Recall Closer",
    description="Starter API scaffold; application features are not implemented.",
    version="0.1.0",
)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a simple status message for the starter API."""
    return {
        "name": "Recall Closer",
        "status": "starter",
        "message": "Application functionality is not implemented yet.",
    }
