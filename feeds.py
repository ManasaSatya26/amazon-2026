"""Recall Closer feed integrations for CPSC recall data."""

import json
from pathlib import Path
from typing import Any

import httpx

CPSC_API_URL = "https://www.saferproducts.gov/RestWebServices/Recall"
DEFAULT_TIMEOUT = 8.0
DEFAULT_USER_AGENT = "RecallCloser/0.1.0"
DEFAULT_SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "cpsc_snapshot.json"


def normalize_cpsc_recall(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize actual CPSC source fields preserving all essential attributes."""
    recall_id = item.get("RecallID")
    recall_number = item.get("RecallNumber")

    products = [
        p.get("Name", "")
        for p in item.get("Products", [])
        if isinstance(p, dict) and p.get("Name")
    ]
    retailers = [
        r.get("Name", "")
        for r in item.get("Retailers", [])
        if isinstance(r, dict) and r.get("Name")
    ]
    hazards = [
        h.get("Name", "")
        for h in item.get("Hazards", [])
        if isinstance(h, dict) and h.get("Name")
    ]
    remedies = [
        rem.get("Name", "")
        for rem in item.get("Remedies", [])
        if isinstance(rem, dict) and rem.get("Name")
    ]

    description = item.get("Description") or ""
    url = item.get("URL") or ""
    title = item.get("Title") or ""
    recall_date = item.get("RecallDate") or ""

    # Check if sold on Amazon
    sold_on_amazon = any("amazon" in r.lower() for r in retailers) or "amazon" in description.lower()

    return {
        "source": "CPSC",
        "recall_id": str(recall_id) if recall_id is not None else "",
        "recall_number": str(recall_number) if recall_number is not None else "",
        "recall_date": recall_date,
        "title": title,
        "description": description,
        "url": url,
        "products": products,
        "retailers": retailers,
        "hazards": hazards,
        "remedies": remedies,
        "sold_on_amazon": sold_on_amazon,
    }


def load_cpsc_snapshot(
    snapshot_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Load records from the verified local CPSC snapshot."""
    path = Path(snapshot_path) if snapshot_path else DEFAULT_SNAPSHOT_PATH
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        return []


def ingest_cpsc_recalls(
    recall_date_start: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    snapshot_path: Path | str | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Fetch CPSC endpoint using httpx with timeout; fall back to local snapshot on failure."""
    params: dict[str, str] = {"format": "json"}
    if recall_date_start:
        params["RecallDateStart"] = recall_date_start

    raw_items: list[dict[str, Any]] = []
    source = "cpsc_api"
    stale = False

    try:
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/json",
        }
        if client is not None:
            response = client.get(
                CPSC_API_URL, params=params, headers=headers, timeout=timeout
            )
        else:
            with httpx.Client(timeout=timeout) as http_client:
                response = http_client.get(
                    CPSC_API_URL, params=params, headers=headers
                )

        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            raw_items = data
        else:
            raise ValueError("Unexpected CPSC response payload format; expected list.")
    except Exception:
        # Network failure, timeout, bad status code, or JSON parse error
        raw_items = load_cpsc_snapshot(snapshot_path=snapshot_path)
        source = "local_snapshot"
        stale = True

    normalized_records = [normalize_cpsc_recall(item) for item in raw_items]
    return {
        "source": source,
        "stale": stale,
        "count": len(normalized_records),
        "recalls": normalized_records,
    }


def get_feeds(
    recall_date_start: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    snapshot_path: Path | str | None = None,
) -> dict[str, Any]:
    """Retrieve normalized CPSC recall feeds."""
    return ingest_cpsc_recalls(
        recall_date_start=recall_date_start,
        timeout=timeout,
        snapshot_path=snapshot_path,
    )
