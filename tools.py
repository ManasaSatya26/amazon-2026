"""Recall Closer application tools.

Provides six deterministic, evidence-preserving tools:
- add_item: Add a household product to inventory.
- check_recalls: Run deterministic recall matching on one or all items.
- list_cases: List tracked cases with optional item filter.
- advance_case: Move a case to a new state with audit logging and evidence requirements.
- replay_new_recall: Ingest a simulated or new recall notice and run matching.
- feed_status: Inspect local cached recall status and snapshot availability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core import (
    CaseNotFoundError,
    CaseState,
    InvalidTransitionError,
    ItemNotFoundError,
    RecallCloserCore,
    ValidationError,
    initialize,
)
from feeds import load_cpsc_snapshot


def _get_core(core: RecallCloserCore | None) -> tuple[RecallCloserCore, bool]:
    """Return an active core instance and a flag indicating if it should be closed."""
    if core is not None:
        return core, False
    return initialize(), True


def add_item(
    name: str,
    brand: str | None = None,
    model: str | None = None,
    upc: str | None = None,
    *,
    core: RecallCloserCore | None = None,
) -> dict[str, Any]:
    """Add a household product to inventory."""
    if not isinstance(name, str) or not name.strip():
        return {
            "success": False,
            "error": "name must be a non-empty string.",
            "error_type": "ValidationError",
        }

    engine, should_close = _get_core(core)
    try:
        item_id = engine.add_item(name=name, brand=brand, model=model, upc=upc)
        return {
            "success": True,
            "item_id": item_id,
            "name": name.strip(),
            "brand": brand.strip() if isinstance(brand, str) and brand.strip() else None,
            "model": model.strip() if isinstance(model, str) and model.strip() else None,
            "upc": upc.strip() if isinstance(upc, str) and upc.strip() else None,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }
    finally:
        if should_close:
            engine.close()


def check_recalls(
    item_id: int | None = None,
    *,
    core: RecallCloserCore | None = None,
) -> dict[str, Any]:
    """Run deterministic recall matching against persisted recalls."""
    if item_id is not None:
        if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id < 1:
            return {
                "success": False,
                "error": "item_id must be a positive integer.",
                "error_type": "ValidationError",
            }

    engine, should_close = _get_core(core)
    try:
        if item_id is not None:
            matches = engine.match_item(item_id)
            return {
                "success": True,
                "item_id": item_id,
                "matches": matches,
                "match_count": len(matches),
            }
        else:
            items = engine.list_items()
            all_matches: list[dict[str, Any]] = []
            for item in items:
                item_matches = engine.match_item(int(item["id"]))
                all_matches.extend(item_matches)
            return {
                "success": True,
                "all_matches": all_matches,
                "items_checked": len(items),
                "match_count": len(all_matches),
            }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }
    finally:
        if should_close:
            engine.close()


def list_cases(
    item_id: int | None = None,
    *,
    core: RecallCloserCore | None = None,
) -> dict[str, Any]:
    """List tracked recall cases with optional item filter."""
    if item_id is not None:
        if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id < 1:
            return {
                "success": False,
                "error": "item_id must be a positive integer.",
                "error_type": "ValidationError",
            }

    engine, should_close = _get_core(core)
    try:
        cases = engine.list_cases(item_id=item_id)
        return {
            "success": True,
            "cases": cases,
            "count": len(cases),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }
    finally:
        if should_close:
            engine.close()


def advance_case(
    case_id: int,
    new_state: str,
    label_evidence: str | None = None,
    remedy_note: str | None = None,
    *,
    core: RecallCloserCore | None = None,
) -> dict[str, Any]:
    """Move a case through an allowed state while enforcing evidence requirements and audit logging."""
    if isinstance(case_id, bool) or not isinstance(case_id, int) or case_id < 1:
        return {
            "success": False,
            "error": "case_id must be a positive integer.",
            "error_type": "ValidationError",
        }
    if not isinstance(new_state, str) or not new_state.strip():
        return {
            "success": False,
            "error": "new_state must be a non-empty string.",
            "error_type": "ValidationError",
        }

    engine, should_close = _get_core(core)
    try:
        updated_case = engine.transition_case(
            case_id=case_id,
            new_state=new_state.strip(),
            label_evidence=label_evidence,
            remedy_note=remedy_note,
        )
        return {
            "success": True,
            "case": updated_case,
        }
    except (CaseNotFoundError, InvalidTransitionError, ValidationError) as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }
    finally:
        if should_close:
            engine.close()


def replay_new_recall(
    recall_id: str,
    title: str,
    description: str | None = None,
    source: str = "SIMULATED",
    check_items: bool = True,
    *,
    core: RecallCloserCore | None = None,
) -> dict[str, Any]:
    """Ingest a new or simulated recall notice and run matching against inventory."""
    if not isinstance(recall_id, str) or not recall_id.strip():
        return {
            "success": False,
            "error": "recall_id must be a non-empty string.",
            "error_type": "ValidationError",
        }
    if not isinstance(title, str) or not title.strip():
        return {
            "success": False,
            "error": "title must be a non-empty string.",
            "error_type": "ValidationError",
        }

    engine, should_close = _get_core(core)
    try:
        engine.cache_recall(
            recall_id=recall_id.strip(),
            title=title.strip(),
            source=source.strip() if isinstance(source, str) and source.strip() else "SIMULATED",
            description=description.strip() if isinstance(description, str) and description.strip() else None,
        )

        matched_cases: list[dict[str, Any]] = []
        if check_items:
            for item in engine.list_items():
                matches = engine.match_item(int(item["id"]))
                for match in matches:
                    if str(match.get("recall_id")) == recall_id.strip():
                        matched_cases.append(match)

        return {
            "success": True,
            "recall_id": recall_id.strip(),
            "title": title.strip(),
            "source": source.strip() if isinstance(source, str) and source.strip() else "SIMULATED",
            "matched_cases": matched_cases,
            "matched_case_count": len(matched_cases),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }
    finally:
        if should_close:
            engine.close()


def feed_status(
    *,
    snapshot_path: Path | str | None = None,
    core: RecallCloserCore | None = None,
) -> dict[str, Any]:
    """Inspect local cached recall status and snapshot availability."""
    engine, should_close = _get_core(core)
    try:
        cached = engine.list_cached_recalls()
        sources = sorted(list({str(r["source"]) for r in cached}))
        latest_cached_at = max((str(r["cached_at"]) for r in cached), default=None)

        snapshot_records = load_cpsc_snapshot(snapshot_path)

        return {
            "success": True,
            "cached_recalls_count": len(cached),
            "sources": sources,
            "latest_cached_at": latest_cached_at,
            "snapshot_available": len(snapshot_records) > 0,
            "snapshot_records_count": len(snapshot_records),
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "error_type": exc.__class__.__name__,
        }
    finally:
        if should_close:
            engine.close()


TOOL_REGISTRY: dict[str, Any] = {
    "add_item": add_item,
    "check_recalls": check_recalls,
    "list_cases": list_cases,
    "advance_case": advance_case,
    "replay_new_recall": replay_new_recall,
    "feed_status": feed_status,
}

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "add_item",
        "description": "Add a household product to the inventory.",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Product name"},
                "brand": {"type": "string", "description": "Brand name"},
                "model": {"type": "string", "description": "Model identifier or number"},
                "upc": {"type": "string", "description": "UPC or barcode"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "check_recalls",
        "description": "Check inventory against persisted recalls using the deterministic matcher.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "integer",
                    "description": "Optional product ID to check; if omitted, checks all inventory items.",
                },
            },
        },
    },
    {
        "name": "list_cases",
        "description": "List tracked recall cases, optionally filtered by product ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "integer",
                    "description": "Optional product ID to filter cases.",
                },
            },
        },
    },
    {
        "name": "advance_case",
        "description": "Advance a recall case through permitted states (POSSIBLE_MATCH, CONFIRMED_AFFECTED, NOT_LISTED, REMEDY_STARTED, CLOSED).",
        "parameters": {
            "type": "object",
            "properties": {
                "case_id": {"type": "integer", "description": "The case ID to advance"},
                "new_state": {"type": "string", "description": "Target state"},
                "label_evidence": {
                    "type": "string",
                    "description": "Required when resolving POSSIBLE_MATCH to CONFIRMED_AFFECTED or NOT_LISTED",
                },
                "remedy_note": {
                    "type": "string",
                    "description": "Required when transitioning to REMEDY_STARTED or CLOSED",
                },
            },
            "required": ["case_id", "new_state"],
        },
    },
    {
        "name": "replay_new_recall",
        "description": "Ingest a new or simulated recall notice and immediately check inventory for matches.",
        "parameters": {
            "type": "object",
            "properties": {
                "recall_id": {"type": "string", "description": "Unique recall identifier"},
                "title": {"type": "string", "description": "Recall title"},
                "description": {"type": "string", "description": "Recall details and hazard text"},
                "source": {"type": "string", "description": "Source name, defaults to SIMULATED"},
                "check_items": {
                    "type": "boolean",
                    "description": "Whether to check inventory immediately, defaults to true",
                },
            },
            "required": ["recall_id", "title"],
        },
    },
    {
        "name": "feed_status",
        "description": "Check recall feed cache status, count of cached records, and local snapshot availability.",
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
]


def run_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    core: RecallCloserCore | None = None,
) -> dict[str, Any]:
    """Execute a named tool with input arguments and structured error handling."""
    if name not in TOOL_REGISTRY:
        return {
            "success": False,
            "error": f"Tool '{name}' is not recognized.",
            "error_type": "UnknownToolError",
        }

    func = TOOL_REGISTRY[name]
    args = arguments.copy() if arguments else {}
    return func(**args, core=core)
