"""Recall Closer FastAPI Server.

Exposes endpoints for web clients, MCP protocol, and chat interactions:
- GET / : starter health check / status
- GET /api/status : application, database, and feed status
- GET /api/cases : recall cases with optional item_id filter
- GET /api/audit : state transition audit log with optional case_id filter
- POST /mcp : Model Context Protocol JSON-RPC endpoint
- POST /chat : LLM chat endpoint where Gemini selects tools and narrates results
"""

from __future__ import annotations

import json
import os
import re
from contextlib import asynccontextmanager
from typing import Any

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core import DEFAULT_DATABASE_PATH, RecallCloserCore, initialize
from tools import TOOL_DEFINITIONS, feed_status, run_tool


def _redact_secrets(text: str) -> str:
    """Ensure API keys and secrets are never exposed in responses or logs."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    redacted = text
    if api_key and len(api_key) > 3:
        redacted = redacted.replace(api_key, "[REDACTED_API_KEY]")
    # Scrub common Gemini key formats (starts with AIza followed by alphanumeric / - / _)
    redacted = re.sub(r"AIza[0-9A-Za-z\-_]{10,}", "[REDACTED_API_KEY]", redacted)
    return redacted


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and shut down the SQLite core database connection."""
    db_path = os.environ.get("RECALL_CLOSER_DB", str(DEFAULT_DATABASE_PATH))
    app.state.core = initialize(db_path)
    try:
        cached = app.state.core.list_cached_recalls()
        if not cached:
            from feeds import sync_cpsc_recalls
            sync_cpsc_recalls(core=app.state.core)
    except Exception:
        pass
    yield
    if hasattr(app.state, "core") and app.state.core is not None:
        app.state.core.close()


app = FastAPI(
    title="Recall Closer",
    description="Deterministic household product recall tracking with MCP and LLM narration.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(__file__).resolve().parent / "static"
if static_path.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


def get_core(request: Request) -> RecallCloserCore:
    """Retrieve the shared RecallCloserCore database instance from app state."""
    if hasattr(request.app.state, "core") and request.app.state.core is not None:
        return request.app.state.core
    return initialize()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt or inquiry")


class ItemCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Product name")
    brand: str | None = Field(None, description="Brand name")
    model: str | None = Field(None, description="Model identifier")
    upc: str | None = Field(None, description="UPC or serial number")


class CheckRecallsRequest(BaseModel):
    item_id: int | None = Field(None, description="Optional product ID to check")


class AdvanceCaseRequest(BaseModel):
    new_state: str = Field(..., min_length=1, description="Target case state")
    label_evidence: str | None = Field(None, description="Label evidence")
    remedy_note: str | None = Field(None, description="Remedy note")


class McpRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Any = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return status check for the API."""
    return {
        "name": "Recall Closer",
        "status": "ready",
        "version": "0.1.0",
    }


@app.get("/api/status")
def get_status(core: RecallCloserCore = Depends(get_core)) -> dict[str, Any]:
    """Return status of the database, inventory, and recall feeds."""
    feeds = feed_status(core=core)
    items = core.list_items()
    cases = core.list_cases()
    audit_entries = core.list_audit_log()

    return {
        "status": "healthy",
        "inventory_count": len(items),
        "active_cases_count": len(cases),
        "audit_records_count": len(audit_entries),
        "feed_status": feeds,
    }


@app.get("/api/cases")
def get_cases(
    item_id: int | None = Query(None, description="Optional product ID to filter cases"),
    core: RecallCloserCore = Depends(get_core),
) -> dict[str, Any]:
    """Retrieve tracked recall cases."""
    result = run_tool("list_cases", {"item_id": item_id}, core=core)
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


@app.get("/api/audit")
def get_audit(
    case_id: int | None = Query(None, description="Optional case ID to filter audit entries"),
    core: RecallCloserCore = Depends(get_core),
) -> dict[str, Any]:
    """Retrieve case transition audit log."""
    try:
        entries = core.list_audit_log(case_id=case_id)
        return {
            "success": True,
            "count": len(entries),
            "audit_log": entries,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=_redact_secrets(str(exc)))


@app.get("/api/items")
def get_items(core: RecallCloserCore = Depends(get_core)) -> dict[str, Any]:
    """Retrieve household product inventory."""
    items = core.list_items()
    return {"success": True, "count": len(items), "items": items}


@app.post("/api/items")
def create_item(
    item_req: ItemCreateRequest,
    core: RecallCloserCore = Depends(get_core),
) -> dict[str, Any]:
    """Add a household product to inventory."""
    result = run_tool("add_item", item_req.model_dump(), core=core)
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


@app.post("/api/check-recalls")
def check_recalls_endpoint(
    req: CheckRecallsRequest | None = None,
    core: RecallCloserCore = Depends(get_core),
) -> dict[str, Any]:
    """Check inventory against cached recalls using deterministic matcher."""
    item_id = req.item_id if req is not None else None
    result = run_tool("check_recalls", {"item_id": item_id}, core=core)
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


@app.post("/api/cases/{case_id}/advance")
def advance_case_endpoint(
    case_id: int,
    req: AdvanceCaseRequest,
    core: RecallCloserCore = Depends(get_core),
) -> dict[str, Any]:
    """Advance case state with evidence validation and audit logging."""
    args = {
        "case_id": case_id,
        "new_state": req.new_state,
        "label_evidence": req.label_evidence,
        "remedy_note": req.remedy_note,
    }
    result = run_tool("advance_case", args, core=core)
    if not result.get("success", False):
        raise HTTPException(status_code=400, detail=result.get("error"))
    return result


@app.post("/api/feeds/sync")
def sync_feeds_endpoint(
    core: RecallCloserCore = Depends(get_core),
) -> dict[str, Any]:
    """Fetch live CPSC recalls or fall back to verified local snapshot."""
    from feeds import sync_cpsc_recalls

    sync_result = sync_cpsc_recalls(core=core)
    status_res = run_tool("feed_status", core=core)
    return {
        "success": True,
        "sync": sync_result,
        "feed_status": status_res,
    }


@app.post("/mcp")
def handle_mcp(
    payload: dict[str, Any],
    core: RecallCloserCore = Depends(get_core),
) -> dict[str, Any]:
    """Model Context Protocol (MCP) JSON-RPC 2.0 endpoint."""
    method = payload.get("method", "")
    req_id = payload.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "recall-closer", "version": "0.1.0"},
            },
        }

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOL_DEFINITIONS},
        }

    if method == "tools/call":
        params = payload.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        result = run_tool(tool_name, arguments, core=core)
        is_error = not result.get("success", False)

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2),
                    }
                ],
                "isError": is_error,
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {
            "code": -32601,
            "message": f"Method '{method}' not found.",
        },
    }


@app.post("/chat")
def handle_chat(
    chat_req: ChatRequest,
    core: RecallCloserCore = Depends(get_core),
) -> dict[str, Any]:
    """Handle chat messages where Gemini selects tools and narrates results.

    Gemini is only responsible for tool selection and user narration.
    Recall matching decisions and case-state transitions are executed
    strictly through deterministic core tools.
    """
    user_message = chat_req.message.strip()
    api_key = os.environ.get("GEMINI_API_KEY", "")

    # Check if a custom or mocked client has been injected on app.state
    custom_client = getattr(app.state, "genai_client", None)

    if not api_key and custom_client is None:
        # Graceful fallback when no Gemini API key is configured
        # Handles basic routing directly to tools without requiring network/credentials
        tools_called = []
        lower_msg = user_message.lower()
        if "status" in lower_msg or "feed" in lower_msg:
            tool_res = run_tool("feed_status", core=core)
            tools_called.append({"tool": "feed_status", "result": tool_res})
            feed_src = tool_res.get("feed_source", "local_snapshot")
            stale_str = "stale snapshot fallback" if tool_res.get("stale") else "live fresh data"
            reply = f"Recall feed status: {tool_res.get('cached_recalls_count', 0)} recalls cached from {feed_src} ({stale_str})."
        elif "cases" in lower_msg or "case" in lower_msg:
            tool_res = run_tool("list_cases", core=core)
            tools_called.append({"tool": "list_cases", "result": tool_res})
            reply = f"Found {tool_res.get('count', 0)} tracked recall case(s) in the database."
        elif "check" in lower_msg or "recall" in lower_msg:
            tool_res = run_tool("check_recalls", core=core)
            tools_called.append({"tool": "check_recalls", "result": tool_res})
            reply = f"Deterministic recall check executed against {tool_res.get('items_checked', 0)} item(s). Found {tool_res.get('match_count', 0)} match(es)."
        elif "item" in lower_msg or "inventory" in lower_msg or "product" in lower_msg:
            items = core.list_items()
            reply = f"Inventory contains {len(items)} registered product(s)."
        else:
            reply = (
                "Recall Closer Assistant: Gemini API key is not configured. "
                "Available tools: add_item, check_recalls, list_cases, advance_case, replay_new_recall, feed_status."
            )

        return {
            "response": reply,
            "tools_called": tools_called,
        }

    try:
        from google import genai
        from google.genai import types

        client = custom_client or genai.Client(api_key=api_key)

        system_instruction = (
            "You are Recall Closer Assistant. You help users manage household inventory, "
            "check for product recalls, and track remedies. You MUST use the provided tools "
            "for any data operations. You are NEVER allowed to invent recall matches or change "
            "case states without using the tools. Keep responses clear and factual."
        )

        # Call Gemini with function calling declarations
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=[{"function_declarations": TOOL_DEFINITIONS}],
            ),
        )

        tools_called = []
        function_calls = response.function_calls if hasattr(response, "function_calls") else []

        if function_calls:
            tool_results = []
            for call in function_calls:
                call_name = getattr(call, "name", "")
                call_args = getattr(call, "args", {}) or {}
                tool_output = run_tool(call_name, call_args, core=core)
                tools_called.append({"tool": call_name, "args": call_args, "result": tool_output})
                tool_results.append(tool_output)

            # Second turn to narrate the tool results
            narration_prompt = (
                f"User request: {user_message}\n"
                f"Tool outputs: {json.dumps(tool_results)}\n"
                "Please narrate these deterministic findings clearly to the user."
            )
            narration = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=narration_prompt,
            )
            final_text = getattr(narration, "text", "") or json.dumps(tool_results)
        else:
            final_text = getattr(response, "text", "") or "No response generated."

        return {
            "response": _redact_secrets(final_text),
            "tools_called": tools_called,
        }

    except Exception as exc:
        sanitized_error = _redact_secrets(str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "error": "Failed to process chat message with Gemini.",
                "detail": sanitized_error,
            },
        )
