"""Tests for FastAPI endpoints in server.py."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient

from core import CaseState, RecallCloserCore
from server import app


class TestServerRoutes(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_api.db"
        self.core = RecallCloserCore(self.db_path)
        app.state.core = self.core
        app.state.genai_client = None
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.core.close()
        self.tmp_dir.cleanup()

    def test_read_root(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "Recall Closer")
        self.assertEqual(data["status"], "ready")

    def test_api_status(self) -> None:
        self.core.add_item("Electric Kettle", brand="BoilFast", model="BF-1")
        self.core.cache_recall("CPSC-KETTLE", "Kettle recall", "CPSC", "Hazard description.")

        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["inventory_count"], 1)
        self.assertIn("feed_status", data)
        self.assertEqual(data["feed_status"]["cached_recalls_count"], 1)

    def test_api_cases_and_filter(self) -> None:
        item_id1 = self.core.add_item("Item 1")
        item_id2 = self.core.add_item("Item 2")
        self.core.cache_recall("REC-A", "Recall A", "CPSC", "Desc")
        self.core.cache_recall("REC-B", "Recall B", "CPSC", "Desc")
        self.core.create_case(item_id1, "REC-A")
        self.core.create_case(item_id2, "REC-B")

        # Get all cases
        resp_all = self.client.get("/api/cases")
        self.assertEqual(resp_all.status_code, 200)
        self.assertEqual(resp_all.json()["count"], 2)

        # Filter by item_id
        resp_filtered = self.client.get(f"/api/cases?item_id={item_id1}")
        self.assertEqual(resp_filtered.status_code, 200)
        self.assertEqual(resp_filtered.json()["count"], 1)
        self.assertEqual(resp_filtered.json()["cases"][0]["item_id"], item_id1)

    def test_api_audit_and_filter(self) -> None:
        item_id = self.core.add_item("Fan", brand="CoolBreeze", model="CB-2")
        self.core.cache_recall("REC-FAN", "Fan Recall", "CPSC", "Desc")
        case_id = self.core.create_case(item_id, "REC-FAN")

        # Successful transition
        self.core.transition_case(
            case_id,
            CaseState.CONFIRMED_AFFECTED,
            label_evidence="Model label matches recall notice.",
        )

        resp = self.client.get(f"/api/audit?case_id={case_id}")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["audit_log"][0]["outcome"], "SUCCESS")

    def test_mcp_tools_list_and_call(self) -> None:
        # 1. tools/list
        list_resp = self.client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual(list_resp.status_code, 200)
        tools = list_resp.json()["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        self.assertIn("add_item", tool_names)
        self.assertIn("check_recalls", tool_names)
        self.assertIn("advance_case", tool_names)

        # 2. tools/call add_item
        call_resp = self.client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "add_item",
                    "arguments": {"name": "Smart Thermostat", "brand": "ThermoTech", "model": "TT-1"},
                },
            },
        )
        self.assertEqual(call_resp.status_code, 200)
        call_result = call_resp.json()["result"]
        self.assertFalse(call_result["isError"])
        content_text = call_result["content"][0]["text"]
        parsed = json.loads(content_text)
        self.assertTrue(parsed["success"])
        self.assertEqual(parsed["name"], "Smart Thermostat")

    def test_chat_without_api_key_graceful_fallback(self) -> None:
        response = self.client.post("/chat", json={"message": "What is the status of the feed?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("response", data)
        self.assertIn("recalls cached", data["response"])

    def test_chat_with_mocked_gemini_client(self) -> None:
        # Mock Gemini client that selects check_recalls tool
        mock_client = MagicMock()
        mock_model = MagicMock()

        # Turn 1: Gemini decides to call check_recalls
        mock_function_call = MagicMock()
        mock_function_call.name = "check_recalls"
        mock_function_call.args = {}

        turn1_response = MagicMock()
        turn1_response.function_calls = [mock_function_call]
        turn1_response.text = None

        # Turn 2: Gemini narrates the result
        turn2_response = MagicMock()
        turn2_response.function_calls = []
        turn2_response.text = "I checked your inventory and found no active recall matches."

        mock_model.generate_content.side_effect = [turn1_response, turn2_response]
        mock_client.models = mock_model

        app.state.genai_client = mock_client

        response = self.client.post("/chat", json={"message": "Can you check my recalls?"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["response"], "I checked your inventory and found no active recall matches.")
        self.assertEqual(len(data["tools_called"]), 1)
        self.assertEqual(data["tools_called"][0]["tool"], "check_recalls")

    def test_api_keys_never_exposed_in_error_responses(self) -> None:
        fake_api_key = "AIzaSyD-FakeTestSecretKey1234567890"

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            f"Authentication failed with key {fake_api_key}."
        )
        app.state.genai_client = mock_client

        response = self.client.post("/chat", json={"message": "Hello"})
        content_str = response.text
        self.assertNotIn(fake_api_key, content_str)
        self.assertIn("[REDACTED_API_KEY]", content_str)

    def test_post_and_get_items(self) -> None:
        create_resp = self.client.post(
            "/api/items",
            json={"name": "Stand Mixer", "brand": "MixPro", "model": "MP-500", "upc": "999888"},
        )
        self.assertEqual(create_resp.status_code, 200)
        created = create_resp.json()
        self.assertTrue(created["success"])
        item_id = created["item_id"]
        self.assertEqual(created["name"], "Stand Mixer")

        list_resp = self.client.get("/api/items")
        self.assertEqual(list_resp.status_code, 200)
        items_data = list_resp.json()
        self.assertTrue(items_data["success"])
        self.assertEqual(items_data["count"], 1)
        self.assertEqual(items_data["items"][0]["id"], item_id)

    def test_check_recalls_endpoint(self) -> None:
        item_id = self.core.add_item("Electric Toaster", brand="ToastMaster", model="TM-2")
        self.core.cache_recall(
            "REC-TOAST",
            "ToastMaster Electric Toaster Recall",
            "CPSC",
            "Recalls model TM-2 due to burn risk.",
        )

        resp = self.client.post("/api/check-recalls", json={"item_id": item_id})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["match_count"], 1)
        match = data["matches"][0]
        self.assertEqual(match["match_type"], "EXACT_MODEL")
        self.assertEqual(match["state"], CaseState.CONFIRMED_AFFECTED.value)

    def test_advance_case_valid_and_invalid(self) -> None:
        item_id = self.core.add_item("Coffee Maker", brand="BrewCo", model="BC-1")
        self.core.cache_recall("REC-BREW", "BrewCo Recall", "CPSC", "Hazard description.")
        case_id = self.core.create_case(item_id, "REC-BREW")

        # Invalid transition: missing required label_evidence from POSSIBLE_MATCH
        bad_resp = self.client.post(
            f"/api/cases/{case_id}/advance",
            json={"new_state": CaseState.CONFIRMED_AFFECTED.value},
        )
        self.assertEqual(bad_resp.status_code, 400)
        self.assertIn("Label evidence is required", bad_resp.json()["detail"])

        # Check audit log captures rejected attempt
        audits = self.core.list_audit_log(case_id=case_id)
        self.assertTrue(any(a["outcome"] == "REJECTED" for a in audits))

        # Valid transition with label_evidence
        good_resp = self.client.post(
            f"/api/cases/{case_id}/advance",
            json={
                "new_state": CaseState.CONFIRMED_AFFECTED.value,
                "label_evidence": "Confirmed label on unit matches model BC-1 in recall notice.",
            },
        )
        self.assertEqual(good_resp.status_code, 200)
        good_data = good_resp.json()
        self.assertTrue(good_data["success"])
        self.assertEqual(good_data["case"]["state"], CaseState.CONFIRMED_AFFECTED.value)

    def test_sync_feeds_endpoint(self) -> None:
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.ConnectError("Simulated offline for test")
        with patch("httpx.Client", return_value=mock_client):
            resp = self.client.post("/api/feeds/sync")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertIn("sync", data)
        self.assertIn("feed_status", data)
        self.assertIn("feed_source", data["feed_status"])

    def test_cors_and_static_files(self) -> None:
        # Test CORS headers on API
        resp = self.client.get("/api/status", headers={"Origin": "http://localhost:3000"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("access-control-allow-origin"), "http://localhost:3000")

        # Test StaticFiles mount
        static_resp = self.client.get("/static/index.html")
        self.assertEqual(static_resp.status_code, 200)
        self.assertIn("Recall Closer", static_resp.text)

    def test_chat_fallback_handles_check_recalls(self) -> None:
        self.core.add_item("Pressure Cooker", brand="CookFast", model="CF-10")
        resp = self.client.post("/chat", json={"message": "Please check my recalls"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("Deterministic recall check executed", data["response"])
        self.assertEqual(len(data["tools_called"]), 1)
        self.assertEqual(data["tools_called"][0]["tool"], "check_recalls")


if __name__ == "__main__":
    unittest.main()
