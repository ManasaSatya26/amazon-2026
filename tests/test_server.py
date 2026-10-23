"""Tests for FastAPI endpoints in server.py."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

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


if __name__ == "__main__":
    unittest.main()
