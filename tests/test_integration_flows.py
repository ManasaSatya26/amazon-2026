"""End-to-end integration and workflow regression tests."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import httpx

from core import CaseState, RecallCloserCore, initialize
from feeds import sync_cpsc_recalls
from server import app


class TestIntegrationFlows(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_flow.db"
        self.core = initialize(self.db_path)
        app.state.core = self.core
        app.state.genai_client = None
        self.client = TestClient(app)

        # Seed with local CPSC snapshot (offline fallback mode)
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.ConnectError("Offline test mode")
        with patch("httpx.Client", return_value=mock_client):
            sync_cpsc_recalls(core=self.core)

    def tearDown(self) -> None:
        self.core.close()
        self.tmp_dir.cleanup()

    def test_flow_1_register_product_and_confirm_persistence(self) -> None:
        """Flow 1: Register a product and confirm it persists in SQLite."""
        resp = self.client.post(
            "/api/items",
            json={
                "name": "18-Drawer Dresser",
                "brand": "Vasicar",
                "model": "VR-US18D",
                "upc": "VAS-001",
            },
        )
        self.assertEqual(resp.status_code, 200)
        item_id = resp.json()["item_id"]

        items = self.core.list_items()
        item = next((i for i in items if i["id"] == item_id), None)
        self.assertIsNotNone(item)
        self.assertEqual(item["name"], "18-Drawer Dresser")
        self.assertEqual(item["brand"], "Vasicar")
        self.assertEqual(item["model"], "VR-US18D")

    def test_flow_2_and_3_check_recalls_and_retrieve_cases(self) -> None:
        """Flow 2 & 3: Check recalls, retrieve cases and audit log with genuine evidence."""
        item_id = self.core.add_item("Adult Portable Bed Rails", brand="YOLAAH", model="BR-01")

        check_resp = self.client.post("/api/check-recalls", json={"item_id": item_id})
        self.assertEqual(check_resp.status_code, 200)
        check_data = check_resp.json()
        self.assertTrue(check_data["success"])
        self.assertGreaterEqual(check_data["match_count"], 1)

        match = check_data["matches"][0]
        self.assertIn("BR-01", match["evidence"])

        # Flow 3: retrieve cases from backend
        cases_resp = self.client.get(f"/api/cases?item_id={item_id}")
        self.assertEqual(cases_resp.status_code, 200)
        self.assertEqual(cases_resp.json()["count"], 1)

        # Retrieve audit entries
        audit_resp = self.client.get("/api/audit")
        self.assertEqual(audit_resp.status_code, 200)

    def test_flow_4_and_5_valid_and_invalid_transitions(self) -> None:
        """Flow 4 & 5: Attempt invalid transition (displays rejection reason) and valid transition (persists)."""
        item_id = self.core.add_item("Heater", brand="BrandX", model="BX-1")
        self.core.cache_recall("REC-X", "BrandX Heater Recall", "CPSC", "Hazardous heating element.")
        case_id = self.core.create_case(item_id, "REC-X")

        # Invalid transition: missing evidence
        inv_resp = self.client.post(
            f"/api/cases/{case_id}/advance",
            json={"new_state": "CONFIRMED_AFFECTED"},
        )
        self.assertEqual(inv_resp.status_code, 400)
        self.assertIn("Label evidence is required", inv_resp.json()["detail"])

        # Check audit log recorded rejection
        audits = self.core.list_audit_log(case_id=case_id)
        self.assertTrue(any(a["outcome"] == "REJECTED" for a in audits))

        # Valid transition
        val_resp = self.client.post(
            f"/api/cases/{case_id}/advance",
            json={
                "new_state": "CONFIRMED_AFFECTED",
                "label_evidence": "Inspected label matches model BX-1.",
            },
        )
        self.assertEqual(val_resp.status_code, 200)
        self.assertEqual(val_resp.json()["case"]["state"], "CONFIRMED_AFFECTED")

        # Verify SQLite persistence
        db_case = self.core.get_case(case_id)
        self.assertEqual(db_case["state"], "CONFIRMED_AFFECTED")

    def test_flow_6_feed_status_distinguishes_snapshot_fallback(self) -> None:
        """Flow 6: Check feed status and distinguish live data from snapshot fallback."""
        status_resp = self.client.get("/api/status")
        self.assertEqual(status_resp.status_code, 200)
        data = status_resp.json()
        fs = data["feed_status"]
        self.assertEqual(fs["feed_source"], "local_snapshot")
        self.assertTrue(fs["stale"])
        self.assertFalse(fs["is_live"])
        self.assertGreater(fs["cached_recalls_count"], 0)

    def test_flow_7_chat_request_tool_execution(self) -> None:
        """Flow 7: Send chat request and verify tool execution and response rendering."""
        chat_resp = self.client.post("/chat", json={"message": "What is the status of the feeds?"})
        self.assertEqual(chat_resp.status_code, 200)
        chat_data = chat_resp.json()
        self.assertIn("response", chat_data)
        self.assertEqual(len(chat_data["tools_called"]), 1)
        self.assertEqual(chat_data["tools_called"][0]["tool"], "feed_status")


if __name__ == "__main__":
    unittest.main()
