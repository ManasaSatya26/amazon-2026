"""Unit tests for Recall Closer application tools."""

from pathlib import Path
import tempfile
import unittest

from core import CaseState, RecallCloserCore
from tools import (
    advance_case,
    add_item,
    check_recalls,
    feed_status,
    list_cases,
    replay_new_recall,
    run_tool,
)


class TestTools(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "test_tools.db"
        self.core = RecallCloserCore(self.db_path)

    def tearDown(self) -> None:
        self.core.close()
        self.tmp_dir.cleanup()

    def test_add_item_success(self) -> None:
        result = add_item(
            name="Smart Blender",
            brand="BlendCo",
            model="BC-100",
            upc="123456789012",
            core=self.core,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["item_id"], 1)
        self.assertEqual(result["name"], "Smart Blender")

        items = self.core.list_items()
        self.assertEqual(len(items), 1)

    def test_add_item_validation_error(self) -> None:
        result = add_item(name="", core=self.core)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "ValidationError")

    def test_check_recalls_with_persisted_record_and_matcher(self) -> None:
        # Add item
        item_id = self.core.add_item("Smart Blender", brand="BlendCo", model="BC-100")

        # Cache a matching recall
        self.core.cache_recall(
            recall_id="REC-001",
            title="BlendCo Smart Blender Recall",
            source="CPSC",
            description="Recalls model BC-100 due to fire hazard.",
        )

        # Run check_recalls tool
        result = check_recalls(item_id=item_id, core=self.core)
        self.assertTrue(result["success"])
        self.assertEqual(result["match_count"], 1)
        match = result["matches"][0]
        self.assertEqual(match["match_type"], "EXACT_MODEL")
        self.assertEqual(match["state"], CaseState.CONFIRMED_AFFECTED.value)

    def test_check_recalls_all_items(self) -> None:
        self.core.add_item("Air Fryer", brand="FryMax", model="AF-50")
        self.core.add_item("Toaster", brand="ToastKing", model="TK-2")
        self.core.cache_recall(
            recall_id="REC-002",
            title="FryMax Air Fryer Recall",
            source="CPSC",
            description="Recalls model AF-50.",
        )

        result = check_recalls(item_id=None, core=self.core)
        self.assertTrue(result["success"])
        self.assertEqual(result["items_checked"], 2)
        self.assertEqual(result["match_count"], 1)

    def test_check_recalls_invalid_item_id(self) -> None:
        result = check_recalls(item_id=-5, core=self.core)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "ValidationError")

    def test_list_cases(self) -> None:
        item_id = self.core.add_item("Hair Dryer", brand="BlowPro", model="HD-1")
        self.core.cache_recall("REC-003", "BlowPro Dryer Recall", "CPSC", "Hazard description.")
        self.core.create_case(item_id, "REC-003")

        result = list_cases(core=self.core)
        self.assertTrue(result["success"])
        self.assertEqual(result["count"], 1)

    def test_advance_case_evidence_requirement_and_audit(self) -> None:
        item_id = self.core.add_item("Space Heater", brand="WarmLife", model="WH-9")
        self.core.cache_recall("REC-004", "WarmLife Heater", "CPSC", "Fire risk.")
        case_id = self.core.create_case(item_id, "REC-004")

        # Attempting advance without label_evidence must fail
        fail_result = advance_case(
            case_id=case_id,
            new_state=CaseState.CONFIRMED_AFFECTED.value,
            label_evidence=None,
            core=self.core,
        )
        self.assertFalse(fail_result["success"])
        self.assertEqual(fail_result["error_type"], "ValidationError")

        # Advance with label_evidence succeeds
        success_result = advance_case(
            case_id=case_id,
            new_state=CaseState.CONFIRMED_AFFECTED.value,
            label_evidence="Inspected serial tag WH-9 verified on unit.",
            core=self.core,
        )
        self.assertTrue(success_result["success"])
        self.assertEqual(success_result["case"]["state"], CaseState.CONFIRMED_AFFECTED.value)

        # Verify audit log captures both rejected and successful transitions
        audits = self.core.list_audit_log(case_id=case_id)
        self.assertEqual(len(audits), 2)
        self.assertEqual(audits[0]["outcome"], "REJECTED")
        self.assertEqual(audits[1]["outcome"], "SUCCESS")

    def test_replay_new_recall_caches_and_matches_immediately(self) -> None:
        item_id = self.core.add_item("Kid Cot", brand="SleepSafe", model="SS-COT-1")

        result = replay_new_recall(
            recall_id="SIM-RECALL-99",
            title="SleepSafe Kid Cot Recall",
            description="Recalls SleepSafe Kid Cot model SS-COT-1 due to collapse hazard.",
            source="SIMULATED",
            check_items=True,
            core=self.core,
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["recall_id"], "SIM-RECALL-99")
        self.assertEqual(result["matched_case_count"], 1)

        # Check that recall is in cache
        cached = self.core.list_cached_recalls()
        self.assertTrue(any(r["recall_id"] == "SIM-RECALL-99" for r in cached))

    def test_feed_status(self) -> None:
        self.core.cache_recall("REC-005", "Test Recall", "CPSC", "Description")
        status = feed_status(core=self.core)
        self.assertTrue(status["success"])
        self.assertEqual(status["cached_recalls_count"], 1)
        self.assertIn("CPSC", status["sources"])

    def test_run_tool_dispatcher(self) -> None:
        res = run_tool("add_item", {"name": "Coffee Maker"}, core=self.core)
        self.assertTrue(res["success"])

        bad_res = run_tool("nonexistent_tool", {}, core=self.core)
        self.assertFalse(bad_res["success"])
        self.assertEqual(bad_res["error_type"], "UnknownToolError")


if __name__ == "__main__":
    unittest.main()
