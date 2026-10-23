"""Unit tests for CPSC feed ingestion and normalization."""

import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import httpx

from feeds import (
    DEFAULT_TIMEOUT,
    get_feeds,
    ingest_cpsc_recalls,
    load_cpsc_snapshot,
    normalize_cpsc_recall,
)


class TestCPSCFeeds(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_cpsc_record = {
            "RecallID": 10548,
            "RecallNumber": "26174",
            "RecallDate": "2026-01-08T00:00:00",
            "Title": "Vasicar 18-Drawer Dressers Recalled",
            "Description": "This recall involves Vasicar 18-Drawer Dressers with wooden top.",
            "URL": "https://www.cpsc.gov/Recalls/2026/Vasicar",
            "Products": [
                {"Name": "Vasicar 18-Drawer Dressers", "Model": "VR-US18D"}
            ],
            "Retailers": [
                {"Name": "Online at Amazon.com from September 2025 through November 2025"}
            ],
            "Hazards": [
                {"Name": "The recalled dressers are unstable if they are not anchored to the wall."}
            ],
            "Remedies": [
                {"Name": "Consumers should immediately stop using the recalled dressers."}
            ],
        }

    def test_timeout_is_eight_seconds(self) -> None:
        """Verify the configured default timeout is 8.0 seconds."""
        self.assertEqual(DEFAULT_TIMEOUT, 8.0)

    def test_normalization_preserves_required_fields(self) -> None:
        """Verify normalization preserves recall ID, description, hazards, remedies, URL, and retailers."""
        normalized = normalize_cpsc_recall(self.sample_cpsc_record)

        self.assertEqual(normalized["source"], "CPSC")
        self.assertEqual(normalized["recall_id"], "10548")
        self.assertEqual(normalized["recall_number"], "26174")
        self.assertEqual(normalized["recall_date"], "2026-01-08T00:00:00")
        self.assertEqual(normalized["title"], "Vasicar 18-Drawer Dressers Recalled")
        self.assertIn("wooden top", normalized["description"])
        self.assertEqual(normalized["url"], "https://www.cpsc.gov/Recalls/2026/Vasicar")
        self.assertEqual(normalized["products"], ["Vasicar 18-Drawer Dressers"])
        self.assertEqual(
            normalized["hazards"],
            ["The recalled dressers are unstable if they are not anchored to the wall."],
        )
        self.assertEqual(
            normalized["remedies"],
            ["Consumers should immediately stop using the recalled dressers."],
        )
        self.assertEqual(
            normalized["retailers"],
            ["Online at Amazon.com from September 2025 through November 2025"],
        )
        self.assertTrue(normalized["sold_on_amazon"])

    def test_load_cpsc_snapshot_from_disk(self) -> None:
        """Verify that the local snapshot file can be loaded directly."""
        snapshot_records = load_cpsc_snapshot()
        self.assertIsInstance(snapshot_records, list)
        self.assertGreater(len(snapshot_records), 0)
        self.assertEqual(snapshot_records[0]["RecallID"], 10548)

    def test_successful_httpx_ingestion(self) -> None:
        """Verify successful API fetch returns source='cpsc_api' and stale=False."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [self.sample_cpsc_record]
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response

        result = ingest_cpsc_recalls(client=mock_client)

        self.assertEqual(result["source"], "cpsc_api")
        self.assertFalse(result["stale"])
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["recalls"][0]["recall_id"], "10548")
        self.assertEqual(result["recalls"][0]["products"], ["Vasicar 18-Drawer Dressers"])

    def test_simulated_network_timeout_fallback(self) -> None:
        """Verify network timeout triggers fallback to local snapshot with stale=True."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.ConnectTimeout("Connection timed out after 8.0s")

        result = ingest_cpsc_recalls(client=mock_client)

        self.assertEqual(result["source"], "local_snapshot")
        self.assertTrue(result["stale"])
        self.assertGreater(result["count"], 0)
        self.assertEqual(result["recalls"][0]["recall_id"], "10548")

    def test_simulated_request_error_fallback(self) -> None:
        """Verify HTTP 500 error triggers fallback to local snapshot with stale=True."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_client.get.return_value = mock_response

        result = ingest_cpsc_recalls(client=mock_client)

        self.assertEqual(result["source"], "local_snapshot")
        self.assertTrue(result["stale"])
        self.assertGreater(result["count"], 0)

    def test_simulated_invalid_json_fallback(self) -> None:
        """Verify malformed JSON payload triggers fallback to local snapshot with stale=True."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
        mock_client.get.return_value = mock_response

        result = ingest_cpsc_recalls(client=mock_client)

        self.assertEqual(result["source"], "local_snapshot")
        self.assertTrue(result["stale"])
        self.assertGreater(result["count"], 0)

    def test_get_feeds_wrapper(self) -> None:
        """Verify get_feeds wrapper delegates correctly to ingest_cpsc_recalls."""
        mock_client = MagicMock(spec=httpx.Client)
        mock_client.get.side_effect = httpx.ConnectError("Simulated offline")

        with patch("httpx.Client", return_value=mock_client):
            result = get_feeds()

        self.assertEqual(result["source"], "local_snapshot")
        self.assertTrue(result["stale"])


if __name__ == "__main__":
    unittest.main()
