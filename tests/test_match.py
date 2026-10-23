"""Tests for deterministic matching against cached recall records."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from core import CaseState, RecallCloserCore


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[RecallCloserCore]:
    """Provide an engine backed by a temporary SQLite database."""
    core = RecallCloserCore(tmp_path / "recall_match.sqlite3")
    try:
        yield core
    finally:
        core.close()


def test_exact_model_match_confirms_case_with_quoted_recall_evidence(
    engine: RecallCloserCore,
) -> None:
    """Normalized exact model matches include a quote from cached recall text."""
    item_id = engine.add_item(
        "ACME Air-Fryer AF 100",
        brand="acme",
        model="af 100",
    )
    engine.cache_recall(
        recall_id="R-EXACT",
        title="ACME Air Fryer Recall",
        source="Example consumer agency",
        description="Models: AF-100 and AF-200. The heating element may fail.",
    )

    matches = engine.match_item(item_id)

    assert len(matches) == 1
    assert matches[0]["match_type"] == "EXACT_MODEL"
    assert matches[0]["state"] == CaseState.CONFIRMED_AFFECTED.value
    assert matches[0]["recall_id"] == "R-EXACT"
    assert matches[0]["evidence"] == '"Models: AF-100 and AF-200."'
    assert matches[0]["evidence"] in str(matches[0]["label_evidence"])


def test_brand_and_product_type_without_clear_model_match_is_possible(
    engine: RecallCloserCore,
) -> None:
    """A contextual brand/type match without explicit models stays possible."""
    item_id = engine.add_item(
        "Acme Air Fryer AF-999",
        brand="ACME",
        model="AF-999",
    )
    engine.cache_recall(
        recall_id="R-POSSIBLE",
        title="Acme Air Fryer Recall",
        source="Example consumer agency",
        description="Some units may overheat; check the product label.",
    )

    matches = engine.match_item(item_id)

    assert len(matches) == 1
    assert matches[0]["match_type"] == "POSSIBLE_MATCH"
    assert matches[0]["state"] == CaseState.POSSIBLE_MATCH.value
    assert matches[0]["evidence"] == '"Acme Air Fryer Recall"'


def test_explicit_model_mismatch_is_not_listed_with_evidence(
    engine: RecallCloserCore,
) -> None:
    """NOT_LISTED requires a model list explicitly present in the recall text."""
    item_id = engine.add_item(
        "Acme Air Fryer AF-999",
        brand="Acme",
        model="AF-999",
    )
    engine.cache_recall(
        recall_id="R-MODELS",
        title="Acme Air Fryer Recall",
        source="Example consumer agency",
        description="Affected models: AF-100, AF-200. Check the serial number.",
    )

    matches = engine.match_item(item_id)

    assert len(matches) == 1
    assert matches[0]["match_type"] == "MODEL_NOT_LISTED"
    assert matches[0]["state"] == CaseState.NOT_LISTED.value
    assert matches[0]["evidence"] == '"Affected models: AF-100, AF-200."'
    assert matches[0]["evidence"] in str(matches[0]["label_evidence"])


def test_no_relevant_cached_recall_returns_no_matches_without_safety_claim(
    engine: RecallCloserCore,
) -> None:
    """No matching cache entry returns an empty list without creating a case."""
    item_id = engine.add_item(
        "Northstar Electric Kettle NK-10",
        brand="Northstar",
        model="NK-10",
    )
    engine.cache_recall(
        recall_id="R-UNRELATED",
        title="Acme Air Fryer Recall",
        source="Example consumer agency",
        description="Affected models: AF-100, AF-200.",
    )

    matches = engine.match_item(item_id)

    assert matches == []
    assert engine.list_cases(item_id) == []
