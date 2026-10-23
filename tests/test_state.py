"""Tests for Recall Closer's SQLite-backed case-state engine."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from core import (
    CaseState,
    InvalidTransitionError,
    RecallCloserCore,
    ValidationError,
)


@pytest.fixture
def engine(tmp_path: Path) -> Iterator[RecallCloserCore]:
    """Provide an engine backed by a temporary SQLite database."""
    core = RecallCloserCore(tmp_path / "recall_closer.sqlite3")
    try:
        yield core
    finally:
        core.close()


def create_case(engine: RecallCloserCore) -> int:
    """Create a case for a household product using the public API."""
    item_id = engine.add_item("Example household product")
    return engine.create_case(item_id)


def test_valid_transition_chain(engine: RecallCloserCore) -> None:
    """A case can progress through the full affected-product remedy chain."""
    case_id = create_case(engine)

    confirmed = engine.transition_case(
        case_id,
        CaseState.CONFIRMED_AFFECTED,
        label_evidence="Model and lot number match the recall notice.",
    )
    assert confirmed["state"] == CaseState.CONFIRMED_AFFECTED.value

    remedy_started = engine.transition_case(
        case_id,
        CaseState.REMEDY_STARTED,
        remedy_note="Replacement requested from the manufacturer.",
    )
    assert remedy_started["state"] == CaseState.REMEDY_STARTED.value

    closed = engine.transition_case(
        case_id,
        CaseState.CLOSED,
        remedy_note="Replacement received.",
    )
    assert closed["state"] == CaseState.CLOSED.value


def test_possible_match_can_be_marked_not_listed_with_label_evidence(
    engine: RecallCloserCore,
) -> None:
    """A label-evidenced check can resolve a possible match as not listed."""
    case_id = create_case(engine)

    case = engine.transition_case(
        case_id,
        CaseState.NOT_LISTED,
        label_evidence="Product label checked against the available recall notice.",
    )

    assert case["state"] == CaseState.NOT_LISTED.value
    assert case["label_evidence"] == (
        "Product label checked against the available recall notice."
    )


def test_missing_required_evidence_is_rejected(engine: RecallCloserCore) -> None:
    """Missing label evidence and remedy notes prevent their required transitions."""
    case_id = create_case(engine)

    with pytest.raises(ValidationError):
        engine.transition_case(case_id, CaseState.CONFIRMED_AFFECTED)
    assert engine.get_case(case_id)["state"] == CaseState.POSSIBLE_MATCH.value

    engine.transition_case(
        case_id,
        CaseState.CONFIRMED_AFFECTED,
        label_evidence="Model number matches.",
    )
    with pytest.raises(ValidationError):
        engine.transition_case(case_id, CaseState.REMEDY_STARTED)
    assert engine.get_case(case_id)["state"] == CaseState.CONFIRMED_AFFECTED.value

    engine.transition_case(
        case_id,
        CaseState.REMEDY_STARTED,
        remedy_note="Refund requested.",
    )
    with pytest.raises(ValidationError):
        engine.transition_case(case_id, CaseState.CLOSED)
    assert engine.get_case(case_id)["state"] == CaseState.REMEDY_STARTED.value


def test_illegal_transition_is_rejected(engine: RecallCloserCore) -> None:
    """A direct transition from a possible match to closed is not permitted."""
    case_id = create_case(engine)

    with pytest.raises(InvalidTransitionError):
        engine.transition_case(
            case_id,
            CaseState.CLOSED,
            remedy_note="This note does not make the transition valid.",
        )

    assert engine.get_case(case_id)["state"] == CaseState.POSSIBLE_MATCH.value


def test_successful_and_rejected_transitions_are_audited(
    engine: RecallCloserCore,
) -> None:
    """The public audit listing contains rejected and successful attempts."""
    case_id = create_case(engine)

    with pytest.raises(ValidationError):
        engine.transition_case(case_id, CaseState.CONFIRMED_AFFECTED)
    engine.transition_case(
        case_id,
        CaseState.CONFIRMED_AFFECTED,
        label_evidence="Model number matches.",
    )

    audit_entries = engine.list_audit_log(case_id)
    assert [entry["outcome"] for entry in audit_entries] == ["REJECTED", "SUCCESS"]
    assert audit_entries[0]["from_state"] == CaseState.POSSIBLE_MATCH.value
    assert audit_entries[0]["to_state"] == CaseState.CONFIRMED_AFFECTED.value
    assert audit_entries[1]["from_state"] == CaseState.POSSIBLE_MATCH.value
    assert audit_entries[1]["to_state"] == CaseState.CONFIRMED_AFFECTED.value


def test_file_database_records_persist_across_connections(
    tmp_path: Path,
) -> None:
    """Products, cases, and transition audit entries survive reopening the file DB."""
    database_path = tmp_path / "persistent.sqlite3"
    first_connection = RecallCloserCore(database_path)
    try:
        item_id = first_connection.add_item("Persisted household product")
        case_id = first_connection.create_case(item_id)
        first_connection.transition_case(
            case_id,
            CaseState.CONFIRMED_AFFECTED,
            label_evidence="Label evidence recorded.",
        )
    finally:
        first_connection.close()

    second_connection = RecallCloserCore(database_path)
    try:
        assert second_connection.list_items()[0]["name"] == (
            "Persisted household product"
        )
        assert second_connection.get_case(case_id)["state"] == (
            CaseState.CONFIRMED_AFFECTED.value
        )
        audit_entries = second_connection.list_audit_log(case_id)
        assert len(audit_entries) == 1
        assert audit_entries[0]["outcome"] == "SUCCESS"
    finally:
        second_connection.close()
