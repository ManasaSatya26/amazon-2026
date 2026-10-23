"""Deterministic SQLite-backed core engine for Recall Closer.

This module has no web-framework or MCP dependencies. A ``NOT_LISTED`` case
records only that a product was not found in the checked recall data; it does
not mean that the product is safe.
"""

from __future__ import annotations

import sqlite3
import re
import threading
import unicodedata
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TypeAlias

DatabasePath: TypeAlias = str | Path
Record: TypeAlias = dict[str, object]
MatchResult: TypeAlias = dict[str, object]

DEFAULT_DATABASE_PATH = Path("recall_closer.db")
_MODEL_LIST_RE = re.compile(
    r"\bmodels?\s*(?:(?:numbers?|nos?)\s*)?(?:are|include|:|#|=)\s*([^.!?;\n]+)",
    re.IGNORECASE,
)
_MODEL_TOKEN_RE = re.compile(
    r"(?<![a-z0-9])[a-z0-9]+(?:[-/][a-z0-9]+)*(?![a-z0-9])",
    re.IGNORECASE,
)


class CaseState(str, Enum):
    """Supported states for a product recall case."""

    POSSIBLE_MATCH = "POSSIBLE_MATCH"
    CONFIRMED_AFFECTED = "CONFIRMED_AFFECTED"
    NOT_LISTED = "NOT_LISTED"
    REMEDY_STARTED = "REMEDY_STARTED"
    CLOSED = "CLOSED"


class RecallCloserError(Exception):
    """Base exception for core engine errors."""


class ValidationError(RecallCloserError, ValueError):
    """Raised when a supplied field is invalid or required evidence is absent."""


class CaseNotFoundError(RecallCloserError, LookupError):
    """Raised when an operation refers to a case that does not exist."""


class ItemNotFoundError(RecallCloserError, LookupError):
    """Raised when an operation refers to a product that does not exist."""


class CachedRecallNotFoundError(RecallCloserError, LookupError):
    """Raised when a case refers to a recall absent from the local cache."""


class InvalidTransitionError(RecallCloserError, ValueError):
    """Raised when a requested case-state transition is not permitted."""


_ALLOWED_TRANSITIONS: dict[CaseState, frozenset[CaseState]] = {
    CaseState.POSSIBLE_MATCH: frozenset(
        {CaseState.CONFIRMED_AFFECTED, CaseState.NOT_LISTED}
    ),
    CaseState.CONFIRMED_AFFECTED: frozenset({CaseState.REMEDY_STARTED}),
    CaseState.REMEDY_STARTED: frozenset({CaseState.CLOSED}),
    CaseState.NOT_LISTED: frozenset(),
    CaseState.CLOSED: frozenset(),
}


def _utc_now() -> str:
    """Return a sortable UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _required_text(value: str, field_name: str) -> str:
    """Validate and normalize a required string value."""
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _optional_text(value: str | None, field_name: str) -> str | None:
    """Normalize an optional string, rejecting non-string values."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{field_name} must be a string or None.")
    return value.strip() or None


def _normalize_text(value: str) -> str:
    """Case-fold text and normalize punctuation and spacing for comparisons."""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.sub(r"[\W_]+", " ", ascii_text).split())


def _contains_normalized(text: str, phrase: str) -> bool:
    """Return whether a normalized phrase occurs as complete words in text."""
    normalized_text = _normalize_text(text)
    normalized_phrase = _normalize_text(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


def _product_type(name: str, brand: str | None, model: str | None) -> str:
    """Derive a product-type phrase from the item name, excluding brand/model."""
    normalized_name = _normalize_text(name)
    remaining = normalized_name.split()
    for value in (brand, model):
        phrase = _normalize_text(value or "")
        words = phrase.split()
        if not words:
            continue
        for index in range(len(remaining) - len(words) + 1):
            if remaining[index : index + len(words)] == words:
                del remaining[index : index + len(words)]
                break
    return " ".join(remaining)


def _quoted_sentence(text: str, phrase: str) -> str | None:
    """Return a quoted source sentence containing the normalized phrase."""
    for sentence in re.split(r"(?<=[.!?])\s+|\r?\n+", text):
        candidate = sentence.strip()
        if candidate and _contains_normalized(candidate, phrase):
            return f'"{candidate}"'
    return None


def _explicit_model_lists(text: str) -> list[tuple[tuple[str, ...], str]]:
    """Extract model identifiers only from explicitly labeled model lists."""
    found: list[tuple[tuple[str, ...], str]] = []
    for match in _MODEL_LIST_RE.finditer(text):
        identifiers = tuple(
            dict.fromkeys(
                _normalize_text(token)
                for token in _MODEL_TOKEN_RE.findall(match.group(1))
                if any(char.isdigit() for char in token)
            )
        )
        if not identifiers:
            continue
        sentence_start = max(
            text.rfind(mark, 0, match.start()) for mark in (".", "!", "?", "\n")
        )
        sentence_end_candidates = [
            (index, mark)
            for mark in (".", "!", "?", "\n")
            if (index := text.find(mark, match.end())) != -1
        ]
        if sentence_end_candidates:
            sentence_end, delimiter = min(sentence_end_candidates)
            if delimiter != "\n":
                sentence_end += 1
        else:
            sentence_end = len(text)
        sentence = text[sentence_start + 1 : sentence_end].strip()
        if sentence:
            found.append((identifiers, f'"{sentence}"'))
    return found


class RecallCloserCore:
    """SQLite persistence and deterministic case-state operations.

    The instance owns one SQLite connection and serializes its operations with
    a lock. Use :meth:`close` when the instance is no longer needed.
    """

    def __init__(self, database_path: DatabasePath = DEFAULT_DATABASE_PATH) -> None:
        """Open or create the SQLite database and initialize its schema."""
        self.database_path = str(database_path)
        self._lock = threading.RLock()
        try:
            self._connection = sqlite3.connect(
                self.database_path,
                check_same_thread=False,
                timeout=5.0,
            )
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._create_schema()
        except sqlite3.Error as exc:
            if hasattr(self, "_connection"):
                self._connection.close()
            raise RecallCloserError(f"Could not initialize SQLite database: {exc}") from exc

    def _create_schema(self) -> None:
        """Create the four core tables if they do not already exist."""
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    brand TEXT,
                    model TEXT,
                    upc TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recalls_cache (
                    recall_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source TEXT NOT NULL,
                    description TEXT,
                    cached_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE RESTRICT,
                    recall_id TEXT REFERENCES recalls_cache(recall_id) ON DELETE SET NULL,
                    state TEXT NOT NULL CHECK (
                        state IN (
                            'POSSIBLE_MATCH',
                            'CONFIRMED_AFFECTED',
                            'NOT_LISTED',
                            'REMEDY_STARTED',
                            'CLOSED'
                        )
                    ),
                    label_evidence TEXT,
                    remedy_note TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id INTEGER,
                    from_state TEXT,
                    to_state TEXT NOT NULL,
                    outcome TEXT NOT NULL CHECK (outcome IN ('SUCCESS', 'REJECTED')),
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def close(self) -> None:
        """Close the owned SQLite connection."""
        with self._lock:
            self._connection.close()

    def _fetch_one(self, query: str, parameters: tuple[object, ...]) -> Record | None:
        """Run a parameterized query and convert one row into a dictionary."""
        row = self._connection.execute(query, parameters).fetchone()
        return dict(row) if row is not None else None

    def _fetch_all(self, query: str, parameters: tuple[object, ...] = ()) -> list[Record]:
        """Run a parameterized query and convert its rows into dictionaries."""
        rows = self._connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def add_item(
        self,
        name: str,
        brand: str | None = None,
        model: str | None = None,
        upc: str | None = None,
    ) -> int:
        """Add a household product and return its database ID."""
        clean_name = _required_text(name, "name")
        clean_brand = _optional_text(brand, "brand")
        clean_model = _optional_text(model, "model")
        clean_upc = _optional_text(upc, "upc")
        try:
            with self._lock, self._connection:
                cursor = self._connection.execute(
                    """
                    INSERT INTO items (name, brand, model, upc, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (clean_name, clean_brand, clean_model, clean_upc, _utc_now()),
                )
                return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise RecallCloserError(f"Could not add product: {exc}") from exc

    def list_items(self) -> list[Record]:
        """Return household products in insertion order."""
        try:
            with self._lock:
                return self._fetch_all(
                    "SELECT id, name, brand, model, upc, created_at FROM items ORDER BY id"
                )
        except sqlite3.Error as exc:
            raise RecallCloserError(f"Could not list products: {exc}") from exc

    def cache_recall(
        self,
        recall_id: str,
        title: str,
        source: str,
        description: str | None = None,
    ) -> None:
        """Insert or refresh a recall record in the local cache."""
        clean_id = _required_text(recall_id, "recall_id")
        clean_title = _required_text(title, "title")
        clean_source = _required_text(source, "source")
        clean_description = _optional_text(description, "description")
        try:
            with self._lock, self._connection:
                self._connection.execute(
                    """
                    INSERT INTO recalls_cache
                        (recall_id, title, source, description, cached_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(recall_id) DO UPDATE SET
                        title = excluded.title,
                        source = excluded.source,
                        description = excluded.description,
                        cached_at = excluded.cached_at
                    """,
                    (
                        clean_id,
                        clean_title,
                        clean_source,
                        clean_description,
                        _utc_now(),
                    ),
                )
        except sqlite3.Error as exc:
            raise RecallCloserError(f"Could not cache recall: {exc}") from exc

    def list_cached_recalls(self) -> list[Record]:
        """Return cached recall records in stable ID order."""
        try:
            with self._lock:
                return self._fetch_all(
                    """
                    SELECT recall_id, title, source, description, cached_at
                    FROM recalls_cache
                    ORDER BY recall_id
                    """
                )
        except sqlite3.Error as exc:
            raise RecallCloserError(f"Could not list cached recalls: {exc}") from exc

    def create_case(self, item_id: int, recall_id: str | None = None) -> int:
        """Create a case in ``POSSIBLE_MATCH`` for an existing product."""
        if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id < 1:
            raise ValidationError("item_id must be a positive integer.")
        clean_recall_id = _optional_text(recall_id, "recall_id")
        now = _utc_now()
        try:
            with self._lock, self._connection:
                if self._fetch_one("SELECT id FROM items WHERE id = ?", (item_id,)) is None:
                    raise ItemNotFoundError(f"Product {item_id} does not exist.")
                if clean_recall_id is not None and self._fetch_one(
                    "SELECT recall_id FROM recalls_cache WHERE recall_id = ?",
                    (clean_recall_id,),
                ) is None:
                    raise CachedRecallNotFoundError(
                        f"Recall {clean_recall_id!r} is not in the local cache."
                    )
                cursor = self._connection.execute(
                    """
                    INSERT INTO cases (item_id, recall_id, state, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (item_id, clean_recall_id, CaseState.POSSIBLE_MATCH.value, now, now),
                )
                return int(cursor.lastrowid)
        except (ItemNotFoundError, CachedRecallNotFoundError):
            raise
        except sqlite3.IntegrityError as exc:
            raise RecallCloserError(f"Could not create case: {exc}") from exc
        except sqlite3.Error as exc:
            raise RecallCloserError(f"Could not create case: {exc}") from exc

    def get_case(self, case_id: int) -> Record:
        """Return a case, raising :class:`CaseNotFoundError` if it is absent."""
        try:
            with self._lock:
                case = self._fetch_one(
                    """
                    SELECT id, item_id, recall_id, state, label_evidence,
                           remedy_note, created_at, updated_at
                    FROM cases WHERE id = ?
                    """,
                    (case_id,),
                )
        except sqlite3.Error as exc:
            raise RecallCloserError(f"Could not retrieve case: {exc}") from exc
        if case is None:
            raise CaseNotFoundError(f"Case {case_id} does not exist.")
        return case

    def list_cases(self, item_id: int | None = None) -> list[Record]:
        """List cases, optionally filtered by product ID."""
        query = """
            SELECT id, item_id, recall_id, state, label_evidence, remedy_note,
                   created_at, updated_at
            FROM cases
        """
        parameters: tuple[object, ...] = ()
        if item_id is not None:
            query += " WHERE item_id = ?"
            parameters = (item_id,)
        query += " ORDER BY id"
        try:
            with self._lock:
                return self._fetch_all(query, parameters)
        except sqlite3.Error as exc:
            raise RecallCloserError(f"Could not list cases: {exc}") from exc

    def match_item(self, item_id: int) -> list[MatchResult]:
        """Compare an item against cached recalls and return evidence-backed cases.

        Product type is derived from the item name after removing its recorded
        brand and model. Recall ``title`` and ``description`` are treated as
        recall text; ``source`` is attribution metadata, not matching evidence.
        An exact model plus matching brand and product type confirms a case.
        Brand/type matches without an exact model remain possible matches.
        A model mismatch is marked ``NOT_LISTED`` only if the recall text
        explicitly labels a list of model identifiers.

        Each result contains the persisted case fields plus ``match_type``
        (``EXACT_MODEL``, ``POSSIBLE_MATCH``, or ``MODEL_NOT_LISTED``) and
        ``evidence`` (a quoted sentence from the cached recall text). An empty
        result means no cached record met these matching rules; it is not a
        statement that the item is safe.
        """
        if isinstance(item_id, bool) or not isinstance(item_id, int) or item_id < 1:
            raise ValidationError("item_id must be a positive integer.")
        item = next(
            (candidate for candidate in self.list_items() if candidate["id"] == item_id),
            None,
        )
        if item is None:
            raise ItemNotFoundError(f"Product {item_id} does not exist.")

        name = str(item["name"])
        brand = str(item["brand"]) if item["brand"] is not None else None
        model = str(item["model"]) if item["model"] is not None else None
        product_type = _product_type(name, brand, model)
        recalls = self.list_cached_recalls()
        existing_cases = {
            str(case["recall_id"]): case
            for case in self.list_cases(item_id)
            if case["recall_id"] is not None
        }
        results: list[MatchResult] = []

        for recall in recalls:
            title = str(recall["title"])
            description = str(recall["description"] or "")
            recall_text = "\n".join(part for part in (title, description) if part)
            brand_matches = bool(brand and _contains_normalized(recall_text, brand))
            type_matches = bool(
                product_type and _contains_normalized(recall_text, product_type)
            )
            context_matches = brand_matches and type_matches
            exact_model = bool(model and _contains_normalized(recall_text, model))
            if not context_matches and not exact_model:
                continue

            model_quote = _quoted_sentence(recall_text, model) if model else None
            listed_model_groups = _explicit_model_lists(recall_text)
            normalized_model = _normalize_text(model or "")
            mismatch_quote = next(
                (
                    quote
                    for listed_models, quote in listed_model_groups
                    if normalized_model and normalized_model not in listed_models
                ),
                None,
            )
            if exact_model:
                match_type = "EXACT_MODEL"
                evidence = model_quote
            elif context_matches and mismatch_quote is not None:
                match_type = "MODEL_NOT_LISTED"
                evidence = mismatch_quote
            else:
                match_type = "POSSIBLE_MATCH"
                evidence = next(
                    (
                        f'"{sentence.strip()}"'
                        for sentence in re.split(r"(?<=[.!?])\s+|\r?\n+", recall_text)
                        if sentence.strip()
                        and _contains_normalized(sentence, brand or "")
                        and _contains_normalized(sentence, product_type)
                    ),
                    None,
                )
                if evidence is None:
                    evidence = _quoted_sentence(recall_text, brand or product_type)

            recall_id = str(recall["recall_id"])
            case = existing_cases.get(recall_id)
            if case is None:
                case_id = self.create_case(item_id, recall_id)
                case = self.get_case(case_id)

            current_state = CaseState(str(case["state"]))
            if current_state is CaseState.POSSIBLE_MATCH:
                if match_type == "EXACT_MODEL" and context_matches and evidence is not None:
                    case = self.transition_case(
                        int(case["id"]),
                        CaseState.CONFIRMED_AFFECTED,
                        label_evidence=(
                            f'Recorded household model "{model}" matches recall record: '
                            f"{evidence}"
                        ),
                    )
                elif match_type == "MODEL_NOT_LISTED" and evidence is not None:
                    case = self.transition_case(
                        int(case["id"]),
                        CaseState.NOT_LISTED,
                        label_evidence=(
                            f'Recorded household model "{model}" differs from the '
                            f"explicitly listed recall models: {evidence}"
                        ),
                    )

            result = dict(case)
            result["match_type"] = match_type
            result["evidence"] = evidence
            results.append(result)

        return results

    def transition_case(
        self,
        case_id: int,
        new_state: CaseState | str,
        *,
        label_evidence: str | None = None,
        remedy_note: str | None = None,
    ) -> Record:
        """Move a case through an allowed state and audit every attempt.

        Label evidence is mandatory for decisions from ``POSSIBLE_MATCH``.
        Remedy notes are mandatory when starting or closing a remedy. Rejected
        transitions are committed to the audit log before the corresponding
        exception is raised.
        """
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                case = self._fetch_one(
                    "SELECT id, state FROM cases WHERE id = ?",
                    (case_id,),
                )
                if case is None:
                    requested = self._requested_state_label(new_state)
                    self._write_audit(
                        case_id, None, requested, "REJECTED", "Case does not exist."
                    )
                    self._connection.commit()
                    raise CaseNotFoundError(f"Case {case_id} does not exist.")

                current = CaseState(str(case["state"]))
                requested = self._requested_state_label(new_state)
                try:
                    clean_label = _optional_text(label_evidence, "label_evidence")
                    clean_remedy = _optional_text(remedy_note, "remedy_note")
                    target = self._coerce_state(new_state)
                except ValidationError as exc:
                    self._write_audit(
                        case_id, current.value, requested, "REJECTED", str(exc)
                    )
                    self._connection.commit()
                    raise

                rejection = self._transition_rejection(
                    current, target, clean_label, clean_remedy
                )
                if rejection is not None:
                    self._write_audit(
                        case_id, current.value, target.value, "REJECTED", rejection
                    )
                    self._connection.commit()
                    if "evidence" in rejection or "remedy_note" in rejection:
                        raise ValidationError(rejection)
                    raise InvalidTransitionError(rejection)

                self._connection.execute(
                    """
                    UPDATE cases
                    SET state = ?,
                        label_evidence = COALESCE(?, label_evidence),
                        remedy_note = COALESCE(?, remedy_note),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (target.value, clean_label, clean_remedy, _utc_now(), case_id),
                )
                self._write_audit(
                    case_id,
                    current.value,
                    target.value,
                    "SUCCESS",
                    "Transition completed.",
                )
                self._connection.commit()
            except (CaseNotFoundError, InvalidTransitionError, ValidationError):
                raise
            except sqlite3.Error as exc:
                self._connection.rollback()
                raise RecallCloserError(f"Could not transition case: {exc}") from exc
            return self.get_case(case_id)

    def list_audit_log(self, case_id: int | None = None) -> list[Record]:
        """List transition audit entries, optionally filtered by case ID."""
        query = """
            SELECT id, case_id, from_state, to_state, outcome, reason, created_at
            FROM audit_log
        """
        parameters: tuple[object, ...] = ()
        if case_id is not None:
            query += " WHERE case_id = ?"
            parameters = (case_id,)
        query += " ORDER BY id"
        try:
            with self._lock:
                return self._fetch_all(query, parameters)
        except sqlite3.Error as exc:
            raise RecallCloserError(f"Could not list audit entries: {exc}") from exc

    def _write_audit(
        self,
        case_id: int | None,
        from_state: str | None,
        to_state: str,
        outcome: str,
        reason: str,
    ) -> None:
        """Insert an audit record within the caller's transaction."""
        self._connection.execute(
            """
            INSERT INTO audit_log
                (case_id, from_state, to_state, outcome, reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (case_id, from_state, to_state, outcome, reason, _utc_now()),
        )

    @staticmethod
    def _coerce_state(value: CaseState | str) -> CaseState:
        """Convert a requested state or raise a clear validation error."""
        try:
            return value if isinstance(value, CaseState) else CaseState(value)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"Unsupported case state: {value!r}.") from exc

    @staticmethod
    def _requested_state_label(value: object) -> str:
        """Return a safe string for recording an attempted state in the audit log."""
        return value.value if isinstance(value, CaseState) else str(value)

    @staticmethod
    def _transition_rejection(
        current: CaseState,
        target: CaseState,
        label_evidence: str | None,
        remedy_note: str | None,
    ) -> str | None:
        """Describe why a transition is invalid, or return ``None`` if allowed."""
        if target not in _ALLOWED_TRANSITIONS[current]:
            return f"Transition from {current.value} to {target.value} is not allowed."
        if current is CaseState.POSSIBLE_MATCH and not label_evidence:
            return "Label evidence is required to resolve a possible match."
        if target in {CaseState.REMEDY_STARTED, CaseState.CLOSED} and not remedy_note:
            return f"A remedy_note is required to transition to {target.value}."
        return None


def initialize(database_path: DatabasePath = DEFAULT_DATABASE_PATH) -> RecallCloserCore:
    """Open a core engine and ensure its SQLite schema is initialized."""
    return RecallCloserCore(database_path)
