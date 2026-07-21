"""F-1.4 상태 머신 단위 테스트.

전이 규칙(validate_transition)은 DB 없이 순수 검증. set_status/fetch_status 는
psycopg Connection 을 mock 으로 대체해 SQL 배선과 검증 경로를 확인(실 DB 불필요).
"""

from __future__ import annotations

import unittest
from unittest import mock

from processing.ingest.status import (
    ALLOWED_TRANSITIONS,
    TERMINAL,
    AssetStatus,
    InvalidTransitionError,
    fetch_status,
    mark_failed,
    set_status,
    validate_transition,
)


def _conn_with_status(status: str | None):
    """fetchone 이 {'status': ...}(또는 None) 를 반환하는 mock Connection."""
    conn = mock.MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = None if status is None else {"status": status}
    return conn, cur


class TestEnumAndMap(unittest.TestCase):
    def test_status_values_match_db_check(self) -> None:
        self.assertEqual(
            {s.value for s in AssetStatus},
            {"received", "routing", "classifying", "extracting", "registered", "failed", "deferred"},
        )

    def test_classifying_to_deferred_allowed(self) -> None:
        validate_transition("classifying", "deferred")  # 예외 없어야

    def test_deferred_is_terminal(self) -> None:
        self.assertEqual(ALLOWED_TRANSITIONS[AssetStatus.DEFERRED], frozenset())
        with self.assertRaises(InvalidTransitionError):
            validate_transition("deferred", "extracting")

    def test_terminal_states_have_no_outgoing(self) -> None:
        for s in TERMINAL:
            self.assertEqual(ALLOWED_TRANSITIONS[s], frozenset())


class TestValidateTransition(unittest.TestCase):
    def test_happy_path_transitions(self) -> None:
        for a, b in [
            ("received", "routing"),
            ("routing", "classifying"),
            ("classifying", "extracting"),
            ("extracting", "registered"),
        ]:
            validate_transition(a, b)  # 예외 없어야 함

    def test_failed_allowed_from_every_non_terminal(self) -> None:
        for s in ("received", "routing", "classifying", "extracting"):
            validate_transition(s, "failed")

    def test_skipping_steps_rejected(self) -> None:
        for a, b in [("received", "extracting"), ("received", "registered"), ("routing", "registered")]:
            with self.assertRaises(InvalidTransitionError):
                validate_transition(a, b)

    def test_from_terminal_rejected(self) -> None:
        for a in ("registered", "failed"):
            with self.assertRaises(InvalidTransitionError):
                validate_transition(a, "routing")
            with self.assertRaises(InvalidTransitionError):
                validate_transition(a, "failed")

    def test_accepts_enum_and_str(self) -> None:
        validate_transition(AssetStatus.RECEIVED, AssetStatus.ROUTING)
        validate_transition(AssetStatus.RECEIVED, "routing")


class TestDbHelpers(unittest.TestCase):
    def test_fetch_status_missing_raises(self) -> None:
        conn, _ = _conn_with_status(None)
        with self.assertRaises(LookupError):
            fetch_status(conn, 123)

    def test_set_status_valid_issues_update(self) -> None:
        conn, cur = _conn_with_status("received")
        set_status(conn, 7, "routing")
        # 마지막 execute 는 조건부 UPDATE — 파라미터에 target/None/asset_id + 기대 현재상태(WHERE 가드).
        # 009: 원자성 위해 WHERE status=<기대 현재상태> 추가 → 파라미터 끝에 current('received').
        update_calls = [c for c in cur.execute.call_args_list if "UPDATE asset" in c.args[0]]
        self.assertEqual(len(update_calls), 1)
        self.assertEqual(update_calls[0].args[1], ("routing", None, 7, "received"))

    def test_set_status_invalid_raises_before_update(self) -> None:
        conn, cur = _conn_with_status("registered")
        with self.assertRaises(InvalidTransitionError):
            set_status(conn, 7, "routing")
        self.assertEqual([c for c in cur.execute.call_args_list if "UPDATE asset" in c.args[0]], [])

    def test_mark_failed_sets_reason(self) -> None:
        conn, cur = _conn_with_status("extracting")
        mark_failed(conn, 9, "지원하지 않는 modality")
        update_calls = [c for c in cur.execute.call_args_list if "UPDATE asset" in c.args[0]]
        self.assertEqual(len(update_calls), 1)
        # 009: 조건부 UPDATE — 파라미터 끝에 기대 현재상태('extracting') WHERE 가드 추가.
        self.assertEqual(update_calls[0].args[1], ("failed", "지원하지 않는 modality", 9, "extracting"))

    def test_mark_failed_from_terminal_raises(self) -> None:
        conn, _ = _conn_with_status("registered")
        with self.assertRaises(InvalidTransitionError):
            mark_failed(conn, 9, "사유")


if __name__ == "__main__":
    unittest.main()
