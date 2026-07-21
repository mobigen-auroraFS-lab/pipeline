"""US2(#4) dedup 상태 보강 단위 테스트.

내용 해시 중복 판정 범위가 ``registered`` 뿐 아니라 ``deferred`` 도 포함하는지를
실 DB 없이 검증한다. ``find_registered_asset_by_hash`` 가 ``conn.cursor().execute``
로 실행하는 SQL 문자열을 mock conn 으로 가로채, 다음을 단언한다:

  - dedup 대상에 ``registered`` 와 ``deferred`` 가 **둘 다** 포함된다(중복 식별).
  - ``failed`` 는 dedup 범위에 **없다**(실패 복구 재처리 허용 — FR-004, SC-004).

DICOM 등 의료 표준 포맷이 ``deferred`` 로 보류된 뒤 동일 파일을 재수집할 때,
중복 ``deferred`` 행이 쌓이지 않도록 하는 게 본 기능의 목표다(SC-003).
"""

from __future__ import annotations

import re
import unittest
import uuid
from unittest import mock

from processing.ingest.asset_persist import find_registered_asset_by_hash

_EXISTING = uuid.UUID("018f0000-0000-7000-8000-000000000018")


class _FakeCursor:
    """conn.cursor(row_factory=...) 컨텍스트매니저를 흉내내는 가짜 커서.

    execute 로 들어온 SQL 문자열을 기록하고, fetchone 으로 미리 정한 행을 돌려준다.
    """

    def __init__(self, fetch_row):
        self.fetch_row = fetch_row
        self.executed_sql = None
        self.executed_params = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed_sql = sql
        self.executed_params = params

    def fetchone(self):
        return self.fetch_row


class _FakeConn:
    """conn.cursor(...) 가 항상 같은 _FakeCursor 를 돌려주는 가짜 커넥션."""

    def __init__(self, fetch_row=None):
        self.cursor_obj = _FakeCursor(fetch_row)

    def cursor(self, *args, **kwargs):
        return self.cursor_obj


def _normalize(sql: str) -> str:
    """공백 정규화 + 소문자화 — 따옴표 안 상태 리터럴 비교를 위해 토큰만 본다."""
    return re.sub(r"\s+", " ", sql).strip().lower()


class TestDedupDeferred(unittest.TestCase):
    def test_dedup_sql_includes_registered_and_deferred(self) -> None:
        # 중복 식별 범위는 registered + deferred 둘 다 포함해야 한다(FR-004).
        conn = _FakeConn(fetch_row={"asset_id": _EXISTING})
        find_registered_asset_by_hash(conn, "h-dicom")
        sql = _normalize(conn.cursor_obj.executed_sql)
        self.assertIn("registered", sql)
        self.assertIn("deferred", sql)

    def test_dedup_sql_excludes_failed(self) -> None:
        # failed 상태의 같은 해시는 중복으로 보지 않아 재처리를 허용한다(SC-004).
        conn = _FakeConn(fetch_row=None)
        find_registered_asset_by_hash(conn, "h-failed")
        sql = _normalize(conn.cursor_obj.executed_sql)
        self.assertNotIn("failed", sql)

    def test_deferred_hash_identified_as_duplicate(self) -> None:
        # deferred 자산이 존재하면(행 반환) 중복으로 식별돼 asset_id 를 돌려준다.
        conn = _FakeConn(fetch_row={"asset_id": _EXISTING})
        got = find_registered_asset_by_hash(conn, "h-dicom")
        self.assertEqual(got, _EXISTING)

    def test_no_match_returns_none(self) -> None:
        # registered/deferred 어디에도 없으면 None(신규 적재 허용).
        conn = _FakeConn(fetch_row=None)
        self.assertIsNone(find_registered_asset_by_hash(conn, "h-new"))

    def test_hash_passed_as_query_param(self) -> None:
        # file_hash 는 SQL 인젝션 방지 위해 파라미터로 바인딩된다(리터럴 직삽 금지).
        conn = _FakeConn(fetch_row=None)
        find_registered_asset_by_hash(conn, "h-param")
        self.assertEqual(conn.cursor_obj.executed_params, ("h-param",))


if __name__ == "__main__":
    unittest.main()
