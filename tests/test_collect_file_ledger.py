"""collect_file — 장부 파일은 DB 를 건드리기 전에 skip 된다(자산 행이 생기면 안 된다)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from processing.ingest.pipeline_steps import collect_file
from processing.ingest.router import REASON_LEDGER_FILE


class _NeverUsedConn:
    """DB 연결 대역 — 어떤 속성이든 건드리면 실패한다(장부 파일은 DB 전에 끝나야 한다)."""

    def __getattr__(self, name: str):  # noqa: D401
        raise AssertionError(f"장부 파일 skip 이 DB 를 건드렸다: {name}")


class TestCollectFileLedger(unittest.TestCase):
    def test_ledger_file_skipped_before_db(self) -> None:
        # 라우터 판정('ledger_file')만으로는 부족하다 — collect_file 이 'missing' 만 건너뛰고 나머지는 등록하던
        # 구멍(2026-09-09 발견)을 막는다. 해시·중복 검사·create_asset 어느 것도 호출되지 않아야 한다.
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "manifest.json"
            f.write_text('{"files": ["a.txt"]}', encoding="utf-8")
            r = collect_file(_NeverUsedConn(), str(f))  # type: ignore[arg-type]
        self.assertIsNone(r.asset_id)
        self.assertEqual(r.skip_reason, REASON_LEDGER_FILE)
        self.assertFalse(r.route.routable)


if __name__ == "__main__":
    unittest.main()
