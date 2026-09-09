"""F-1.2 라우터 단위 테스트.

``detect_file_kind`` 는 libmagic 의존이므로 mock 으로 대체해 라우팅 로직만 검증한다
(없는 파일 분기는 실제 경로로 검증, magic 불필요).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from processing.ingest import router
from processing.ingest.router import (
    LEDGER_FILE_NAMES,
    REASON_LEDGER_FILE,
    REASON_MISSING,
    REASON_UNKNOWN_MODALITY,
    route_file,
)


class RouterTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.f = Path(self._tmp.name) / "sample.bin"
        self.f.write_text("x", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestRouteFile(RouterTestBase):
    def test_missing_file_not_routable(self) -> None:
        r = route_file("/no/such/file.txt")
        self.assertFalse(r.routable)
        self.assertEqual(r.reason, REASON_MISSING)
        self.assertEqual(r.modality, "unknown")

    def test_unknown_modality_not_routable(self) -> None:
        with mock.patch.object(router, "detect_file_kind", return_value="unknown"):
            r = route_file(self.f)
        self.assertFalse(r.routable)
        self.assertEqual(r.reason, REASON_UNKNOWN_MODALITY)

    def test_supported_modalities_routable(self) -> None:
        for kind in ("txt", "pdf", "json", "word", "excel", "powerpoint", "image", "video", "audio"):
            with mock.patch.object(router, "detect_file_kind", return_value=kind):
                r = route_file(self.f)
            self.assertTrue(r.routable, f"{kind} 가 routable 이어야 함")
            self.assertEqual(r.modality, kind)
            self.assertEqual(r.reason, "")
            self.assertEqual(r.domain, "general")

    def test_ledger_file_not_routable(self) -> None:
        # 수집 도구의 목록 파일(manifest.json · _manifest.json)은 자산이 아니다 — 종류 판정 전에 걸러
        # 'ledger_file' 사유로 skip 한다. 통과시키면 json 텍스트 자산이 돼 모든 검색에 걸린다(2026-09-09 실측).
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            for name in sorted(LEDGER_FILE_NAMES) + ["MANIFEST.JSON"]:
                f = Path(d) / name
                f.write_text('{"files": []}', encoding="utf-8")
                r = route_file(f)
                self.assertFalse(r.routable, name)
                self.assertEqual(r.reason, REASON_LEDGER_FILE, name)
            # 이름이 정확히 같을 때만 — 비슷한 이름의 정상 자료는 걸리지 않는다(닫힌 목록).
            g = Path(d) / "데이터_manifest.json"
            g.write_text('{"a": 1}', encoding="utf-8")
            self.assertNotEqual(route_file(g).reason, REASON_LEDGER_FILE)

    def test_domain_passthrough(self) -> None:
        with mock.patch.object(router, "detect_file_kind", return_value="txt"):
            r = route_file(self.f, domain="medical")
        self.assertEqual(r.domain, "medical")
        self.assertTrue(r.routable)

    def test_toctou_filenotfound_treated_missing(self) -> None:
        with mock.patch.object(router, "detect_file_kind", side_effect=FileNotFoundError):
            r = route_file(self.f)
        self.assertFalse(r.routable)
        self.assertEqual(r.reason, REASON_MISSING)


if __name__ == "__main__":
    unittest.main()
