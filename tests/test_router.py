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
