"""v2 단계 B — 기본 전략 등록 + 분리 함수 디스패치 테스트."""
from __future__ import annotations

import unittest
from unittest import mock

from processing.dispatch.types import AssetRecord, ExtractContext


class TestDispatchSplit(unittest.TestCase):
    def test_dispatch_extract_meta_routes_to_text(self) -> None:
        from processing.dispatch import dispatcher
        with mock.patch("processing.dispatch.dispatcher._extract_text_meta", return_value=AssetRecord(tags=["t"])) as m:
            rec = dispatcher.dispatch_extract_meta(ExtractContext(file_path="/a.txt", modality="txt"))
        m.assert_called_once()
        self.assertEqual(rec.tags, ["t"])

    def test_dispatch_embed_routes_to_text(self) -> None:
        from processing.dispatch import dispatcher
        with mock.patch("processing.dispatch.dispatcher._embed_text", return_value=[]) as m:
            out = dispatcher.dispatch_embed(ExtractContext(file_path="/a.txt", modality="txt"), AssetRecord())
        m.assert_called_once()
        self.assertEqual(out, [])

    def test_unsupported_modality_raises(self) -> None:
        from processing.dispatch import dispatcher
        with self.assertRaises(dispatcher.UnsupportedModalityError):
            dispatcher.dispatch_extract_meta(ExtractContext(file_path="/a.bin", modality="unknown"))


class TestBuiltinsRegistration(unittest.TestCase):
    def test_defaults_registered(self) -> None:
        from processing.pipeline import builtins  # noqa: F401 — 등록 부수효과
        from processing.pipeline.registry import DEFAULT_REGISTRY
        self.assertTrue(callable(DEFAULT_REGISTRY.resolve("classify", "cascade_v1")))
        self.assertTrue(callable(DEFAULT_REGISTRY.resolve("extract", "by_modality")))
        self.assertTrue(callable(DEFAULT_REGISTRY.resolve("embed", "by_modality")))
        self.assertTrue(callable(DEFAULT_REGISTRY.resolve("persist", "asset_upsert")))
        # by_modality 추출/임베딩은 온프레미스 — external_llm 태그가 없어야 의료 정책 통과
        self.assertNotIn("external_llm", DEFAULT_REGISTRY.tags("extract", "by_modality"))


if __name__ == "__main__":
    unittest.main()
