"""F-3.4 디스패처 단위 테스트 — modality → 분리 함수(_extract_*_meta / _embed_*) 분기."""
from __future__ import annotations

import unittest
from unittest import mock

from processing.dispatch import dispatcher
from processing.dispatch.dispatcher import UnsupportedModalityError, dispatch_embed, dispatch_extract_meta
from processing.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind


def _ctx(modality: str, domain: str = "general") -> ExtractContext:
    return ExtractContext(file_path="/tmp/x", modality=modality, domain=domain)


class TestTypes(unittest.TestCase):
    def test_assetrecord_defaults(self) -> None:
        r = AssetRecord()
        self.assertEqual((r.core_meta, r.ext_meta, r.tags, r.embeddings), ({}, {}, [], []))
        # 037: fts_plain 필드 제거(OS 전용·PG FTS 폐기) — 더 이상 존재하지 않아야 한다.
        self.assertFalse(hasattr(r, "fts_plain"))

    def test_embeddingitem_defaults(self) -> None:
        e = EmbeddingItem(channel="clip", vector=[0.0], model_name="legacy")
        self.assertEqual(e.chunk_index, 0)
        self.assertIsNone(e.model_version)


class TestExtractMetaRouting(unittest.TestCase):
    def test_text_kinds_route(self) -> None:
        for kind in ALLOWED_TEXT_META_FILE_KINDS:
            with mock.patch.object(dispatcher, "_extract_text_meta", return_value=AssetRecord(tags=["t"])) as m:
                out = dispatch_extract_meta(_ctx(kind))
            self.assertEqual(out.tags, ["t"], kind)
            m.assert_called_once()

    def test_media_routes(self) -> None:
        for mod, fn in [(MediaKind.IMAGE.value, "_extract_image_meta"),
                        (MediaKind.VIDEO.value, "_extract_video_meta"),
                        (MediaKind.AUDIO.value, "_extract_audio_meta")]:
            with mock.patch.object(dispatcher, fn, return_value=AssetRecord(tags=[mod])) as m:
                out = dispatch_extract_meta(_ctx(mod))
            self.assertEqual(out.tags, [mod])
            m.assert_called_once()

    def test_unsupported_raises(self) -> None:
        for bad in (MediaKind.UNKNOWN.value, "bogus", ""):
            with self.assertRaises(UnsupportedModalityError):
                dispatch_extract_meta(_ctx(bad))


class TestEmbedRouting(unittest.TestCase):
    def test_routes_and_unsupported(self) -> None:
        with mock.patch.object(dispatcher, "_embed_text", return_value=[]) as m:
            self.assertEqual(dispatch_embed(_ctx("txt"), AssetRecord()), [])
        m.assert_called_once()
        with self.assertRaises(UnsupportedModalityError):
            dispatch_embed(_ctx("bogus"), AssetRecord())


if __name__ == "__main__":
    unittest.main()
