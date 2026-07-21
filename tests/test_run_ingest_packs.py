"""v2 단계 B — run_ingest 팩 경로(override 없음) + 정책 검증 테스트(목 영속화)."""
from __future__ import annotations

import contextlib
import unittest
import uuid
from unittest import mock

from processing.app import run_ingest as ri
from processing.classify.types import ClassificationResult
from processing.dispatch.types import AssetRecord, EmbeddingItem
from processing.ingest import pipeline_steps as ps  # FR-E3: 스텝 정본(seam patch)
from processing.ingest.router import RouteResult
from processing.pipeline.registry import StrategyRegistry


def _route(modality="txt", domain="general"):
    return RouteResult("/d/a." + modality, modality, domain, True, "")


def _fake_registry(captured):
    r = StrategyRegistry()
    r.register("classify", "cascade_v1", lambda ctx: ClassificationResult(final_label="general", confidence=0.7, decided_stage=2))

    def _extract(ctx):
        return AssetRecord()

    def _embed(ctx, rec):
        captured["embed_called"] = True
        return [EmbeddingItem(channel="st", vector=[0.1], model_name="m")]

    r.register("extract", "by_modality", _extract)
    r.register("embed", "by_modality", _embed)
    r.register("persist", "asset_upsert", lambda *a: None)
    return r


class TestRunIngestPackPath(unittest.TestCase):
    def _patch(self, stack):
        stack.enter_context(mock.patch.object(ps, "route_file", return_value=_route()))
        stack.enter_context(mock.patch.object(ps, "file_hash_and_size", return_value=("h", 1)))
        stack.enter_context(mock.patch.object(ps, "find_registered_asset_by_hash", return_value=None))
        stack.enter_context(mock.patch.object(ps, "create_asset", return_value=uuid.UUID(int=1)))
        stack.enter_context(mock.patch.object(ps, "record_classification"))
        stack.enter_context(mock.patch.object(ps, "set_status"))
        stack.enter_context(mock.patch.object(ps, "finalize_asset"))

    def test_pack_path_runs_extract_embed_and_validates_policy(self) -> None:
        captured: dict = {}
        reg = _fake_registry(captured)
        with contextlib.ExitStack() as stack:
            self._patch(stack)
            with mock.patch.object(ps, "policy_validate") as mval:
                res = ri.run_ingest(["/d/a.txt"], db=mock.MagicMock(), registry=reg, settings=mock.MagicMock(opensearch=mock.MagicMock(sync_enabled=False)))
        self.assertEqual(len(res["registered"]), 1)
        self.assertTrue(captured["embed_called"])      # embed 슬롯 호출됨(분리 경로)
        mval.assert_called()                            # 정책 검증 호출됨

    def test_override_extract_fn_skips_embed_slot(self) -> None:
        captured: dict = {}
        reg = _fake_registry(captured)
        with contextlib.ExitStack() as stack:
            self._patch(stack)
            ri.run_ingest(["/d/a.txt"], db=mock.MagicMock(), registry=reg,
                          extract_fn=lambda ctx: AssetRecord(),
                          classify_fn=lambda p, m: ClassificationResult(final_label="general", confidence=0.7, decided_stage=2),
                          settings=mock.MagicMock(opensearch=mock.MagicMock(sync_enabled=False)))
        self.assertNotIn("embed_called", captured)      # override 경로는 embed 슬롯 미호출

    def test_stage1_signature_defers_medical_format(self) -> None:
        """cascade stage1 중첩 구조({domain: {signature}})로 DICOM 파일이 deferred 처리되어야 한다.

        회귀: 평면 .get("signature")는 중첩 구조에서 항상 None → deferred 누락.
        """
        captured: dict = {}
        reg = _fake_registry(captured)

        # 새 중첩 형태의 stage1 단락 결과
        def stage1_medical_fn(p, m):
            return ClassificationResult(
                final_label="medical",
                confidence=1.0,
                decided_stage=1,
                stage1_scores={"medical": {"signature": "dicom"}},
            )

        with contextlib.ExitStack() as stack:
            self._patch(stack)
            stack.enter_context(mock.patch.object(ps, "record_lineage"))
            res = ri.run_ingest(
                ["/d/a.dcm"],
                db=mock.MagicMock(),
                registry=reg,
                settings=mock.MagicMock(opensearch=mock.MagicMock(sync_enabled=False)),
                classify_fn=stage1_medical_fn,
            )

        # deferred 에 asset_id 가 들어가야 하고 registered 는 비어 있어야 한다
        self.assertEqual(len(res["deferred"]), 1, "DICOM 파일은 deferred 로 분류돼야 함")
        self.assertEqual(len(res["registered"]), 0, "registered 는 비어 있어야 함")
        # extract/embed 슬롯이 호출되지 않아야 한다
        self.assertNotIn("embed_called", captured, "deferred 분기에서 embed 슬롯이 호출되면 안 됨")


if __name__ == "__main__":
    unittest.main()
