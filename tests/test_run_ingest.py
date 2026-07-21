"""F-1.x run_ingest 오케스트레이션 단위 테스트 (모델 A + F-5.1 분류 통합).

persist/status/route/해시/분류 함수는 mock 으로 대체하고, 오케스트레이터의 제어 흐름
(성공/실패 흡수/skip/중복/분류 영속화/도메인 주입)만 검증한다. 실 DB·LLM·파일 불필요.
"""

from __future__ import annotations

import contextlib
import unittest
import uuid
from unittest import mock

from processing.app import run_ingest as ri
from processing.classify.types import ClassificationResult
from processing.dispatch.dispatcher import UnsupportedModalityError
from processing.dispatch.types import AssetRecord
from processing.ingest import pipeline_steps as ps  # 069 FR-E3: 수집·처리 스텝 정본(patch 대상)
from processing.ingest.router import RouteResult

_EXISTING = uuid.UUID("018f0000-0000-7000-8000-000000000018")


def _route(modality="txt", routable=True, reason="", domain="general"):
    return RouteResult("/d/a." + modality, modality, domain, routable, reason)


def _cls(label="general", stage=2, conf=0.7):
    return ClassificationResult(final_label=label, confidence=conf, decided_stage=stage)


def _patch_all(stack: contextlib.ExitStack, route_result, *, dup=None) -> dict:
    """수집·처리 스텝 의존 함수 전부 patch 후 mock dict 반환.

    069 FR-E3: collect_file/process_asset 가 pipeline_steps(ps) 로 이관돼 내부 seam(route_file·set_status
    ·finalize_asset 등)을 ps 이름공간에서 호출한다 → **ps 에서 patch**. run_ingest CLI 가 직접 쓰는
    ``mark_failed`` 는 ri 에서, ``record_lineage`` 는 스텝(ps)·실패핸들러(ri) 양쪽에서 호출되므로 **같은
    mock 으로 둘 다** patch 해 호출을 한 곳에 집계한다(assert 호환).
    """
    m: dict = {}
    step_specs = {
        "route_file": mock.patch.object(ps, "route_file", return_value=route_result),
        "file_hash_and_size": mock.patch.object(ps, "file_hash_and_size", return_value=("h0", 10)),
        "find_registered_asset_by_hash":
            mock.patch.object(ps, "find_registered_asset_by_hash", return_value=dup),
        "create_asset": mock.patch.object(ps, "create_asset", return_value=1),
        "record_classification": mock.patch.object(ps, "record_classification"),
        "set_status": mock.patch.object(ps, "set_status"),
        "finalize_asset": mock.patch.object(ps, "finalize_asset"),
        "validate_ext_meta": mock.patch.object(ps, "validate_ext_meta"),
    }
    for k, p in step_specs.items():
        m[k] = stack.enter_context(p)
    m["mark_failed"] = stack.enter_context(mock.patch.object(ri, "mark_failed"))
    # record_lineage 는 ps(process_asset·collect_file)·ri(run_ingest 실패핸들러) 양쪽 호출 → 같은 mock.
    rl = mock.MagicMock()
    stack.enter_context(mock.patch.object(ps, "record_lineage", rl))
    stack.enter_context(mock.patch.object(ri, "record_lineage", rl))
    m["record_lineage"] = rl
    return m


class TestRunIngest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = mock.MagicMock(opensearch=mock.MagicMock(sync_enabled=False))
        self.db = mock.MagicMock()

    def _ingest(self, files, m_route, *, extract_fn, classify=None, dup=None, configure=None):
        with contextlib.ExitStack() as stack:
            m = _patch_all(stack, m_route, dup=dup)
            if configure:
                configure(m)
            res = ri.run_ingest(
                files, db=self.db, extract_fn=extract_fn,
                classify_fn=lambda p, mod: (classify or _cls()), settings=self.settings,
            )
        return res, m

    def test_success_path(self) -> None:
        res, m = self._ingest(["/d/a.txt"], _route(), extract_fn=lambda ctx: AssetRecord())
        self.assertEqual(res["registered"], [1])
        m["create_asset"].assert_called_once()
        m["record_classification"].assert_called_once()
        m["finalize_asset"].assert_called_once()
        m["mark_failed"].assert_not_called()
        self.assertEqual(m["set_status"].call_count, 3)  # routing→classifying→extracting

    def test_received_lineage_generated_modality_canonical(self) -> None:
        # 053(FR-203): received lineage 의 generated.modality 도 canonical('text') 로 일관.
        # route.modality 는 file_kind('json') 이지만 감사 기록은 저장값과 같은 canonical.
        res, m = self._ingest(
            ["/x/a.json"], _route(modality="json"), extract_fn=lambda ctx: AssetRecord()
        )
        self.assertEqual(res["registered"], [1])
        received = [
            c for c in m["record_lineage"].call_args_list
            if c.kwargs.get("activity") == "ingest.received.v1"
        ]
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].kwargs["generated"], {"modality": "text"})

    def test_missing_file_skipped(self) -> None:
        res, m = self._ingest(
            ["/no/x.txt"], _route(routable=False, reason=ri.REASON_MISSING), extract_fn=lambda ctx: AssetRecord()
        )
        self.assertEqual(len(res["skipped"]), 1)
        m["create_asset"].assert_not_called()
        m["record_classification"].assert_not_called()

    def test_duplicate_skipped(self) -> None:
        res, m = self._ingest(["/d/a.txt"], _route(), extract_fn=lambda ctx: AssetRecord(), dup=_EXISTING)
        self.assertEqual(res["registered"], [])
        self.assertEqual(len(res["skipped"]), 1)
        self.assertTrue(res["skipped"][0][1].startswith(ri.REASON_DUPLICATE))
        m["create_asset"].assert_not_called()

    def test_deferred_duplicate_skipped_no_new_row(self) -> None:
        # 009(#4): 이미 deferred 로 보류된 동일 해시(DICOM 재수집)는 find_registered_asset_by_hash
        # 가 기존 asset_id 를 dup 으로 돌려준다 → skip, 새 deferred/asset 행 미생성(중복 0).
        # run_ingest 의 skip 분기는 dup is not None 만 보므로 dedup 확장과 동일하게 동작한다.
        res, m = self._ingest(["/d/scan.dcm"], _route(modality="unknown"),
                              extract_fn=lambda ctx: AssetRecord(), dup=_EXISTING)
        self.assertEqual(res["registered"], [])
        self.assertEqual(res["deferred"], [])
        self.assertEqual(len(res["skipped"]), 1)
        self.assertTrue(res["skipped"][0][1].startswith(ri.REASON_DUPLICATE))
        m["create_asset"].assert_not_called()
        m["record_classification"].assert_not_called()

    def test_medical_standard_format_deferred(self) -> None:
        # DICOM 등 stage1 시그니처 → 추출 보류(deferred), 실패 아님, 추출 미호출.
        called = {"extract": False}

        def extract(ctx):
            called["extract"] = True
            return AssetRecord()

        cls = ClassificationResult(
            final_label="medical", confidence=1.0, decided_stage=1,
            stage1_scores={"medical": {"signature": "dicom"}},
        )
        res, m = self._ingest(
            ["/d/scan.dcm"], _route(modality="unknown", routable=False, reason="unknown_modality"),
            extract_fn=extract, classify=cls,
        )
        self.assertEqual(res["deferred"], [1])
        self.assertEqual(res["registered"], [])
        self.assertFalse(called["extract"])
        m["finalize_asset"].assert_not_called()
        m["mark_failed"].assert_not_called()
        m["record_classification"].assert_called_once()

    def test_classification_medical_sets_domain(self) -> None:
        seen = {}

        def capture(ctx):
            seen["domain"] = ctx.domain
            return AssetRecord()

        _, m = self._ingest(["/d/a.txt"], _route(), extract_fn=capture, classify=_cls(label="medical", stage=1, conf=1.0))
        self.assertEqual(seen["domain"], "medical")
        m["record_classification"].assert_called_once()

    def test_extract_exception_absorbed_as_failed(self) -> None:
        def boom(ctx):
            raise RuntimeError("추출 실패")
        res, m = self._ingest(["/d/a.txt"], _route(), extract_fn=boom)
        self.assertEqual(len(res["failed"]), 1)
        self.assertEqual(res["failed"][0][0], 1)
        m["finalize_asset"].assert_not_called()
        m["mark_failed"].assert_called_once()

    def test_unknown_modality_not_skipped_but_failed(self) -> None:
        def unsupported(ctx):
            raise UnsupportedModalityError(ctx.modality)
        res, m = self._ingest(
            ["/d/a.bin"], _route(modality="unknown", routable=False, reason="unknown_modality"), extract_fn=unsupported
        )
        m["create_asset"].assert_called_once()
        m["mark_failed"].assert_called_once()
        self.assertEqual(len(res["failed"]), 1)

    def test_create_asset_failure_isolated(self) -> None:
        res, m = self._ingest(
            ["/d/a.txt"], _route(), extract_fn=lambda ctx: AssetRecord(),
            configure=lambda m: setattr(m["create_asset"], "side_effect", RuntimeError("DB down")),
        )
        self.assertEqual(len(res["failed"]), 1)
        self.assertIsNone(res["failed"][0][0])
        m["finalize_asset"].assert_not_called()
        m["mark_failed"].assert_not_called()

    def test_batch_continues_after_one_failure(self) -> None:
        calls = {"n": 0}

        def extract(ctx):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("첫 파일 실패")
            return AssetRecord()

        res, m = self._ingest(
            ["/d/a.txt", "/d/b.txt"], _route(), extract_fn=extract,
            configure=lambda m: setattr(m["create_asset"], "side_effect", [1, 2]),
        )
        self.assertEqual(res["registered"], [2])
        self.assertEqual(len(res["failed"]), 1)
        self.assertEqual(res["failed"][0][0], 1)


class TestRunIngestSelfTopicWiring(unittest.TestCase):
    """T301(065) — 수집 배선: registered 직후·OS 색인 전 자기주제 분류(FR-301/302)·실패 격리(FR-204).

    ``process_asset`` 이 finalize(등록) 커밋 뒤·``os_index`` 앞에서 ``classify_asset_topic`` 을
    호출한다(색인 시점에 OS doc 이 topics 를 포함하도록 — FR-302). 자기 텍스트는 in-memory
    ``record.ext_meta`` 에서 구성해 주입한다(DB 재조회 회피). 분류 실패는 완전 격리해 registered 를
    유지하고 색인을 진행한다(FR-204). ``classify_asset_topic``·``_make_opensearch_indexer`` 를 patch 해
    실 DB·LLM 없이 호출 순서·인자·격리만 검증한다.
    """

    def setUp(self) -> None:
        self.settings = mock.MagicMock(opensearch=mock.MagicMock(sync_enabled=False))
        self.db = mock.MagicMock()

    def _run(self, *, extract_fn, classify_topic, os_recorder):
        with contextlib.ExitStack() as stack:
            m = _patch_all(stack, _route())
            stack.enter_context(
                mock.patch.object(ps, "classify_asset_topic", side_effect=classify_topic)
            )
            # os_index 배치기를 레코더로 대체 — 분류/색인 호출 순서를 관측한다.
            # _make_opensearch_indexer 는 run_ingest() CLI 가 호출하므로 ri 에서 patch(재export).
            stack.enter_context(
                mock.patch.object(ri, "_make_opensearch_indexer", return_value=os_recorder)
            )
            res = ri.run_ingest(
                ["/d/a.txt"], db=self.db, extract_fn=extract_fn,
                classify_fn=lambda p, mod: _cls(), settings=self.settings,
            )
        return res, m

    def test_classify_called_once_before_os_index(self) -> None:
        order: list[str] = []

        def classify_topic(conn, asset_id, *, self_text=None, settings=None, client=None):
            order.append("classify")

        def os_recorder(aid):
            order.append("os_index")

        res, _ = self._run(
            extract_fn=lambda ctx: AssetRecord(ext_meta={"summary": "요약"}),
            classify_topic=classify_topic, os_recorder=os_recorder,
        )
        self.assertEqual(res["registered"], [1])
        # 색인 전 분류(FR-302) — 정확히 이 순서.
        self.assertEqual(order, ["classify", "os_index"])
        self.assertEqual(order.count("classify"), 1)  # 자산당 정확히 1회 분류

    def test_self_text_built_from_inmemory_record(self) -> None:
        seen: dict = {}

        def classify_topic(conn, asset_id, *, self_text=None, settings=None, client=None):
            seen["self_text"] = self_text
            seen["settings"] = settings
            seen["asset_id"] = asset_id

        rec = AssetRecord(ext_meta={
            "summary": "농구 경기", "keywords": ["농구"],
            "labels": [{"label": "sport", "score": 0.9}],
        })
        self._run(extract_fn=lambda ctx: rec, classify_topic=classify_topic,
                  os_recorder=lambda aid: None)
        # in-memory ext_meta 로 구성된 자기 텍스트를 주입(DB 재조회 없이)·settings 전달.
        self.assertIn("농구 경기", seen["self_text"])
        self.assertIn("농구", seen["self_text"])
        self.assertIn("sport", seen["self_text"])  # labels 상위 라벨도 자기 텍스트에 포함
        self.assertIs(seen["settings"], self.settings)
        self.assertEqual(seen["asset_id"], 1)

    def test_classify_failure_isolated_registered_and_indexed(self) -> None:
        called = {"os": False}

        def classify_topic(conn, asset_id, *, self_text=None, settings=None, client=None):
            raise RuntimeError("분류 실패")

        def os_recorder(aid):
            called["os"] = True

        with self.assertLogs(ri._LOG, level="WARNING"):
            res, _ = self._run(
                extract_fn=lambda ctx: AssetRecord(ext_meta={"summary": "요약"}),
                classify_topic=classify_topic, os_recorder=os_recorder,
            )
        # 분류 실패에도 registered 유지·os_index 진행(FR-204 완전 격리).
        self.assertEqual(res["registered"], [1])
        self.assertEqual(res["failed"], [])
        self.assertTrue(called["os"])


if __name__ == "__main__":
    unittest.main()
