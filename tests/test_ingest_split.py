"""030 G1(T001) — run_ingest → collect_file + process_asset 분할 단위 테스트.

목적
    `run_ingest` 의 파일 단위 루프를 두 순수 함수로 분할하되 **오케스트레이션만 분리**하고
    추출·분류·임베딩·상태전이·lineage·dedup·os_index 로직은 **무변경**임을 단언한다(헌법 8조).

    · ``collect_file(conn, fs_path) -> CollectResult``: route_file 탐지 + create_asset(received)
      + ingest.received.v1 lineage. 파일 누락/동일 해시 중복이면 스킵(asset 미생성).
    · ``process_asset(asset_id, *, db, ...) -> 'registered'|'deferred'``: 그 자산을
      routing→classifying→extracting→registered(또는 deferred)로 전이 + 단계별 lineage + 인라인 색인.

봉인(동작 불변)
    분할된 ``run_ingest`` CLI 가 분할 전과 **같은 자산/상태/lineage/os_index 호출 시퀀스**를 내는지
    ``test_run_ingest_split_preserves_operation_sequence`` 로 연산 순서를 그대로 단언한다. 기존
    ``tests/test_run_ingest*`` 는 무변경 통과(같은 mock.patch.object(ri, ...) seam 유지)가 추가 봉인.

모두 실 DB·LLM·파일·Airflow 불필요(가짜 conn/db·주입 fn). FR-011: Airflow import 0 도 별도 단언.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import unittest
import uuid
from unittest import mock

from processing.app import run_ingest as ri
from processing.classify.types import ClassificationResult
from processing.dispatch.types import AssetRecord
from processing.ingest import pipeline_steps as ps  # 069 FR-E3: 수집·처리 스텝 정본(내부 seam patch 대상)
from processing.ingest.router import RouteResult

_EXISTING = uuid.UUID("018f0000-0000-7000-8000-000000000018")
_AID = uuid.UUID("018f0000-0000-7000-8000-000000000001")


def _route(modality="txt", routable=True, reason="", domain="general"):
    return RouteResult("/d/a." + modality, modality, domain, routable, reason)


def _cls(label="general", stage=2, conf=0.7, sig=None):
    scores = {label: {"signature": sig}} if sig else {}
    return ClassificationResult(final_label=label, confidence=conf, decided_stage=stage, stage1_scores=scores)


# ── collect_file 단위 ────────────────────────────────────────────────────────


class TestCollectFile(unittest.TestCase):
    """collect_file: 파일 1건을 received 자산으로 만들거나(또는 스킵) — route+create+lineage."""

    def setUp(self) -> None:
        self.conn = mock.MagicMock()

    def _patch(self, stack, *, route, dup=None):
        # FR-E3: collect_file 는 pipeline_steps(ps) 내부 seam(route_file·create_asset 등)을 호출 → ps 에서 patch.
        return {
            "route_file": stack.enter_context(mock.patch.object(ps, "route_file", return_value=route)),
            "file_hash_and_size": stack.enter_context(
                mock.patch.object(ps, "file_hash_and_size", return_value=("h0", 10))
            ),
            "find_registered_asset_by_hash": stack.enter_context(
                mock.patch.object(ps, "find_registered_asset_by_hash", return_value=dup)
            ),
            "create_asset": stack.enter_context(mock.patch.object(ps, "create_asset", return_value=_AID)),
            "record_lineage": stack.enter_context(mock.patch.object(ps, "record_lineage")),
        }

    def test_routable_file_creates_received_asset_with_lineage(self) -> None:
        with contextlib.ExitStack() as stack:
            m = self._patch(stack, route=_route(modality="txt"))
            res = ri.collect_file(self.conn, "/d/a.txt")
        # received 자산 1건 — asset_id 반환·스킵 아님
        self.assertEqual(res.asset_id, _AID)
        self.assertIsNone(res.skip_reason)
        # 모달리티 탐지값이 create_asset 으로 전달됨
        m["create_asset"].assert_called_once()
        self.assertEqual(m["create_asset"].call_args.kwargs["modality"], "txt")
        # ingest.received.v1 lineage 정확히 1건
        m["record_lineage"].assert_called_once()
        self.assertEqual(m["record_lineage"].call_args.kwargs["activity"], "ingest.received.v1")
        # 탐지된 route 가 결과에 실려 process 단계로 전달 가능
        self.assertEqual(res.route.modality, "txt")

    def test_missing_file_skipped_without_create(self) -> None:
        with contextlib.ExitStack() as stack:
            m = self._patch(stack, route=_route(routable=False, reason=ri.REASON_MISSING))
            res = ri.collect_file(self.conn, "/no/x.txt")
        self.assertIsNone(res.asset_id)
        self.assertEqual(res.skip_reason, ri.REASON_MISSING)
        m["create_asset"].assert_not_called()
        m["record_lineage"].assert_not_called()
        # 누락 파일은 해시조차 계산하지 않는다(기존 동작 — route 직후 스킵)
        m["file_hash_and_size"].assert_not_called()

    def test_duplicate_hash_skipped_without_create(self) -> None:
        with contextlib.ExitStack() as stack:
            m = self._patch(stack, route=_route(), dup=_EXISTING)
            res = ri.collect_file(self.conn, "/d/a.txt")
        self.assertIsNone(res.asset_id)
        self.assertTrue(res.skip_reason.startswith(ri.REASON_DUPLICATE))
        self.assertIn(str(_EXISTING), res.skip_reason)
        m["create_asset"].assert_not_called()
        m["record_lineage"].assert_not_called()

    def test_unknown_modality_still_creates_received(self) -> None:
        # unknown_modality(=routable False, reason!=missing)는 스킵하지 않고 received 생성 →
        # 이후 process 단계 추출에서 UnsupportedModalityError 로 failed 처리되는 기존 동작 보존.
        with contextlib.ExitStack() as stack:
            m = self._patch(stack, route=_route(modality="unknown", routable=False, reason="unknown_modality"))
            res = ri.collect_file(self.conn, "/d/a.bin")
        self.assertEqual(res.asset_id, _AID)
        self.assertIsNone(res.skip_reason)
        m["create_asset"].assert_called_once()


# ── process_asset 단위 ───────────────────────────────────────────────────────


class TestProcessAsset(unittest.TestCase):
    """process_asset: received 자산을 registered/deferred 까지 — 단계별 전이·lineage·인라인 색인."""

    def setUp(self) -> None:
        self.db = mock.MagicMock()

    def _patch(self, stack):
        # FR-E3: process_asset 는 pipeline_steps(ps) 내부 seam(set_status·finalize_asset 등)을 호출 → ps 에서 patch.
        return {
            "set_status": stack.enter_context(mock.patch.object(ps, "set_status")),
            "record_classification": stack.enter_context(mock.patch.object(ps, "record_classification")),
            "validate_ext_meta": stack.enter_context(mock.patch.object(ps, "validate_ext_meta")),
            "finalize_asset": stack.enter_context(mock.patch.object(ps, "finalize_asset")),
            "record_lineage": stack.enter_context(mock.patch.object(ps, "record_lineage")),
        }

    def test_registered_transition_with_inline_index(self) -> None:
        os_index = mock.MagicMock()
        with contextlib.ExitStack() as stack:
            m = self._patch(stack)
            outcome = ri.process_asset(
                _AID, db=self.db, fs_path="/d/a.txt", modality="txt", domain="general",
                classify_fn=lambda p, mod: _cls(), extract_fn=lambda ctx: AssetRecord(),
                settings=object(), os_index=os_index,
            )
        self.assertEqual(outcome, "registered")
        # routing→classifying→extracting 3회 전이(registered 는 finalize_asset 내부, 여기선 mock)
        self.assertEqual(m["set_status"].call_count, 3)
        m["record_classification"].assert_called_once()
        m["finalize_asset"].assert_called_once()
        # 인라인 색인이 해당 asset_id 로 finalize 직후 호출됨
        os_index.assert_called_once_with(_AID)
        # 단계별 lineage activity 가 순서대로 기록됨
        acts = [c.kwargs["activity"] for c in m["record_lineage"].call_args_list]
        self.assertEqual(
            acts,
            ["ingest.routing.v1", "ingest.classifying.v1", "ingest.classified.v1",
             "ingest.extracting.v1", "ingest.registered.v1"],
        )

    def test_medical_signature_deferred_no_extract_no_index(self) -> None:
        os_index = mock.MagicMock()
        called = {"extract": False}

        def extract(ctx):
            called["extract"] = True
            return AssetRecord()

        with contextlib.ExitStack() as stack:
            m = self._patch(stack)
            outcome = ri.process_asset(
                _AID, db=self.db, fs_path="/d/scan.dcm", modality="unknown", domain="general",
                classify_fn=lambda p, mod: _cls(label="medical", stage=1, conf=1.0, sig="dicom"),
                extract_fn=extract, settings=object(), os_index=os_index,
            )
        self.assertEqual(outcome, "deferred")
        self.assertFalse(called["extract"])          # 추출기 미호출(계획적 보류)
        m["finalize_asset"].assert_not_called()
        os_index.assert_not_called()                  # deferred 는 색인 안 함
        # routing→classifying→deferred 전이(extracting 없음)
        self.assertEqual(m["set_status"].call_count, 3)

    def test_os_index_optional_defaults_to_noop(self) -> None:
        # os_index 미주입이어도(배치 외 호출) 예외 없이 registered 까지 완주 — 주입 seam 안전 기본값.
        with contextlib.ExitStack() as stack:
            self._patch(stack)
            outcome = ri.process_asset(
                _AID, db=self.db, fs_path="/d/a.txt", modality="txt", domain="general",
                classify_fn=lambda p, mod: _cls(), extract_fn=lambda ctx: AssetRecord(),
                settings=object(),
            )
        self.assertEqual(outcome, "registered")


# ── run_ingest 동작 불변 봉인 ─────────────────────────────────────────────────


class TestRunIngestSplitSeal(unittest.TestCase):
    """분할된 run_ingest CLI 가 분할 전과 동일한 연산 시퀀스를 내는지 봉인한다(바이트 동일)."""

    def _record_seq(self, stack, seq, *, route, dup=None):
        """run_ingest 의존 seam 을 순서 기록 side_effect 로 patch."""
        stack.enter_context(mock.patch.object(ps, "route_file", return_value=route))
        stack.enter_context(mock.patch.object(ps, "file_hash_and_size", return_value=("h0", 10)))
        stack.enter_context(
            mock.patch.object(
                ps, "find_registered_asset_by_hash",
                side_effect=lambda c, h: (seq.append("dedup"), dup)[1],
            )
        )
        stack.enter_context(
            mock.patch.object(ps, "create_asset", side_effect=lambda c, **k: (seq.append("create_asset"), _AID)[1])
        )
        stack.enter_context(
            mock.patch.object(
                ps, "set_status",
                side_effect=lambda c, aid, tgt, **k: seq.append(f"set_status:{getattr(tgt, 'value', tgt)}"),
            )
        )
        stack.enter_context(
            mock.patch.object(
                ps, "record_lineage", side_effect=lambda c, aid, *, activity, **k: seq.append(f"lineage:{activity}")
            )
        )
        stack.enter_context(
            mock.patch.object(ps, "record_classification", side_effect=lambda c, aid, cls: seq.append("record_classification"))
        )
        stack.enter_context(
            mock.patch.object(ps, "validate_ext_meta", side_effect=lambda c, d, ext: seq.append("validate_ext_meta"))
        )
        stack.enter_context(
            mock.patch.object(ps, "finalize_asset", side_effect=lambda c, aid, rec: seq.append("finalize_asset"))
        )
        stack.enter_context(mock.patch.object(ri, "mark_failed", side_effect=lambda c, aid, reason: seq.append("mark_failed")))
        spy_os = mock.MagicMock(side_effect=lambda aid: seq.append("os_index"))
        stack.enter_context(mock.patch.object(ri, "_make_opensearch_indexer", return_value=spy_os))

    def test_run_ingest_split_preserves_operation_sequence(self) -> None:
        seq: list[str] = []
        with contextlib.ExitStack() as stack:
            self._record_seq(stack, seq, route=_route())
            res = ri.run_ingest(
                ["/d/a.txt"], db=mock.MagicMock(),
                extract_fn=lambda ctx: AssetRecord(),
                classify_fn=lambda p, m: _cls(), settings=object(),
            )
        self.assertEqual(res["registered"], [_AID])
        self.assertEqual(
            seq,
            [
                "dedup", "create_asset", "lineage:ingest.received.v1",
                "set_status:routing", "lineage:ingest.routing.v1",
                "set_status:classifying", "lineage:ingest.classifying.v1",
                "record_classification", "lineage:ingest.classified.v1",
                "set_status:extracting", "lineage:ingest.extracting.v1",
                "validate_ext_meta", "finalize_asset", "lineage:ingest.registered.v1",
                "os_index",
            ],
        )

    def test_run_ingest_split_deferred_sequence(self) -> None:
        seq: list[str] = []
        with contextlib.ExitStack() as stack:
            self._record_seq(stack, seq, route=_route(modality="unknown"))
            res = ri.run_ingest(
                ["/d/scan.dcm"], db=mock.MagicMock(),
                extract_fn=lambda ctx: AssetRecord(),
                classify_fn=lambda p, m: _cls(label="medical", stage=1, conf=1.0, sig="dicom"),
                settings=object(),
            )
        self.assertEqual(res["deferred"], [_AID])
        self.assertEqual(res["registered"], [])
        self.assertEqual(
            seq,
            [
                "dedup", "create_asset", "lineage:ingest.received.v1",
                "set_status:routing", "lineage:ingest.routing.v1",
                "set_status:classifying", "lineage:ingest.classifying.v1",
                "record_classification", "lineage:ingest.classified.v1",
                "set_status:deferred", "lineage:ingest.deferred.v1",
            ],
        )

    def test_run_ingest_split_duplicate_skips_before_create(self) -> None:
        seq: list[str] = []
        with contextlib.ExitStack() as stack:
            self._record_seq(stack, seq, route=_route(), dup=_EXISTING)
            res = ri.run_ingest(
                ["/d/a.txt"], db=mock.MagicMock(),
                extract_fn=lambda ctx: AssetRecord(),
                classify_fn=lambda p, m: _cls(), settings=object(),
            )
        self.assertEqual(res["registered"], [])
        self.assertEqual(len(res["skipped"]), 1)
        self.assertTrue(res["skipped"][0][1].startswith(ri.REASON_DUPLICATE))
        # dedup 만 일어나고 create_asset 이후 어떤 전이도 없다(중복은 자산 미생성)
        self.assertEqual(seq, ["dedup"])


# ── FR-011: Airflow import 0 ─────────────────────────────────────────────────


class TestNoAirflowImport(unittest.TestCase):
    """분할 모듈은 Airflow 없이 단위 가능해야 한다 — import 만으로 airflow 가 들어오지 않음."""

    def test_run_ingest_module_does_not_import_airflow(self) -> None:
        # 격리된 서브프로세스에서 run_ingest 와 분할 함수를 import 한 뒤 airflow 미적재를 단언.
        code = (
            "import sys\n"
            "import processing.app.run_ingest as ri\n"
            "assert hasattr(ri, 'collect_file') and hasattr(ri, 'process_asset')\n"
            "assert 'airflow' not in sys.modules, 'run_ingest import 가 airflow 를 끌어옴'\n"
            "print('ok')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()
