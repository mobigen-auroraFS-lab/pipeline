"""T3-1 관계 후보 검색 단위 테스트 (asset_candidates). DB·LLM 불필요.

엣지 영속화는 단계 C에서 graph_edge 로 이관됨(graph_persist) — tests/test_graph_persist.py 참조.

[008 그룹3] run_relations 의 cross_asset 슬롯 resolve 전환(T009~T012) 단위 테스트를 함께 둔다 —
일반 팩 슬롯 경유 결과가 기존 propose_relations_for_asset 위임과 **동치**(FR-001·SC-001),
미배선 cross_asset 전략 가드(FR-002), 도메인 폴백(FR-003) 검증. DB·LLM 불필요(seam 주입).
"""
from __future__ import annotations

import math
import os
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

from src.relations.asset_candidates import _channels_param, find_embedding_candidates
from src.relations.llm_propose import parse_and_normalize_edges

_SRC = "018f0000-0000-7000-8000-000000000001"
_T1 = "018f0000-0000-7000-8000-000000000007"
_T2 = "018f0000-0000-7000-8000-000000000008"


def _edge(confidence: object) -> dict:
    """confidence 값만 바꿔 가며 검증할 최소 LLM 엣지 dict."""
    return {
        "target_media_item_id": _T1,
        "relation_type_code": "same_series",
        "confidence": confidence,
        "reason": "테스트",
    }


class TestConfidenceClamp(unittest.TestCase):
    """T004 [US4, FR-010, #2] — confidence 를 [0,1] 로 클램프하고, 비정상값은 결정적 0.0."""

    def _conf(self, raw: object) -> float:
        out = parse_and_normalize_edges({"edges": [_edge(raw)]})
        self.assertEqual(len(out), 1)
        return out[0]["confidence"]

    def test_above_one_clamped_to_one(self) -> None:
        # 1.5 → 1.0: 자동승인 임계 판정이 1.0 을 넘지 않도록 상한 클램프.
        self.assertEqual(self._conf(1.5), 1.0)

    def test_below_zero_clamped_to_zero(self) -> None:
        # -0.3 → 0.0: 음수 confidence 를 하한 0.0 으로 클램프.
        self.assertEqual(self._conf(-0.3), 0.0)

    def test_in_range_preserved(self) -> None:
        # 정상 범위 값은 그대로 보존.
        self.assertEqual(self._conf(0.42), 0.42)

    def test_nan_falls_back_to_zero(self) -> None:
        # NaN → 0.0: 비교 불가능한 값은 결정적 기본값으로(헌법 3조).
        self.assertEqual(self._conf(float("nan")), 0.0)

    def test_unparsable_string_falls_back_to_zero(self) -> None:
        # 파싱 불가 문자열 → 0.0.
        self.assertEqual(self._conf("높음"), 0.0)

    def test_missing_falls_back_to_zero(self) -> None:
        # confidence 키 누락 → 0.0.
        edge = _edge(0.5)
        del edge["confidence"]
        out = parse_and_normalize_edges({"edges": [edge]})
        self.assertEqual(out[0]["confidence"], 0.0)

    def test_numeric_string_in_range_parsed(self) -> None:
        # 숫자 문자열도 float 로 파싱되어 보존.
        self.assertEqual(self._conf("0.8"), 0.8)


def _mock_conn(rows):
    conn = mock.MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = rows
    return conn, cur


class TestChannelsParam(unittest.TestCase):
    def test_st_clip_both(self) -> None:
        self.assertEqual(_channels_param("st"), ["st"])
        self.assertEqual(_channels_param("clip"), ["clip"])
        self.assertEqual(set(_channels_param("both")), {"st", "clip"})


class TestFindCandidates(unittest.TestCase):
    def test_maps_rows_to_str_id_candidates(self) -> None:
        rows = [
            {"id": uuid.UUID(_T1), "file_uri": "/d/a.png", "media_type": "image", "emb_score": 0.91, "summary": "요약A"},
            {"id": uuid.UUID(_T2), "file_uri": "/d/b.txt", "media_type": "txt", "emb_score": 0.42, "summary": None},
        ]
        conn, cur = _mock_conn(rows)
        out = find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5, embedding_kind="both")
        self.assertEqual([c["id"] for c in out], [_T1, _T2])
        self.assertTrue(all(isinstance(c["id"], str) for c in out))
        self.assertEqual(out[1]["summary"], "")  # None → ''
        params = cur.execute.call_args.args[1]
        self.assertEqual(params[0], _SRC)
        self.assertEqual(set(params[1]), {"st", "clip"})
        self.assertEqual(params[3], 0.0)  # min_sim 기본값
        self.assertEqual(params[4], 5)    # top_k


class TestDeterministicOrdering(unittest.TestCase):
    """T003 [US2, FR-007] — best_sim 동률 시 후보 id ASC tiebreaker 로 결정적 정렬(헌법 3조)."""

    def test_order_by_has_id_tiebreaker(self) -> None:
        # ORDER BY 가 best_sim DESC 만이면 동률 후보 순서가 비결정적 — id ASC 보조 정렬 필수.
        conn, cur = _mock_conn([])
        find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5)
        sql = cur.execute.call_args.args[0]
        # 정규화: 공백을 단일화해 줄바꿈·들여쓰기에 무관하게 부분 문자열 검사.
        norm = " ".join(sql.split())
        self.assertIn("ORDER BY p.best_sim DESC, p.id ASC", norm)

    def test_id_tiebreaker_is_after_best_sim(self) -> None:
        # best_sim 가 1순위, id 가 2순위여야 유사도 우선순위가 보존된다.
        conn, cur = _mock_conn([])
        find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5)
        norm = " ".join(cur.execute.call_args.args[0].split())
        order_clause = norm[norm.index("ORDER BY"):]
        self.assertLess(order_clause.index("best_sim"), order_clause.index("p.id"))


class TestZeroNormGuard(unittest.TestCase):
    """034 — 영노름(vector_norm=0) 임베딩을 후보 비교에서 제외(NaN 코사인 오염 차단).

    영벡터 타깃과의 코사인 1-(a<=>b)=NaN → PG가 HAVING NaN>=min_sim TRUE·ORDER BY DESC 최댓값
    정렬로 top_k 점령 → 진짜 후보 축출. 소스·타깃 모두 vector_norm>0 으로 영벡터를 비교에서 뺀다.
    """

    def test_sql_excludes_zero_norm_source_and_target(self) -> None:
        conn, cur = _mock_conn([])
        find_embedding_candidates(conn, source_asset_id=_SRC, top_k=5)
        norm = " ".join(cur.execute.call_args.args[0].split())
        self.assertIn("vector_norm(embedding) > 0", norm)     # src_vecs 소스 제외
        self.assertIn("vector_norm(ae.embedding) > 0", norm)  # cand 타깃 제외


class TestRelationDefaultEmbeddingKind(unittest.TestCase):
    """036 — 관계 후보 기본 embedding_kind = bge-only('st'). 4모달리티 캡션 공유 단일 공간·척도 일관."""

    def test_propose_default_is_st(self) -> None:
        import inspect

        from src.relations.asset_entry import propose_relations_for_asset
        sig = inspect.signature(propose_relations_for_asset)
        self.assertEqual(sig.parameters["embedding_kind"].default, "st")

    def test_runner_default_is_st(self) -> None:
        import inspect

        from processing.app.run_relations import run_relations
        sig = inspect.signature(run_relations)
        self.assertEqual(sig.parameters["embedding_kind"].default, "st")


# ── [008 그룹3] T009~T012: run_relations cross_asset 슬롯 resolve 전환 ──────────
from processing.pipeline.packs import GENERAL_PACK, MEDICAL_PACK, for_domain  # noqa: E402
from processing.pipeline.registry import StrategyRegistry  # noqa: E402


# ── [016 G3] T011~T012: SAMPLE_PACK + 샘플 4전략 등록 ─────────────────────────
class TestSamplePackRegistration(unittest.TestCase):
    """T011·T012 [016 US1] — 샘플 도메인 팩이 등록되고 cross_asset 슬롯이 샘플 전략으로 resolve.

    목적: 008 이 만든 cross_asset 슬롯 resolve seam 을 **일반과 다른 전략**(else 분기로 가는
    비일반 팩)으로 처음 실제 배선한다. 헌법 4조: 갈림은 for_domain(팩 선택) + cross_asset
    데이터 비교로만 — 도메인명 코드 분기 없음.
    """

    def _registry_with_defaults(self) -> StrategyRegistry:
        from processing.pipeline.builtins import register_defaults
        reg = StrategyRegistry()
        register_defaults(reg)
        return reg

    def test_for_domain_sample_returns_sample_pack(self) -> None:
        # for_domain('sample') → SAMPLE_PACK(name='sample'). _PACKS 에 등록돼 폴백이 아니어야 한다.
        from processing.pipeline.packs import SAMPLE_PACK, for_domain
        pack = for_domain("sample")
        self.assertIs(pack, SAMPLE_PACK)
        self.assertEqual(pack.name, "sample")

    def test_sample_pack_cross_asset_differs_from_general(self) -> None:
        # else(비일반) 분기로 가려면 cross_asset 묶음이 GENERAL_PACK 과 달라야 한다(핵심 라우팅 조건).
        from processing.pipeline.packs import GENERAL_PACK, SAMPLE_PACK
        self.assertNotEqual(SAMPLE_PACK.cross_asset, GENERAL_PACK.cross_asset)
        # decide 포함 4슬롯 모두 샘플 전략명을 가리킨다.
        self.assertEqual(
            SAMPLE_PACK.cross_asset,
            {
                "candidates": "sample_candidates",
                "score": "sample_score",
                "decide": "sample_decide",
                "persist_edges": "sample_graph_upsert",
            },
        )

    def test_sample_pack_per_asset_same_as_general(self) -> None:
        # per_asset 은 일반과 동일(샘플은 cross_asset 만 데모로 갈림).
        from processing.pipeline.packs import GENERAL_PACK, SAMPLE_PACK
        self.assertEqual(SAMPLE_PACK.per_asset, GENERAL_PACK.per_asset)

    def test_sample_strategies_registered_and_resolvable(self) -> None:
        # 샘플 4전략(슬롯명→함수)이 레지스트리에서 resolve 되고, 일반과 다른 Callable 이어야 한다.
        from processing.pipeline import sample_strategies as ss
        reg = self._registry_with_defaults()
        # 슬롯명(persist_edges 의 등록명은 'sample_graph_upsert', 함수는 sample_persist_edges).
        self.assertIs(reg.resolve("candidates", "sample_candidates"), ss.sample_candidates)
        self.assertIs(reg.resolve("score", "sample_score"), ss.sample_score)
        self.assertIs(reg.resolve("decide", "sample_decide"), ss.sample_decide)
        self.assertIs(reg.resolve("persist_edges", "sample_graph_upsert"), ss.sample_persist_edges)
        # 일반 전략과 다른 Callable.
        self.assertIsNot(
            reg.resolve("candidates", "sample_candidates"),
            reg.resolve("candidates", "embedding_topk"),
        )
        self.assertIsNot(
            reg.resolve("persist_edges", "sample_graph_upsert"),
            reg.resolve("persist_edges", "graph_upsert"),
        )

    def test_sample_candidates_tagged_deterministic(self) -> None:
        # 결정성 태그(헌법 3조) — candidates/score/decide 는 deterministic.
        reg = self._registry_with_defaults()
        self.assertIn("deterministic", reg.tags("candidates", "sample_candidates"))
        self.assertIn("deterministic", reg.tags("score", "sample_score"))
        self.assertIn("deterministic", reg.tags("decide", "sample_decide"))


class TestResolveCrossAssetSlots(unittest.TestCase):
    """T009·T010 [US1, FR-001·FR-002] — 팩의 cross_asset 슬롯을 레지스트리에서 resolve.

    슬롯 resolve 는 "의료 ER(단계 D)이 코드 분기 없이 끼워질 자리 만들기"이며, 일반 팩의
    슬롯이 모두 등록돼 있는지(미배선 가드)를 진입부에서 검증하는 chokepoint 다.
    """

    def _registry_with_defaults(self) -> StrategyRegistry:
        from processing.pipeline.builtins import register_defaults
        reg = StrategyRegistry()
        register_defaults(reg)
        return reg

    def test_general_pack_slots_all_resolvable(self) -> None:
        # 일반 팩의 cross_asset 슬롯(candidates/score/persist_edges)이 전부 등록돼 있어야 한다.
        from processing.app.run_relations import _resolve_cross_asset_slots
        reg = self._registry_with_defaults()
        resolved = _resolve_cross_asset_slots(GENERAL_PACK, registry=reg)
        # candidates/score/persist_edges 는 Callable 로 resolve 된다.
        for slot in ("candidates", "score", "persist_edges"):
            self.assertTrue(callable(resolved[slot]), f"슬롯 {slot} 미해결")

    def test_unwired_strategy_raises_notimplemented(self) -> None:
        # 미등록 cross_asset 전략(의료 단계 D 전 상태)을 가리키는 팩은 명시적 오류로 차단(FR-002).
        from processing.app.run_relations import _resolve_cross_asset_slots
        from processing.pipeline.packs import DomainPack
        reg = self._registry_with_defaults()
        unwired = DomainPack(
            name="medical",
            per_asset=dict(MEDICAL_PACK.per_asset),
            cross_asset={  # candidates 가 미등록 전략을 가리킴
                "candidates": "blocking_5keys",
                "score": "llm_propose",
                "decide": "confidence",
                "persist_edges": "graph_upsert",
            },
            policy="medical_strict",
        )
        with self.assertRaises(NotImplementedError):
            _resolve_cross_asset_slots(unwired, registry=reg)


class _FakeDB:
    """run_relations 가 _fetch_domain_label 로 호출하는 db.execute_in_transaction 만 흉내낸다.

    실제 트랜잭션 대신, domain_label 조회용 콜백에 자산별 라벨을 돌려주는 가짜 커서를 주입한다.
    """

    def __init__(self, labels: dict[str, str]) -> None:
        self._labels = labels
        self._current: str | None = None

    def execute_in_transaction(self, fn, *, idempotent: bool = True):
        # _fetch_domain_label 의 _run(conn) 만 사용 — domain_label SELECT 1행을 반환하는 conn mock.
        conn = mock.MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value

        def _execute(sql, params=None):
            # _fetch_domain_label 의 domain_label SELECT → 라벨 반환.
            # 009 병합: run_relations 가 자산별 _record_resolution(_fetch_attempts SELECT
            # relation_resolution / upsert)을 호출하므로, 큐 경로는 행 없음(None)→attempts 0 으로 흘린다.
            if "domain_label" in sql:
                aid = params[0]
                label = self._labels.get(aid)
                cur.fetchone.return_value = (label,) if label is not None else None
            else:
                cur.fetchone.return_value = None

        cur.execute.side_effect = _execute
        return fn(conn)


class TestRunRelationsSlotRouting(unittest.TestCase):
    """T009·T010·T011 [US1] — run_relations 가 도메인 팩 슬롯 resolve 로 라우팅하되,
    일반 경로는 기존 propose_relations_for_asset 위임과 **동치**(FR-001)."""

    _A_GEN = "018f0000-0000-7000-8000-000000000011"
    _A_MED = "018f0000-0000-7000-8000-000000000012"
    _A_REVIEW = "018f0000-0000-7000-8000-000000000013"

    def test_general_delegates_to_propose_with_same_args(self) -> None:
        # FR-001/SC-001: 일반 자산은 기존 propose_relations_for_asset 에 동일 인자로 위임되어
        # 결과(엣지 수치)가 슬롯 미경유 기존 경로와 동일해야 한다.
        # T015(016): 일반 경로는 제네릭 러너를 **타지 않음**을 함께 단언해 동작 불변을 못박는다.
        from processing.app import run_relations as rr
        db = _FakeDB({self._A_GEN: "general"})
        with mock.patch.object(
            rr, "propose_relations_for_asset", return_value=(1, 2, 3, 4)
        ) as m, mock.patch.object(rr, "run_cross_asset") as runner:
            result = rr.run_relations(
                [self._A_GEN], db=db, top_k=7, embedding_kind="st", max_attempts=3
            )
        m.assert_called_once_with(db, self._A_GEN, top_k=7, embedding_kind="st")
        # 일반 경로는 러너(016)를 절대 호출하지 않는다 — propose 위임 경로 그대로(회귀 0).
        runner.assert_not_called()
        # done 에 (asset_id, edges_upserted, edges_skipped) = (aid, 3, 4) 가 그대로 실려야 한다.
        self.assertEqual(result["done"], [(self._A_GEN, 3, 4)])
        self.assertEqual(result["failed"], [])

    def test_review_label_falls_back_to_general(self) -> None:
        # FR-003: review/미지정 라벨은 일반 팩으로 보수적 폴백 → propose 위임 경로를 탄다.
        from processing.app import run_relations as rr
        db = _FakeDB({self._A_REVIEW: "review"})
        with mock.patch.object(
            rr, "propose_relations_for_asset", return_value=(0, 0, 0, 0)
        ) as m:
            result = rr.run_relations([self._A_REVIEW], db=db, max_attempts=3)
        m.assert_called_once()
        self.assertEqual(result["failed"], [])
        self.assertEqual(len(result["done"]), 1)

    def test_unspecified_label_falls_back_to_general(self) -> None:
        # FR-003: domain_label NULL/자산 미존재 → 'general' 폴백.
        from processing.app import run_relations as rr
        db = _FakeDB({})  # 라벨 없음 → fetchone None → 'general'
        with mock.patch.object(
            rr, "propose_relations_for_asset", return_value=(0, 0, 1, 0)
        ) as m:
            result = rr.run_relations([self._A_GEN], db=db, max_attempts=3)
        m.assert_called_once()
        self.assertEqual(result["done"], [(self._A_GEN, 1, 0)])

    def test_unwired_medical_isolated_as_failed_batch_continues(self) -> None:
        # FR-002: 의료 cross_asset 전략 미배선이면 그 자산만 failed 격리, 일반 자산은 계속 처리.
        from processing.app import run_relations as rr
        db = _FakeDB({self._A_MED: "medical", self._A_GEN: "general"})

        # 의료 팩이 미등록 전략을 가리키도록 for_domain 을 패치(단계 D 전 미배선 시뮬레이션).
        from processing.pipeline.packs import DomainPack
        unwired_medical = DomainPack(
            name="medical",
            per_asset=dict(MEDICAL_PACK.per_asset),
            cross_asset={
                "candidates": "blocking_5keys",  # 미등록
                "score": "llm_propose",
                "decide": "confidence",
                "persist_edges": "graph_upsert",
            },
            policy="medical_strict",
        )

        def _for_domain(label: str) -> DomainPack:
            return unwired_medical if label == "medical" else GENERAL_PACK

        with mock.patch.object(rr, "for_domain", side_effect=_for_domain), \
             mock.patch.object(
                 rr, "propose_relations_for_asset", return_value=(0, 0, 5, 1)
             ) as m:
            result = rr.run_relations([self._A_MED, self._A_GEN], db=db, max_attempts=3)

        # 의료는 failed 로 격리, 일반은 done. 배치는 중단되지 않는다.
        failed_ids = [aid for aid, _ in result["failed"]]
        self.assertIn(self._A_MED, failed_ids)
        self.assertEqual(result["done"], [(self._A_GEN, 5, 1)])
        # 일반 자산에 대해서만 propose 위임이 일어난다(의료는 resolve 단계에서 차단).
        m.assert_called_once_with(db, self._A_GEN, top_k=None, embedding_kind="st")


# ── [016 G3] T013~T015: 비일반 팩 → 제네릭 러너 라우팅 + _domain_fn seam ──────
class _RunnerFakeDB:
    """run_relations 의 두 종류 execute_in_transaction 호출을 모두 흉내내는 가짜 DB.

    1) _fetch_domain_label / _record_resolution(_fetch_attempts·upsert) — SELECT/None 흘림.
    2) 러너 경로 ``execute_in_transaction(lambda conn: run_cross_asset(...), idempotent=False)``
       — fn(conn) 을 그대로 실행해 콜백 안의 run_cross_asset 가 fake conn 으로 돌게 한다.

    _domain_fn seam 을 주입해 라벨 조회는 우회하므로(테스트 단순화), 이 fake conn 은 큐 경로의
    SELECT(행 없음)와 러너 콜백 실행만 담당한다.
    """

    def execute_in_transaction(self, fn, *, idempotent: bool = True):
        conn = mock.MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        cur.fetchone.return_value = None   # 큐 경로 SELECT → attempts 0
        cur.fetchall.return_value = []
        return fn(conn)


class TestRunRelationsRunnerRouting(unittest.TestCase):
    """T013·T014 [016 US1] — 비일반(샘플) 팩은 NotImplementedError 가 아니라 제네릭 러너로 라우팅.

    헌법 4조: 라우팅 갈림은 _domain_fn(라벨) → for_domain(팩 선택) + cross_asset 데이터 비교로만.
    Acceptance 2(미등록 전략 팩 → NotImplementedError → failed)는 유지된다.
    """

    _A_SAMPLE = "018f0000-0000-7000-8000-000000000015"
    _A_GEN = "018f0000-0000-7000-8000-000000000016"
    _A_MED = "018f0000-0000-7000-8000-000000000017"

    def test_sample_pack_routes_to_runner_not_notimplemented(self) -> None:
        # 비일반 샘플 팩 → run_cross_asset 호출(NotImplementedError 아님). _domain_fn seam 으로 'sample' 주입.
        from processing.app import run_relations as rr
        db = _RunnerFakeDB()
        with mock.patch.object(rr, "run_cross_asset", return_value=3) as runner:
            result = rr.run_relations(
                [self._A_SAMPLE],
                db=db,
                max_attempts=3,
                _domain_fn=lambda _db, _aid: "sample",
            )
        # 러너가 정확히 한 번, source_asset_id=해당 자산으로 호출됐다.
        runner.assert_called_once()
        call = runner.call_args
        # run_cross_asset(resolved, conn, source_asset_id)
        self.assertEqual(call.args[2], self._A_SAMPLE)
        resolved = call.args[0]
        # 4슬롯(decide 포함)이 모두 resolve 돼 러너에 전달됐다.
        self.assertEqual(set(resolved.keys()), {"candidates", "score", "decide", "persist_edges"})
        # 적재 엣지 수(러너 반환=3)가 done 에 실린다. 카탈로그 카운트는 N/A(0).
        self.assertEqual(result["done"], [(self._A_SAMPLE, 3, 0)])
        self.assertEqual(result["failed"], [])

    def test_sample_pack_resolves_actual_sample_strategies(self) -> None:
        # 러너에 전달된 4 Callable 이 실제 샘플 전략(sample_strategies)이어야 한다(실배선 증명).
        from processing.app import run_relations as rr
        from processing.pipeline import sample_strategies as ss
        db = _RunnerFakeDB()
        captured: dict = {}

        def _spy(resolved, conn, source_asset_id):
            captured.update(resolved)
            return 0

        with mock.patch.object(rr, "run_cross_asset", side_effect=_spy):
            rr.run_relations(
                [self._A_SAMPLE], db=db, max_attempts=3,
                _domain_fn=lambda _db, _aid: "sample",
            )
        self.assertIs(captured["candidates"], ss.sample_candidates)
        self.assertIs(captured["score"], ss.sample_score)
        self.assertIs(captured["decide"], ss.sample_decide)
        self.assertIs(captured["persist_edges"], ss.sample_persist_edges)

    def test_domain_fn_seam_exists_with_none_default(self) -> None:
        # seam 파라미터가 존재하고 기본값이 None(미주입) — 미주입 시 모듈 _fetch_domain_label 로 폴백.
        import inspect

        from processing.app import run_relations as rr
        sig = inspect.signature(rr.run_relations)
        self.assertIn("_domain_fn", sig.parameters)
        self.assertIsNone(sig.parameters["_domain_fn"].default)

    def test_default_seam_honors_module_patch_of_fetch_domain_label(self) -> None:
        # 회귀 0 핵심: _domain_fn 미주입 시 mock.patch.object(rr,"_fetch_domain_label",...) 가
        # 그대로 적용돼야 한다(def 기본값 직접 바인딩이면 패치가 무시돼 기존 테스트가 깨짐).
        from processing.app import run_relations as rr
        db = _RunnerFakeDB()
        with mock.patch.object(rr, "_fetch_domain_label", return_value="general") as fdl, \
             mock.patch.object(
                 rr, "propose_relations_for_asset", return_value=(0, 0, 1, 0)
             ) as prop:
            result = rr.run_relations([self._A_GEN], db=db, max_attempts=3)
        fdl.assert_called_once_with(db, self._A_GEN)
        prop.assert_called_once()
        self.assertEqual(result["done"], [(self._A_GEN, 1, 0)])

    def test_unwired_pack_still_notimplemented_failed(self) -> None:
        # Acceptance 2 유지: 미등록 전략 팩(blocking_5keys)은 여전히 NotImplementedError → failed.
        from processing.app import run_relations as rr
        from processing.pipeline.packs import DomainPack
        db = _RunnerFakeDB()
        unwired = DomainPack(
            name="medical",
            per_asset=dict(MEDICAL_PACK.per_asset),
            cross_asset={
                "candidates": "blocking_5keys",  # 미등록
                "score": "llm_propose",
                "decide": "confidence",
                "persist_edges": "graph_upsert",
            },
            policy="medical_strict",
        )
        with mock.patch.object(rr, "for_domain", side_effect=lambda _l: unwired), \
             mock.patch.object(rr, "run_cross_asset") as runner:
            result = rr.run_relations(
                [self._A_MED], db=db, max_attempts=3,
                _domain_fn=lambda _db, _aid: "medical",
            )
        # 러너는 호출되지 않고(resolve 단계에서 차단), failed 로 격리된다.
        runner.assert_not_called()
        failed_ids = [aid for aid, _ in result["failed"]]
        self.assertIn(self._A_MED, failed_ids)
        self.assertEqual(result["done"], [])

    def test_sample_and_general_mixed_batch_routes_each(self) -> None:
        # 혼합 배치: 샘플은 러너, 일반은 propose 위임 — 라우팅이 자산별로 정확히 갈린다.
        from processing.app import run_relations as rr
        db = _RunnerFakeDB()
        labels = {self._A_SAMPLE: "sample", self._A_GEN: "general"}
        with mock.patch.object(rr, "run_cross_asset", return_value=2) as runner, \
             mock.patch.object(
                 rr, "propose_relations_for_asset", return_value=(0, 0, 5, 1)
             ) as prop:
            result = rr.run_relations(
                [self._A_SAMPLE, self._A_GEN], db=db, max_attempts=3,
                _domain_fn=lambda _db, aid: labels[aid],
            )
        # 샘플 → 러너 1회, 일반 → propose 1회.
        runner.assert_called_once()
        prop.assert_called_once_with(db, self._A_GEN, top_k=None, embedding_kind="st")
        self.assertEqual(result["failed"], [])
        # done 순서: 샘플(러너 반환 2, 카탈로그 N/A 0) → 일반(propose 3·4 위치 → 5,1).
        self.assertEqual(
            result["done"], [(self._A_SAMPLE, 2, 0), (self._A_GEN, 5, 1)]
        )


# ── [008 그룹5] T017: cross-asset end-to-end 후보 흐름 통합 가드 ─────────────
class _SingleConnDB:
    """propose_relations_for_asset 의 execute_in_transaction(_run, ...) 만 흉내내는 가짜 DB.

    실 DB·트랜잭션 없이 _run(conn) 을 그대로 실행해, 그 안의 seam(후보 조회·LLM·엣지 upsert)을
    mock 으로 가로채 union→candidate_ids 흐름을 순수 단위로 검증하기 위한 최소 더블.
    """

    def execute_in_transaction(self, fn, *, idempotent: bool = True):
        return fn(mock.MagicMock())


# get_current_settings 가 요구하는 env 17개 없이도 순수 단위로 돌리기 위한 최소 cfg 더블.
# propose_relations_for_asset 이 읽는 4개 설정만 채운다(나머지는 본 경로에서 미사용).
import types  # noqa: E402

_FAKE_CFG = types.SimpleNamespace(relations=types.SimpleNamespace(
    top_k=10,
    min_sim=0.2,
    path_top_k=10,
    auto_approve_min=0.75,
    auto_approve_emb_min=0.0,  # 033: 무력 기본값(자동승인 emb 게이트 미적용)
))


class TestCrossAssetCandidateFlowIntegration(unittest.TestCase):
    """T017 [US4, FR-011, SC-007, #10] — cross-asset 경로 **통합** 관점 회귀 가드.

    기존 단위 테스트와 **중복되지 않는** end-to-end seam 흐름만 본다:
      - TestUnionCandidates(test_path_signal.py): union_candidates 순수 함수 자체 → 본 테스트는
        그 union 결과가 **propose_relations_for_asset 안에서 sync_graph_edges 의 allowed_target_ids
        (=candidate_ids)로 실제로 흐르는지**(FR-006 환각 화이트리스트 자동 확장)를 검증.
      - TestConfidenceClamp: parse_and_normalize_edges 단위 → 중복 안 함.
      - TestRunRelationsSlotRouting: 슬롯 라우팅·폴백 → 중복 안 함.
    DB·LLM 불필요(execute_in_transaction·후보 seam·LLM seam 전부 주입/mock).
    """

    def test_path_candidate_id_flows_into_allowed_target_ids(self) -> None:
        # FR-006/SC-004: min_sim 미달이라 임베딩 후보엔 없던 경로 신호 후보(_T2)가
        # union 되어 sync_graph_edges 의 allowed_target_ids(화이트리스트)에 자동 포함돼야 한다.
        from src.relations import asset_entry as ae

        emb_rows = [
            {"id": _T1, "file_uri": "/d/a.txt", "media_type": "txt",
             "emb_score": 0.83, "summary": ""},
        ]
        path_rows = [
            {"id": _T2, "file_uri": "/d/a_summary.txt", "media_type": "txt",
             "emb_score": 0.0, "summary": ""},  # 경로 신호 전용(min_sim 미달 가정)
        ]
        captured: dict = {}

        def _fake_sync(conn, *, source_asset_id, edges, allowed_target_ids, auto_approve_min,
                       target_emb_scores=None, auto_approve_emb_min=0.0,
                       collect=None):  # 033 FR-003 + 013 collect(계보 관계쌍) 신규 kwargs
            captured["allowed"] = allowed_target_ids
            captured["emb_scores"] = target_emb_scores
            return 1, 0

        with mock.patch.object(ae, "get_current_settings", return_value=_FAKE_CFG), \
             mock.patch.object(ae, "_fetch_source_row",
                               return_value={"fs_path": "/d/a.txt", "modality": "txt", "summary": ""}), \
             mock.patch.object(ae, "fetch_asset_topic",
                               return_value=[{"topic_ko": "음식·요리", "subtopic_ko": "라면"}]), \
             mock.patch.object(ae, "find_embedding_candidates", return_value=emb_rows), \
             mock.patch.object(ae, "find_path_signal_candidates", return_value=path_rows), \
             mock.patch.object(ae, "fetch_active_relation_kinds", return_value=[]), \
             mock.patch.object(ae, "register_new_relation_kinds", return_value=(0, 0)), \
             mock.patch.object(ae, "sync_graph_edges", side_effect=_fake_sync), \
             mock.patch.object(ae, "record_lineage", return_value=None):
            ae.propose_relations_for_asset(
                _SingleConnDB(), _SRC, top_k=5,
                llm_fn=lambda _prompt: {"edges": []},
            )

        allowed = captured["allowed"]
        # 임베딩 후보(_T1)와 경로 신호 후보(_T2)가 모두 화이트리스트에 들어가야 환각 차단을 통과.
        self.assertIn(_T1, allowed)
        self.assertIn(_T2, allowed)  # FR-006: 경로 신호 후보 자동 확장

    def test_overlap_keeps_embedding_emb_score_in_prompt_candidates(self) -> None:
        # C-3 회귀: 같은 asset_id 가 임베딩·경로 양쪽에 있으면 임베딩 실측 emb_score 가
        # 프롬프트 후보로 흐른다(0.0 sentinel 로 덮어쓰지 않음). build_prompt 에 넘어간
        # candidates 를 가로채 검증한다.
        from src.relations import asset_entry as ae

        emb_rows = [
            {"id": _T1, "file_uri": "/d/a.txt", "media_type": "txt",
             "emb_score": 0.77, "summary": ""},
        ]
        path_rows = [
            {"id": _T1, "file_uri": "/d/a.txt", "media_type": "txt",
             "emb_score": 0.0, "summary": ""},  # 같은 _T1 (겹침)
        ]
        captured: dict = {}

        # 066: build_relation_proposal_prompt 가 source_topic 키워드를 받게 되어 fake 시그니처도 확장.
        def _fake_prompt(*, source_summary, source_media_type, candidates,
                         relation_kinds_catalog, source_topic=None):
            captured["candidates"] = list(candidates)
            return "PROMPT"

        with mock.patch.object(ae, "get_current_settings", return_value=_FAKE_CFG), \
             mock.patch.object(ae, "_fetch_source_row",
                               return_value={"fs_path": "/d/a.txt", "modality": "txt", "summary": ""}), \
             mock.patch.object(ae, "fetch_asset_topic",
                               return_value=[{"topic_ko": "음식·요리", "subtopic_ko": "라면"}]), \
             mock.patch.object(ae, "find_embedding_candidates", return_value=emb_rows), \
             mock.patch.object(ae, "find_path_signal_candidates", return_value=path_rows), \
             mock.patch.object(ae, "fetch_active_relation_kinds", return_value=[]), \
             mock.patch.object(ae, "register_new_relation_kinds", return_value=(0, 0)), \
             mock.patch.object(ae, "build_relation_proposal_prompt", side_effect=_fake_prompt), \
             mock.patch.object(ae, "sync_graph_edges", return_value=(0, 0)), \
             mock.patch.object(ae, "record_lineage", return_value=None):
            ae.propose_relations_for_asset(
                _SingleConnDB(), _SRC, top_k=5,
                llm_fn=lambda _prompt: {"edges": []},
            )

        cands = captured["candidates"]
        # 겹친 _T1 은 1건만, emb_score 는 임베딩 실측값(0.77) 유지.
        t1 = [c for c in cands if c["id"] == _T1]
        self.assertEqual(len(t1), 1)
        self.assertEqual(t1[0]["emb_score"], 0.77)


@unittest.skipUnless(os.environ.get("RUN_DB_E2E") == "1", "실 DB 게이트(RUN_DB_E2E=1)")
class TestZeroNormGuardDB(unittest.TestCase):
    """034 SC-001/002 실 DB — 영노름 타깃이 후보에서 빠지고 결과 emb_score 에 NaN 이 없다.

    실 dev DB 에는 영노름 임베딩(st_bge 31·st 11)이 있어, 가드 전이면 정상 소스의 후보에도
    NaN 이 끼어 상위를 점령한다. 가드 후엔 영노름이 제외돼 결과 emb_score 가 전부 유한이어야 한다.
    """

    @classmethod
    def setUpClass(cls):
        from pathlib import Path

        from dotenv import load_dotenv

        from src.config.settings import init_settings
        from src.database.postgres_util import PostgresUtil

        load_dotenv(Path(__file__).resolve().parents[1] / ".env.dev", override=False)
        init_settings("dev")
        cls.db = PostgresUtil()
        cls.db.open_pool()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def _insert(self, conn, vec_str: str) -> str:
        from src.config.settings import active_embed_channel
        from src.database.ids import uuid7_str

        aid = uuid7_str()
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO asset (asset_id, fs_path, modality, status, file_hash, created_at) "
                "VALUES (%s, %s, 'text', 'registered', %s, now())",  # 053: canonical modality
                (aid, f"/tmp/zn_{aid}.txt", f"hash_{aid}"),
            )
            cur.execute(
                "INSERT INTO asset_embedding (asset_id, channel, chunk_index, embedding, model_name) "
                "VALUES (%s, %s, 0, %s::vector, 'test')",
                (aid, active_embed_channel(), vec_str),
            )
        return aid

    def test_zero_norm_excluded_and_no_nan(self) -> None:
        from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION

        d = FIX_EMBEDDING_DIMENSION
        normal = "[" + ",".join(["1"] + ["0"] * (d - 1)) + "]"  # 노름 1
        zero = "[" + ",".join(["0"] * d) + "]"                   # 영노름
        src_id = zero_id = None
        try:
            with self.db.transaction() as conn:
                src_id = self._insert(conn, normal)
                zero_id = self._insert(conn, zero)
            with self.db.transaction() as conn:
                cands = find_embedding_candidates(
                    conn, source_asset_id=src_id, top_k=50, embedding_kind="st", min_sim=0.2)
            scores = [c["emb_score"] for c in cands]
            ids = [c["id"] for c in cands]
            # SC-002: 결과 emb_score 에 NaN 없음(영노름 자산이 NaN 으로 끼지 않음).
            self.assertFalse(
                any(isinstance(s, float) and math.isnan(s) for s in scores),
                msg=f"후보 emb_score 에 NaN 존재: {scores[:5]}")
            # SC-001: 방금 넣은 영노름 타깃이 후보에 없음.
            self.assertNotIn(zero_id, ids)
        finally:
            # 자산별 독립 트랜잭션 — 한 자산 정리 실패가 다른 자산 정리를 롤백하지 않도록(리뷰 권고).
            for aid in (src_id, zero_id):
                if aid:
                    with self.db.transaction() as conn:
                        with conn.cursor() as cur:
                            cur.execute("DELETE FROM asset_embedding WHERE asset_id=%s", (aid,))
                            cur.execute("DELETE FROM asset WHERE asset_id=%s", (aid,))


class TestLineageRecordsEdgePairs(unittest.TestCase):
    """013: relations.proposed.v1 계보 generated 에 **관계 쌍**(상대자산·관계유형)을 기록(건수만 → 쌍 포함).

    DB·LLM 불필요 — propose_relations_for_asset 의 내부 seam 을 모두 주입한다.
    """

    def test_generated_includes_sorted_edge_pairs(self) -> None:
        from src.relations import asset_entry

        captured: dict = {}

        def _fake_sync(conn, *, source_asset_id, edges, allowed_target_ids, auto_approve_min,
                       target_emb_scores=None, auto_approve_emb_min=0.0, collect=None):
            # sync_graph_edges 가 upsert 된 쌍을 collect 에 적재(정렬 안 된 순서로) + (upserted, skipped) 반환.
            # 실제 함수 시그니처와 동일한 기본값을 둬, 향후 호출부 변경 시 fake 불일치를 드러낸다.
            if collect is not None:
                collect.append({"target_asset_id": "t-bbb", "kind_code": "same_domain",
                                "confidence": 0.7, "status": "proposed"})
                collect.append({"target_asset_id": "t-aaa", "kind_code": "duplicate_near",
                                "confidence": 0.95, "status": "active"})
            return 2, 1

        def _fake_record(conn, aid, *, activity, agent, generated, payload):
            captured["activity"] = activity
            captured["generated"] = generated

        class _DB:
            def execute_in_transaction(self, fn, *, idempotent):
                return fn(object())  # fake conn — seam 들이 전부 mock 이라 미사용

        cfg = SimpleNamespace(relations=SimpleNamespace(top_k=10, min_sim=0.2, path_top_k=5,
                              auto_approve_min=0.9, auto_approve_emb_min=0.0))
        with mock.patch.object(asset_entry, "get_current_settings", return_value=cfg), \
             mock.patch.object(asset_entry, "_fetch_source_row",
                               return_value={"summary": "s", "modality": "text"}), \
             mock.patch.object(asset_entry, "fetch_asset_topic",
                               return_value=[{"topic_ko": "음식·요리", "subtopic_ko": "라면"}]), \
             mock.patch.object(asset_entry, "find_embedding_candidates", return_value=[]), \
             mock.patch.object(asset_entry, "find_path_signal_candidates", return_value=[]), \
             mock.patch.object(asset_entry, "union_candidates", return_value=[]), \
             mock.patch.object(asset_entry, "target_emb_score_map", return_value={}), \
             mock.patch.object(asset_entry, "fetch_active_relation_kinds", return_value=[]), \
             mock.patch.object(asset_entry, "build_relation_proposal_prompt", return_value="p"), \
             mock.patch.object(asset_entry, "parse_and_normalize_edges", return_value=[]), \
             mock.patch.object(asset_entry, "register_new_relation_kinds", return_value=(0, 0)), \
             mock.patch.object(asset_entry, "sync_graph_edges", side_effect=_fake_sync), \
             mock.patch.object(asset_entry, "record_lineage", side_effect=_fake_record):
            asset_entry.propose_relations_for_asset(_DB(), _SRC, llm_fn=lambda p: {})

        self.assertEqual(captured["activity"], "relations.proposed.v1")
        gen = captured["generated"]
        # 카운트는 그대로(하위호환). 이 값들은 _fake_sync 고정 반환(2,1)이며,
        # 실제 upsert/skip 필터 로직 검증은 test_graph_persist.test_collect_captures_upserted_pairs_only 에 위임.
        self.assertEqual(gen["edges_upserted"], 2)
        self.assertEqual(gen["edges_skipped"], 1)
        # 신규: 관계 쌍 — target_asset_id ASC 로 결정적 정렬(헌법 3조)
        self.assertEqual([e["target_asset_id"] for e in gen["edges"]], ["t-aaa", "t-bbb"])
        self.assertEqual(gen["edges"][0]["kind_code"], "duplicate_near")
        self.assertEqual(gen["edges"][0]["status"], "active")


if __name__ == "__main__":
    unittest.main()
