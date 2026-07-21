"""샘플 도메인 팩 cross_asset 전략(G1) 단위 테스트.

검증 대상: ``src/pipeline/sample_strategies.py`` 의 4전략(candidates/score/decide/persist).
모두 **DB·LLM 불필요한 순수 단위**다 — conn 은 fake(MagicMock)로 주입한다.

헌법 불변식(반드시 챙김):
- 결정성(3조): 동일 입력 2회 동일 출력, 정렬 tiebreak = target asset_id 오름차순.
- 비학습(1조)·LLM 미사용(2조): 샘플 전략은 학습/네트워크/LLM seam 없이 동작한다.
- 스키마 변경 0(6조): persist 는 기존 graph_edge·relation_kind 어휘만 재사용한다.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from processing.pipeline.cross_types import Candidate, Decision, Evidence, ScoredPair


# ── 공용 fake conn 헬퍼 ─────────────────────────────────────────────────────
def _conn_returning(rows: list[dict]) -> MagicMock:
    """샘플 전략의 dict_row 커서 조회(fetchall)를 흉내내는 fake conn.

    sample_candidates/score 는 conn 으로 SQL 을 실행하므로, fake conn 이 fetchall() 로
    행을 돌려주게 설계해 DB 없이 단위 테스트한다.
    """
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.fetchall.return_value = rows
    conn.cursor.return_value = cur
    return conn


# ── T001: sample_candidates ────────────────────────────────────────────────
class TestSampleCandidates(unittest.TestCase):
    _SRC = "018f0000-0000-7000-8000-000000000001"

    def _rows(self) -> list[dict]:
        # sample_candidates 의 경로 마커 SQL 이 SELECT 하는 행 형태(id 만). 일부러 순서를 뒤섞어
        # sample_candidates 가 target asset_id 오름차순으로 정렬하는지 본다.
        return [
            {"id": "018f0000-0000-7000-8000-000000000017"},
            {"id": "018f0000-0000-7000-8000-000000000007"},
            {"id": "018f0000-0000-7000-8000-000000000012"},
        ]

    def test_returns_candidate_list_with_source_and_target(self) -> None:
        from processing.pipeline.sample_strategies import sample_candidates
        out = sample_candidates(_conn_returning(self._rows()), self._SRC)
        self.assertTrue(out)
        self.assertTrue(all(isinstance(c, Candidate) for c in out))
        for c in out:
            self.assertEqual(c.source_id, self._SRC)
            self.assertTrue(c.target_id)

    def test_sorted_by_target_id_ascending(self) -> None:
        # 결정성(헌법 3조): target asset_id 오름차순 고정.
        from processing.pipeline.sample_strategies import sample_candidates
        out = sample_candidates(_conn_returning(self._rows()), self._SRC)
        targets = [c.target_id for c in out]
        self.assertEqual(targets, sorted(targets))

    def test_deterministic_same_input_twice(self) -> None:
        from processing.pipeline.sample_strategies import sample_candidates
        out1 = sample_candidates(_conn_returning(self._rows()), self._SRC)
        out2 = sample_candidates(_conn_returning(self._rows()), self._SRC)
        self.assertEqual(out1, out2)

    def test_method_and_block_key_tagged(self) -> None:
        # 추적용 block_key='sample_pack'(경로 마커 블로킹), method 는 비지 않아야(샘플 전략 식별).
        from processing.pipeline.sample_strategies import sample_candidates
        out = sample_candidates(_conn_returning(self._rows()), self._SRC)
        for c in out:
            self.assertEqual(c.block_key, "sample_pack")
            self.assertTrue(c.method)

    def test_empty_when_no_neighbors(self) -> None:
        from processing.pipeline.sample_strategies import sample_candidates
        self.assertEqual(sample_candidates(_conn_returning([]), self._SRC), [])


# ── T003: sample_score ─────────────────────────────────────────────────────
class TestSampleScore(unittest.TestCase):
    _SRC = "018f0000-0000-7000-8000-000000000001"
    _T1 = "018f0000-0000-7000-8000-000000000007"
    _T2 = "018f0000-0000-7000-8000-000000000012"

    def _pairs(self) -> list[Candidate]:
        return [
            Candidate(self._SRC, self._T1, block_key="embedding", method="sample"),
            Candidate(self._SRC, self._T2, block_key="embedding", method="sample"),
        ]

    def _conn_scores(self) -> MagicMock:
        # sample_score 가 후보 집합 한정으로 소스↔후보 쌍 MAX 코사인을 조회한다고 가정한 fake.
        # 반환 행 형태는 (id, sim) — 픽스처-격리 쌍 코사인 쿼리의 결과.
        rows = [
            {"id": self._T1, "sim": 0.80},
            {"id": self._T2, "sim": 0.40},
        ]
        return _conn_returning(rows)

    def test_maps_candidate_to_scored_pair(self) -> None:
        from processing.pipeline.sample_strategies import sample_score
        out = sample_score(self._conn_scores(), self._pairs())
        self.assertEqual(len(out), 2)
        self.assertTrue(all(isinstance(s, ScoredPair) for s in out))
        # 각 ScoredPair 는 원래 Candidate 를 보존한다.
        self.assertEqual({s.candidate.target_id for s in out}, {self._T1, self._T2})

    def test_no_llm_call(self) -> None:
        # 헌법 2조: 샘플 점수는 LLM seam 을 호출하지 않는다.
        # complete_text/json/vision_json 어디도 부르지 않음을 patch 로 확인.
        from unittest import mock

        from processing.pipeline.sample_strategies import sample_score
        with mock.patch("src.llm.client.complete_text") as mt, \
             mock.patch("src.llm.client.complete_json") as mj, \
             mock.patch("src.llm.client.complete_vision_json") as mv:
            sample_score(self._conn_scores(), self._pairs())
        mt.assert_not_called()
        mj.assert_not_called()
        mv.assert_not_called()

    def test_deterministic_same_input_twice(self) -> None:
        from processing.pipeline.sample_strategies import sample_score
        out1 = sample_score(self._conn_scores(), self._pairs())
        out2 = sample_score(self._conn_scores(), self._pairs())
        self.assertEqual(out1, out2)

    def test_evidence_attached_with_embedding_cosine(self) -> None:
        from processing.pipeline.sample_strategies import sample_score
        out = sample_score(self._conn_scores(), self._pairs())
        for s in out:
            self.assertTrue(s.evidence)
            ev = s.evidence[0]
            self.assertIsInstance(ev, Evidence)
            self.assertEqual(ev.field, "embedding")
            self.assertEqual(ev.comparator, "embedding_cosine")
            self.assertEqual(ev.similarity, s.score)

    def test_score_reflects_similarity(self) -> None:
        # T1 유사도 0.80 > T2 0.40 이 점수에 반영(결정적 규칙).
        from processing.pipeline.sample_strategies import sample_score
        out = {s.candidate.target_id: s.score for s in sample_score(self._conn_scores(), self._pairs())}
        self.assertGreater(out[self._T1], out[self._T2])

    def test_empty_pairs(self) -> None:
        from processing.pipeline.sample_strategies import sample_score
        self.assertEqual(sample_score(self._conn_scores(), []), [])


# ── T005: sample_decide ────────────────────────────────────────────────────
class TestSampleDecide(unittest.TestCase):
    _SRC = "018f0000-0000-7000-8000-000000000001"

    def _scored(self, score: float, tid: str = "018f0000-0000-7000-8000-000000000007") -> ScoredPair:
        c = Candidate(self._SRC, tid, block_key="embedding", method="sample")
        return ScoredPair(candidate=c, score=score, evidence=[])

    def test_is_pure_function_no_conn(self) -> None:
        # decide 는 conn 을 받지 않는 순수 함수(DecideStage 계약).
        from processing.pipeline.sample_strategies import sample_decide
        out = sample_decide([self._scored(0.9)])
        self.assertTrue(all(isinstance(d, Decision) for d in out))

    def test_above_threshold_is_match(self) -> None:
        from processing.pipeline.sample_strategies import SAMPLE_DECIDE_TAU, sample_decide
        out = sample_decide([self._scored(SAMPLE_DECIDE_TAU + 0.1)])
        self.assertEqual(out[0].verdict, "match")

    def test_below_threshold_is_non_match(self) -> None:
        from processing.pipeline.sample_strategies import SAMPLE_DECIDE_TAU, sample_decide
        out = sample_decide([self._scored(SAMPLE_DECIDE_TAU - 0.1)])
        self.assertEqual(out[0].verdict, "non_match")

    def test_boundary_at_threshold_is_match(self) -> None:
        # 경계값(score == τ)은 match 쪽에 결정적으로 포함(>= 규칙).
        from processing.pipeline.sample_strategies import SAMPLE_DECIDE_TAU, sample_decide
        out = sample_decide([self._scored(SAMPLE_DECIDE_TAU)])
        self.assertEqual(out[0].verdict, "match")

    def test_tie_scores_deterministic(self) -> None:
        # 동점이라도 입력 순서·결과가 결정적(2회 동일).
        from processing.pipeline.sample_strategies import SAMPLE_DECIDE_TAU, sample_decide
        scored = [
            self._scored(SAMPLE_DECIDE_TAU, "018f0000-0000-7000-8000-000000000012"),
            self._scored(SAMPLE_DECIDE_TAU, "018f0000-0000-7000-8000-000000000007"),
        ]
        out1 = sample_decide(scored)
        out2 = sample_decide(scored)
        self.assertEqual(out1, out2)
        self.assertTrue(all(d.verdict == "match" for d in out1))

    def test_score_preserved_in_decision(self) -> None:
        from processing.pipeline.sample_strategies import sample_decide
        out = sample_decide([self._scored(0.73)])
        self.assertEqual(out[0].score, 0.73)
        self.assertEqual(out[0].candidate.target_id, "018f0000-0000-7000-8000-000000000007")

    def test_empty_scored(self) -> None:
        from processing.pipeline.sample_strategies import sample_decide
        self.assertEqual(sample_decide([]), [])


# ── T007: sample_persist_edges ─────────────────────────────────────────────
class TestSamplePersistEdges(unittest.TestCase):
    _SRC = "018f0000-0000-7000-8000-000000000001"
    _T1 = "018f0000-0000-7000-8000-000000000007"
    _T2 = "018f0000-0000-7000-8000-000000000012"

    def _decision(self, tid: str, verdict: str, score: float = 0.8) -> Decision:
        c = Candidate(self._SRC, tid, block_key="embedding", method="sample")
        return Decision(candidate=c, verdict=verdict, score=score)

    def _run_capturing(self, decisions: list[Decision]):
        """sample_persist_edges 가 graph_persist.sync_graph_edges 를 어떻게 호출하는지 캡처.

        graph_persist 헬퍼 재사용 검증 — 신규 스키마/카탈로그 경로를 만들지 않고
        기존 sync_graph_edges 한 곳으로만 적재함을 단언한다.
        """
        from unittest import mock

        from processing.pipeline import sample_strategies
        conn = MagicMock()
        with mock.patch.object(sample_strategies, "sync_graph_edges", return_value=(0, 0)) as spy:
            sample_strategies.sample_persist_edges(conn, decisions)
        return conn, spy

    def test_only_match_decisions_upserted(self) -> None:
        decisions = [
            self._decision(self._T1, "match"),
            self._decision(self._T2, "non_match"),
        ]
        conn, spy = self._run_capturing(decisions)
        spy.assert_called_once()
        kwargs = spy.call_args.kwargs
        # edges 에는 match 타깃만, non_match 는 제외.
        edge_targets = {e["target_media_item_id"] for e in kwargs["edges"]}
        self.assertEqual(edge_targets, {self._T1})
        self.assertNotIn(self._T2, edge_targets)

    def test_source_passed_through(self) -> None:
        conn, spy = self._run_capturing([self._decision(self._T1, "match")])
        self.assertEqual(spy.call_args.kwargs["source_asset_id"], self._SRC)

    def test_uses_existing_relation_kind_vocab(self) -> None:
        # 신규 카탈로그 등록 금지 — 기존 active 어휘(same_series 등)만 사용.
        from processing.pipeline.sample_strategies import SAMPLE_RELATION_KIND
        self.assertIn(SAMPLE_RELATION_KIND, {"same_domain", "same_series", "duplicate_near"})
        conn, spy = self._run_capturing([self._decision(self._T1, "match")])
        for e in spy.call_args.kwargs["edges"]:
            self.assertEqual(e["relation_type_code"], SAMPLE_RELATION_KIND)

    def test_allowed_targets_are_match_targets(self) -> None:
        # allowed_target_ids 는 match 타깃 집합과 일치(환각 방지 게이트와 정합).
        conn, spy = self._run_capturing([
            self._decision(self._T1, "match"),
            self._decision(self._T2, "match"),
        ])
        self.assertEqual(set(spy.call_args.kwargs["allowed_target_ids"]), {self._T1, self._T2})

    def test_no_match_skips_persist_call_or_empty(self) -> None:
        # non_match 만 있으면 적재할 엣지가 없다 — sync_graph_edges 미호출 또는 빈 edges.
        from unittest import mock

        from processing.pipeline import sample_strategies
        conn = MagicMock()
        with mock.patch.object(sample_strategies, "sync_graph_edges", return_value=(0, 0)) as spy:
            sample_strategies.sample_persist_edges(conn, [self._decision(self._T1, "non_match")])
        if spy.called:
            self.assertEqual(list(spy.call_args.kwargs["edges"]), [])

    def test_returns_none(self) -> None:
        # EdgePersistStage 계약: 반환값 None.
        conn, spy = self._run_capturing([self._decision(self._T1, "match")])
        # _run_capturing 은 반환을 버리므로 직접 한 번 더 확인.
        from unittest import mock

        from processing.pipeline import sample_strategies
        c = MagicMock()
        with mock.patch.object(sample_strategies, "sync_graph_edges", return_value=(1, 0)):
            ret = sample_strategies.sample_persist_edges(c, [self._decision(self._T1, "match")])
        self.assertIsNone(ret)

    def test_no_new_schema_or_catalog_calls(self) -> None:
        # 신규 스키마·카탈로그 금지: ensure_relation_kind_for_llm_proposal 같은 등록 경로를
        # 호출하지 않음을 단언(기존 sync_graph_edges 만 경유).
        from unittest import mock

        from processing.pipeline import sample_strategies
        conn = MagicMock()
        with mock.patch.object(sample_strategies, "sync_graph_edges", return_value=(1, 0)), \
             mock.patch(
                 "src.relations.relation_type_catalog.ensure_relation_kind_for_llm_proposal"
             ) as reg:
            sample_strategies.sample_persist_edges(conn, [self._decision(self._T1, "match")])
        reg.assert_not_called()


if __name__ == "__main__":
    unittest.main()
