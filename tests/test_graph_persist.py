"""v2 단계 C — 공용 그래프 영속화 테스트.

- TestSyncGraphEdgesUnit: mock conn 으로 필터 로직 검증(DB 불필요).
- Test*DB: RUN_DB_E2E=1 일 때만, 실 PostgreSQL 에 node/graph_edge 적재 검증.
"""
from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path
from unittest import mock

from dotenv import load_dotenv

_RUN = os.getenv("RUN_DB_E2E") == "1"
_ENV = Path(__file__).resolve().parents[1] / ".env.dev"

_SRC = "018f0000-0000-7000-8000-000000000001"
_T1 = "018f0000-0000-7000-8000-000000000007"
_T2 = "018f0000-0000-7000-8000-000000000008"


# ── 순수 단위(mock conn) — DB 불필요 ────────────────────────────────────────
class TestCanonicalOrdering(unittest.TestCase):
    def test_symmetric_kind_orders_pair(self):
        from src.relations.graph_persist import _canonical_pair
        a, b = "018f0000-0000-7000-8000-000000000006", "018f0000-0000-7000-8000-000000000005"
        # 대칭: 항상 (min, max)
        self.assertEqual(_canonical_pair(a, b, symmetric=True), (b, a))
        self.assertEqual(_canonical_pair(b, a, symmetric=True), (b, a))
        # 비대칭: 입력 방향 유지
        self.assertEqual(_canonical_pair(a, b, symmetric=False), (a, b))


class TestDecideStatus(unittest.TestCase):
    """033 T003(FR-001): AND 게이트 순수 헬퍼 `_decide_status` — emb_min<=0.0 무력=현행."""

    def test_decide_status_and_gate(self) -> None:
        from src.relations.graph_persist import _decide_status
        # 무력 기본값: emb_min=0.0 → conf 단독 결정(현행 동일)
        self.assertEqual(_decide_status(0.95, 0.10, 0.9, 0.0), "active")
        self.assertEqual(_decide_status(0.80, 0.99, 0.9, 0.0), "proposed")
        # emb_min>0: conf 충분해도 emb 미달이면 proposed (SC-002)
        self.assertEqual(_decide_status(0.95, 0.40, 0.9, 0.5), "proposed")
        # 둘 다 통과해야 active
        self.assertEqual(_decide_status(0.95, 0.60, 0.9, 0.5), "active")

    def test_decide_status_none_conf_is_proposed(self) -> None:
        from src.relations.graph_persist import _decide_status
        # conf 미상(None)은 자동승인 대상 아님(현행 status_val 식과 동일).
        self.assertEqual(_decide_status(None, 0.99, 0.9, 0.0), "proposed")


class TestSyncGraphEdgesUnit(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = mock.MagicMock()
        self.cur = self.conn.cursor.return_value.__enter__.return_value
        self.allowed = frozenset({_T1, _T2})

    def _edge(self, **kw):
        e = {"target_media_item_id": _T1, "relation_type_code": "same_domain", "reason": "유사"}
        e.update(kw)
        return e

    def _run(self, edges, kind=("k1", True)):
        """kind=(relation_kind_id, is_symmetric) 또는 None(미해소 kind)."""
        from src.relations import graph_persist
        kdict = None if kind is None else {"relation_kind_id": kind[0], "is_symmetric": kind[1]}
        with mock.patch.object(graph_persist, "ensure_asset_node", side_effect=lambda conn, aid: "n_" + aid), \
             mock.patch.object(graph_persist, "fetch_relation_kind", return_value=kdict):
            return graph_persist.sync_graph_edges(
                self.conn, source_asset_id=_SRC, edges=edges, allowed_target_ids=self.allowed
            )

    def test_valid_edge_upserted(self) -> None:
        up, sk = self._run([self._edge()])
        self.assertEqual((up, sk), (1, 0))
        sql = " ".join(str(c.args[0]) for c in self.cur.execute.call_args_list)
        self.assertIn("INSERT INTO graph_edge", sql)

    def test_collect_captures_upserted_pairs_only(self) -> None:
        # collect 전달 시 upsert 된 쌍만 수집(skip 제외) — 계보 관계쌍 기록용(013).
        from src.relations import graph_persist
        collected: list = []
        # 069 B4: collect status 는 INSERT…RETURNING(DB 실제값)에서 온다 — mock 커서가 그 값을 돌려주게 설정.
        self.cur.fetchone.return_value = ("active",)
        with mock.patch.object(graph_persist, "ensure_asset_node", side_effect=lambda conn, aid: "n_" + aid), \
             mock.patch.object(graph_persist, "fetch_relation_kind",
                               return_value={"relation_kind_id": "k1", "is_symmetric": True}):
            up, sk = graph_persist.sync_graph_edges(
                self.conn, source_asset_id=_SRC,
                edges=[self._edge(target_media_item_id=_T1, confidence=0.95),
                       self._edge(target_media_item_id=_T2, confidence=0.6)],
                allowed_target_ids=frozenset({_T1}),  # _T2 는 미허용 → skip
                auto_approve_min=0.9, collect=collected)
        self.assertEqual((up, sk), (1, 1))
        self.assertEqual([c["target_asset_id"] for c in collected], [_T1])  # skip 제외
        self.assertEqual(collected[0]["kind_code"], "same_domain")
        self.assertEqual(collected[0]["confidence"], 0.95)
        self.assertEqual(collected[0]["status"], "active")  # 0.95 >= 0.9

    def test_collect_none_is_noop(self) -> None:
        # collect 미전달(기본 None)이면 기존 (upserted, skipped) 계약·동작 그대로(하위호환).
        up, sk = self._run([self._edge()])
        self.assertEqual((up, sk), (1, 0))

    def test_collect_uses_db_returned_status_not_computed(self) -> None:
        # 069 B4(P2-4): 계보(collect)에 계산 status_val 이 아니라 INSERT…ON CONFLICT RETURNING 이
        # 돌려준 **DB 실제 status** 를 기록해야 한다. 반려(rejected) 엣지가 재제안돼도(계산상 active)
        # ON CONFLICT 는 status 를 미갱신하므로 DB 는 rejected 로 남는다 → 계보도 rejected 여야 한다.
        from src.relations import graph_persist
        collected: list = []
        self.cur.fetchone.return_value = ("rejected",)  # DB 가 보존한 반려 상태
        with mock.patch.object(graph_persist, "ensure_asset_node", side_effect=lambda conn, aid: "n_" + aid), \
             mock.patch.object(graph_persist, "fetch_relation_kind",
                               return_value={"relation_kind_id": "k1", "is_symmetric": True}):
            graph_persist.sync_graph_edges(
                self.conn, source_asset_id=_SRC,
                edges=[self._edge(target_media_item_id=_T1, confidence=0.99)],
                allowed_target_ids=frozenset({_T1}),
                auto_approve_min=0.9, collect=collected)
        # 계산상 active(0.99≥0.9)이지만 계보엔 DB RETURNING 값(rejected).
        self.assertEqual(collected[0]["status"], "rejected")
        insert_sql = next(str(c.args[0]) for c in self.cur.execute.call_args_list
                          if "INSERT INTO graph_edge" in str(c.args[0]))
        self.assertIn("RETURNING status", insert_sql)

    def test_returning_none_raises_instead_of_silent_fallback(self) -> None:
        # 069 B4 재발 차단(code-reviewer): 현재 SQL(DO UPDATE)은 항상 1행을 돌려주므로 이 경로는
        # 도달 불가지만, 누군가 INSERT 를 DO NOTHING 으로 바꾸면 RETURNING 이 None 이 된다. 그때
        # status_val 로 조용히 폴백하면 B4 가 고친 계보 오염 버그가 소리 없이 부활한다 → 조용한
        # 폴백 대신 RuntimeError 로 못박아, SQL 변경이 이 불변식을 깨면 즉시 드러나게 한다.
        from src.relations import graph_persist
        self.cur.fetchone.return_value = None  # DO NOTHING 이 무행을 돌려주는 가상 상황
        with mock.patch.object(graph_persist, "ensure_asset_node", side_effect=lambda conn, aid: "n_" + aid), \
             mock.patch.object(graph_persist, "fetch_relation_kind",
                               return_value={"relation_kind_id": "k1", "is_symmetric": True}):
            with self.assertRaises(RuntimeError):
                graph_persist.sync_graph_edges(
                    self.conn, source_asset_id=_SRC,
                    edges=[self._edge(target_media_item_id=_T1, confidence=0.99)],
                    allowed_target_ids=frozenset({_T1}), collect=[])

    def _insert_params(self, edges, kind=("k1", True), **kwargs):
        """단일 엣지 upsert 의 INSERT 바인딩 파라미터(ensure_asset_node mock — INSERT 만 self.cur 사용)."""
        from src.relations import graph_persist
        kdict = {"relation_kind_id": kind[0], "is_symmetric": kind[1]}
        with mock.patch.object(graph_persist, "ensure_asset_node", side_effect=lambda conn, aid: "n_" + aid), \
             mock.patch.object(graph_persist, "fetch_relation_kind", return_value=kdict):
            graph_persist.sync_graph_edges(
                self.conn, source_asset_id=_SRC, edges=edges, allowed_target_ids=self.allowed, **kwargs)
        return self.cur.execute.call_args[0][1]

    def test_status_proposed_below_auto_approve(self) -> None:
        params = self._insert_params([self._edge(confidence=0.5)], auto_approve_min=0.9)
        self.assertEqual(params[-1], "proposed")  # status_val 은 INSERT 마지막 바인딩

    def test_status_active_at_or_above_auto_approve(self) -> None:
        params = self._insert_params([self._edge(confidence=0.95)], auto_approve_min=0.9)
        self.assertEqual(params[-1], "active")

    def test_status_proposed_when_emb_below_emb_min(self) -> None:
        # 033 T003: 고conf(0.95)여도 타깃 emb_score(0.40)가 emb_min(0.5) 미달이면 proposed (SC-002).
        params = self._insert_params(
            [self._edge(confidence=0.95)], auto_approve_min=0.9,
            target_emb_scores={_T1: 0.40}, auto_approve_emb_min=0.5)
        self.assertEqual(params[-1], "proposed")

    def test_status_active_when_both_pass(self) -> None:
        # conf·emb 둘 다 통과하면 active.
        params = self._insert_params(
            [self._edge(confidence=0.95)], auto_approve_min=0.9,
            target_emb_scores={_T1: 0.60}, auto_approve_emb_min=0.5)
        self.assertEqual(params[-1], "active")

    def test_status_unchanged_when_emb_args_omitted(self) -> None:
        # 무력 기본값(emb 인자 미전달) → conf 단독 결정(현행 동일, 동작 보존).
        params = self._insert_params([self._edge(confidence=0.95)], auto_approve_min=0.9)
        self.assertEqual(params[-1], "active")

    def test_topic_jsonb_serialized(self) -> None:
        import json
        params = self._insert_params([self._edge(topic_ko="게임", topic_en="gaming")])
        topic = json.loads(params[6])  # (edge_id,a,b,kind,conf,reason,topic,status) 중 7번째
        self.assertEqual(topic["topic_ko"], "게임")
        self.assertEqual(topic["topic_en"], "gaming")

    def test_target_not_in_allowed_skipped(self) -> None:
        self.assertEqual(self._run([self._edge(target_media_item_id="018f0000-0000-7000-8000-000000000014")]), (0, 1))

    def test_invalid_uuid_target_skipped(self) -> None:
        self.assertEqual(self._run([self._edge(target_media_item_id="42")]), (0, 1))

    def test_self_reference_skipped(self) -> None:
        self.assertEqual(self._run([self._edge(target_media_item_id=_SRC)]), (0, 1))

    def test_missing_relation_code_skipped(self) -> None:
        self.assertEqual(self._run([self._edge(relation_type_code="")]), (0, 1))

    def test_unresolved_relation_kind_skipped(self) -> None:
        self.assertEqual(self._run([self._edge()], kind=None), (0, 1))


# ── 058 G4(FR-401): topic 정본화 배선 — 플래그 게이트·동작 불변(T401/T403) ─────
class TestTopicCanonicalizeGate(unittest.TestCase):
    """``_topic_canonicalize_enabled`` — TOPIC_CANONICALIZE_ENABLED 설정 조회 헬퍼.

    settings 미초기화(순수 단위 등)에서는 보수적 False 폴백이라 현행 경로(canonicalize 미배선)를
    보존한다 — 기존 graph_persist 단위 테스트가 init_settings 없이도 동작 불변이도록(다른 선택 설정
    조회 헬퍼의 미초기화 보수 폴백과 동형).
    """

    def test_gate_false_when_settings_uninitialized(self) -> None:
        from src.config import settings as settings_mod
        from src.relations import graph_persist
        with mock.patch.object(settings_mod, "get_current_settings", side_effect=RuntimeError):
            self.assertIs(graph_persist._topic_canonicalize_enabled(), False)

    def test_gate_reads_setting(self) -> None:
        from src.config import settings as settings_mod
        from src.relations import graph_persist
        fake = mock.MagicMock(topic_canonicalize_enabled=True)
        with mock.patch.object(settings_mod, "get_current_settings", return_value=fake):
            self.assertIs(graph_persist._topic_canonicalize_enabled(), True)


class TestSyncGraphEdgesCanonicalize(unittest.TestCase):
    """플래그 게이트로 persist 직전 topic/subtopic 정본화(FR-401). 기본 off=동작 불변(T403)."""

    def setUp(self) -> None:
        self.conn = mock.MagicMock()
        self.cur = self.conn.cursor.return_value.__enter__.return_value

    def _edge(self, **kw):
        e = {"target_media_item_id": _T1, "relation_type_code": "same_domain",
             "topic_ko": "식품", "topic_en": "food",
             "subtopic_ko": "김밥", "subtopic_en": "gimbap",
             "reason": "유사", "confidence": 0.5}
        e.update(kw)
        return e

    def _run(self, *, enabled, sub_return="김밥"):
        """canonicalize seam 을 mock 으로 주입하고 INSERT topic jsonb·mock 을 돌려준다."""
        import json

        from src.relations import graph_persist
        kdict = {"relation_kind_id": "k1", "is_symmetric": True}
        with mock.patch.object(graph_persist, "ensure_asset_node", side_effect=lambda conn, aid: "n_" + aid), \
             mock.patch.object(graph_persist, "fetch_relation_kind", return_value=kdict), \
             mock.patch.object(graph_persist, "_topic_canonicalize_enabled", return_value=enabled), \
             mock.patch.object(graph_persist, "canonicalize_topic") as m_topic, \
             mock.patch.object(graph_persist, "canonicalize_subtopic") as m_sub:
            m_topic.return_value = {"canonical_ko": "요리", "canonical_en": "cooking",
                                    "decided_by": "exact"}
            m_sub.return_value = sub_return
            graph_persist.sync_graph_edges(
                self.conn, source_asset_id=_SRC, edges=[self._edge()],
                allowed_target_ids=frozenset({_T1}))
        params = self.cur.execute.call_args[0][1]  # 마지막 INSERT 바인딩
        return m_topic, m_sub, json.loads(params[6])  # topic jsonb 는 7번째 바인딩

    def test_flag_off_no_canonicalize_and_topic_unchanged(self) -> None:
        # T401/T403 동작 불변: flag off → canonicalize 미호출·저장 topic == coerce 결과(식품/food).
        m_topic, m_sub, topic = self._run(enabled=False)
        m_topic.assert_not_called()
        m_sub.assert_not_called()
        self.assertEqual(topic["topic_ko"], "식품")
        self.assertEqual(topic["topic_en"], "food")
        self.assertEqual(topic["subtopic_ko"], "김밥")
        self.assertEqual(topic["subtopic_en"], "gimbap")

    def test_flag_on_canonicalizes_topic_and_subtopic(self) -> None:
        # flag on → canonicalize 호출·정본(요리/cooking) 저장.
        m_topic, m_sub, topic = self._run(enabled=True)
        m_topic.assert_called_once()
        m_sub.assert_called_once()
        self.assertEqual(topic["topic_ko"], "요리")
        self.assertEqual(topic["topic_en"], "cooking")
        self.assertEqual(topic["subtopic_ko"], "김밥")

    def test_flag_on_subtopic_none_empties_subtopic(self) -> None:
        # flag on + canonicalize_subtopic None(모달리티/계층 규칙) → subtopic ko/en 비움.
        _, _, topic = self._run(enabled=True, sub_return=None)
        self.assertEqual(topic["subtopic_ko"], "")
        self.assertEqual(topic["subtopic_en"], "")

    def test_flag_on_subtopic_receives_canonical_topic(self) -> None:
        # canonicalize_subtopic 는 정본화된 상위 topic 을 문맥으로 받는다(계약 대칭).
        from src.relations import graph_persist
        kdict = {"relation_kind_id": "k1", "is_symmetric": True}
        with mock.patch.object(graph_persist, "ensure_asset_node", side_effect=lambda conn, aid: "n_" + aid), \
             mock.patch.object(graph_persist, "fetch_relation_kind", return_value=kdict), \
             mock.patch.object(graph_persist, "_topic_canonicalize_enabled", return_value=True), \
             mock.patch.object(graph_persist, "canonicalize_topic",
                               return_value={"canonical_ko": "요리", "canonical_en": "cooking",
                                             "decided_by": "exact"}), \
             mock.patch.object(graph_persist, "canonicalize_subtopic", return_value="김밥") as m_sub:
            graph_persist.sync_graph_edges(
                self.conn, source_asset_id=_SRC, edges=[self._edge()],
                allowed_target_ids=frozenset({_T1}))
        # 두 번째 위치인자 = 정본 topic_ko('요리'), 세 번째 = 원본 subtopic('김밥').
        _, call_args, _ = m_sub.mock_calls[0]
        self.assertEqual(call_args[1], "요리")
        self.assertEqual(call_args[2], "김밥")


# ── 실 DB 통합(RUN_DB_E2E=1) ────────────────────────────────────────────────
def _vec():
    from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
    v = [0.0] * FIX_EMBEDDING_DIMENSION
    v[0] = 0.5  # 비영 벡터(코사인 정의)
    return v


def _make_registered_asset(db, ids: list) -> str:
    """registered + st 임베딩 보유 자산 1건 생성(테스트 헬퍼). 생성 id 를 ids 에 누적."""
    from processing.dispatch.types import AssetRecord, EmbeddingItem
    from processing.ingest.asset_persist import create_asset, finalize_asset
    from processing.ingest.status import AssetStatus, set_status

    with db.transaction() as conn:
        aid = create_asset(conn, fs_path=f"/t/{uuid.uuid4().hex}.txt", modality="txt", file_hash=uuid.uuid4().hex)
    ids.append(aid)
    with db.transaction() as conn:
        set_status(conn, aid, AssetStatus.ROUTING)
        set_status(conn, aid, AssetStatus.CLASSIFYING)
        set_status(conn, aid, AssetStatus.EXTRACTING)
    with db.transaction() as conn:
        finalize_asset(conn, aid, AssetRecord(
            embeddings=[EmbeddingItem(channel="st", vector=_vec(), model_name="m")]))
    return str(aid)


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만")
class TestGraphPersistDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(_ENV, override=False)
        from src.database.postgres_util import PostgresUtil
        cls.db = PostgresUtil()
        cls.db.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.db.__exit__(None, None, None)

    def setUp(self):
        self._ids: list = []

    def tearDown(self):
        if self._ids:
            with self.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM asset WHERE asset_id = ANY(%s)", (self._ids,))

    def test_schema_exists(self):
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.node'), to_regclass('public.graph_edge')")
            n, e = cur.fetchone()
        self.assertIsNotNone(n)
        self.assertIsNotNone(e)

    def test_sync_graph_edges_creates_nodes_and_edge(self):
        from src.relations.graph_persist import sync_graph_edges
        src_id = _make_registered_asset(self.db, self._ids)
        dst_id = _make_registered_asset(self.db, self._ids)
        edges = [{
            "target_media_item_id": dst_id, "relation_type_code": "same_domain",
            "topic_ko": "일반", "topic_en": "general", "subtopic_ko": "", "subtopic_en": "",
            "confidence": 0.8, "reason": "테스트",
        }]

        def _run(conn):
            return sync_graph_edges(conn, source_asset_id=src_id, edges=edges, allowed_target_ids=frozenset({dst_id}))

        up, sk = self.db.execute_in_transaction(_run, idempotent=False)
        self.assertEqual((up, sk), (1, 0))
        # 멱등 — 같은 엣지 재실행
        up2, _ = self.db.execute_in_transaction(_run, idempotent=False)
        self.assertEqual(up2, 1)
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT count(*) FROM graph_edge ge
                JOIN node sn ON sn.node_id = ge.src_node AND sn.asset_id = %s
                JOIN node dn ON dn.node_id = ge.dst_node AND dn.asset_id = %s
            """, (src_id, dst_id))
            (n,) = cur.fetchone()
        self.assertEqual(n, 1)   # 멱등: 1건 유지

    # ── 032: 충돌 시 confidence 더 큰 제안의 topic·reason 갱신(status 보존) ──
    def _sync_edge(self, src_id, dst_id, *, confidence, topic_ko, reason):
        """단일 derived_from(비대칭 — canonical 방향 유지) 엣지 sync. (upserted, skipped) 반환."""
        from src.relations.graph_persist import sync_graph_edges
        edges = [{
            "target_media_item_id": dst_id, "relation_type_code": "derived_from",
            "topic_ko": topic_ko, "topic_en": "", "subtopic_ko": "", "subtopic_en": "",
            "confidence": confidence, "reason": reason,
        }]
        return self.db.execute_in_transaction(
            lambda conn: sync_graph_edges(
                conn, source_asset_id=src_id, edges=edges, allowed_target_ids=frozenset({dst_id})),
            idempotent=False)

    def _edge_row(self, src_id, dst_id):
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT ge.topic->>'topic_ko', ge.reason, ge.confidence, ge.status FROM graph_edge ge "
                "JOIN node sn ON sn.node_id = ge.src_node AND sn.asset_id = %s "
                "JOIN node dn ON dn.node_id = ge.dst_node AND dn.asset_id = %s",
                (src_id, dst_id))
            return cur.fetchone()

    def test_conflict_refreshes_topic_when_higher_confidence(self):
        src_id = _make_registered_asset(self.db, self._ids)
        dst_id = _make_registered_asset(self.db, self._ids)
        self._sync_edge(src_id, dst_id, confidence=0.5, topic_ko="기타", reason="r1")
        self._sync_edge(src_id, dst_id, confidence=0.9, topic_ko="게임", reason="r2")  # 더 높음 → 갱신
        topic, reason, conf, _ = self._edge_row(src_id, dst_id)
        self.assertEqual(topic, "게임")
        self.assertEqual(reason, "r2")
        self.assertEqual(float(conf), 0.9)
        self._sync_edge(src_id, dst_id, confidence=0.3, topic_ko="잡담", reason="r3")  # 더 낮음 → 보존
        topic2, reason2, conf2, _ = self._edge_row(src_id, dst_id)
        self.assertEqual(topic2, "게임")            # 보존
        self.assertEqual(reason2, "r2")
        self.assertEqual(float(conf2), 0.9)         # GREATEST
        self._sync_edge(src_id, dst_id, confidence=0.9, topic_ko="동률", reason="r4")  # 동률(==) → strict > 라 보존
        topic3, reason3, _, _ = self._edge_row(src_id, dst_id)
        self.assertEqual(topic3, "게임")            # 동률은 갱신 안 함(strict > 경계·결정적)
        self.assertEqual(reason3, "r2")

    def test_conflict_preserves_status_even_higher_confidence(self):
        src_id = _make_registered_asset(self.db, self._ids)
        dst_id = _make_registered_asset(self.db, self._ids)
        self._sync_edge(src_id, dst_id, confidence=0.5, topic_ko="기타", reason="r1")
        with self.db.transaction() as conn, conn.cursor() as cur:  # 사람이 rejected 처리
            cur.execute(
                "UPDATE graph_edge ge SET status='rejected' FROM node sn, node dn "
                "WHERE sn.node_id=ge.src_node AND sn.asset_id=%s "
                "AND dn.node_id=ge.dst_node AND dn.asset_id=%s",
                (src_id, dst_id))
        self._sync_edge(src_id, dst_id, confidence=0.95, topic_ko="게임", reason="r2")  # 높아도
        _, _, _, status = self._edge_row(src_id, dst_id)
        self.assertEqual(status, "rejected")        # status 보존(사람 결정 무손상)

    # ── 033 T007: 자동승인 AND 게이트 실 DB e2e(SC-001/SC-002) ──
    def _sync_gated_edge(self, src_id, dst_id, *, confidence, target_emb_scores=None,
                         auto_approve_emb_min=0.0, auto_approve_min=0.9):
        """단일 same_domain 엣지를 AND 게이트 인자와 함께 sync. (upserted, skipped) 반환.

        auto_approve_min 기본 0.9 — 고conf(0.95) 엣지가 conf 게이트는 통과하도록 낮춘다.
        emb 인자(target_emb_scores·auto_approve_emb_min)로 emb 게이트 동작을 검증한다.
        """
        from src.relations.graph_persist import sync_graph_edges
        edges = [{
            "target_media_item_id": dst_id, "relation_type_code": "same_domain",
            "topic_ko": "일반", "topic_en": "general", "subtopic_ko": "", "subtopic_en": "",
            "confidence": confidence, "reason": "게이트 e2e",
        }]
        return self.db.execute_in_transaction(
            lambda conn: sync_graph_edges(
                conn, source_asset_id=src_id, edges=edges,
                allowed_target_ids=frozenset({dst_id}),
                auto_approve_min=auto_approve_min,
                target_emb_scores=target_emb_scores,
                auto_approve_emb_min=auto_approve_emb_min),
            idempotent=False)

    def test_auto_approve_default_high_conf_active(self):
        # SC-001: emb 인자 미전달(기본 0.0=무력) + 고conf(0.95≥auto_approve_min 0.9) → 현행대로 active.
        src_id = _make_registered_asset(self.db, self._ids)
        dst_id = _make_registered_asset(self.db, self._ids)
        up, sk = self._sync_gated_edge(src_id, dst_id, confidence=0.95)
        self.assertEqual((up, sk), (1, 0))
        _, _, _, status = self._edge_row(src_id, dst_id)
        self.assertEqual(status, "active")           # emb 게이트 무력 → conf 단독 자동승인

    def test_auto_approve_emb_below_min_proposed(self):
        # SC-002: auto_approve_emb_min=0.5 + 타깃 emb_score=0.4(미달) → 고conf(0.95)여도 proposed.
        src_id = _make_registered_asset(self.db, self._ids)
        dst_id = _make_registered_asset(self.db, self._ids)
        up, sk = self._sync_gated_edge(
            src_id, dst_id, confidence=0.95,
            target_emb_scores={dst_id: 0.4}, auto_approve_emb_min=0.5)
        self.assertEqual((up, sk), (1, 0))
        _, _, _, status = self._edge_row(src_id, dst_id)
        self.assertEqual(status, "proposed")         # emb 미달 → 자동승인 차단(AND 게이트)


if __name__ == "__main__":
    unittest.main()
