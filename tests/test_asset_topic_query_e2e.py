"""065 — 자기주제 정본 소비 함수 실 DB 라운드트립 e2e(RUN_DB_E2E 게이트).

무DB 환경에서는 자동 skip(다른 ``*_e2e`` 관례 일치). ``RUN_DB_E2E=1`` + 로컬 PostgreSQL(v299
``asset_topic`` 적용)에서만 실행한다(사람 게이트). 구 ``tests/test_topic_query_e2e.py``(이웃-엣지
투영 e2e)를 자기주제 정본(``asset_topic``) 기준으로 개작한 것이다.

검증 의도 (FR-402·FR-403)
    단위 테스트(``tests/test_asset_topic_consumers.py``·``test_asset_topic_classify.py``)는 mock cursor 로
    파이썬 로직만 본다. 여기서는 **SQL 자체의 정합**(asset_topic 조인·topic/subtopic 필터·의료 제외·
    already_linked 대칭 엣지)을 실 DB 로 검증한다.

시나리오
    A·B·C(일반)·M(의료) 자산 + 자기주제 정본 (topic, 제빵) 4행 시딩(충돌 방지 유니크 topic_ko).
    already_linked 검증용으로 A—B active 엣지 1건 시딩.
    - ``fetch_asset_topic(A)``       → [{topic, 제빵, ..., weight:1}] · 미부여 자산 []
    - ``find_same_topic_groups(A)``  → 주제›제빵 그룹 {B,C} · M(의료) 제외 · A—B already_linked
    - ``assets_in_topic(topic)``     → {A,B,C} total=3 · M 제외 · 페이징 · subtopic 필터
    - ``list_topics()``              → (topic, 제빵) 엔트리 asset_count=3 · M 제외
"""
from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path

from dotenv import load_dotenv

_RUN = os.getenv("RUN_DB_E2E") == "1"
_ENV = Path(__file__).resolve().parents[1] / ".env.dev"


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만")
class TestAssetTopicQueryDB(unittest.TestCase):
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
        # 충돌 방지 유니크 topic_ko — 실 DB 의 기존 asset_topic 행과 겹치지 않아 카운트 단언이 견고하다.
        self._topic = "e2e주제_" + uuid.uuid4().hex[:8]
        self._subtopic = "제빵"

    def tearDown(self):
        # asset 삭제 → asset_topic(ON DELETE CASCADE)·node/graph_edge(CASCADE) 연쇄 정리.
        if self._ids:
            with self.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM asset WHERE asset_id = ANY(%s)", (self._ids,))

    def _make_asset(self, *, domain: str = "general") -> str:
        from processing.ingest.asset_persist import create_asset
        with self.db.transaction() as conn:
            aid = create_asset(
                conn, fs_path=f"/t/{uuid.uuid4().hex}.txt", modality="txt",
                domain=domain, file_hash=uuid.uuid4().hex)
        self._ids.append(aid)
        return str(aid)

    def _seed_topic(self, asset_id: str) -> None:
        """자기주제 정본 1행 시딩(asset_topic upsert) — (self._topic, 제빵)."""
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO asset_topic
                    (asset_id, topic_ko, topic_en, subtopic_ko, subtopic_en,
                     confidence, decided_by, policy_version)
                VALUES (%s, %s, 'e2e_topic', %s, 'baking', 0.9, 'hybrid', 'asset_topic.v1')
                ON CONFLICT (asset_id) DO UPDATE SET topic_ko = EXCLUDED.topic_ko
                """,
                (asset_id, self._topic, self._subtopic),
            )

    def _link_active(self, src_id: str, dst_id: str) -> None:
        """src→dst active 엣지 1건 시딩(already_linked 검증용). confidence=1.0·auto_approve=1.0."""
        from src.relations.graph_persist import sync_graph_edges
        edges = [{
            "target_media_item_id": dst_id, "relation_type_code": "same_domain",
            "topic_ko": self._topic, "topic_en": "e2e_topic",
            "subtopic_ko": self._subtopic, "subtopic_en": "baking",
            "confidence": 1.0, "reason": "065 e2e already_linked 픽스처",
        }]
        up, _ = self.db.execute_in_transaction(
            lambda conn: sync_graph_edges(
                conn, source_asset_id=src_id, edges=edges,
                allowed_target_ids=frozenset({dst_id}), auto_approve_min=1.0),
            idempotent=False)
        self.assertEqual(up, 1)

    def _seed(self):
        """A·B·C(일반)·M(의료) + 자기주제 정본 4행 + A—B active 엣지."""
        a = self._make_asset()
        b = self._make_asset()
        c = self._make_asset()
        m = self._make_asset(domain="medical")
        for aid in (a, b, c, m):
            self._seed_topic(aid)
        self._link_active(a, b)  # A—B active → B.already_linked True
        return a, b, c, m

    def test_fetch_asset_topic_roundtrip(self):
        from src.topic.asset_topic_query import fetch_asset_topic
        a, _b, _c, _m = self._seed()

        out = self.db.execute_in_transaction(
            lambda conn: fetch_asset_topic(conn, a), idempotent=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["topic_ko"], self._topic)
        self.assertEqual(out[0]["subtopic_ko"], self._subtopic)
        self.assertEqual(out[0]["topic_en"], "e2e_topic")
        self.assertEqual(out[0]["subtopic_en"], "baking")
        self.assertEqual(out[0]["weight"], 1)
        # 구 project_asset_topics 형상과 필드명 동일(소비처 무변경 스왑 계약).
        self.assertEqual(
            set(out[0].keys()),
            {"topic_ko", "subtopic_ko", "topic_en", "subtopic_en", "weight"},
        )

        # 미부여 자산(정본 행 없음) → []
        empty = self.db.execute_in_transaction(
            lambda conn: fetch_asset_topic(conn, str(uuid.uuid4())), idempotent=True)
        self.assertEqual(empty, [])

    def test_find_same_topic_groups_pairs_and_medical_excluded(self):
        from src.topic.asset_topic_query import find_same_topic_groups
        a, b, c, m = self._seed()

        out = self.db.execute_in_transaction(
            lambda conn: find_same_topic_groups(conn, a), idempotent=True)

        groups = [g for g in out if g["topic_ko"] == self._topic]
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual(g["asset_count"], 2)   # {B,C} distinct — 대상 A·M(의료) 제외
        self.assertEqual([s["subtopic_ko"] for s in g["subtopics"]], [self._subtopic])
        sub = g["subtopics"][0]
        self.assertEqual(sub["asset_count"], 2)
        by_id = {x["asset_id"]: x for x in sub["assets"]}
        self.assertEqual(set(by_id), {b, c})
        self.assertTrue(by_id[b]["already_linked"])   # A—B active
        self.assertFalse(by_id[c]["already_linked"])  # A—C 없음
        self.assertTrue(all(isinstance(x["asset_id"], str) for x in sub["assets"]))
        # PHI: 의료 M·대상 A 는 어디에도 없다.
        all_ids = {x["asset_id"] for gr in out for s in gr["subtopics"] for x in s["assets"]}
        self.assertNotIn(m, all_ids)
        self.assertNotIn(a, all_ids)

    def test_assets_in_topic_paging_and_medical_excluded(self):
        from src.topic.asset_topic_query import assets_in_topic
        a, b, c, m = self._seed()

        full = self.db.execute_in_transaction(
            lambda conn: assets_in_topic(conn, topic_ko=self._topic, limit=50, offset=0),
            idempotent=True)
        ids = [r["asset_id"] for r in full["rows"]]
        self.assertEqual(full["total"], 3)          # {A,B,C} — M 제외
        self.assertEqual(set(ids), {a, b, c})
        self.assertNotIn(m, ids)
        self.assertEqual(ids, sorted(ids))          # asset_id asc 결정적

        p1 = self.db.execute_in_transaction(
            lambda conn: assets_in_topic(conn, topic_ko=self._topic, limit=2, offset=0),
            idempotent=True)
        p2 = self.db.execute_in_transaction(
            lambda conn: assets_in_topic(conn, topic_ko=self._topic, limit=2, offset=2),
            idempotent=True)
        self.assertEqual(len(p1["rows"]), 2)
        self.assertEqual(len(p2["rows"]), 1)
        self.assertEqual(p1["total"], 3)

        hit = self.db.execute_in_transaction(
            lambda conn: assets_in_topic(conn, topic_ko=self._topic, subtopic_ko=self._subtopic),
            idempotent=True)
        self.assertEqual(hit["total"], 3)
        miss = self.db.execute_in_transaction(
            lambda conn: assets_in_topic(conn, topic_ko=self._topic, subtopic_ko="__없는것__"),
            idempotent=True)
        self.assertEqual(miss["total"], 0)

    def test_list_topics_asset_count_and_medical_excluded(self):
        from src.topic.asset_topic_query import list_topics
        a, b, c, m = self._seed()

        out = self.db.execute_in_transaction(
            lambda conn: list_topics(conn), idempotent=True)
        entries = [
            o for o in out
            if o["topic_ko"] == self._topic and o["subtopic_ko"] == self._subtopic
        ]
        self.assertEqual(len(entries), 1)               # 유니크 topic — 단일 엔트리
        self.assertEqual(entries[0]["asset_count"], 3)  # {A,B,C} distinct — M 제외
        self.assertEqual(entries[0]["topic_asset_count"], 3)
        self.assertEqual(entries[0]["subtopic_ko"], self._subtopic)
        # 의료 자산 M 은 어떤 엔트리에도 자기 topic 을 노출하지 않는다(SQL 의료 제외).
        self.assertTrue(all("topic_ko" in o for o in out))


if __name__ == "__main__":
    unittest.main()
