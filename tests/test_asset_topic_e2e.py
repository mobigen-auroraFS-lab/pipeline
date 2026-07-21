"""065 T501 — 자산 자기주제 분류 실 DB e2e(RUN_DB_E2E 게이트·SC-05).

무DB 환경에서는 자동 skip(다른 ``*_e2e`` 관례 일치). 게이트 2단(실 LLM 필요 구간 명시 — T501):
  - ``TestAssetTopicClassifyDB``    : ``RUN_DB_E2E=1`` 만 — LLM 은 가짜 주입(``client=``)·kNN 은
    모듈 상단 import seam patch(``asset_topic`` 모듈 docstring 이 명시한 patch 지점) → 네트워크 0.
  - ``TestAssetTopicClassifyRealLLM``: ``RUN_DB_E2E=1`` **+ ``RUN_LLM_E2E=1``** — 활성 채널 임베딩
    API·온프레미스 LLM(temp=0)까지 실제 경로 전체. 기본 e2e 실행을 느리게/불안정하게 만들지 않도록
    별도 게이트로 분리한다.

검증 의도 (T501·SC-05)
    단위 테스트(``test_asset_topic_classify``)는 mock cursor 로 파이썬 분기만 본다. 여기서는
    **실 스키마(v299)와의 정합**을 본다: 테스트 자산 1건 분류 → ``asset_topic`` 행 생성(upsert SQL·
    policy_version 기록) → ``fetch_asset_topic`` 왕복(소비 계약 형상) → **멱등 재실행 동일**(SC-05·
    ON CONFLICT DO UPDATE·updated_at 전이) → 미부여 경로(메타 없음 → LLM 미호출·행 없음).

동시 실행 안전
    유니크 topic_ko·유니크 자산만 만들고 그 스코프만 단언/정리하므로, 라이브 수집(드레인)이 같은 DB 에
    동시 적재 중이어도 서로 간섭하지 않는다. tearDown 은 asset 삭제(CASCADE)로 asset_topic 까지 정리.
"""
from __future__ import annotations

import json
import os
import unittest
import uuid
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

from dotenv import load_dotenv

_RUN = os.getenv("RUN_DB_E2E") == "1"
# 실 LLM 필요 구간의 게이트(T501 "게이트 명시") — 임베딩 API + 온프레미스 LLM 실호출.
_RUN_LLM = os.getenv("RUN_LLM_E2E") == "1"
_ENV = Path(__file__).resolve().parents[1] / ".env.dev"


def _client_returning(content: str) -> MagicMock:
    """OpenAI 호환 응답을 흉내내는 가짜 LLM 클라이언트(테스트 가이드 §2)."""
    c = MagicMock()
    c.chat.completions.create.return_value.choices = [MagicMock()]
    c.chat.completions.create.return_value.choices[0].message.content = content
    return c


class _AssetTopicE2EBase(unittest.TestCase):
    """공통 픽스처 — 유니크 자산 생성·메타 시딩·CASCADE 정리(두 게이트 클래스가 공유)."""

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
        # asset 삭제 → asset_metadata·asset_topic(ON DELETE CASCADE) 연쇄 정리.
        if self._ids:
            with self.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM asset WHERE asset_id = ANY(%s)", (self._ids,))

    def _make_asset(self, *, ext_meta: dict | None = None) -> str:
        """테스트 자산 1건(+선택 메타) 생성. ext_meta=None 이면 asset_metadata 행 자체를 안 만든다."""
        from processing.ingest.asset_persist import create_asset
        with self.db.transaction() as conn:
            aid = create_asset(
                conn, fs_path=f"/t/{uuid.uuid4().hex}.txt", modality="txt",
                domain="general", file_hash=uuid.uuid4().hex)
            if ext_meta is not None:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO asset_metadata (asset_id, core_meta, ext_meta) "
                        "VALUES (%s, %s::jsonb, %s::jsonb)",
                        (aid, "{}", json.dumps(ext_meta, ensure_ascii=False)),
                    )
        self._ids.append(aid)
        return str(aid)

    def _topic_row(self, asset_id: str) -> dict | None:
        """asset_topic 실 행 조회(단언용) — 없으면 None."""
        from psycopg.rows import dict_row
        with self.db.transaction() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                "SELECT topic_ko, topic_en, subtopic_ko, subtopic_en, confidence, "
                "       decided_by, policy_version, created_at, updated_at "
                "FROM asset_topic WHERE asset_id = %s",
                (asset_id,),
            )
            return cur.fetchone()


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만(실 DB·LLM 은 가짜 주입)")
class TestAssetTopicClassifyDB(_AssetTopicE2EBase):
    """가짜 LLM + kNN patch 로 **DB 라운드트립만** 실검증(네트워크 0).

    LLM 판정 분기(재질의·none 도피처 등)는 순수 단위가 덮는다. 여기서는 v299 스키마에 대한
    upsert/fetch SQL 정합·멱등(SC-05)·미부여 경로의 실 DB 동작을 본다.
    """

    def setUp(self):
        super().setUp()
        # 충돌 방지 유니크 topic — 라이브 데이터·동시 수집과 절대 안 겹친다.
        self._topic = "e2e주제_" + uuid.uuid4().hex[:8]
        self._llm_content = json.dumps(
            {"topic_ko": self._topic, "topic_en": "e2e_topic_en",
             "subtopic_ko": None, "subtopic_en": None, "confidence": 0.83},
            ensure_ascii=False,
        )
        self._meta = {
            "summary": "경주 불국사와 석굴암을 둘러보는 여행 기록",
            "keywords": ["여행", "경주", "불국사"],
            "labels": [{"label": "temple", "score": 0.91}],
        }

    def _classify(self, asset_id: str, client) -> dict | None:
        """topic 후보 seam 을 patch 하고 분류 1회 실행 — 임베딩 API 미사용.

        068 G1 이후 후보 seam 은 ``knn_topic_candidates`` 가 아니라 레지스트리 전체-27 조회
        ``topic_candidates_for_self_text`` 다(2026-07-15 B1: 옛 seam patch 는 대상 부재로
        AttributeError·게이트 뒤라 CI 미탐이었음). seam 존재는 비-DB 가드 테스트가 봉인.
        """
        from processing.classify.asset_topic import classify_asset_topic
        # 후보에 catch-all '미분류' 를 섞어 FR-702 배제 필터가 실경로에서도 무해함을 겸사 확인.
        with mock.patch(
            "processing.classify.asset_topic.topic_candidates_for_self_text",
            return_value=[self._topic, "미분류"],
        ):
            return self.db.execute_in_transaction(
                lambda conn: classify_asset_topic(conn, asset_id, client=client),
                idempotent=False,
            )

    def test_classify_creates_row_and_fetch_roundtrip(self):
        """분류 1회 → 행 생성(policy_version 기록) → fetch 왕복 형상(T501 본문)."""
        from processing.classify.asset_topic import POLICY_VERSION
        from src.topic.asset_topic_query import fetch_asset_topic
        aid = self._make_asset(ext_meta=self._meta)
        client = _client_returning(self._llm_content)

        result = self._classify(aid, client)

        # 반환 계약 — 유니크 topic 은 registry 에 없으므로 topic_en 은 LLM 값 폴백(FR-102 역방향).
        self.assertIsNotNone(result)
        self.assertEqual(result["topic_ko"], self._topic)
        self.assertEqual(result["topic_en"], "e2e_topic_en")
        self.assertIsNone(result["subtopic_ko"])
        self.assertEqual(result["decided_by"], "hybrid")
        # topic 확정 1회만 — 후보 내 응답이라 재질의 0 + subtopic None 이라 canonicalize LLM 0.
        self.assertEqual(client.chat.completions.create.call_count, 1)

        # 실 행 — 최초 insert 는 updated_at NULL·policy_version 기록(FR-601).
        row = self._topic_row(aid)
        self.assertIsNotNone(row)
        self.assertEqual(row["topic_ko"], self._topic)
        self.assertEqual(row["policy_version"], POLICY_VERSION)
        self.assertEqual(row["decided_by"], "hybrid")
        self.assertAlmostEqual(float(row["confidence"]), 0.83, places=6)
        self.assertIsNone(row["updated_at"])

        # fetch 왕복 — 구 project_asset_topics 형상(소비처 무변경 스왑 계약).
        out = self.db.execute_in_transaction(
            lambda conn: fetch_asset_topic(conn, aid), idempotent=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(
            set(out[0].keys()),
            {"topic_ko", "subtopic_ko", "topic_en", "subtopic_en", "weight"},
        )
        self.assertEqual(out[0]["topic_ko"], self._topic)
        self.assertEqual(out[0]["weight"], 1)

    def test_rerun_is_idempotent_same_result(self):
        """같은 입력 재실행 → 같은 결과·1행 유지·updated_at 전이(SC-05 멱등·결정성)."""
        aid = self._make_asset(ext_meta=self._meta)

        r1 = self._classify(aid, _client_returning(self._llm_content))
        row1 = self._topic_row(aid)
        r2 = self._classify(aid, _client_returning(self._llm_content))
        row2 = self._topic_row(aid)

        # 결과 동일(결정성) + 행은 여전히 1개(PK upsert — 새 행이 아니라 DO UPDATE).
        self.assertEqual(r1, r2)
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM asset_topic WHERE asset_id = %s", (aid,))
            self.assertEqual(cur.fetchone()[0], 1)
        # created_at 불변·updated_at 은 NULL → now() 전이(재분류 추적·FR-601).
        self.assertEqual(row1["created_at"], row2["created_at"])
        self.assertIsNone(row1["updated_at"])
        self.assertIsNotNone(row2["updated_at"])
        self.assertEqual(
            (row1["topic_ko"], row1["subtopic_ko"]),
            (row2["topic_ko"], row2["subtopic_ko"]),
        )

    def test_no_meta_skips_llm_and_creates_no_row(self):
        """메타 없는 자산 → 자기 텍스트 '' → LLM 미호출·행 없음·fetch [](FR-201 미부여 격리)."""
        from src.topic.asset_topic_query import fetch_asset_topic
        aid = self._make_asset(ext_meta=None)
        client = _client_returning(self._llm_content)

        result = self._classify(aid, client)

        self.assertIsNone(result)
        self.assertEqual(client.chat.completions.create.call_count, 0)
        self.assertIsNone(self._topic_row(aid))
        out = self.db.execute_in_transaction(
            lambda conn: fetch_asset_topic(conn, aid), idempotent=True)
        self.assertEqual(out, [])


@unittest.skipUnless(
    _RUN and _RUN_LLM,
    "RUN_DB_E2E=1 + RUN_LLM_E2E=1 일 때만(활성 채널 임베딩 API·온프레미스 LLM 실호출)",
)
class TestAssetTopicClassifyRealLLM(_AssetTopicE2EBase):
    """실 LLM 골든 — 전체 실경로(임베딩 kNN → LLM temp=0 → canonicalize → upsert)·SC-05.

    뚜렷한 요리 텍스트 1건을 실제로 분류해 ① 행 생성 ② topic 이 registry 닫힌 topic 층에 존재
    (FR-203 닫힌집합) ③ '미분류' 미배정(FR-702) ④ 재실행 동일(SC-05: temp=0 + subtopic alias
    동결 캐시)을 단언한다. canonicalize 가 신규 등록했을 수 있는 subtopic registry/alias 행은
    분류 **전 스냅샷과의 diff**(확정 parent 스코프 한정)로 이 런이 만든 것만 tearDown 에서
    정리한다 — 선존 행·동시 수집이 만든 다른 스코프 행은 건드리지 않는다.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # 실 LLM·활성 채널 임베딩은 settings 프로파일 필요(운영 클라이언트 생성 경로).
        from src.config.settings import init_settings
        init_settings("dev")

    def setUp(self):
        super().setUp()
        self._parent_scope: str | None = None  # 정리 스코프(확정 topic) — 분류 성공 시 기록
        self._pre_subs: set[tuple] = set()     # 분류 전 (parent, subtopic) 스냅샷
        self._pre_alias: set[tuple] = set()    # 분류 전 (parent, raw_ko) 스냅샷

    def tearDown(self):
        # 스냅샷 diff 중 이 런의 parent 스코프 행만 정리(선존·타 스코프 보존 — 라이브 데이터 보호).
        if self._parent_scope:
            parent = self._parent_scope
            with self.db.transaction() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT parent_topic, raw_ko FROM topic_alias WHERE parent_topic = %s",
                    (parent,),
                )
                new_alias = [r for r in cur.fetchall() if tuple(r) not in self._pre_alias]
                for _, raw_ko in new_alias:
                    cur.execute(
                        "DELETE FROM topic_alias WHERE parent_topic = %s AND raw_ko = %s",
                        (parent, raw_ko),
                    )
                cur.execute(
                    "SELECT parent_topic, topic_ko FROM topic_registry "
                    "WHERE parent_topic = %s",
                    (parent,),
                )
                new_subs = [r for r in cur.fetchall() if tuple(r) not in self._pre_subs]
                for _, sub_ko in new_subs:
                    cur.execute(
                        "DELETE FROM topic_registry "
                        "WHERE parent_topic = %s AND topic_ko = %s AND source = 'auto'",
                        (parent, sub_ko),
                    )
        super().tearDown()

    def _snapshot_registry(self) -> None:
        """분류 전 subtopic 층·alias 키 스냅샷(diff 기반 정리의 기준선)."""
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT parent_topic, topic_ko FROM topic_registry "
                "WHERE parent_topic IS NOT NULL"
            )
            self._pre_subs = {tuple(r) for r in cur.fetchall()}
            cur.execute(
                "SELECT parent_topic, raw_ko FROM topic_alias "
                "WHERE parent_topic IS NOT NULL"
            )
            self._pre_alias = {tuple(r) for r in cur.fetchall()}

    def test_real_classify_creates_row_and_rerun_identical(self):
        from processing.classify.asset_topic import classify_asset_topic
        from src.topic.asset_topic_query import fetch_asset_topic
        aid = self._make_asset(ext_meta={
            "summary": "김치찌개를 맛있게 끓이는 법을 단계별로 설명하는 요리 레시피. "
                       "돼지고기와 신김치를 볶아 육수를 붓고 끓여 완성한다.",
            "keywords": ["요리", "레시피", "김치찌개", "한식"],
        })
        self._snapshot_registry()  # tearDown diff 정리의 기준선(분류 전)

        r1 = self.db.execute_in_transaction(
            lambda conn: classify_asset_topic(conn, aid), idempotent=False)

        # 뚜렷한 요리 텍스트가 미부여면 실경로 품질 회귀로 본다.
        self.assertIsNotNone(r1, "실 LLM 분류가 미부여(None) — 실경로 품질 회귀 의심")
        self.assertNotEqual(r1["topic_ko"], "미분류")  # FR-702 catch-all 배제

        # 닫힌집합(FR-203): 확정 topic 은 registry topic 층(닫힌 어휘)에 실존해야 한다.
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM topic_registry "
                "WHERE topic_ko = %s AND parent_topic IS NULL",
                (r1["topic_ko"],),
            )
            self.assertIsNotNone(cur.fetchone(), f"topic '{r1['topic_ko']}' 이 registry 밖")

        # 정리 스코프 기록 — tearDown 이 이 parent 아래의 "스냅샷 이후 신규" 행만 지운다.
        self._parent_scope = r1["topic_ko"]

        # SC-05 멱등·결정성: 같은 입력 재실행 → 같은 (topic, subtopic).
        #   temp=0 + subtopic 은 1차 런이 동결한 alias 캐시 히트(LLM 0)로 수렴한다.
        r2 = self.db.execute_in_transaction(
            lambda conn: classify_asset_topic(conn, aid), idempotent=False)
        self.assertIsNotNone(r2)
        self.assertEqual(
            (r1["topic_ko"], r1["subtopic_ko"]),
            (r2["topic_ko"], r2["subtopic_ko"]),
            "재실행 결과 불일치 — SC-05 결정성 위반",
        )

        # fetch 왕복 — 행 1개·소비 계약 형상.
        out = self.db.execute_in_transaction(
            lambda conn: fetch_asset_topic(conn, aid), idempotent=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["topic_ko"], r1["topic_ko"])
        self.assertEqual(out[0]["weight"], 1)


if __name__ == "__main__":
    unittest.main()
