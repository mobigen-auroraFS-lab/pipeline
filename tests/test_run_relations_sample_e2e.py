"""016 G4 — 샘플 팩 cross_asset 제네릭 러너 실DB e2e (RUN_DB_E2E=1 일 때만).

검증
    SC-001: ``for_domain('sample')`` → 샘플 전략(제네릭 러너) 실행 → ``graph_edge`` 적재
            (core 도메인 코드 분기 0; 008 死코드였던 else 가 실제로 동작함을 실 DB 로 증명).
    SC-003: 동일 입력 2회 실행 결과 동일(결정성·멱등).

설계 메모(driver)
    - domain_label CHECK 가 'sample' 을 불허(plan D-5)하므로, 픽스처 자산은 domain_label='general'
      (create_asset 기본)로 저장하고 ``_domain_fn`` seam 으로 'sample' 을 주입해 비일반 러너 경로를
      강제한다(DB 에 'sample' 미저장 — 스키마·CHECK 무변경).
    - 실 dev DB 에는 운영 자산이 다수 있어 ``find_embedding_candidates`` 가 그들을 후보로 끌 수
      있다. 픽스처 소스/타깃은 동일 one-hot 벡터(코사인 1.0)라 타깃이 항상 top-k 에 들고 match 된다.
    - tearDown 의 자산 DELETE 가 node(asset_id ON DELETE CASCADE) → graph_edge(src/dst_node ON
      DELETE CASCADE) → relation_resolution(asset_id ON DELETE CASCADE)로 흘러 **테스트가 만든
      엣지/큐를 전부 정리한다**(프로덕션 오염 0, FR-004). 실 운영 자산은 건드리지 않는다.
    - SC-002(일반은 propose 위임)는 propose 가 온프레미스 LLM 을 호출해 e2e 가 느리고 흔들리므로
      여기서 재증명하지 않는다 — 단위(test_asset_relations: 일반 경로 runner 미호출 + 라우팅 분기)로
      이미 증명됨. 본 e2e 는 샘플 성공 경로(SC-001/003)에 집중한다.
"""
from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path

from dotenv import load_dotenv

_RUN = os.getenv("RUN_DB_E2E") == "1"
_ENV = Path(__file__).resolve().parents[1] / ".env.dev"


def _onehot(idx: int = 1500) -> list[float]:
    """idx 위치만 1.0 인 1536D one-hot. 운영 임베딩과의 코사인 충돌을 줄이려 고차원 인덱스 사용."""
    from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION

    v = [0.0] * FIX_EMBEDDING_DIMENSION
    v[idx] = 1.0
    return v


def _make_sample_asset(db, ids: list, vec: list[float]) -> str:
    """``/sample_pack/`` 마커 경로 + st 임베딩 보유 registered 자산(FR-004 픽스처). id 를 ids 에 누적."""
    from processing.dispatch.types import AssetRecord, EmbeddingItem
    from processing.ingest.asset_persist import create_asset, finalize_asset
    from processing.ingest.status import AssetStatus, set_status

    with db.transaction() as conn:
        aid = create_asset(
            conn,
            fs_path=f"/sample_pack/{uuid.uuid4().hex}.txt",
            modality="txt",
            file_hash=uuid.uuid4().hex,
        )
    ids.append(aid)
    with db.transaction() as conn:
        set_status(conn, aid, AssetStatus.ROUTING)
        set_status(conn, aid, AssetStatus.CLASSIFYING)
        set_status(conn, aid, AssetStatus.EXTRACTING)
    with db.transaction() as conn:
        finalize_asset(
            conn,
            aid,
            AssetRecord(
                embeddings=[EmbeddingItem(channel="st", vector=vec, model_name="m")],
            ),
        )
    return str(aid)


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만")
class TestRunRelationsSampleE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(_ENV, override=False)
        # run_relations 가 max_attempts=None 일 때 get_current_settings 를 쓰므로 초기화한다
        # (009 e2e 교훈: load_dotenv 후 init_settings 없으면 META_MODEL 등 누락 에러). 본 테스트는
        # max_attempts 를 명시하지만, 진입 경로 안정성을 위해 설정을 초기화해 둔다.
        from src.config.settings import init_settings

        init_settings("dev")
        from src.database.postgres_util import PostgresUtil

        cls.db = PostgresUtil()
        cls.db.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.db.__exit__(None, None, None)

    def setUp(self):
        self._ids: list = []

    def tearDown(self):
        # 자산 삭제 → node/graph_edge/relation_resolution ON DELETE CASCADE 로 테스트 흔적 전부 정리.
        if self._ids:
            with self.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM asset WHERE asset_id = ANY(%s)", (self._ids,))

    def _edge_count(self, src_id: str, dst_id: str) -> int:
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM graph_edge ge
                JOIN node sn ON sn.node_id = ge.src_node AND sn.asset_id = %s
                JOIN node dn ON dn.node_id = ge.dst_node AND dn.asset_id = %s
                """,
                (src_id, dst_id),
            )
            (n,) = cur.fetchone()
        return n

    def test_sample_pack_routes_to_runner_and_persists_edge(self):
        """SC-001: 샘플 팩이 제네릭 러너로 실행돼 graph_edge(src→dst) 적재."""
        from processing.app.run_relations import run_relations

        vec = _onehot()
        src_id = _make_sample_asset(self.db, self._ids, vec)
        dst_id = _make_sample_asset(self.db, self._ids, list(vec))  # 동일 벡터 → 코사인 1.0 → match

        result = run_relations(
            [src_id], db=self.db, max_attempts=3, _domain_fn=lambda db, aid: "sample"
        )
        self.assertEqual(result["failed"], [], f"실패 격리됨: {result['failed']}")
        self.assertGreaterEqual(
            self._edge_count(src_id, dst_id), 1, "샘플 러너가 src→dst graph_edge 를 적재해야 한다"
        )

    def test_deterministic_two_runs(self):
        """SC-003: 동일 입력 2회 실행 — 멱등·결정적(중복 적재 0)."""
        from processing.app.run_relations import run_relations

        vec = _onehot()
        src_id = _make_sample_asset(self.db, self._ids, vec)
        dst_id = _make_sample_asset(self.db, self._ids, list(vec))

        def _run():
            return run_relations(
                [src_id], db=self.db, max_attempts=3, _domain_fn=lambda db, aid: "sample"
            )

        r1 = _run()
        n1 = self._edge_count(src_id, dst_id)
        r2 = _run()
        n2 = self._edge_count(src_id, dst_id)

        self.assertEqual(r1["failed"], [])
        self.assertEqual(r2["failed"], [])
        self.assertGreaterEqual(n1, 1)
        self.assertEqual(n1, n2, "2회 실행 후 동일 엣지 수(멱등·결정성)")


if __name__ == "__main__":
    unittest.main()
