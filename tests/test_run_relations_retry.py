"""009 US3 — run_relations --retry / 미해소 큐 갱신 단위 테스트.

순수 단위(LLM/DB 불필요): ``propose_relations_for_asset`` 주입/monkeypatch 로 자산별 결과를
3케이스(엣지0/예외/성공)로 만들고, relation_resolution 큐 status·attempts 갱신과
큐 upsert 실패 격리(다른 자산 미롤백)를 검증한다.

실 DB 라운드트립(T018)은 ``@skipUnless(RUN_DB_E2E)`` 게이트로 분리한다 — v250 적용 전제.
"""

from __future__ import annotations

import os
import unittest
import uuid
from unittest import mock

from psycopg import Rollback

from processing.app import run_relations as rr
from src.relations.resolution_persist import (
    decide_resolution_status,
    fetch_unresolved_asset_ids,
    upsert_resolution,
)

_A1 = str(uuid.UUID("018f0000-0000-7000-8000-000000000001"))
_A2 = str(uuid.UUID("018f0000-0000-7000-8000-000000000002"))
_A3 = str(uuid.UUID("018f0000-0000-7000-8000-000000000003"))


class _FakeDB:
    """execute_in_transaction(콜백, ...) 을 fake conn 으로 즉시 실행하는 최소 더블.

    실제 PostgresUtil 처럼 콜백 _run(conn) 을 호출하되, conn 동작(attempts 조회·upsert)을
    인메모리 dict 로 흉내 낸다. 큐 upsert 시 fail_on 에 든 asset_id 면 예외를 던져
    격리(다른 자산 미롤백) 검증에 쓴다.
    """

    def __init__(self, *, attempts: dict[str, int] | None = None, fail_on: set[str] | None = None):
        self.attempts = dict(attempts or {})
        self.fail_on = set(fail_on or set())
        self.upserts: list[tuple[str, str, int, str | None]] = []  # (aid, status, attempts, reason)

    def execute_in_transaction(self, fn, idempotent=True):  # noqa: ARG002
        conn = _FakeConn(self)
        return fn(conn)


class _FakeConn:
    def __init__(self, db: _FakeDB):
        self._db = db

    def cursor(self):
        return _FakeCursor(self._db)


class _FakeCursor:
    """upsert_resolution / _fetch_attempts 가 부르는 cursor.execute 를 가로채 인메모리 반영."""

    def __init__(self, db: _FakeDB):
        self._db = db
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        s = " ".join(sql.split())
        if s.startswith("SELECT attempts FROM relation_resolution"):
            aid = params[0]
            n = self._db.attempts.get(aid)
            self._row = (n,) if n is not None else None
            return
        if s.startswith("INSERT INTO relation_resolution"):
            aid, status, attempts, reason = params
            if aid in self._db.fail_on:
                raise RuntimeError("simulated queue upsert failure")
            self._db.attempts[aid] = attempts
            self._db.upserts.append((aid, status, attempts, reason))
            return
        raise AssertionError(f"unexpected SQL in fake cursor: {s[:60]}")

    def fetchone(self):
        return self._row


class TestRecordResolutionIsolation(unittest.TestCase):
    """_record_resolution: 자산별 fresh 트랜잭션 격리 + 비식별 last_reason."""

    def test_edges_zero_writes_isolated_status_and_reason(self) -> None:
        # 035 #2(B): 고립(엣지0·예외 없음)은 status='isolated'(attempts 불변)·last_reason=고립 표식.
        db = _FakeDB(attempts={})
        rr._record_resolution(db, _A1, 0, error=None, max_attempts=3)
        self.assertEqual(len(db.upserts), 1)
        aid, status, attempts, reason = db.upserts[0]
        self.assertEqual((aid, status, attempts), (_A1, "isolated", 0))
        self.assertEqual(reason, rr._REASON_ISOLATED)  # 고립 비식별 표식

    def test_exception_writes_pending_type_name_only(self) -> None:
        db = _FakeDB(attempts={})
        # 예외 메시지에 풀경로/PHI 가 있어도 큐에는 예외 타입명만 들어가야 한다(헌법 10조).
        err = RuntimeError("/secret/path/patient_홍길동.dcm 실패")
        rr._record_resolution(db, _A1, 0, error=err, max_attempts=3)
        aid, status, attempts, reason = db.upserts[0]
        self.assertEqual((status, attempts), ("pending", 1))
        self.assertEqual(reason, "RuntimeError")
        self.assertNotIn("홍길동", reason)
        self.assertNotIn("/secret/", reason)

    def test_edges_one_writes_resolved_attempts_unchanged(self) -> None:
        db = _FakeDB(attempts={_A1: 2})  # 이미 2회 시도
        rr._record_resolution(db, _A1, 1, error=None, max_attempts=3)
        aid, status, attempts, reason = db.upserts[0]
        self.assertEqual((status, attempts), ("resolved", 2))  # attempts 불변

    def test_attempts_at_max_escalates_failed(self) -> None:
        db = _FakeDB(attempts={_A1: 2})  # +1=3=max → failed(DLQ)
        rr._record_resolution(db, _A1, 0, error=RuntimeError("e"), max_attempts=3)
        aid, status, attempts, reason = db.upserts[0]
        self.assertEqual((status, attempts), ("failed", 3))

    def test_queue_upsert_failure_swallowed(self) -> None:
        # 큐 upsert 가 실패해도 예외가 전파되지 않아야 한다(다른 자산 미롤백 전제).
        db = _FakeDB(attempts={}, fail_on={_A1})
        rr._record_resolution(db, _A1, 0, error=None, max_attempts=3)  # 예외 안 남
        self.assertEqual(db.upserts, [])  # 실패라 기록 안 됨


class TestRunRelationsQueueWiring(unittest.TestCase):
    """run_relations 루프: 3케이스(엣지0/예외/성공) → 각 자산 큐 갱신 + 격리."""

    def _run(self, db, *, propose, domain="general"):
        with mock.patch.object(rr, "propose_relations_for_asset", side_effect=propose), \
             mock.patch.object(rr, "_fetch_domain_label", return_value=domain):
            return rr.run_relations([_A1, _A2, _A3], db=db, max_attempts=3)

    def test_three_cases_queue_status(self) -> None:
        db = _FakeDB(attempts={})

        def propose(db_, aid, *, top_k=None, embedding_kind="both"):
            if aid == _A1:  # 엣지0(고립)
                return (0, 0, 0, 0)
            if aid == _A2:  # 예외(일시 실패)
                raise RuntimeError("LLM timeout")
            return (1, 1, 2, 0)  # 성공(엣지2)

        res = self._run(db, propose=propose)
        self.assertEqual(len(res["done"]), 2)   # A1, A3
        self.assertEqual(len(res["failed"]), 1)  # A2
        by_aid = {u[0]: u for u in db.upserts}
        self.assertEqual(by_aid[_A1][1:3], ("isolated", 0))  # 035 #2: 고립=isolated(재시도 없음)
        self.assertEqual(by_aid[_A2][1:3], ("pending", 1))   # 예외(일시 실패)는 재시도 유지
        self.assertEqual(by_aid[_A2][3], "RuntimeError")     # 비식별 타입명
        self.assertEqual(by_aid[_A3][1:3], ("resolved", 0))  # 성공

    def test_queue_failure_does_not_stop_other_assets(self) -> None:
        # A1 큐 upsert 가 실패해도 A2·A3 처리·큐 갱신은 계속된다(SC-008 격리).
        db = _FakeDB(attempts={}, fail_on={_A1})

        def propose(db_, aid, *, top_k=None, embedding_kind="both"):
            return (1, 1, 1, 0)  # 전부 성공

        res = self._run(db, propose=propose)
        self.assertEqual(len(res["done"]), 3)  # 모두 관계 적재 성공(큐 실패와 무관)
        upserted_aids = {u[0] for u in db.upserts}
        self.assertNotIn(_A1, upserted_aids)  # A1 큐는 실패(미기록)
        self.assertIn(_A2, upserted_aids)     # A2·A3 큐는 정상 갱신
        self.assertIn(_A3, upserted_aids)

    def test_max_attempts_from_settings_when_none(self) -> None:
        # max_attempts 미지정 시 현재 설정값을 사용한다(여기선 1로 주입).
        db = _FakeDB(attempts={})
        fake_settings = mock.MagicMock(relations=mock.MagicMock(retry_max_attempts=1))

        def propose(db_, aid, *, top_k=None, embedding_kind="both"):
            raise RuntimeError("boom")  # 첫 실패 → attempts 0→1 == max(1) → failed

        with mock.patch.object(rr, "propose_relations_for_asset", side_effect=propose), \
             mock.patch.object(rr, "_fetch_domain_label", return_value="general"), \
             mock.patch("src.config.settings.get_current_settings", return_value=fake_settings):
            rr.run_relations([_A1], db=db)
        self.assertEqual(db.upserts[0][1:3], ("failed", 1))


# =============================================================================
# T018 실 DB 라운드트립(게이트) — v250 적용 전제. RUN_DB_E2E=1 에서만 실행.
#   * upsert/조회 라운드트립(now() updated_at 갱신·attempts 증분)
#   * --retry(fetch_unresolved_asset_ids) 가 pending+미시도만 반환, resolved/failed 제외
#   * 선택 순서 결정성(attempts ASC, created_at ASC) — 2회 실행 동일
# 실 DB 실행·v250 적용은 driver 게이트 몫. 본 워커는 게이트 테스트 작성만 한다.
# =============================================================================
@unittest.skipUnless(os.environ.get("RUN_DB_E2E") == "1", "실 DB 게이트(RUN_DB_E2E=1)")
class TestResolutionPersistDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path

        from dotenv import load_dotenv

        from src.config.settings import init_settings
        from src.database.postgres_util import PostgresUtil

        # 다른 e2e 와 동일 패턴: init_settings 전에 .env.dev 로드(META_MODEL 등 필수 env).
        load_dotenv(Path(__file__).resolve().parents[1] / ".env.dev", override=False)
        init_settings("dev")
        cls.db = PostgresUtil()
        cls.db.open_pool()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def _insert_registered_asset_with_embedding(self, conn, created_at_offset_sec: int) -> str:
        """테스트용 registered 자산 + 임베딩 1건 적재. asset_id 반환."""
        from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
        from src.database.ids import uuid7_str

        aid = uuid7_str()
        zero_vec = "[" + ",".join(["0"] * FIX_EMBEDDING_DIMENSION) + "]"
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO asset (asset_id, fs_path, modality, status, file_hash, created_at)
                VALUES (%s, %s, 'text', 'registered', %s,
                        now() - make_interval(secs => %s))
                """,
                (aid, f"/tmp/relret_{aid}.txt", f"hash_{aid}", created_at_offset_sec),
            )
            cur.execute(
                """
                INSERT INTO asset_embedding (asset_id, channel, chunk_index, embedding, model_name)
                VALUES (%s, 'st', 0, %s::vector, 'test')
                """,
                (aid, zero_vec),
            )
        return aid

    def test_upsert_and_fetch_roundtrip(self) -> None:
        """upsert→재upsert(updated_at 갱신·attempts 증분), --retry 선택·제외·결정적 순서."""
        with self.db.transaction() as conn:
            # 미시도(큐 행 없음) 2건 + pending 1건 + resolved 1건 + failed 1건.
            a_untried_old = self._insert_registered_asset_with_embedding(conn, 100)  # 가장 오래됨
            a_untried_new = self._insert_registered_asset_with_embedding(conn, 50)
            a_pending = self._insert_registered_asset_with_embedding(conn, 80)
            a_resolved = self._insert_registered_asset_with_embedding(conn, 70)
            a_failed = self._insert_registered_asset_with_embedding(conn, 60)

            upsert_resolution(conn, a_pending, status="pending", attempts=1, reason="isolated:no_edges")
            upsert_resolution(conn, a_resolved, status="resolved", attempts=1, reason=None)
            upsert_resolution(conn, a_failed, status="failed", attempts=3, reason="RuntimeError")

            # 라운드트립: 재upsert 시 updated_at 갱신 + attempts 증분 반영.
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT updated_at FROM relation_resolution WHERE asset_id=%s", (a_pending,)
                )
                t1 = cur.fetchone()[0]
            upsert_resolution(conn, a_pending, status="pending", attempts=2, reason="ValueError")
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status, attempts, last_reason, updated_at "
                    "FROM relation_resolution WHERE asset_id=%s",
                    (a_pending,),
                )
                status, attempts, reason, t2 = cur.fetchone()
            self.assertEqual((status, attempts, reason), ("pending", 2, "ValueError"))
            self.assertGreaterEqual(t2, t1)  # updated_at = now() 명시 갱신

            # --retry 선택: pending + 미시도만, resolved/failed 제외.
            selected = fetch_unresolved_asset_ids(conn)
            self.assertIn(a_pending, selected)
            self.assertIn(a_untried_old, selected)
            self.assertIn(a_untried_new, selected)
            self.assertNotIn(a_resolved, selected)
            self.assertNotIn(a_failed, selected)

            # 결정성: 2회 실행 동일 집합·순서.
            selected2 = fetch_unresolved_asset_ids(conn)
            self.assertEqual(selected, selected2)

            # 정렬: 미시도(attempts NULL→FIRST)가 pending(attempts=2)보다 앞.
            #       미시도 동률은 created_at ASC(오래된 a_untried_old 가 a_untried_new 보다 앞).
            idx_old = selected.index(a_untried_old)
            idx_new = selected.index(a_untried_new)
            idx_pending = selected.index(a_pending)
            self.assertLess(idx_old, idx_new)
            self.assertLess(idx_new, idx_pending)

            raise Rollback()  # 테스트 격리: 변경 폐기(PostgresUtil transaction 이 Rollback 을 잡아 롤백)

    def test_isolation_isolated_and_excluded_from_rescan(self) -> None:
        """035 #2 SC-005: 고립(엣지0·예외없음) → 순수 결정·실 DB 영속 모두 isolated,
        미해소 스캔에서 제외(재시도·DLQ 없음)."""
        with self.db.transaction() as conn:
            aid = self._insert_registered_asset_with_embedding(conn, 10)
            # 전제: 큐 행 없으면 미해소로 선택된다.
            self.assertIn(aid, fetch_unresolved_asset_ids(conn))
            status, attempts = decide_resolution_status(0, 0, error=None, max_attempts=3)
            self.assertEqual((status, attempts), ("isolated", 0))  # 035 #2: 고립=isolated(attempts 불변)
            upsert_resolution(conn, aid, status=status, attempts=attempts, reason=rr._REASON_ISOLATED)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status, attempts, last_reason FROM relation_resolution WHERE asset_id=%s",
                    (aid,),
                )
                self.assertEqual(tuple(cur.fetchone()), ("isolated", 0, "isolated:no_edges"))
            # isolated 라 미해소 스캔에서 제외 → 다음 tick 재선택 안 됨(고립 무한 재시도·DLQ 방지).
            self.assertNotIn(aid, fetch_unresolved_asset_ids(conn))
            raise Rollback()  # 테스트 격리: 변경 폐기


if __name__ == "__main__":
    unittest.main()
