"""009 US1 — 상태 전이 원자성(조건부 UPDATE) 단위 + 실 DB 라운드트립 테스트.

목적
    `set_status`/`mark_failed` 가 read-modify-write(SELECT→검증→UPDATE)의 lost update 를
    막도록 **조건부 `UPDATE ... WHERE asset_id=%s AND status=<기대 현재상태>`** 로 원자화됐는지
    검증한다. 동시 워커가 같은 현재 상태를 읽어도 DB 가 한쪽만 적용(나머지 0행)하고,
    0행이면 명시적 충돌 오류(`ConcurrentTransitionError`)를 낸다.

    헌법 8조: FSM 정의(`validate_transition`/`ALLOWED_TRANSITIONS`/`AssetStatus`)·CHECK 무변경.
    단일 워커 동작은 100% 동일(정상 전이·status_reason·검증 순서 불변, 회귀 0).

설계
    순수 단위: psycopg `Connection` 을 mock 으로 대체. 충돌은 `cursor.rowcount=0` 모킹으로
    "다른 워커가 먼저 상태를 바꿈"을 시뮬레이션한다(SELECT→검증 통과 후 조건부 UPDATE 0행).
    실 DB 라운드트립(`RUN_DB_E2E=1`): 두 커넥션 동시 전이가 단일 종료 상태로 수렴.
"""

from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path
from unittest import mock

from processing.ingest.status import (
    AssetStatus,
    ConcurrentTransitionError,
    InvalidTransitionError,
    mark_failed,
    set_status,
)


def _conn(status: str | None, *, rowcount: int = 1, refetch: str | None = None):
    """mock Connection.

    - fetchone 1회차: set_status 의 fetch_status(현재 상태 SELECT) → {'status': status}.
    - UPDATE 후 cursor.rowcount = rowcount (0 이면 충돌 시뮬레이션).
    - refetch: 충돌 시 현재 상태 재조회 SELECT 가 돌려줄 값(0행 경로 진단용).
    """
    conn = mock.MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.rowcount = rowcount
    # 1회차 fetch(fetch_status, dict_row 커서) → {'status': ...}.
    fetched = [None if status is None else {"status": status}]
    # 충돌(0행) 재조회 fetch(set_status 내부, 일반 커서) → row[0] 로 접근 → 1-튜플.
    if refetch is not None:
        fetched.append((refetch,))
    cur.fetchone.side_effect = fetched
    return conn, cur


def _update_calls(cur) -> list:
    return [c for c in cur.execute.call_args_list if "UPDATE asset" in c.args[0]]


class TestConditionalUpdateAtomicity(unittest.TestCase):
    """T006: 충돌(0행) 시뮬레이션 — 조건부 UPDATE 가 0행이면 충돌 오류."""

    def test_update_is_conditional_on_expected_current_status(self) -> None:
        # 조건부 UPDATE 는 WHERE 에 기대 현재상태를 포함해야 한다(lost update 방지).
        conn, cur = _conn("received")
        set_status(conn, 7, "routing")
        upd = _update_calls(cur)
        self.assertEqual(len(upd), 1)
        sql = upd[0].args[0]
        self.assertIn("status = %s", sql)  # WHERE status=<기대 현재상태> 가드
        # 파라미터에 asset_id 와 기대 현재상태(received)가 모두 포함돼야 한다.
        self.assertIn(7, upd[0].args[1])
        self.assertIn("received", upd[0].args[1])

    def test_zero_rows_raises_concurrent_conflict(self) -> None:
        # 다른 워커가 먼저 상태를 바꿈 → 조건부 UPDATE 0행 → 충돌 오류.
        conn, _ = _conn("extracting", rowcount=0, refetch="registered")
        with self.assertRaises(ConcurrentTransitionError):
            set_status(conn, 9, "registered")

    def test_conflict_error_is_invalid_transition_subclass(self) -> None:
        # 충돌 오류는 InvalidTransitionError 계열이어야 한다
        # (run_ingest 의 fresh 트랜잭션 mark_failed 격리부가 그대로 흡수 — 배치 비중단).
        self.assertTrue(issubclass(ConcurrentTransitionError, InvalidTransitionError))

    def test_zero_rows_message_includes_observed_status(self) -> None:
        # 0행이면 현재 상태를 재조회해 충돌 메시지에 담는다(디버깅 가치, plan Phase0 #2).
        conn, _ = _conn("extracting", rowcount=0, refetch="failed")
        with self.assertRaises(ConcurrentTransitionError) as ctx:
            set_status(conn, 9, "registered")
        self.assertIn("failed", str(ctx.exception))  # 그사이 관측된 실제 상태

    def test_successful_update_does_not_raise(self) -> None:
        # 정상(rowcount=1) 단일 워커 경로: 충돌 없이 통과(회귀 0).
        conn, _ = _conn("received", rowcount=1)
        set_status(conn, 7, "routing")  # 예외 없어야

    def test_validate_runs_before_update_unchanged(self) -> None:
        # FSM 검증은 UPDATE 전에 그대로 수행 — 비허용 전이는 UPDATE 도달 전 거부(검증 순서 불변).
        conn, cur = _conn("registered")  # terminal → routing 불가
        with self.assertRaises(InvalidTransitionError):
            set_status(conn, 7, "routing")
        self.assertEqual(_update_calls(cur), [])  # UPDATE 미발생


class TestMarkFailedAtomicity(unittest.TestCase):
    """T008: mark_failed 도 조건부 UPDATE 와 정합 — 충돌은 InvalidTransitionError 계열."""

    def test_mark_failed_zero_rows_raises_concurrent(self) -> None:
        # 다른 워커가 먼저 registered 로 종료 → mark_failed 의 조건부 UPDATE 0행 → 충돌.
        conn, _ = _conn("extracting", rowcount=0, refetch="registered")
        with self.assertRaises(ConcurrentTransitionError):
            mark_failed(conn, 9, "추출 실패")

    def test_mark_failed_conflict_absorbed_as_invalid_transition(self) -> None:
        # run_ingest 격리부는 `except InvalidTransitionError` 로 잡는다 → 충돌이 흡수되는지(subclass).
        conn, _ = _conn("extracting", rowcount=0, refetch="failed")
        try:
            mark_failed(conn, 9, "추출 실패")
        except InvalidTransitionError:
            pass  # 흡수 가능(배치 비중단)
        else:
            self.fail("충돌이 InvalidTransitionError 로 흡수되지 않음")

    def test_mark_failed_success_unchanged(self) -> None:
        # 단일 워커 정상 실패 기록: status_reason 포함 UPDATE, 충돌 없음(회귀 0).
        conn, cur = _conn("extracting", rowcount=1)
        mark_failed(conn, 9, "지원하지 않는 modality")
        upd = _update_calls(cur)
        self.assertEqual(len(upd), 1)
        self.assertIn("failed", upd[0].args[1])
        self.assertIn("지원하지 않는 modality", upd[0].args[1])


# ── T009: 실 DB 라운드트립(게이트) — 동시 전이 수렴 + 단일 워커 회귀 0 ──────────────
_RUN = os.getenv("RUN_DB_E2E") == "1"
_ENV = Path(__file__).resolve().parents[1] / ".env.dev"


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만 실행하는 실 DB e2e")
class TestStatusTransitionAtomicE2E(unittest.TestCase):
    """동시 전이가 단일 종료 상태로 수렴 + 단일 워커 정상 전이·status_reason 회귀 0.

    두 커넥션(워커 A/B)이 같은 `extracting` 자산에 경쟁 전이(registered vs failed)를
    시도하면 먼저 커밋한 쪽만 적용되고 나머지는 0행 → ConcurrentTransitionError.
    자산은 단 하나의 종료 상태만 갖는다(SC-001).
    """

    db = None  # type: ignore[assignment]

    @classmethod
    def setUpClass(cls) -> None:
        from dotenv import load_dotenv

        load_dotenv(_ENV, override=False)
        from src.database.postgres_util import PostgresUtil

        try:
            cls.db = PostgresUtil()
            cls.db.__enter__()
            with cls.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.asset')")
                if cur.fetchone()[0] is None:
                    raise unittest.SkipTest("asset 스키마 미적용")
        except unittest.SkipTest:
            raise
        except Exception as exc:  # noqa: BLE001 — 접속 불가 시 깔끔히 skip
            raise unittest.SkipTest(f"DB 미접속: {type(exc).__name__}: {exc}") from None

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.db is not None:
            cls.db.__exit__(None, None, None)

    def setUp(self) -> None:
        self._ids: list[uuid.UUID] = []

    def tearDown(self) -> None:
        if self._ids:
            with self.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM asset WHERE asset_id = ANY(%s)", (self._ids,))

    def _seed_extracting(self) -> uuid.UUID:
        """extracting 상태 자산 1행 시드(라우팅 묶음을 거쳐 정상 전이 — 회귀 검증 겸).

        create_asset(received)→routing→classifying→extracting 까지 단일 워커 정상 경로로
        진행한다. 각 set_status 가 예외 없이 진행되면 단일 워커 회귀 0(정상 전이 불변).
        """
        from processing.ingest.asset_persist import create_asset

        with self.db.transaction() as conn:
            aid = create_asset(
                conn,
                fs_path=f"/tmp/atomic-{uuid.uuid4().hex}.txt",
                modality="txt",
                domain="general",
                file_hash=uuid.uuid4().hex,
                file_size=1,
            )
        self._ids.append(aid)
        with self.db.transaction() as conn:
            set_status(conn, aid, AssetStatus.ROUTING)
            set_status(conn, aid, AssetStatus.CLASSIFYING)
            set_status(conn, aid, AssetStatus.EXTRACTING)
        return aid

    def _status(self, aid: uuid.UUID) -> str:
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute("SELECT status FROM asset WHERE asset_id=%s", (aid,))
            return cur.fetchone()[0]

    def test_single_worker_normal_transition_regression(self) -> None:
        # 단일 워커 정상 전이: extracting→registered 가 status_reason=None 으로 적용(회귀 0).
        aid = self._seed_extracting()
        with self.db.transaction() as conn:
            set_status(conn, aid, AssetStatus.REGISTERED)
        self.assertEqual(self._status(aid), "registered")

    def test_serialized_loser_is_rejected_single_terminal(self) -> None:
        # 직렬화된 경우(A 가 B 시작 전 커밋): B 의 set_status 는 현재 상태(registered)를 읽어
        # validate_transition 단계에서 거부(InvalidTransitionError) — 자산은 단 하나의 종료 상태.
        # (충돌이 검증 단계에서 잡히든 조건부 UPDATE 0행에서 잡히든, B 는 결코 이기지 못한다 = SC-001.)
        aid = self._seed_extracting()
        with self.db.transaction() as conn_a:
            set_status(conn_a, aid, AssetStatus.REGISTERED)  # A 먼저 종료(커밋)
        with self.db.transaction() as conn_b:
            with self.assertRaises(InvalidTransitionError):
                set_status(conn_b, aid, AssetStatus.FAILED, reason="추출 실패")
        self.assertEqual(self._status(aid), "registered")

    def test_interleaved_conditional_update_rejects_lost_update(self) -> None:
        # 진짜 동시성(인터리빙): 두 커넥션이 모두 extracting 을 읽은 상태에서, A 가 먼저
        # extracting→registered 를 커밋하면, B 의 조건부 UPDATE(WHERE status='extracting')는
        # 0행이 되어 lost update 가 DB 레벨에서 거부된다(SC-001). set_status 가 내는 것과
        # 동일한 조건부 UPDATE SQL 을 두 커넥션으로 인터리빙해 원자성을 직접 검증한다.
        aid = self._seed_extracting()
        upd = (
            "UPDATE asset SET status=%s, status_reason=%s, updated_at=now() "
            "WHERE asset_id=%s AND status=%s"
        )
        with self.db.connection() as ca, self.db.connection() as cb:
            ca.autocommit = False
            cb.autocommit = False
            # 1) 두 워커가 같은 현재 상태(extracting)를 읽음.
            with cb.cursor() as curb:
                curb.execute("SELECT status FROM asset WHERE asset_id=%s", (aid,))
                self.assertEqual(curb.fetchone()[0], "extracting")
            # 2) 워커 A: extracting→registered 적용 후 커밋(1행).
            with ca.cursor() as cura:
                cura.execute(upd, ("registered", None, aid, "extracting"))
                self.assertEqual(cura.rowcount, 1)
            ca.commit()
            # 3) 워커 B: 여전히 extracting 을 기대한 조건부 UPDATE → 0행(lost update 거부).
            with cb.cursor() as curb:
                curb.execute(upd, ("failed", "추출 실패", aid, "extracting"))
                self.assertEqual(curb.rowcount, 0)  # 충돌 — B 는 적용되지 않음
            cb.rollback()
        # 자산은 단 하나의 종료 상태(registered)만 가짐.
        self.assertEqual(self._status(aid), "registered")

    def test_status_reason_recorded_on_normal_failed(self) -> None:
        # 단일 워커 실패 기록: status_reason 이 그대로 보존(회귀 0).
        aid = self._seed_extracting()
        with self.db.transaction() as conn:
            mark_failed(conn, aid, "지원하지 않는 modality")
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute("SELECT status, status_reason FROM asset WHERE asset_id=%s", (aid,))
            status, reason = cur.fetchone()
        self.assertEqual(status, "failed")
        self.assertEqual(reason, "지원하지 않는 modality")


if __name__ == "__main__":
    unittest.main()
