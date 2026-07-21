"""030 G2(T002·T003·T004) — DAG 로직 함수(batch_runner) 단위 테스트.

목적
    Airflow DAG 가 호출할 **순수 스캔·claim·배치·cap 함수**를 Airflow/실DB 없이(가짜 conn/db·
    주입 fn) 단언한다. 상태 정본은 ``asset.status``(009 FSM)·``relation_resolution``(v250)·
    ``asset_lineage``(재시도 cap 카운트). **신규 큐 테이블 0**(ingest_job 미도입, SC-011).

설계(핵심 불변식 — spec 030 / ADR 2026-06-16)
    · 원자 claim = ``process_asset`` 의 첫 조건부 전이(received→routing, 009). batch 는 received
      자산마다 process_asset 를 호출하고 **첫 전이 0행(InvalidTransitionError 계열)이면 스킵**한다 —
      충돌하는 별도 claim UPDATE 를 두지 않는다(constraint #1).
    · 고착(crash) 자산 = ``claim_asset`` 조건부 UPDATE 로 received 리셋 후 다음 처리에서 재처리.
    · 재시도 cap = ``asset_lineage`` 의 ``ingest.failed.v1`` 수 ≥ N 이면 ``mark_failed``(종료 격리).
    · 종료 계열(registered/failed/deferred)은 received/고착 스캔에서 제외(deferred 는 재시도 아님).

모두 실 DB·LLM·파일·Airflow 불필요. FR-011(Airflow import 0)·SC-011(ingest_job 부재)도 단언.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

from processing.dispatch.types import AssetRecord
from processing.ingest import batch_runner as br
from processing.ingest.router import RouteResult
from processing.ingest.status import (
    TERMINAL,
    AssetStatus,
    ConcurrentTransitionError,
    InvalidTransitionError,
)

_A1 = uuid.UUID("018f0000-0000-7000-8000-000000000001")
_A2 = uuid.UUID("018f0000-0000-7000-8000-000000000002")
_A3 = uuid.UUID("018f0000-0000-7000-8000-000000000003")


def _conn(rows=(), *, rowcount=1, fetchone=None):
    """가짜 psycopg Connection — cursor.execute 인자 기록 + fetchall/fetchone/rowcount 제어."""
    conn = mock.MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchall.return_value = list(rows)
    cur.rowcount = rowcount
    if fetchone is not None:
        cur.fetchone.return_value = fetchone
    return conn, cur


def _last_sql(cur) -> str:
    return cur.execute.call_args.args[0]


def _last_params(cur):
    return cur.execute.call_args.args[1]


def _route(modality="txt", domain="general", routable=True, reason=""):
    return RouteResult("/d/a." + modality, modality, domain, routable, reason)


_NO_OS_SETTINGS = types.SimpleNamespace(opensearch=types.SimpleNamespace(sync_enabled=False))


def _os_settings(*, enabled: bool, channel: str = "st") -> types.SimpleNamespace:
    """process_received_batch 가 ``_make_opensearch_indexer`` 로 넘길 설정 대역(필요 필드만)."""
    return types.SimpleNamespace(
        opensearch=types.SimpleNamespace(
            sync_enabled=enabled,
            url="http://localhost:9200",
            index="assets",
            filename_noise_patterns=(),
        ),
        embed=types.SimpleNamespace(active_channel=channel),
    )


@contextlib.contextmanager
def _patched_os_sync():
    """지연 import 되는 ``src.search.opensearch_sync`` 를 가짜 모듈로 대체 — index_asset/get_client 기록."""
    fake = types.ModuleType("src.search.opensearch_sync")
    fake.index_asset = mock.MagicMock(name="index_asset")
    fake.get_client = mock.MagicMock(name="get_client", return_value=mock.MagicMock())
    with mock.patch.dict(sys.modules, {"src.search.opensearch_sync": fake}):
        yield fake


def _order_by_clause(sql: str) -> str:
    """SQL 에서 ORDER BY 이후 부분(LIMIT 직전까지)을 잘라 정렬키 순서를 단언하기 쉽게 한다."""
    tail = sql[sql.index("ORDER BY"):]
    return tail[: tail.index("LIMIT")] if "LIMIT" in tail else tail


# ── T002: 스캔 함수 SQL 계약 ──────────────────────────────────────────────────


class TestScanReceived(unittest.TestCase):
    def test_selects_received_ordered_with_limit_and_maps_rows(self) -> None:
        conn, cur = _conn(rows=[(_A1, "/d/a.txt"), (_A2, "/d/b.txt")])
        out = br.scan_received_assets(conn, limit=50)
        # (asset_id, fs_path) 튜플 목록으로 매핑
        self.assertEqual(out, [(_A1, "/d/a.txt"), (_A2, "/d/b.txt")])
        sql = _last_sql(cur)
        self.assertIn("status = 'received'", sql)
        self.assertIn("ORDER BY created_at", sql)
        self.assertIn("LIMIT %s", sql)
        self.assertIn(50, _last_params(cur))

    def test_orders_by_created_at_then_asset_id_deterministic(self) -> None:
        # 헌법 3조·FR-012: 동일 created_at·대상>limit 면 보조 정렬(asset_id ASC·UUIDv7 자연 tiebreaker)이
        # 경계 자산을 결정적으로 고정한다 — 형제 scan_unresolved_assets 와 동형.
        conn, cur = _conn(rows=[])
        br.scan_received_assets(conn, limit=50)
        order = _order_by_clause(_last_sql(cur))
        self.assertIn("created_at", order)
        self.assertIn("asset_id", order)
        # 1차 키(created_at) 뒤에 2차 tiebreaker(asset_id) 가 와야 한다.
        self.assertLess(order.index("created_at"), order.index("asset_id"))


class TestScanStuck(unittest.TestCase):
    def test_selects_only_nonterminal_stuck_excludes_terminal(self) -> None:
        conn, cur = _conn(rows=[(_A1, "extracting")])
        out = br.scan_stuck_assets(conn, older_than_s=900, limit=50)
        self.assertEqual(out, [(_A1, "extracting")])
        sql = _last_sql(cur)
        # 비종료 고착만 — routing/classifying/extracting
        self.assertIn("routing", sql)
        self.assertIn("classifying", sql)
        self.assertIn("extracting", sql)
        # 종료 계열은 스캔 대상 아님
        self.assertNotIn("deferred", sql)
        self.assertNotIn("registered", sql)
        self.assertNotIn("'failed'", sql)
        # 임계시간 경과 가드(updated_at < now()-interval)
        self.assertIn("updated_at <", sql)
        params = _last_params(cur)
        self.assertIn(900, params)
        self.assertIn(50, params)

    def test_orders_by_updated_at_then_asset_id_deterministic(self) -> None:
        # 헌법 3조·FR-012: 동일 updated_at·대상>limit 시 asset_id ASC 보조 정렬로 결정적 경계 선택.
        conn, cur = _conn(rows=[])
        br.scan_stuck_assets(conn, older_than_s=900, limit=50)
        order = _order_by_clause(_last_sql(cur))
        self.assertIn("updated_at", order)
        self.assertIn("asset_id", order)
        self.assertLess(order.index("updated_at"), order.index("asset_id"))


class TestScanUnresolved(unittest.TestCase):
    def test_selects_registered_unresolved_left_join(self) -> None:
        conn, cur = _conn(rows=[(_A1,), (_A2,)])
        out = br.scan_unresolved_assets(conn, limit=50)
        self.assertEqual(out, [_A1, _A2])
        sql = _last_sql(cur)
        self.assertIn("LEFT JOIN relation_resolution", sql)
        self.assertIn("status = 'registered'", sql)
        # 미해소(pending) 또는 큐 행 없음(IS NULL) — resolved/failed 자연 제외
        self.assertIn("pending", sql)
        self.assertIn("IS NULL", sql)
        self.assertIn("LIMIT %s", sql)
        self.assertIn(50, _last_params(cur))


# ── T002: 원자 claim 프리미티브 ───────────────────────────────────────────────


class TestClaimAsset(unittest.TestCase):
    def test_conditional_update_returns_true_on_one_row(self) -> None:
        conn, cur = _conn(rowcount=1)
        ok = br.claim_asset(conn, _A1, expected="extracting", next="received")
        self.assertTrue(ok)
        sql = _last_sql(cur)
        self.assertIn("UPDATE asset", sql)
        self.assertIn("WHERE asset_id = %s AND status = %s", sql)
        self.assertIn("updated_at = now()", sql)
        # params 순서: (next, asset_id, expected) — 조건부 기대 현재상태 가드
        self.assertEqual(_last_params(cur), ("received", _A1, "extracting"))

    def test_zero_rows_returns_false(self) -> None:
        # 그사이 다른 run 이 상태를 바꿈 → 조건부 UPDATE 0행 → 점유 실패(스킵 신호)
        conn, _ = _conn(rowcount=0)
        self.assertFalse(br.claim_asset(conn, _A1, expected="extracting", next="received"))

    def test_atomicity_one_of_two_concurrent_wins(self) -> None:
        # 동시 2회 중 1회만 True — 조건부 UPDATE 원자성(winner=1행, loser=0행).
        conn_win, _ = _conn(rowcount=1)
        conn_lose, _ = _conn(rowcount=0)
        r1 = br.claim_asset(conn_win, _A1, expected="received", next="routing")
        r2 = br.claim_asset(conn_lose, _A1, expected="received", next="routing")
        self.assertEqual([r1, r2].count(True), 1)

    def test_accepts_assetstatus_enum(self) -> None:
        conn, cur = _conn(rowcount=1)
        br.claim_asset(conn, _A1, expected=AssetStatus.ROUTING, next=AssetStatus.RECEIVED)
        # enum 도 .value 로 정규화돼 파라미터에 실린다
        self.assertEqual(_last_params(cur), ("received", _A1, "routing"))


# ── T003: 배치 처리(모델 1회·순차·자산별 실패 격리) ───────────────────────────


class TestProcessReceivedBatch(unittest.TestCase):
    def _patch_common(self, stack, received, *, process_side):
        stack.enter_context(mock.patch.object(br, "scan_received_assets", return_value=received))
        stack.enter_context(mock.patch.object(br, "route_file", return_value=_route()))
        return stack.enter_context(mock.patch.object(br, "process_asset", side_effect=process_side))

    def test_model_loader_called_once_across_batch(self) -> None:
        # SC-003: 같은 모델 키 로더가 배치 전체에서 ≤1회 호출(프로세스 수명 캐시 재사용).
        import contextlib
        import functools

        calls = {"n": 0}

        @functools.cache  # 기존 스킬 lru_cache 등가 — 프로세스 수명 캐시(자산마다 재로드 0)
        def _load(_key):
            calls["n"] += 1
            return object()

        def _process(asset_id, **kw):
            _load("st")  # 배치 한 프로세스 — 캐시되면 자산마다 재로드되지 않음
            return "registered"

        with contextlib.ExitStack() as stack:
            self._patch_common(stack, [(_A1, "/d/a"), (_A2, "/d/b"), (_A3, "/d/c")], process_side=_process)
            rep = br.process_received_batch(mock.MagicMock(), limit=50, max_failures=3, settings=_NO_OS_SETTINGS)

        self.assertEqual(calls["n"], 1)               # 3자산 처리에도 로더 1회
        self.assertEqual(set(rep.registered), {_A1, _A2, _A3})

    def test_one_asset_failure_does_not_stop_batch(self) -> None:
        import contextlib

        def _process(asset_id, **kw):
            if asset_id == _A2:
                raise RuntimeError("boom")
            return "registered"

        with contextlib.ExitStack() as stack:
            self._patch_common(stack, [(_A1, "/d/a"), (_A2, "/d/b"), (_A3, "/d/c")], process_side=_process)
            stack.enter_context(mock.patch.object(br, "record_lineage"))
            stack.enter_context(mock.patch.object(br, "failure_count", return_value=1))  # cap 미만
            mf = stack.enter_context(mock.patch.object(br, "mark_failed"))
            rep = br.process_received_batch(mock.MagicMock(), limit=50, max_failures=3, settings=_NO_OS_SETTINGS)

        # 실패 1건이 격리되고 나머지는 정상 등록(배치 무중단)
        self.assertEqual(set(rep.registered), {_A1, _A3})
        self.assertEqual(rep.failed_retry, [_A2])     # cap 미만 → 비종료 유지(다음 run 재시도)
        self.assertEqual(rep.failed_terminal, [])
        mf.assert_not_called()                         # cap 미달이라 종료 격리 안 함

    def test_first_transition_zero_row_is_skipped_not_failed(self) -> None:
        import contextlib

        def _process(asset_id, **kw):
            if asset_id == _A2:
                # 첫 전이(received→routing) 0행 = 다른 run 이 이미 점유/received 아님
                raise ConcurrentTransitionError("이미 점유됨")
            return "registered"

        with contextlib.ExitStack() as stack:
            self._patch_common(stack, [(_A1, "/d/a"), (_A2, "/d/b")], process_side=_process)
            fc = stack.enter_context(mock.patch.object(br, "failure_count", return_value=99))
            mf = stack.enter_context(mock.patch.object(br, "mark_failed"))
            stack.enter_context(mock.patch.object(br, "record_lineage"))
            rep = br.process_received_batch(mock.MagicMock(), limit=50, max_failures=3, settings=_NO_OS_SETTINGS)

        self.assertEqual(rep.registered, [_A1])
        self.assertEqual(rep.skipped, [_A2])           # 경쟁 점유 흡수 — 실패 아님
        self.assertEqual(rep.failed_retry, [])
        self.assertEqual(rep.failed_terminal, [])
        fc.assert_not_called()                          # 스킵은 cap 카운트/실패 lineage 미기록
        mf.assert_not_called()

    def test_deferred_outcome_bucketed_not_registered(self) -> None:
        import contextlib

        with contextlib.ExitStack() as stack:
            self._patch_common(stack, [(_A1, "/d/a")], process_side=lambda asset_id, **kw: "deferred")
            rep = br.process_received_batch(mock.MagicMock(), limit=50, max_failures=3, settings=_NO_OS_SETTINGS)

        self.assertEqual(rep.deferred, [_A1])
        self.assertEqual(rep.registered, [])


# ── 코드리뷰 #1: dag_process 경로 OpenSearch 인라인 색인 배선 [FR-002·US1§2] ──────


class TestProcessBatchOpenSearchWiring(unittest.TestCase):
    """dag_process 자동 적재분도 OpenSearch 인라인 색인을 한다 — CLI run_ingest 와 동형(FR-002·US1§2).

    ``process_received_batch`` 가 ``os_index`` 미주입 시 **배치당 1회**
    ``_make_opensearch_indexer(db=db, settings=settings)`` 를 만들어 자산마다 ``process_asset(os_index=…)``
    에 주입한다(run_ingest 와 동일 함수 재사용). off(기본)면 콜러블이 즉시 반환해 무해(회귀 0).
    실제 finalize-직후 색인 호출은 process_asset 안에 있으므로, 본 단위는 process_asset 를 모킹하되
    side_effect 가 주입받은 ``os_index(asset_id)`` 를 호출해 그 배선이 살아 있음을 단언한다.
    """

    def _patch(self, stack, received, *, settings, process_side):
        stack.enter_context(mock.patch.object(br, "scan_received_assets", return_value=received))
        stack.enter_context(mock.patch.object(br, "route_file", return_value=_route()))
        stack.enter_context(mock.patch.object(br, "process_asset", side_effect=process_side))
        # 실패 경로(만에 하나)도 조용히 — RED 가 색인 단언에서 깨끗이 실패하도록.
        stack.enter_context(mock.patch.object(br, "record_lineage"))
        stack.enter_context(mock.patch.object(br, "failure_count", return_value=0))
        stack.enter_context(mock.patch.object(br, "mark_failed"))

    @staticmethod
    def _process_indexing(asset_id, **kw):
        # process_asset 가 finalize(PG 커밋) 직후 os_index(asset_id) 를 1회 부르는 실제 동작 모사.
        kw["os_index"](asset_id)
        return "registered"

    def test_builds_indexer_once_and_indexes_each_asset_when_enabled(self) -> None:
        settings = _os_settings(enabled=True)
        with contextlib.ExitStack() as stack, _patched_os_sync() as fake:
            self._patch(stack, [(_A1, "/d/a"), (_A2, "/d/b")],
                        settings=settings, process_side=self._process_indexing)
            rep = br.process_received_batch(
                mock.MagicMock(), limit=50, max_failures=3, settings=settings
            )
        self.assertEqual(set(rep.registered), {_A1, _A2})
        # 배치당 1회 클라이언트 생성·재사용(자산마다 새 연결 X) — run_ingest 동형.
        fake.get_client.assert_called_once()
        # 자산마다 인라인 색인.
        self.assertEqual(fake.index_asset.call_count, 2)

    def test_factory_built_exactly_once_per_batch(self) -> None:
        # 3자산 배치에도 인덱서 팩토리는 배치당 1회만 만든다(자산마다 생성 금지).
        settings = _os_settings(enabled=False)  # off → 콜러블 no-op, 팩토리 호출 횟수만 관찰
        with contextlib.ExitStack() as stack:
            self._patch(stack, [(_A1, "/d/a"), (_A2, "/d/b"), (_A3, "/d/c")],
                        settings=settings, process_side=self._process_indexing)
            mk = stack.enter_context(
                mock.patch.object(br, "_make_opensearch_indexer", wraps=br._make_opensearch_indexer)
            )
            br.process_received_batch(mock.MagicMock(), limit=50, max_failures=3, settings=settings)
        mk.assert_called_once()
        # run_ingest 동형 — db·settings 를 키워드로 넘긴다.
        self.assertIs(mk.call_args.kwargs.get("settings"), settings)

    def test_disabled_indexer_does_not_touch_opensearch(self) -> None:
        # off(기본)면 팩토리는 만들어지되 콜러블이 즉시 반환 — OS 모듈 미접촉(미도입 환경 회귀 0).
        settings = _os_settings(enabled=False)
        with contextlib.ExitStack() as stack, _patched_os_sync() as fake:
            self._patch(stack, [(_A1, "/d/a")],
                        settings=settings, process_side=self._process_indexing)
            br.process_received_batch(mock.MagicMock(), limit=50, max_failures=3, settings=settings)
        fake.get_client.assert_not_called()
        fake.index_asset.assert_not_called()

    def test_explicit_os_index_is_honored_without_building_factory(self) -> None:
        # 호출자(또는 테스트)가 os_index 를 직접 주입하면 새 인덱서를 만들지 않고 그대로 쓴다.
        calls: list[uuid.UUID] = []
        with contextlib.ExitStack() as stack:
            self._patch(stack, [(_A1, "/d/a")],
                        settings=object(), process_side=self._process_indexing)
            mk = stack.enter_context(mock.patch.object(br, "_make_opensearch_indexer"))
            br.process_received_batch(
                mock.MagicMock(), limit=50, max_failures=3,
                settings=object(), os_index=calls.append,
            )
        mk.assert_not_called()
        self.assertEqual(calls, [_A1])


# ── 069 B8(P2-9): settings 미주입 폴백 — OS 색인 조용한 off 차단 ─────────────


class TestSettingsFallback(unittest.TestCase):
    """settings=None 이면 get_current_settings() 로 폴백해 _make_opensearch_indexer 가 실제
    설정(opensearch_sync_enabled 등)을 읽게 한다(CLI run_ingest 동형). 미폴백 시 None 이 넘어가
    OS 색인이 조용히 off 되는 결함을 막는다.
    """

    def test_none_settings_falls_back_to_current_settings(self) -> None:
        sentinel = types.SimpleNamespace(opensearch=types.SimpleNamespace(sync_enabled=False))
        captured: dict = {}

        def _fake_indexer(*, db, settings):
            captured["settings"] = settings
            return None

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(br, "scan_received_assets", return_value=[]))
            stack.enter_context(
                mock.patch.object(br, "_make_opensearch_indexer", side_effect=_fake_indexer)
            )
            stack.enter_context(
                mock.patch("src.config.settings.get_current_settings", return_value=sentinel)
            )
            br.process_received_batch(mock.MagicMock(), limit=10, max_failures=3, settings=None)
        # 폴백된 현재 설정이 인덱서 팩토리로 전달됐다(None 아님).
        self.assertIs(captured["settings"], sentinel)


# ── T004: 재시도 cap·종료 격리 ───────────────────────────────────────────────


class TestFailureCount(unittest.TestCase):
    def test_counts_ingest_failed_lineage(self) -> None:
        conn, cur = _conn(fetchone=(3,))
        n = br.failure_count(conn, _A1)
        self.assertEqual(n, 3)
        sql = _last_sql(cur)
        self.assertIn("COUNT(*)", sql.upper())
        self.assertIn("asset_lineage", sql)
        # 활동명은 파라미터로 바인딩(SQL 인젝션 안전) — 카운트 대상이 ingest.failed.v1 인지 검증.
        self.assertEqual(br.FAILED_ACTIVITY, "ingest.failed.v1")
        params = _last_params(cur)
        self.assertIn(_A1, params)
        self.assertIn("ingest.failed.v1", params)


class TestRetryCapTerminalIsolation(unittest.TestCase):
    def _run_failing(self, stack, *, fail_count):
        stack.enter_context(mock.patch.object(br, "scan_received_assets", return_value=[(_A1, "/d/a")]))
        stack.enter_context(mock.patch.object(br, "route_file", return_value=_route()))
        stack.enter_context(
            mock.patch.object(br, "process_asset", side_effect=RuntimeError("boom"))
        )
        rl = stack.enter_context(mock.patch.object(br, "record_lineage"))
        stack.enter_context(mock.patch.object(br, "failure_count", return_value=fail_count))
        mf = stack.enter_context(mock.patch.object(br, "mark_failed"))
        rep = br.process_received_batch(mock.MagicMock(), limit=50, max_failures=3, settings=_NO_OS_SETTINGS)
        return rep, rl, mf

    def test_cap_reached_isolates_to_failed_terminal(self) -> None:
        import contextlib

        with contextlib.ExitStack() as stack:
            rep, rl, mf = self._run_failing(stack, fail_count=3)  # ≥ max_failures
        mf.assert_called_once()                         # 종료 격리(failed)
        self.assertEqual(rep.failed_terminal, [_A1])
        self.assertEqual(rep.failed_retry, [])
        # 실패 사유는 비식별 — 예외 타입명만(헌법 10조)
        self.assertEqual(rl.call_args.kwargs["activity"], "ingest.failed.v1")
        self.assertEqual(rl.call_args.kwargs["payload"]["reason"], "RuntimeError")

    def test_under_cap_keeps_nonterminal_for_retry(self) -> None:
        import contextlib

        with contextlib.ExitStack() as stack:
            rep, _, mf = self._run_failing(stack, fail_count=2)  # < max_failures
        mf.assert_not_called()                          # 비종료 유지 — 다음 run 재시도
        self.assertEqual(rep.failed_retry, [_A1])
        self.assertEqual(rep.failed_terminal, [])

    def test_mark_failed_conflict_absorbed(self) -> None:
        # 이미 종료 상태면 mark_failed 의 충돌(InvalidTransitionError)을 흡수 — 배치 무중단.
        import contextlib

        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(br, "scan_received_assets", return_value=[(_A1, "/d/a")]))
            stack.enter_context(mock.patch.object(br, "route_file", return_value=_route()))
            stack.enter_context(mock.patch.object(br, "process_asset", side_effect=RuntimeError("boom")))
            stack.enter_context(mock.patch.object(br, "record_lineage"))
            stack.enter_context(mock.patch.object(br, "failure_count", return_value=5))
            stack.enter_context(
                mock.patch.object(br, "mark_failed", side_effect=InvalidTransitionError("이미 종료"))
            )
            rep = br.process_received_batch(mock.MagicMock(), limit=50, max_failures=3, settings=_NO_OS_SETTINGS)
        self.assertEqual(rep.failed_terminal, [_A1])    # 충돌 흡수돼도 종료 격리로 집계


class TestTerminalExcludedFromScans(unittest.TestCase):
    def test_deferred_registered_failed_are_terminal(self) -> None:
        # deferred 는 재시도 아님 — registered/failed 와 함께 종료 계열(스캔 대상 제외).
        self.assertEqual(
            TERMINAL, frozenset({AssetStatus.REGISTERED, AssetStatus.FAILED, AssetStatus.DEFERRED})
        )

    def test_stuck_scan_sql_excludes_all_terminal_statuses(self) -> None:
        conn, cur = _conn(rows=[])
        br.scan_stuck_assets(conn, older_than_s=900, limit=50)
        sql = _last_sql(cur)
        for term in ("registered", "deferred"):
            self.assertNotIn(term, sql)  # 종료 계열은 고착 재스캔 대상이 아님


# ── T004: SC-011 — 신규 큐 테이블(ingest_job) 부재 ────────────────────────────


class TestNoIngestJobTable(unittest.TestCase):
    """SC-011: 수집 상태 정본은 asset.status — ingest_job 류 신규 큐 테이블을 만들지 않는다."""

    def test_no_ingest_job_in_migrations(self) -> None:
        root = Path(__file__).resolve().parents[1]
        sql_dir = root / "migrations" / "sql"
        offenders = []
        for path in sorted(sql_dir.glob("*.sql")):
            text = path.read_text(encoding="utf-8").lower()
            if "ingest_job" in text:
                offenders.append(path.name)
        self.assertEqual(offenders, [], msg=f"ingest_job 류 큐 테이블 발견: {offenders}")

    def test_batch_runner_does_not_reference_ingest_job(self) -> None:
        src = Path(br.__file__).read_text(encoding="utf-8").lower()
        self.assertNotIn("ingest_job", src)  # 상태 정본은 asset.status(FR-007)


# ── FR-011: Airflow import 0 ─────────────────────────────────────────────────


class TestNoAirflowImport(unittest.TestCase):
    def test_batch_runner_module_does_not_import_airflow(self) -> None:
        code = (
            "import sys\n"
            "import processing.ingest.batch_runner as br\n"
            "assert hasattr(br, 'process_received_batch') and hasattr(br, 'claim_asset')\n"
            "assert 'airflow' not in sys.modules, 'batch_runner import 가 airflow 를 끌어옴'\n"
            "print('ok')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("ok", proc.stdout)


# AssetRecord 는 process_asset 정상경로 반환 계약 참조용(직접 적재는 e2e 영역).
_ = AssetRecord


if __name__ == "__main__":
    unittest.main()
