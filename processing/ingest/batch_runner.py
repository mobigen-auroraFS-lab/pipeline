"""030 G2 — Airflow DAG 가 호출하는 PG 상태 기반 배치 로직(순수 함수).

큐·브로커·장수명 워커 없이 **주기 배치가 PostgreSQL 상태를 스캔해 미완분을 전진**시키는 엔진의
코어 로직이다(spec 030, ADR 2026-06-16). Airflow DAG 태스크는 이 함수들을 호출하는 얇은 래퍼이며
(FR-011), 본 모듈은 **Airflow 를 import 하지 않아** Airflow 없이 단위 테스트·CLI 디버그가 가능하다.

상태 정본(신규 큐 테이블 0 — SC-011):
    · ``asset.status``       — 수집 FSM(009 조건부 UPDATE 원자성). received→…→registered/failed/deferred.
    · ``relation_resolution`` — 관계 단계(v250). dag_relations 가 미해소 자산을 집어 전진.
    · ``asset_lineage``       — append-only 활동 로그. ``ingest.failed.v1`` 수가 재시도 cap 의 소스.

핵심 설계 불변식(★ 이대로 — 어기면 미묘한 버그)
    1. **원자 claim = ``process_asset`` 의 첫 조건부 전이**다. received 자산은 process_asset 가
       ``set_status(received→routing)``(009 조건부 UPDATE)로 시작하므로 그 자체가 원자 점유다.
       ``process_received_batch`` 는 received 자산마다 process_asset 를 호출하고, **첫 전이가 0행
       (InvalidTransitionError 계열 — 이미 누가 점유했거나 received 가 아님)이면 그 자산을 스킵**한다.
       별도의 충돌하는 claim UPDATE 를 두지 않는다(두면 process_asset 첫 전이가 깨진다).
    2. **고착(crash) 자산 = received 리셋 후 재처리**. 비종료(routing/classifying/extracting)로
       임계시간 넘게 고착된 자산은 ``claim_asset(expected=<고착상태>, next='received')`` 조건부 UPDATE 로
       received 로 되돌린 뒤 다음 처리에서 재처리한다(재추출은 결정적·해시 dedup 이 중복 흡수).
       resumable process_asset(복잡)보다 이 리셋(단순·멱등)을 택한다.
    3. **재시도 cap**: 자산 처리 중 예외를 잡으면 ``ingest.failed.v1`` lineage 기록 +
       ``failure_count`` ≥ N 이면 ``mark_failed``(종료 격리), 미만이면 비종료로 두어 다음 run 고착스캔이
       received 로 리셋·재처리한다. 한 자산 예외가 배치 루프를 멈추지 않는다(자산별 try).
    4. **종료 계열(registered/failed/deferred)** 은 received/고착 스캔 대상에서 제외. deferred 는
       재시도가 아니라 계획적 대기(단계 D 의료 어댑터 대기)이므로 dag_process 재스캔 대상이 아니다.

모든 함수는 ``conn``/``db`` 주입 seam — 트랜잭션 경계는 호출자가 제어하거나(스캔/claim) batch 내부가
짧은 트랜잭션으로 제어한다(process_received_batch). 결정성·온프레미스 LLM·PHI 비식별(헌법 3·2·10조).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from psycopg import Connection

from src.database.lineage_persist import record_lineage
from processing.ingest.pipeline_steps import OsIndexFn, _make_opensearch_indexer, process_asset
from processing.ingest.router import route_file
from processing.ingest.status import AssetStatus, InvalidTransitionError, mark_failed
from processing.pipeline.registry import DEFAULT_REGISTRY

_LOG = logging.getLogger("meta_extract.batch_runner")

# 재시도 cap 카운트 소스 — 자산 처리 실패 활동(run_ingest CLI·dag_process 공통 기록).
FAILED_ACTIVITY = "ingest.failed.v1"

# 비종료(고착 재스캔 대상) 상태 — 종료 계열(registered/failed/deferred)의 여집합(불변식 #4).
_NON_TERMINAL = (AssetStatus.ROUTING, AssetStatus.CLASSIFYING, AssetStatus.EXTRACTING)


def _status_value(status: AssetStatus | str) -> str:
    """AssetStatus enum 이든 문자열이든 DB 비교용 값 문자열로 정규화."""
    return status.value if isinstance(status, AssetStatus) else status


# ── T002: 스캔 함수 ──────────────────────────────────────────────────────────


def scan_received_assets(conn: Connection[Any], *, limit: int) -> list[tuple[uuid.UUID, str]]:
    """``received`` 자산을 생성순으로 ``limit`` 개 집어 ``(asset_id, fs_path)`` 목록 반환.

    dag_process 가 처리할 대상이다. modality 는 호출자(process_received_batch)가 ``route_file`` 로
    재탐지한다(모델 0·결정적). 정렬은 created_at ASC, asset_id ASC — 먼저 들어온 파일을 먼저 처리(FIFO)
    하되 동일 created_at·대상>limit 면 asset_id(UUIDv7) 보조 정렬로 경계를 결정적으로 고정한다
    (헌법 3조·FR-012, 형제 scan_unresolved_assets 와 동형).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT asset_id, fs_path FROM asset "
            "WHERE status = 'received' ORDER BY created_at ASC, asset_id ASC LIMIT %s",
            (limit,),
        )
        return [(r[0], r[1]) for r in cur.fetchall()]


def scan_stuck_assets(
    conn: Connection[Any], *, older_than_s: int, limit: int
) -> list[tuple[uuid.UUID, str]]:
    """비종료(routing/classifying/extracting)로 ``older_than_s`` 초 넘게 고착된 자산 목록.

    이전 run 크래시로 비종료에 멈춘 자산을 ``(asset_id, status)`` 로 돌려준다 — 호출자가
    ``claim_asset(expected=status, next='received')`` 로 received 리셋해 재처리한다(self-healing,
    불변식 #2). 종료 계열(registered/failed/deferred)은 IN 목록에서 빠져 재스캔되지 않는다(불변식 #4).
    ``updated_at`` 이 NULL 이면(전이 전) 비교가 거짓이라 자연 제외된다(received 는 애초에 대상 아님).
    정렬은 updated_at ASC, asset_id ASC — 동일 updated_at·대상>limit 시 asset_id(UUIDv7) 보조 정렬로
    경계를 결정적으로 고정한다(헌법 3조·FR-012).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT asset_id, status FROM asset "
            "WHERE status IN ('routing', 'classifying', 'extracting') "
            "  AND updated_at < now() - make_interval(secs => %s) "
            "ORDER BY updated_at ASC, asset_id ASC LIMIT %s",
            (older_than_s, limit),
        )
        return [(r[0], r[1]) for r in cur.fetchall()]


def scan_unresolved_assets(conn: Connection[Any], *, limit: int) -> list[uuid.UUID]:
    """``registered`` 인데 ``relation_resolution`` 미해소(행 없음 또는 ``pending``)인 자산 목록.

    dag_relations 가 ``propose_relations_for_asset`` 으로 관계를 만들 대상이다. LEFT JOIN 으로
    큐 행 없음(asset_id IS NULL)·pending 만 고르고 resolved/failed(DLQ)는 자연 제외한다(FR-004).
    정렬은 created_at ASC, asset_id ASC — 결정적 tiebreaker(헌법 3조).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT a.asset_id FROM asset a "
            "LEFT JOIN relation_resolution rr ON rr.asset_id = a.asset_id "
            "WHERE a.status = 'registered' "
            "  AND (rr.status = 'pending' OR rr.asset_id IS NULL) "
            "ORDER BY a.created_at ASC, a.asset_id ASC LIMIT %s",
            (limit,),
        )
        return [r[0] for r in cur.fetchall()]


# ── T002: 원자 claim 프리미티브 ───────────────────────────────────────────────


def claim_asset(
    conn: Connection[Any],
    asset_id: uuid.UUID,
    *,
    expected: AssetStatus | str,
    next: AssetStatus | str,  # noqa: A002 — 명시적 다음 상태(전이 의미를 드러내는 호출부 가독성 우선)
) -> bool:
    """조건부 ``UPDATE asset SET status=next WHERE asset_id=%s AND status=expected`` → 점유 성공 여부.

    원자 점유·고착 리셋 프리미티브(009 조건부 UPDATE 원자성). 기대 현재상태(expected)가 그사이
    바뀌면 0행이 되어 ``False`` 를 돌려준다(lost update 거부) — 동시 2회 중 1회만 ``True``.

    ⚠️ ``set_status`` 와 달리 **FSM 검증(ALLOWED_TRANSITIONS)을 거치지 않는다** — 고착 리셋
    (routing→received 등)은 정상 FSM 전이가 아닌 *복구 리셋*이라 의도적으로 검증을 우회한다(불변식 #2).
    DB CHECK 제약은 그대로 적용되므로 받은 상태값은 유효해야 한다. status_reason 은 건드리지 않는다.
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE asset SET status = %s, updated_at = now() "
            "WHERE asset_id = %s AND status = %s",
            (_status_value(next), asset_id, _status_value(expected)),
        )
        return cur.rowcount > 0


# ── T004: 재시도 cap 카운트 ───────────────────────────────────────────────────


def failure_count(conn: Connection[Any], asset_id: uuid.UUID) -> int:
    """그 자산의 ``ingest.failed.v1`` lineage 누적 수(재시도 cap 의 소스, 불변식 #3).

    같은 트랜잭션 안에서 방금 INSERT 한 실패 lineage 도 함께 센다(자기 트랜잭션 가시성) — 따라서
    N번째 실패에서 count==N 이 되어 cap(≥N) 판정이 그 자리에서 성립한다. run 을 가로질러 누적된다.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM asset_lineage WHERE asset_id = %s AND activity = %s",
            (asset_id, FAILED_ACTIVITY),
        )
        return int(cur.fetchone()[0])


# ── T003·T004: 배치 처리 + cap·종료 격리 ─────────────────────────────────────


@dataclass
class BatchReport:
    """``process_received_batch`` 한 run 의 자산별 처리 결과 집계.

    · ``registered``/``deferred`` — 정상 완주.
    · ``skipped``         — 첫 전이 0행(경쟁 점유/received 아님). 실패 아님, cap 미카운트(불변식 #1).
    · ``failed_retry``    — 처리 실패·cap 미달. 비종료 유지 → 다음 run 고착스캔이 리셋·재시도(불변식 #3).
    · ``failed_terminal`` — 처리 실패·cap 도달. ``failed`` 종료 격리(무한 재시도 차단).
    · ``reset``           — 고착(crash) 자산을 received 로 리셋(self-healing, 불변식 #2).
    """

    registered: list[uuid.UUID] = field(default_factory=list)
    deferred: list[uuid.UUID] = field(default_factory=list)
    skipped: list[uuid.UUID] = field(default_factory=list)
    failed_retry: list[uuid.UUID] = field(default_factory=list)
    failed_terminal: list[uuid.UUID] = field(default_factory=list)
    reset: list[uuid.UUID] = field(default_factory=list)


def reset_stuck_assets(db: Any, *, older_than_s: int, limit: int) -> list[uuid.UUID]:
    """고착(crash) 자산을 received 로 리셋(self-healing, 불변식 #2). 리셋된 asset_id 목록 반환.

    한 트랜잭션에서 고착 스캔 + 조건부 claim_asset 리셋을 수행한다. claim 0행(그사이 다른 처리가
    상태를 바꿈)은 자연 무시된다. 리셋된 자산은 received 가 되어 같은/다음 run 의 received 스캔에 잡힌다.
    """
    reset: list[uuid.UUID] = []
    with db.transaction() as conn:
        for asset_id, status in scan_stuck_assets(conn, older_than_s=older_than_s, limit=limit):
            if claim_asset(conn, asset_id, expected=status, next=AssetStatus.RECEIVED):
                reset.append(asset_id)
                _LOG.info("고착 리셋(received): asset_id=%s (%s→received)", asset_id, status)
    return reset


def _handle_failure(
    db: Any, asset_id: uuid.UUID, exc: BaseException, *, max_failures: int, report: BatchReport
) -> None:
    """자산 처리 실패를 cap 정책으로 처리(불변식 #3) — 비식별 사유 lineage + cap 도달 시 종료 격리.

    fresh 트랜잭션으로 ``ingest.failed.v1`` lineage(사유=예외 **타입명**만, 헌법 10조 PHI 비식별)를
    기록하고, 누적 실패 수가 ``max_failures`` 이상이면 ``mark_failed``(종료 격리)·미만이면 비종료로
    둔다(다음 run 재시도). mark_failed 충돌(이미 종료 상태)은 흡수한다(배치 무중단).
    """
    reason = type(exc).__name__  # 예외 타입명만 — 메시지·경로(PHI 가능)를 lineage 에 담지 않는다.
    _LOG.warning("처리 실패: asset_id=%s (%s)", asset_id, reason)
    with db.transaction() as conn:
        record_lineage(
            conn, asset_id, activity=FAILED_ACTIVITY, agent="dag_process",
            payload={"reason": reason},
        )
        if failure_count(conn, asset_id) >= max_failures:
            try:
                mark_failed(conn, asset_id, reason)
            except InvalidTransitionError:
                # 이미 종료 상태(다른 경로가 먼저 종료)면 흡수 — 충돌도 InvalidTransitionError 계열.
                pass
            report.failed_terminal.append(asset_id)
            _LOG.warning("재시도 cap 도달 → failed 격리: asset_id=%s", asset_id)
        else:
            report.failed_retry.append(asset_id)


def process_received_batch(
    db: Any,
    *,
    limit: int,
    max_failures: int,
    older_than_s: int | None = None,
    extract_fn: Any = None,    # 테스트·e2e 전용 override(미주입=팩 기본 extract/embed)
    classify_fn: Any = None,   # 테스트·e2e 전용 override(미주입=cascade_v1)
    registry: Any = DEFAULT_REGISTRY,
    settings: Any = None,
    os_index: OsIndexFn | None = None,
) -> BatchReport:
    """received(+옵션 고착 리셋) 자산을 **단일 프로세스에서 모델 1회 로드·순차** 처리한다(불변식 #1·#3).

    한 run = 한 프로세스(dag_process ``max_active_runs=1``/Pool 1)이므로 인프로세스 모델(ST·CLIP·
    faster-whisper)은 기존 ``lru_cache`` 가 프로세스 수명 동안 1회만 로드해 배치 전체에서 재사용한다
    (자산마다 재로드 0 — SC-003). LLM 은 기존 외부 HTTP seam.

    흐름:
      1. ``older_than_s`` 주어지면 고착 자산을 received 로 리셋(self-healing, 불변식 #2).
      2. received 자산을 ``limit`` 개 스캔.
      3. 자산마다 ``route_file`` 재탐지(modality·domain) 후 ``process_asset`` 순차 호출.
         · 정상 → registered/deferred 집계.
         · 첫 전이 0행(InvalidTransitionError 계열) → **스킵**(경쟁 점유 흡수, cap 미카운트, 불변식 #1).
         · 그 외 예외 → ``_handle_failure``(cap·종료 격리, 불변식 #3). **한 자산 실패가 배치를 멈추지
           않는다**(자산별 try).
    """
    # 069 B8(P2-9): settings 미주입 시 현재 설정으로 폴백(CLI run_ingest L329 동형). 미폴백이면
    # _make_opensearch_indexer 에 None 이 넘어가 opensearch_sync_enabled 를 못 읽고 OS 색인이
    # 조용히 off 된다. get_current_settings 는 운영 진입점(init_settings)에서 활성 — 순수 단위는
    # 호출자가 settings 를 주입하므로 이 폴백 경로를 타지 않는다(지연 import 로 미초기화 오염 방지).
    if settings is None:
        from src.config.settings import get_current_settings

        settings = get_current_settings()

    report = BatchReport()

    # OpenSearch 증분 색인기 — os_index 미주입 시 배치당 1회 생성(run_ingest CLI 동형, FR-002·US1§2).
    # opensearch_sync_enabled off(기본)면 콜러블이 즉시 반환하므로 미도입 환경에서 무해(회귀 0). 클라이언트는
    # 첫 색인에서 만들어 배치 전체 재사용(자산마다 새 연결 X). process_asset 의 finalize 직후 os_index(asset_id)
    # 로 호출된다. 호출자가 os_index 를 직접 주입하면 새로 만들지 않고 그대로 쓴다(중복 생성 회피).
    if os_index is None:
        os_index = _make_opensearch_indexer(db=db, settings=settings)

    # 1) 고착 리셋(옵션) — 비종료 고착 자산을 received 로 되돌려 이번/다음 run 재처리.
    if older_than_s is not None:
        report.reset.extend(reset_stuck_assets(db, older_than_s=older_than_s, limit=limit))

    # 2) received 스캔(짧은 읽기 트랜잭션).
    with db.transaction() as conn:
        received = scan_received_assets(conn, limit=limit)

    # 3) 자산별 순차 처리 — 모델은 프로세스 수명 캐시 재사용, 자산별 try 로 실패 격리.
    for asset_id, fs_path in received:
        try:
            route = route_file(fs_path)  # 모델 0·결정적 — collect 단계와 같은 탐지 seam.
            outcome = process_asset(
                asset_id, db=db, fs_path=fs_path,
                modality=route.modality, domain=route.domain,
                extract_fn=extract_fn, classify_fn=classify_fn,
                registry=registry, settings=settings, os_index=os_index,
            )
            if outcome == "deferred":
                report.deferred.append(asset_id)
            else:
                report.registered.append(asset_id)
        except InvalidTransitionError:
            # 첫 전이(received→routing) 0행 = 다른 run 이 이미 점유했거나 received 아님 → 스킵.
            # ConcurrentTransitionError 도 이 계열이라 함께 흡수된다(불변식 #1, 경쟁=실패 아님).
            # 단일 워커(max_active_runs=1) 경로에서는 점유 자산의 후속 전이가 충돌하지 않으므로
            # 여기 도달은 사실상 '경쟁 점유/이미 종료'뿐이다.
            _LOG.info("스킵(첫 전이 0행·경쟁 점유): asset_id=%s", asset_id)
            report.skipped.append(asset_id)
        except Exception as exc:  # noqa: BLE001 — route/추출/적재 모든 실패 흡수(자산별 격리·cap)
            _handle_failure(db, asset_id, exc, max_failures=max_failures, report=report)

    _LOG.info(
        "batch done: registered=%d deferred=%d skipped=%d failed_retry=%d failed_terminal=%d reset=%d",
        len(report.registered), len(report.deferred), len(report.skipped),
        len(report.failed_retry), len(report.failed_terminal), len(report.reset),
    )
    return report
