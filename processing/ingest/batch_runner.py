"""주기 배치가 DB 상태를 훑어 미완분을 전진시키는 엔진(스케줄러 비의존·순수 함수).

**흐름에서의 위치**: 스케줄러의 태스크는 이 함수들을 부르는 얇은 껍데기다. 이 모듈은
**스케줄러를 import 하지 않는다** — 그래야 스케줄러 없이 단위 테스트와 CLI 디버그가 된다.

큐도 브로커도 장수명 워커도 두지 않는다. **상태는 전부 DB 에 있다**: 자산의 처리 단계,
관계 해소 여부, 그리고 무슨 일이 있었는지를 덧붙이기만 하는 활동 기록. 배치는 그것을
훑어 다음으로 밀 뿐이다.

지켜야 할 규칙 넷 — 어기면 조용히 어긋난다
    1. **점유는 첫 상태 전이가 대신한다.** 처리 함수의 첫 조건부 UPDATE 가 곧 원자 점유이므로,
       별도의 점유 UPDATE 를 두면 안 된다 — 두는 순간 그 첫 전이가 항상 0행이 되어 깨진다.
       첫 전이가 0행이면 남이 이미 집어 간 것이므로 **스킵**이지 실패가 아니다.
    2. **멈춰 있는 자산은 처음으로 되돌려 다시 돌린다.** 중간 단계에서 오래 머문 자산은
       조건부 UPDATE 로 처음 상태로 되돌린다. 중단 지점부터 이어 하는 편이 똑똑해 보이지만
       훨씬 복잡하고, 재추출은 결정적이라 다시 해도 결과가 같다.
    3. **되돌리기도 횟수를 센다.** 프로세스가 통째로 죽으면 실패 처리기가 돌지 못해 실패
       기록이 남지 않는다. 그러면 같은 자산이 영원히 되돌려지며 **배치 선두를 막는다**.
       그래서 되돌릴 때도 기록을 남기고, 실패 수와 합산해 한계에 닿으면 실패로 격리한다.
    4. **끝난 자산은 다시 보지 않는다.** 특히 '보류'는 실패가 아니라 계획된 대기이므로
       재시도 대상이 아니다 — 여기 넣으면 영원히 재시도한다.

트랜잭션 경계는 호출자(스캔·점유)나 배치 내부(짧게 여러 번)가 잡는다. 실패 사유는 **예외
타입명만** 남긴다 — 메시지·경로에는 민감정보가 섞일 수 있다(헌법 10조).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from psycopg import Connection

from processing.ingest.pipeline_steps import OsIndexFn, _make_opensearch_indexer, process_asset
from processing.ingest.router import route_file
from processing.ingest.status import AssetStatus, InvalidTransitionError, mark_failed
from processing.pipeline.registry import DEFAULT_REGISTRY
from src.database.lineage_activity import LineageActivity  # 계보 활동명 정본(3레포 공용)
from src.database.lineage_persist import record_lineage

_LOG = logging.getLogger("meta_extract.batch_runner")

# 재시도 cap 카운트 소스 — 자산 처리 실패 활동(run_ingest CLI·dag_process 공통 기록).
FAILED_ACTIVITY = LineageActivity.INGEST_FAILED  # 값은 "ingest.failed.v1" — 정본은 코어

# 크래시 루프 cap 카운트 소스 — 고착 자산의 received 리셋 활동. 하드 크래시(OOM-kill/SIGKILL/
# 네이티브 segfault)는 예외 핸들러(_handle_failure)가 못 돌아 FAILED_ACTIVITY 가 안 남으므로,
# 리셋마다 이 활동을 남겨 실패 수와 합산해 무한 재처리를 차단한다(불변식 #2·#3 통합).
RESET_ACTIVITY = LineageActivity.INGEST_RESET  # 값은 "ingest.reset.v1" — 정본은 코어
# 리셋 cap 도달 시 종료 격리 사유 — 비식별(헌법 10조·예외 메시지/경로 없음).
RESET_CAP_REASON = "reset_cap_exceeded"

# 비종료(고착 재스캔 대상) 상태 — 종료 계열(registered/failed/deferred)의 여집합(불변식 #4).
_NON_TERMINAL = (AssetStatus.ROUTING, AssetStatus.CLASSIFYING, AssetStatus.EXTRACTING)


def _status_value(status: AssetStatus | str) -> str:
    """열거형이든 문자열이든 DB 비교에 쓸 문자열로 맞춘다.

    Args:
        status: 상태값. 호출부마다 열거형·문자열이 섞여 들어온다.

    Returns:
        DB 에 저장된 것과 같은 형태의 문자열.
    """
    return status.value if isinstance(status, AssetStatus) else status


# ── T002: 스캔 함수 ──────────────────────────────────────────────────────────


def scan_received_assets(conn: Connection[Any], *, limit: int) -> list[tuple[uuid.UUID, str]]:
    """``received`` 자산을 생성순으로 ``limit`` 개 집어 ``(asset_id, fs_path)`` 목록 반환.

    먼저 들어온 것을 먼저 처리하되, 생성 시각이 같으면 자산 id 로 갈라 **자르는 경계를
    고정한다** — 안 그러면 대상이 상한을 넘을 때 매번 다른 묶음이 잡힌다(헌법 3조).

    Args:
        conn: DB 연결.
        limit: 한 번에 집을 최대 건수.

    Returns:
        ``(asset_id, 파일 경로)`` 목록. 모달리티는 담지 않는다 — 호출자가 경로로 다시
        판정한다(모델을 쓰지 않는 결정적 판정이라 저장값에 기댈 이유가 없다).
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

    이전 실행이 죽어 중간 단계에 멈춘 자산을 찾는다. 호출자가 이것을 처음 상태로 되돌려
    다시 돌린다(규칙 #2).

    Args:
        conn: DB 연결.
        older_than_s: 이 시간(초)보다 오래 멈춘 것만. **너무 짧게 주면 정상 처리 중인
            자산을 빼앗는다** — 한 자산 처리에 걸리는 최대 시간보다 넉넉해야 한다.
        limit: 한 번에 집을 최대 건수.

    Returns:
        ``(asset_id, 멈춘 상태)`` 목록. 되돌릴 때 그 상태를 조건으로 걸어야 하므로 함께
        돌려준다. 끝난 자산은 애초에 조회 대상이 아니다(규칙 #4).
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

    ⚠️ **아직 시도조차 안 한 자산과 시도 중인 자산을 함께** 집는다 — 관계 단계는 행이
    없는 상태로 시작하므로, 있는 행만 보면 새 자산이 영원히 빠진다.

    Args:
        conn: DB 연결.
        limit: 한 번에 집을 최대 건수.

    Returns:
        자산 id 목록(생성순·동시각은 id 순으로 고정). 이미 끝났거나 포기한 자산은 빠진다.
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

    현재 상태를 조건에 걸어 갱신하므로, 그사이 남이 바꿨으면 0행이 되어 ``False`` 다.
    동시에 둘이 시도해도 **한쪽만 참**을 받는다.

    ⚠️ **정상 전이 규칙을 검사하지 않는다.** 되돌리기는 정상 흐름이 아니라 *복구*라
    일부러 우회한다 — 대신 아무 값이나 넣으면 DB 제약에 걸리므로 유효한 상태를 줘야 한다.

    Args:
        conn: DB 연결.
        asset_id: 대상 자산.
        expected: **지금 이 상태일 때만** 바꾼다. 이 조건이 곧 원자 점유다.
        next: 바꿀 상태.

    Returns:
        실제로 바꿨으면 ``True``. 사유 컬럼은 건드리지 않는다.
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

    **방금 같은 트랜잭션에서 남긴 실패도 함께 센다** — 그래야 N번째 실패에서 바로 한계
    판정이 선다(다음 실행까지 기다리지 않는다). 실행을 가로질러 누적된다.

    Args:
        conn: DB 연결.
        asset_id: 대상 자산.

    Returns:
        누적 실패 횟수.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM asset_lineage WHERE asset_id = %s AND activity = %s",
            (asset_id, FAILED_ACTIVITY),
        )
        return int(cur.fetchone()[0])


def recovery_attempt_count(conn: Connection[Any], asset_id: uuid.UUID) -> int:
    """복구 시도 누적 = ``ingest.failed.v1``(잡힌 예외) + ``ingest.reset.v1``(고착 리셋).

    잡히는 실패는 실패 처리기가 세지만, **프로세스가 통째로 죽으면 그 처리기가 못 돈다** —
    그런 자산은 실패 기록이 없어 영원히 재시도된다. 되돌리기 기록까지 합산해야 그 경우가
    잡힌다.

    Args:
        conn: DB 연결.
        asset_id: 대상 자산.

    Returns:
        실패 + 되돌리기 누적 횟수.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM asset_lineage WHERE asset_id = %s AND activity IN (%s, %s)",
            (asset_id, FAILED_ACTIVITY, RESET_ACTIVITY),
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
    · ``reset_isolated``  — 고착 리셋 cap 도달 → ``failed`` 종료 격리(크래시 루프 차단, 불변식 #2·#3).
    """

    registered: list[uuid.UUID] = field(default_factory=list)
    deferred: list[uuid.UUID] = field(default_factory=list)
    skipped: list[uuid.UUID] = field(default_factory=list)
    failed_retry: list[uuid.UUID] = field(default_factory=list)
    failed_terminal: list[uuid.UUID] = field(default_factory=list)
    reset: list[uuid.UUID] = field(default_factory=list)
    reset_isolated: list[uuid.UUID] = field(default_factory=list)


def reset_stuck_assets(
    db: Any, *, older_than_s: int, limit: int, max_failures: int
) -> tuple[list[uuid.UUID], list[uuid.UUID]]:
    """고착(crash) 자산을 received 로 리셋(self-healing, 불변식 #2) — 단, 크래시 루프는 cap 으로 차단.

    **DB 에 쓴다.** 되돌리기·격리·기록을 한 트랜잭션에서 처리한다.

    ⚠️ **되돌리기에도 한계를 둔다.** 프로세스가 통째로 죽는 자산은 실패 기록이 안 남아
    한계에 걸리지 않고, 오래된 자산이라 다음 배치에서도 맨 앞에 온다 — 그대로 두면 같은
    자산이 영원히 배치를 막는다. 그래서 되돌릴 때마다 기록을 남기고, 누적 시도가 한계에
    닿으면 되돌리는 대신 실패로 격리한다. 한계 판정은 **이번 기록을 남기기 전에** 세므로
    정확히 N번 시도한 뒤 격리된다.

    Args:
        db: 트랜잭션을 열 수 있는 DB 핸들.
        older_than_s: 이 시간보다 오래 멈춘 것만 대상으로.
        limit: 한 번에 다룰 최대 건수.
        max_failures: 누적 시도가 이 값 이상이면 되돌리지 않고 격리한다.

    Returns:
        ``(되돌린 목록, 격리한 목록)``. 그사이 다른 경로가 이미 끝낸 자산은 조용히 넘어간다.
    """
    reset: list[uuid.UUID] = []
    isolated: list[uuid.UUID] = []
    with db.transaction() as conn:
        for asset_id, status in scan_stuck_assets(conn, older_than_s=older_than_s, limit=limit):
            # 이전까지의 복구 시도(실패 lineage + 리셋 lineage) — 이번 리셋 기록 '전에' 센다.
            attempts = recovery_attempt_count(conn, asset_id)
            if attempts >= max_failures:
                # 크래시 루프 cap 도달 → received 리셋 대신 종료 격리(무한 재처리·head-of-line 정체 차단).
                try:
                    mark_failed(conn, asset_id, RESET_CAP_REASON)
                    record_lineage(
                        conn, asset_id, activity=RESET_ACTIVITY, agent="dag_process",
                        payload={"prior_status": _status_value(status), "attempts": attempts, "terminal": True},
                    )
                    isolated.append(asset_id)
                    _LOG.warning(
                        "고착 리셋 cap 도달 → failed 격리: asset_id=%s (누적 %d회·%s)",
                        asset_id, attempts, status,
                    )
                except InvalidTransitionError:
                    # 그사이 다른 경로가 종료시킴(ConcurrentTransitionError 도 이 계열) — 흡수.
                    pass
            elif claim_asset(conn, asset_id, expected=status, next=AssetStatus.RECEIVED):
                # cap 미달 — received 리셋 + 리셋 lineage(다음 판정의 cap 카운트 소스).
                record_lineage(
                    conn, asset_id, activity=RESET_ACTIVITY, agent="dag_process",
                    payload={"prior_status": _status_value(status), "attempts": attempts},
                )
                reset.append(asset_id)
                _LOG.info("고착 리셋(received): asset_id=%s (%s→received·누적 %d회)", asset_id, status, attempts)
    return reset, isolated


def _handle_failure(
    db: Any, asset_id: uuid.UUID, exc: BaseException, *, max_failures: int, report: BatchReport
) -> None:
    """자산 처리 실패를 cap 정책으로 처리(불변식 #3) — 비식별 사유 lineage + cap 도달 시 종료 격리.

    **DB에 쓴다** — 실패 트랜잭션과 분리된 새 트랜잭션으로 기록한다(실패 때문에 롤백된
    트랜잭션에 기록하면 그 기록도 함께 사라진다).

    ⚠️ 사유는 **예외 타입명만** 남긴다 — 예외 메시지와 경로에는 민감정보가 섞일 수 있고,
    활동 기록은 지우지 않는 테이블이다(헌법 10조).

    Args:
        db: 트랜잭션을 열 수 있는 DB 핸들.
        asset_id: 실패한 자산.
        exc: 잡은 예외. **타입명만 쓰고 내용은 버린다**.
        max_failures: 누적 실패가 이 값 이상이면 실패로 격리하고, 미만이면 그대로 둬
            다음 실행이 되돌려 재시도하게 한다.
        report: 집계 대상. 이 함수가 **여기에 결과를 담는다**(반환값이 아니라 인자를 채운다).
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

    **한 번에 하나씩 순차로** 돈다. 병렬로 돌리지 않는 이유는 임베딩 모델이 프로세스마다
    수 GB 를 잡아먹기 때문이다 — 한 프로세스로 묶으면 모델을 **한 번만 올려** 배치 전체에서
    재사용한다.

    흐름: (선택) 멈춘 자산 되돌리기 → 대기 자산 스캔 → 자산마다 경로 재판정 후 처리.

    **한 자산의 실패가 배치를 멈추지 않는다** — 자산마다 따로 감싼다. 다른 실행이 이미
    집어 간 자산은 실패가 아니라 스킵으로 세고 재시도 횟수에 넣지 않는다.

    Args:
        db: 트랜잭션을 열 수 있는 DB 핸들.
        limit: 한 번에 처리할 최대 자산 수.
        max_failures: 누적 실패·되돌리기가 이 값 이상이면 실패로 격리한다.
        older_than_s: 주면 멈춘 자산 되돌리기를 **먼저** 한 번 돈다. ``None`` 이면 건너뛴다.
        extract_fn: 추출 단계를 갈아끼울 때만. 미주입이면 팩이 고른 전략을 쓴다.
        classify_fn: 분류 단계를 갈아끼울 때만. 미주입이면 팩이 고른 전략을 쓴다.
        registry: 전략을 찾을 레지스트리. 테스트가 격리된 것을 줄 수 있다.
        settings: 설정. **미주입이면 현재 활성 설정으로 채운다** — 비워 두면 색인 설정을
            못 읽어 색인이 조용히 꺼진다.
        os_index: 색인 함수. 미주입이면 배치당 하나를 만들어 전체에서 재사용한다.

    Returns:
        자산별 처리 결과 집계.
    """
    # ⚠️ 설정을 안 받으면 색인 설정을 못 읽어 **색인이 말없이 꺼진다** — 현재 활성 설정으로
    # 메운다. 순수 단위 테스트는 항상 설정을 주입하므로 이 경로를 타지 않고, 지연 import 라
    # 설정 미초기화 환경을 오염시키지도 않는다.
    if settings is None:
        from src.config.settings import get_current_settings

        settings = get_current_settings()

    report = BatchReport()

    # 색인기는 **배치당 하나**만 만든다 — 자산마다 만들면 검색 엔진 연결이 그만큼 열린다.
    # 색인이 꺼져 있으면 이 콜러블은 즉시 반환하므로 검색 엔진 없는 환경에서도 무해하다.
    if os_index is None:
        os_index = _make_opensearch_indexer(db=db, settings=settings)

    # 1) 고착 리셋(옵션) — 비종료 고착 자산을 received 로 되돌려 이번/다음 run 재처리.
    #    단, 크래시 루프(하드 크래시 반복)는 recovery cap 으로 failed 격리한다(불변식 #2·#3).
    if older_than_s is not None:
        _reset, _isolated = reset_stuck_assets(
            db, older_than_s=older_than_s, limit=limit, max_failures=max_failures
        )
        report.reset.extend(_reset)
        report.reset_isolated.extend(_isolated)

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
        "batch done: registered=%d deferred=%d skipped=%d failed_retry=%d "
        "failed_terminal=%d reset=%d reset_isolated=%d",
        len(report.registered), len(report.deferred), len(report.skipped),
        len(report.failed_retry), len(report.failed_terminal), len(report.reset),
        len(report.reset_isolated),
    )
    return report
