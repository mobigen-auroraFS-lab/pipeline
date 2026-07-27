"""인입 디렉터리를 자주 훑어 새 파일을 '대기' 자산으로 등록하는 DAG.

**흐름에서의 위치**: 파이프라인의 **입구**다. 여기서 파일이 자산 행으로 바뀌기만 하고, 실제
처리(분류·추출·임베딩)는 다음 DAG 가 한다. 그래서 무거운 모델을 전혀 쓰지 않아 자주 돌 수 있다.

    (이 DAG) 파일 → 대기 자산  ─트리거→  처리 DAG  ─트리거→  관계 DAG

**로직은 여기 없다.** 라우팅·중복 판정·행 생성·기록은 전부 수집 함수 안에 있고, 이 파일은
"언제 부를지"만 정한다. 그래야 같은 로직을 CLI·테스트가 스케줄러 없이 그대로 쓴다.

**즉시 트리거는 지연 단축용 "얹기"다.** 수집 직후 처리 DAG 를 바로 깨워 다음 정기 실행까지
기다리지 않게 한다. ⚠️ 그러나 **정기 실행을 없애면 안 된다** — 트리거가 유실되거나 스킵돼도
대기 자산은 DB 에 남아 있어 다음 정기 실행이 회수한다(스스로 복구되는 구조).

⚠️ **모듈 최상위는 가볍게 유지한다.** 스케줄러가 이 파일을 아주 자주 다시 읽으므로, 설정 초기화·
DB 연결·처리 함수 import 는 전부 태스크 함수 **안에서** 한다. 최상위에 무거운 import 를 하나
넣으면 스케줄러 전체가 느려진다.

운영 주입: 설정 프로파일·인입 경로·아카이브 경로·스케줄은 모두 환경변수로 받는다.
"""

from __future__ import annotations

import logging
import os

import pendulum
from airflow import DAG
from airflow.providers.standard.operators.python import (  # Airflow 3.x: core 에서 이동
    PythonOperator,
    ShortCircuitOperator,
)
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

_LOG = logging.getLogger("meta_extract.dag_collect")

# ── 운영 설정(env 기본값 — 모듈 최상위는 os.environ.get 만, 가볍게) ──────────────
_DEFAULT_ENV = "prod"
# 자주 돌린다 — 모델을 쓰지 않아 싸고, 새 파일이 들어온 뒤 처리까지의 지연을 줄인다.
_SCHEDULE = os.environ.get("DAG_COLLECT_SCHEDULE", "*/5 * * * *")


def _fs_path_exists(conn, fs_path: str) -> bool:
    """같은 경로의 자산이 이미 있는지 본다(중복 수집 방지).

    ⚠️ **수집 함수의 중복 판정만으로는 부족하다.** 그쪽은 처리가 끝난 자산과 보류 자산만 보므로,
    **아직 처리 중인 자산은 걸러 내지 못한다**. 파일이 인입에 남아 있는 동안 매 주기 같은 경로가
    다시 감지되면 대기 행이 계속 늘어난다 — 이 검사가 그 빈틈을 메운다.

    Args:
        conn: DB 연결.
        fs_path: 검사할 경로.

    Returns:
        이미 있으면 참(그 파일은 건너뛴다).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM asset WHERE fs_path = %s LIMIT 1", (fs_path,))
        return cur.fetchone() is not None


def collect_inbox(**_context) -> int:
    """인입 디렉터리를 훑어 아직 없는 파일만 대기 자산으로 만든다(수집 함수를 부르는 껍데기).

    **DB에 쓴다**(자산 행·기록) **그리고 파일을 옮긴다**(중복으로 판정된 파일).

    파일마다 **짧은 트랜잭션**을 따로 쓴다 — 하나로 묶으면 중간에 실패했을 때 그때까지 수집한
    것이 통째로 사라지고, 트랜잭션이 파일 수만큼 길어진다.

    ⚠️ 중복 파일은 **인입에서 즉시 치운다.** 남겨 두면 매 주기 경로 판정과 해시 계산을 다시
    하며 헛돈다. 파일 이동 실패는 한 건씩 격리해 넘긴다(다음 주기에 다시 시도).

    Returns:
        새로 만든 자산 수. **뒤따르는 게이트가 이 값으로** 처리 DAG 를 깨울지 정하므로,
        중복으로 치운 파일은 여기 세지 않는다.

    Raises:
        RuntimeError: 인입·아카이브 경로 환경변수가 없거나, 아카이브가 인입 하위일 때
            (그러면 옮긴 파일을 다시 수집해 무한히 돈다).
    """
    from datetime import date

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil
    from processing.ingest import archiver
    from processing.ingest.collector import collect_files
    from processing.ingest.pipeline_steps import REASON_DUPLICATE, collect_file

    init_settings(os.environ.get("META_ENV", _DEFAULT_ENV))
    inbox = os.environ.get("WATCHER_INBOX_DIR")
    if not inbox:
        raise RuntimeError("WATCHER_INBOX_DIR 환경변수(인입 디렉터리)가 필요합니다.")
    # 아카이브 뿌리 — 중복 파일과 처리완료 파일이 함께 쓴다. 없으면 그 자리에서 멈춘다.
    archive = os.environ.get("WATCHER_ARCHIVE_DIR")
    if not archive:
        raise RuntimeError("WATCHER_ARCHIVE_DIR 환경변수(아카이브 루트)가 필요합니다.")
    archiver.assert_archive_separate(inbox, archive)  # 아카이브⊄인입 fail-fast(자기수렴 불변식)
    when = date.today()

    collected = 0
    archived_dup = 0
    db = PostgresUtil()
    with db:
        for fs_path in collect_files(input_dir=inbox):
            # 짧은 트랜잭션·파일 단위 — asset.fs_path 에 없는 파일만 received 화(중복 방지).
            with db.transaction() as conn:
                if _fs_path_exists(conn, fs_path):
                    continue
                result = collect_file(conn, fs_path)
            # 내용이 같은 자산이 이미 있으면 행을 만들지 않는다 — 그 파일은 **인입에서 즉시 치운다**.
            # 남겨 두면 매 주기 경로 판정과 해시 계산을 다시 하며 헛돈다.
            # 파일 이동은 트랜잭션 **밖**에서 한다(파일시스템은 롤백되지 않는다).
            if result.asset_id is not None:
                collected += 1
            elif result.skip_reason and result.skip_reason.startswith(REASON_DUPLICATE):
                # 한 파일 이동 실패가 이번 주기 전체를 막지 않게 격리한다(다음 주기에 다시 시도).
                dest = archiver.archive_dest(archive, os.path.basename(fs_path), when=when, subdir="dup")
                try:
                    archiver.execute_move(fs_path, dest)
                    archived_dup += 1
                except OSError as exc:
                    _LOG.warning("중복 파일 아카이브 이동 실패(스킵·다음 tick 재시도): %s → %s (%r)", fs_path, dest, exc)
    _LOG.info("dag_collect 완료: collected=%d dup_archived=%d (inbox=%s)", collected, archived_dup, inbox)
    return collected


def _inbox_nonempty(**_context) -> bool:
    """인입에 파일이 하나라도 있을 때만 참 — 빈 인입에서 수집을 헛돌리지 않는 게이트.

    ⚠️ **파일을 기다리는 센서를 쓰지 않는다.** 센서는 파일이 나타날 때까지 워커 슬롯을 붙잡고
    있는데, 이 DAG 는 몇 분마다 도는 폴링이라 슬롯이 사실상 영구 점유된다. 대신 "지금 있나"만
    보고 없으면 아래 단계를 건너뛴다.

    Returns:
        파일이 있으면 참. 인입 디렉터리 자체가 없으면 거짓(오류가 아니다 — 아직 안 만든 상태).

    Raises:
        RuntimeError: 인입 경로 환경변수가 없을 때.
    """
    inbox = os.environ.get("WATCHER_INBOX_DIR")
    if not inbox:
        raise RuntimeError("WATCHER_INBOX_DIR 환경변수(인입 디렉터리)가 필요합니다.")
    try:
        with os.scandir(inbox) as it:
            return any(entry.is_file() for entry in it)
    except FileNotFoundError:
        return False


def _has_new_received(**context) -> bool:
    """새로 수집한 것이 있을 때만 참 — 없으면 무거운 처리 DAG 를 깨우지 않는다.

    처리 DAG 는 GPU 에 모델을 올리므로 헛 기동이 비싸다. 앞 태스크가 돌려준 수집 건수를 받아
    0이면 아래 트리거를 건너뛴다.

    트리거를 건너뛰어도 **아무것도 잃지 않는다** — 처리 DAG 의 정기 실행이 살아 있어 대기 자산은
    다음 주기에 회수된다.

    Returns:
        새로 수집한 자산이 있으면 참.
    """
    collected = context["ti"].xcom_pull(task_ids="collect_inbox")
    return bool(collected and collected > 0)


with DAG(
    dag_id="dag_collect",
    description="인입 디렉터리 폴링 → received 자산화(030 G3·FR-001)",
    schedule=_SCHEDULE,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,            # 과거분 보충 금지 — 인입은 '지금 있는 파일'만 의미가 있다
    max_active_runs=1,        # 동시 1개 — 둘이 돌면 같은 파일을 함께 집어 중복 행이 생긴다
    tags=["030", "pipeline", "collect"],
):
    # 파일이 있을 때만 수집으로 진행 — 빈 인입에서 헛 스캔·DB 왕복을 피한다.
    gate_inbox = ShortCircuitOperator(task_id="gate_inbox_nonempty", python_callable=_inbox_nonempty)
    collect = PythonOperator(task_id="collect_inbox", python_callable=collect_inbox)
    # 신규 수집이 있을 때만 통과 — 빈 인입의 헛 트리거(=GPU 배치 헛 기동) 차단(게이트).
    gate = ShortCircuitOperator(task_id="gate_new_received", python_callable=_has_new_received)
    # 수집 직후 처리 DAG 를 바로 깨운다(정기 실행까지 기다리지 않게). 정기 실행은 안전망으로 유지.
    trigger_process = TriggerDagRunOperator(task_id="trigger_process", trigger_dag_id="dag_process")
    gate_inbox >> collect >> gate >> trigger_process
