"""030 G3 — dag_collect: 인입 디렉터리 폴링 → received 자산화 [US1·FR-001·011].

주기(잦게)로 고정 인입 디렉터리를 스캔해 **asset 테이블에 아직 없는(fs_path 기준)** 파일을
G1 순수 함수 ``collect_file`` 로 ``received`` 자산화한다(모델 0·저비용). 이 DAG 는 *스케줄 껍데기*일
뿐이며 수집 로직(라우팅·해시 dedup·create_asset·lineage)은 전부 ``collect_file`` 안에 있다 —
태스크 callable 은 인입 스캔 + 짧은 트랜잭션 루프뿐, **비즈니스 로직 0**(FR-011).

★ (a) push 체이닝 — 수집 직후 ``TriggerDagRunOperator`` 로 ``dag_process`` 를 즉시 트리거해 처리 지연을
줄인다(다음 cron tick — 기본 1시간(dag_process "0 * * * *") — 대기 회피). 단 ``ShortCircuitOperator`` 게이트가 **신규 수집이 있을 때만**
트리거를 통과시켜 빈 인입에 GPU 배치를 헛 깨우지 않는다. 트리거는 *얹기*일 뿐 — dag_process 의 cron 은
그대로 살아 있어 트리거가 스킵·유실돼도 received 는 다음 주기에 PG 상태로 집힌다(self-healing 유지).

★ 모듈 최상위는 가볍게(스케줄러가 자주 파싱) — 정의만 둔다. 무거운 일(``init_settings``·
``PostgresUtil``·G1 함수 import)은 전부 callable 안에서 **런타임에** 한다. 설정 프로파일은
환경변수 ``META_ENV``(기본 'prod'), 인입 경로는 ``WATCHER_INBOX_DIR``, 스케줄은
``DAG_COLLECT_SCHEDULE`` 로 운영에서 주입한다(plan Clarifications).
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
# 인입 폴링은 잦게(예 5분) — collect 는 모델 0·저비용이라 자주 돌려 신규 파일 지연을 줄인다.
_SCHEDULE = os.environ.get("DAG_COLLECT_SCHEDULE", "*/5 * * * *")


def _fs_path_exists(conn, fs_path: str) -> bool:
    """이미 asset 테이블에 같은 경로가 있으면 True(중복 collect 방지 가드 — FR-001·008).

    인입에 파일을 남겨두면 매 tick 같은 경로가 재감지되는데, ``collect_file`` 의 해시 dedup 은
    registered/deferred 만 본다(received 중간상태는 못 막음). 이 fs_path 존재 검사가 그 빈틈을
    메워 같은 파일이 ``received`` 행을 중복 생성하지 않게 한다(멱등). 수집 자체가 아닌 *선택 가드*다.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM asset WHERE fs_path = %s LIMIT 1", (fs_path,))
        return cur.fetchone() is not None


def collect_inbox(**_context) -> int:
    """인입 디렉터리를 스캔해 미수집 파일을 G1 ``collect_file`` 로 received 화하는 얇은 래퍼(FR-011).

    무거운 의존(설정·DB·수집 함수)은 여기서 런타임에 import 한다(모듈 최상위 경량 유지).
    파일마다 **짧은 트랜잭션**으로 fs_path 가드 검사 후 ``collect_file(conn, path)`` 만 호출한다 —
    라우팅·해시 dedup·create_asset·lineage 는 전부 collect_file 안에 있어 DAG 는 로직 0.
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
    # 061: 아카이브 루트(중복 파일 즉시 이동·처리완료 파일 이동 공용). 미설정이면 fail-fast.
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
            # 061(FR-105·C2): 중복(내용 해시가 기존 registered/deferred 자산과 동일 → asset 미생성)은
            # 트랜잭션 밖(IO)에서 archive/dup 로 즉시 이동해, 감시 디렉터리에 남아 매 tick route+해시
            # 재계산되던 churn 을 없앤다. asset 을 만든 경우(신규 received)만 collected 로 세어 트리거 게이트에 쓴다.
            if result.asset_id is not None:
                collected += 1
            elif result.skip_reason and result.skip_reason.startswith(REASON_DUPLICATE):
                # 파일별 격리(batch_runner 자산별 try 원칙) — 한 파일 이동 실패가 tick 전체를 막지 않게.
                dest = archiver.archive_dest(archive, os.path.basename(fs_path), when=when, subdir="dup")
                try:
                    archiver.execute_move(fs_path, dest)
                    archived_dup += 1
                except OSError as exc:
                    _LOG.warning("중복 파일 아카이브 이동 실패(스킵·다음 tick 재시도): %s → %s (%r)", fs_path, dest, exc)
    _LOG.info("dag_collect 완료: collected=%d dup_archived=%d (inbox=%s)", collected, archived_dup, inbox)
    return collected


def _inbox_nonempty(**_context) -> bool:
    """인입 디렉터리에 파일이 하나라도 있을 때만 True — 빈 인입에서 collect 를 헛돌리지 않는 게이트(061·C5).

    "FileSensor(파일 있을 때만 수집)" 의도를 **비차단**으로 실현한다 — 차단 FileSensor 는 폴링 DAG 에서
    워커 슬롯을 폴 간격 내내 점유해 부적합. 인입 부재/빈 경우 False 로 하위(collect)를 스킵한다(repo
    ShortCircuit 관례 일관·모델 0·저비용). 무거운 의존 없이 os.scandir 만 쓴다(모듈 최상위 경량 유지).
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
    """collect_inbox 가 신규 received 를 만들었을 때만 True — 빈 인입에 GPU dag_process 를 헛 깨우지 않는 게이트.

    collect_inbox 의 반환(수집 건수)을 XCom 으로 받아 0 이면 ShortCircuit 으로 하위(트리거)를 스킵한다.
    트리거가 스킵돼도 dag_process 의 cron(안전망·기본 1시간)이 살아 있어 received 는 다음 주기에 집힌다(self-healing).
    """
    collected = context["ti"].xcom_pull(task_ids="collect_inbox")
    return bool(collected and collected > 0)


with DAG(
    dag_id="dag_collect",
    description="인입 디렉터리 폴링 → received 자산화(030 G3·FR-001)",
    schedule=_SCHEDULE,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,            # 과거 미실행 보충 금지 — 인입은 '지금 있는 파일'만 의미(US1)
    max_active_runs=1,        # 동시 1 run — 인입 이중 감지·중복 collect 차단(FR-008)
    tags=["030", "pipeline", "collect"],
):
    # 061(C5): 인입에 파일이 있을 때만 collect 로 진행 — 빈 인입에서 헛 스캔/DB 왕복 회피(비차단 게이트).
    gate_inbox = ShortCircuitOperator(task_id="gate_inbox_nonempty", python_callable=_inbox_nonempty)
    collect = PythonOperator(task_id="collect_inbox", python_callable=collect_inbox)
    # 신규 수집이 있을 때만 통과 — 빈 인입의 헛 트리거(=GPU 배치 헛 기동) 차단(게이트).
    gate = ShortCircuitOperator(task_id="gate_new_received", python_callable=_has_new_received)
    # (a) push 체이닝 — 수집 직후 dag_process 를 즉시 트리거(cron 대기 단축). cron 은 안전망으로 유지.
    trigger_process = TriggerDagRunOperator(task_id="trigger_process", trigger_dag_id="dag_process")
    gate_inbox >> collect >> gate >> trigger_process
