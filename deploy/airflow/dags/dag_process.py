"""030 G3 — dag_process: received(+고착) 자산을 단일 배치로 registered 전이 [US1·US4·FR-002·010·011].

더 긴 주기로, ``received`` 와 비종료 고착 자산을 **단일 태스크(단일 프로세스)에서 모델 1회 로드·순차**
처리해 ``routing→classifying→extracting→registered``(+인라인 색인)까지 끌고 간다. 처리 로직은 전부
G2 ``process_received_batch`` 안에 있다(고착 리셋·received 스캔·자산별 try·재시도 cap·종료 격리) —
DAG 태스크 callable 은 그 배치 함수를 1회 호출하는 얇은 래퍼다(비즈니스 로직 0, FR-011).

★ 단일 GPU OOM 구조적 차단: ``max_active_runs=1`` + 크기 1 Pool('gpu')로 **모델 적재 프로세스를
항상 1개**로 제한한다(FR-010·US4). 인프로세스 모델(ST·CLIP·faster-whisper)은 기존 ``lru_cache`` 가
그 run(프로세스) 수명 동안 1회만 로드해 배치 전체에서 재사용한다(SC-003).

★ (a) push 체이닝 — registered 산출 직후 ``TriggerDagRunOperator`` 로 ``dag_relations`` 를 즉시 트리거해
관계 생성 지연을 줄인다(다음 cron tick — 기본 1시간(dag_relations "30 * * * *") — 대기 회피). ``ShortCircuitOperator`` 게이트가 **신규 registered 가
있을 때만** 트리거를 통과시킨다(빈 산출 시 헛 트리거 회피). 게이트·트리거는 모델 미사용이라 gpu Pool 불요.
트리거는 *얹기*일 뿐 — dag_relations 의 cron 은 유지되어 트리거 스킵·유실에도 미해소 자산은 다음 주기에 집힌다.

★ (b) 연속 드레인(self-retrigger) — registered 를 냈으면(=진행 중) ``dag_process`` 를 **자기 자신**도 즉시
재트리거해 남은 ``received`` 배치(50/run)를 매시 cron 대기 없이 back-to-back 소화한다. received 가 바닥나면
다음 run 의 registered=0 으로 게이트가 차단해 안전 종료(빈 꼬리 run 1회). 정체(registered=0)면 재트리거하지
않아 tight-loop 을 막고 cron·고착리셋 안전망에 위임한다. ``max_active_runs=1`` 이라 재트리거 run 은 현재 run
종료 뒤 순차 실행 — 단일 GPU 1프로세스 보장은 그대로다.

★ 모듈 최상위는 가볍게 — 무거운 일(설정·DB·배치 함수)은 callable 안에서 런타임에. 설정 프로파일은
``META_ENV``(기본 'prod'), 배치 한도·cap·고착 임계·스케줄·Pool 은 env 로 주입(plan Clarifications).
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

_LOG = logging.getLogger("meta_extract.dag_process")

# ── 운영 설정(env 기본값 — 모듈 최상위는 경량) ───────────────────────────────────
_DEFAULT_ENV = "prod"
# 처리 cron 은 '안전망'만 — 정상 경로는 dag_collect 트리거. 뜸하게(기본 1시간): GPU 배치라 헛돌면 비싸고,
#   트리거가 놓친 잔여(크래시·재시도분)만 주기적으로 회수하면 충분(env DAG_PROCESS_SCHEDULE 로 더 길게 조정).
_SCHEDULE = os.environ.get("DAG_PROCESS_SCHEDULE", "0 * * * *")
# 크기 1 Pool — 단일 GPU 에 모델 적재 프로세스가 둘이 되지 않게 슬롯을 1로 묶는다(배포에서 pool 생성).
_POOL = os.environ.get("DAG_PROCESS_POOL", "gpu")


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    """정수 env(미설정/형식오류/범위위반=기본값) — 배치 한도·cap·고착 임계 주입용.

    A1: ``minimum`` 미만(기본값 하 음수·0)이면 기본값으로 되돌린다. 방치하면 오설정이 조용히 진행돼
    ``DAG_PROCESS_LIMIT`` 음수는 PG 'LIMIT must not be negative' 로 배치 태스크가 크래시, 0 은 매 run
    0건 처리(무음 스톨), ``DAG_PROCESS_MAX_FAILURES`` 0 은 첫 실패 즉시 격리를 유발한다. 세 소비처
    (limit·max_failures·older_than_s) 모두 ≥1 이 유효값이라 기본 ``minimum=1``."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        val = int(raw)
    except ValueError:
        _LOG.warning("env %s 정수 변환 실패(기본값 %d 사용): %r", name, default, raw)
        return default
    if val < minimum:
        _LOG.warning("env %s=%d 가 최소 %d 미만 → 기본값 %d 사용", name, val, minimum, default)
        return default
    return val


def process_batch(**_context) -> dict[str, int]:
    """received(+고착 리셋) 자산을 G2 ``process_received_batch`` 로 1회 배치 처리하는 얇은 래퍼(FR-011).

    무거운 의존(설정·DB·배치 함수)은 여기서 런타임에 import 한다. 배치 스캔·claim·순차 처리·자산별
    실패 격리·재시도 cap·종료 격리는 전부 ``process_received_batch`` 안에 있어 DAG 는 로직 0.
    XCom 친화적으로 자산별 집계 카운트(dict)만 돌려준다(BatchReport 원본 대신).
    """
    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil
    from processing.ingest.batch_runner import process_received_batch

    # 활성 설정을 잡아 배치로 넘긴다 — 배치가 _make_opensearch_indexer(settings=…) 로 인라인 색인을
    # 배선하기 위함(FR-002·US1§2, run_ingest CLI 동형). off(기본)면 색인기는 no-op 라 무해.
    settings = init_settings(os.environ.get("META_ENV", _DEFAULT_ENV))
    db = PostgresUtil()
    with db:
        report = process_received_batch(
            db,
            limit=_int_env("DAG_PROCESS_LIMIT", 50),                 # 한 run 배치 한도
            max_failures=_int_env("DAG_PROCESS_MAX_FAILURES", 3),    # 재시도 cap N(≥N → failed 격리)
            older_than_s=_int_env("DAG_PROCESS_STUCK_OLDER_THAN_S", 900),  # 고착 리셋 임계(초)
            settings=settings,                                        # 인라인 OpenSearch 색인 배선용
        )
    summary = {
        "registered": len(report.registered),
        "deferred": len(report.deferred),
        "skipped": len(report.skipped),
        "failed_retry": len(report.failed_retry),
        "failed_terminal": len(report.failed_terminal),
        "reset": len(report.reset),
        "reset_isolated": len(report.reset_isolated),
    }
    _LOG.info("dag_process 완료: %s", summary)
    return summary


def _has_new_registered(**context) -> bool:
    """process_batch 가 신규 registered 를 만들었을 때만 True — 새 registered 없으면 dag_relations 트리거 스킵.

    process_batch 의 반환(dict: registered 등 집계)을 XCom 으로 받아 registered==0 이면 ShortCircuit 으로
    하위(트리거)를 스킵한다. 트리거가 스킵돼도 dag_relations 의 cron(안전망·기본 1시간)이 살아 있어 미해소 자산은
    다음 주기에 집힌다(self-healing).
    """
    summary = context["ti"].xcom_pull(task_ids="process_batch") or {}
    return summary.get("registered", 0) > 0


def archive_processed(**_context) -> int:
    """처리완료(registered) 자산의 **인입-잔류 파일**을 archive/ 로 이동 + ``fs_path`` 갱신하는 꼬리 태스크(061·C3/C4).

    ``status='registered'`` 이면서 ``fs_path`` 가 인입 하위인 자산만 스윕한다(``plan_archive_moves`` 가 필터).
    이동 후 fs_path 가 인입 밖이라 다음 스윕서 제외돼 **멱등·자기수렴**. 이동→fs_path 갱신 순(C4: 크래시로
    갱신 누락 시 dest 가 asset_id 키 결정적이라 다음 스윕이 같은 dest 재현→``execute_move`` no-op 후 갱신 재시도).
    GPU 미사용(파일 이동·DB UPDATE 뿐)이라 pool 불요. 무거운 의존은 런타임 import(모듈 최상위 경량 유지).
    """
    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil
    from processing.ingest import archiver

    init_settings(os.environ.get("META_ENV", _DEFAULT_ENV))
    inbox = os.environ.get("WATCHER_INBOX_DIR")
    archive = os.environ.get("WATCHER_ARCHIVE_DIR")
    if not inbox or not archive:
        raise RuntimeError("WATCHER_INBOX_DIR·WATCHER_ARCHIVE_DIR 환경변수(인입·아카이브)가 필요합니다.")

    db = PostgresUtil()
    with db:
        moved = archiver.archive_registered_assets(db, inbox_root=inbox, archive_root=archive)
    _LOG.info("dag_process archive_processed 완료: moved=%d", moved)
    return moved


with DAG(
    dag_id="dag_process",
    description="received(+고착) 자산 배치 처리 → registered(030 G3·FR-002·010)",
    schedule=_SCHEDULE,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,            # 과거 미실행 보충 금지 — 상태는 PG 가 정본(US3 self-healing)
    max_active_runs=1,        # 동시 1 run — 단일 GPU 모델 적재 프로세스 1개 보장(FR-010·US4)
    tags=["030", "pipeline", "process"],
):
    # 크기 1 Pool('gpu')로 한 번 더 직렬화 — max_active_runs 와 이중 가드(단일 GPU OOM 차단).
    process = PythonOperator(task_id="process_batch", python_callable=process_batch, pool=_POOL)
    # 신규 registered 가 있을 때만 통과 — 게이트·트리거는 모델 미사용이라 gpu Pool 에 묶지 않는다.
    gate = ShortCircuitOperator(task_id="gate_new_registered", python_callable=_has_new_registered)
    # (a) push 체이닝 — registered 직후 dag_relations 를 즉시 트리거(cron 대기 단축). cron 은 안전망으로 유지.
    trigger_relations = TriggerDagRunOperator(task_id="trigger_relations", trigger_dag_id="dag_relations")
    # (b) 연속 드레인(self-retrigger) — 이번 run 이 registered 를 냈으면(=진행 중) dag_process 를 즉시 자기
    #   재트리거해 남은 received 배치(50/run)를 매시 cron 대기 없이 back-to-back 소화한다. received 가 바닥나면
    #   다음 run 이 registered=0 → 이 게이트가 차단해 안전 종료(빈 꼬리 run 1회). registered=0(전량 실패/deferred)
    #   정체 상황엔 재트리거 안 해 tight-loop 을 막고 cron·고착리셋 안전망에 위임. max_active_runs=1 이라 재트리거
    #   run 은 현재 run(archive 포함) 종료 후 순차 실행 — 단일 GPU 1프로세스 보장 불변.
    gate_more = ShortCircuitOperator(task_id="gate_more_received", python_callable=_has_new_registered)
    trigger_more = TriggerDagRunOperator(task_id="trigger_more", trigger_dag_id="dag_process")
    # 061(C3): 처리완료 파일 아카이브 — process 뒤 독립 꼬리(gate/trigger 와 병렬·GPU pool 미사용).
    archive = PythonOperator(task_id="archive_processed", python_callable=archive_processed)
    process >> gate >> trigger_relations
    process >> gate_more >> trigger_more
    process >> archive
