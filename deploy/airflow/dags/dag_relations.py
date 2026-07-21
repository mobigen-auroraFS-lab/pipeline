"""030 G3 — dag_relations: registered-미해소 자산 → 관계 생성 [US2·FR-004·011].

주기로 "``registered`` 인데 ``relation_resolution`` 미해소(행 없음/pending)"인 자산을 G2
``scan_unresolved_assets`` 로 집어 관계를 생성한다. 관계 생성은 G1 레벨의 ``run_relations`` 로 위임한다 —
``run_relations`` 가 자산마다 ``propose_relations_for_asset``(후보→온프레미스 LLM 제안→graph_edge)을
**짧은 트랜잭션·자산별 격리**로 호출하고, 결과로 ``relation_resolution`` 큐를 pending→resolved/failed 로
전이시킨다(별도 fresh 트랜잭션). 이 큐 전이가 있어야 성공 자산이 다음 tick 에 재선택되지 않는다
(US2§2·SC-002). DAG 태스크 callable 은 스캔→위임뿐, 관계 로직 0(FR-011).

★ propose_relations_for_asset 를 raw 로 부르지 않고 run_relations 로 위임하는 이유: propose 단독은
relation_resolution 큐를 건드리지 않아 성공 자산이 매 tick 무한 재선택된다. run_relations 가
"자산별 propose + 큐 전이 + 격리"를 한데 묶은 정본 오케스트레이터다(헌법 8조 재사용).

★ 모듈 최상위는 가볍게 — 무거운 일(설정·DB·스캔/관계 함수)은 callable 안에서 런타임에. 설정
프로파일은 ``META_ENV``(기본 'prod'), 스캔 한도·스케줄은 env 주입(plan Clarifications).
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

_LOG = logging.getLogger("meta_extract.dag_relations")

# ── 운영 설정(env 기본값 — 모듈 최상위는 경량) ───────────────────────────────────
_DEFAULT_ENV = "prod"
# 관계 cron 도 '안전망'만 — 정상 경로는 dag_process 트리거. 뜸하게(기본 1시간, process 와 어긋나게 :30):
#   트리거가 놓친 미해소(pending)·잔여만 주기적으로 회수(env DAG_RELATIONS_SCHEDULE 로 더 길게 조정).
_SCHEDULE = os.environ.get("DAG_RELATIONS_SCHEDULE", "30 * * * *")


def _int_env(name: str, default: int) -> int:
    """정수 env(미설정/형식오류=기본값) — 스캔 한도 주입용."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        _LOG.warning("env %s 정수 변환 실패(기본값 %d 사용): %r", name, default, raw)
        return default


def propose_relations(**_context) -> dict[str, int]:
    """registered-미해소 자산을 스캔해 ``run_relations`` 로 관계를 생성하는 얇은 래퍼(FR-011).

    무거운 의존(설정·DB·스캔/관계 함수)은 여기서 런타임에 import 한다. 짧은 읽기 트랜잭션으로
    G2 ``scan_unresolved_assets`` 한 번, 이후 ``run_relations`` 가 자산별 propose + 큐 전이 + 격리를
    수행한다(DAG 는 관계 로직 0). 미해소 0건이면 빈 배치 호출을 피하고 즉시 반환한다.
    """
    from processing.app.run_relations import run_relations
    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil
    from processing.ingest.batch_runner import scan_unresolved_assets

    init_settings(os.environ.get("META_ENV", _DEFAULT_ENV))
    db = PostgresUtil()
    with db:
        with db.transaction() as conn:  # 짧은 읽기 트랜잭션 — 미해소 자산 id 만 집는다.
            asset_ids = [str(a) for a in scan_unresolved_assets(
                conn, limit=_int_env("DAG_RELATIONS_LIMIT", 50))]
        if not asset_ids:
            _LOG.info("dag_relations 완료: 미해소 0건")
            return {"done": 0, "failed": 0}
        # run_relations: 자산별 propose_relations_for_asset(짧은 트랜잭션·격리) + relation_resolution 전이.
        result = run_relations(asset_ids, db=db)
    summary = {"done": len(result["done"]), "failed": len(result["failed"])}
    _LOG.info("dag_relations 완료: %s (scanned=%d)", summary, len(asset_ids))
    return summary


def _has_more_unresolved(**context) -> bool:
    """이번 run 이 관계를 생성했으면(done>0=진행 중) dag_relations 를 자기 재트리거한다 — 미해소
    (registered-unresolved) 배치(50/run)를 매시 cron 대기 없이 back-to-back 소화(연속 드레인).

    미해소가 바닥나면 다음 run 이 ``scan_unresolved_assets`` 0건 → done=0 → 이 게이트가 차단해 안전
    종료(빈 꼬리 run 1회). done=0(대상 없음/전량 실패)엔 재트리거하지 않아 tight-loop 을 막고 cron 안전망에
    위임한다. ``max_active_runs=1`` 이라 재트리거 run 은 현재 run 종료 뒤 순차 실행(같은 자산 동시 제안 회피 불변).
    """
    summary = context["ti"].xcom_pull(task_ids="propose_relations") or {}
    return summary.get("done", 0) > 0


with DAG(
    dag_id="dag_relations",
    description="registered-미해소 자산 → graph_edge 관계 생성(030 G3·FR-004)",
    schedule=_SCHEDULE,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,            # 과거 미실행 보충 금지 — 미해소 집합은 PG 상태가 정본(US2)
    max_active_runs=1,        # 동시 1 run — 같은 자산 동시 제안·중복 엣지 제안 회피
    tags=["030", "pipeline", "relations"],
):
    propose = PythonOperator(task_id="propose_relations", python_callable=propose_relations)
    # 연속 드레인(self-retrigger) — done>0 이면 dag_relations 를 즉시 자기 재트리거해 미해소를 끝까지 소화.
    gate_more = ShortCircuitOperator(task_id="gate_more_unresolved", python_callable=_has_more_unresolved)
    trigger_more = TriggerDagRunOperator(task_id="trigger_more", trigger_dag_id="dag_relations")
    propose >> gate_more >> trigger_more
