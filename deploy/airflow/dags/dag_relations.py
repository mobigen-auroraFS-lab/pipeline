"""등록은 됐지만 아직 관계를 만들지 않은 자산을 찾아 관계를 생성하는 DAG.

**흐름에서의 위치**: 파이프라인의 마지막 단계다. 처리 DAG 가 등록을 끝낸 자산을 받아 이웃을
찾고 관계 엣지를 만든다.

⚠️ **관계 제안 함수를 직접 부르면 안 된다.** 그 함수는 엣지만 만들고 "이 자산은 처리했다"는
표시를 남기지 않는다 — 그러면 성공한 자산이 매 주기 다시 선택돼 **영원히 같은 일을 반복한다**.
반드시 그 표시까지 함께 처리하는 오케스트레이터를 거쳐야 한다.

**연속 드레인**: 한 번에 정해진 개수만 처리하므로, 처리한 것이 있으면 자기 자신을 다시 깨워
이어서 소화한다. ⚠️ 조건은 "이번에 뭔가 했는가"다 — 아무것도 못 했는데 다시 깨우면 진전 없이
빙빙 돈다.

정기 실행은 **안전망**이다(정상 경로는 처리 DAG 의 트리거). 트리거가 놓친 잔여만 회수한다.

⚠️ 모듈 최상위는 가볍게 — 설정·DB·관계 함수 import 는 태스크 함수 **안에서** 한다.
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
# 정기 실행은 안전망 — 정상 경로는 처리 DAG 의 트리거다. 처리 DAG 와 **시각을 어긋나게** 둔다
# (같은 시각에 겹치면 아직 등록 중인 자산을 보게 된다).
_SCHEDULE = os.environ.get("DAG_RELATIONS_SCHEDULE", "30 * * * *")


def _int_env(name: str, default: int) -> int:
    """환경변수를 정수로 읽는다 — 못 읽으면 기본값을 쓴다.

    Args:
        name: 환경변수 이름.
        default: 못 읽었을 때 쓸 값.

    Returns:
        정수 값. 변환 실패는 **경고를 남기고** 기본값으로 — 조용히 넘기면 왜 기대와 다르게
        도는지 알 수 없다.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        _LOG.warning("env %s 정수 변환 실패(기본값 %d 사용): %r", name, default, raw)
        return default


def propose_relations(**_context) -> dict[str, int]:
    """관계를 아직 안 만든 자산을 훑어 관계 생성 오케스트레이터에 넘기는 껍데기.

    **DB에 쓴다**(관계 엣지·처리 이력).

    대상을 집는 것은 **짧은 읽기 트랜잭션 한 번**으로 끝낸다 — 관계 생성이 오래 걸리므로
    그 트랜잭션을 열어 둔 채 진행하면 다른 작업을 오래 막는다.

    Returns:
        성공·실패 건수 dict. 대상이 0건이면 배치를 부르지 않고 바로 0을 돌려준다.
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
        # 이 함수가 자산별 관계 제안(짧은 트랜잭션·격리)과 "처리했다" 표시를 함께 처리한다.
        result = run_relations(asset_ids, db=db)
    summary = {"done": len(result["done"]), "failed": len(result["failed"])}
    _LOG.info("dag_relations 완료: %s (scanned=%d)", summary, len(asset_ids))
    return summary


def _has_more_unresolved(**context) -> bool:
    """이번에 관계를 만든 것이 있으면 참 — 자기 자신을 다시 깨워 남은 것을 이어서 소화한다.

    한 번에 정해진 개수만 처리하므로, 정기 실행을 기다리면 밀린 자산을 다 소화하는 데 오래 걸린다.
    대상이 바닥나면 다음 실행이 0건이 되어 이 게이트가 차단한다(빈 실행 한 번으로 끝).

    ⚠️ 조건이 "이번에 뭔가 했는가"인 것이 핵심 — 전량 실패로 정체된 상황에서 다시 깨우면 진전
    없이 빙빙 돈다. 그때는 정기 실행 안전망에 맡긴다.

    Returns:
        관계를 만든 자산이 있으면 참.
    """
    summary = context["ti"].xcom_pull(task_ids="propose_relations") or {}
    return summary.get("done", 0) > 0


with DAG(
    dag_id="dag_relations",
    description="registered-미해소 자산 → graph_edge 관계 생성(030 G3·FR-004)",
    schedule=_SCHEDULE,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,            # 과거분 보충 금지 — 무엇이 남았는지는 DB 상태가 정본이다
    max_active_runs=1,        # 동시 1개 — 둘이 돌면 같은 자산에 관계를 겹쳐 제안한다
    tags=["030", "pipeline", "relations"],
):
    propose = PythonOperator(task_id="propose_relations", python_callable=propose_relations)
    # 연속 드레인 — 이번에 뭔가 처리했으면 자기 자신을 다시 깨워 남은 것을 끝까지 소화한다.
    gate_more = ShortCircuitOperator(task_id="gate_more_unresolved", python_callable=_has_more_unresolved)
    trigger_more = TriggerDagRunOperator(task_id="trigger_more", trigger_dag_id="dag_relations")
    propose >> gate_more >> trigger_more
