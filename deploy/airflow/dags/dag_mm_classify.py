"""등록된 **분류 스킬**로 자산을 분류하고 패싯 색인을 맞추는 DAG(085).

**흐름에서의 위치**: 적재·관계 파이프라인과 나란히 도는 별도 배치다(084 소속 배치와 동형). 사람이
등록한 분류 체계(스킬)로 이미 저장된 요약·키워드를 판정해 라벨을 붙이고, OpenSearch 의
``mm_skill_labels`` 칸만 부분 갱신한다(전체 재색인 아님).

⚠️ **활성 스킬이 없으면 아무 일도 하지 않는다.** 스킬 등록은 ``scripts/register_mm_skill.py``
(미리보기 → 확인 코드 → 등록)로 하는 사람 작업이며 이 DAG 의 일이 아니다.

⚠️ **배치 함수를 직접 조립하지 않는다.** 스킬 로드·대상 선별·판정·저장·색인 순서는 러너의
``run_batch`` 한 곳에 있다 — 여기에 복사하면 한쪽만 고쳐져 갈라진다(대표 사고: 색인 값을 방금
판정한 스킬 라벨로만 만들어 다른 스킬 라벨을 색인에서 지우는 것).

**연속 드레인**: 한 번에 스킬당 정해진 개수만 처리하므로, 판정한 것이 있으면 자기 자신을 다시
깨워 이어서 소화한다. ⚠️ 조건은 "이번에 판정했는가"다(전량 실패 정체에서 헛도는 것을 막는다).

⚠️ 모듈 최상위는 가볍게 — 설정·DB·배치 러너 import 는 태스크 함수 **안에서** 한다.
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

_LOG = logging.getLogger("meta_extract.dag_mm_classify")

# ── 운영 설정(env 기본값 — 모듈 최상위는 경량) ───────────────────────────────────
_DEFAULT_ENV = "prod"
# 정기 실행이 정상 경로다(적재와 독립이라 밀어 주는 트리거가 없다). 관계(30분)·소속(45분) DAG 와
# **시각을 어긋나게** 둔다 — 같은 시각에 겹치면 LLM 을 여러 배치가 동시에 두드린다.
_SCHEDULE = os.environ.get("DAG_MM_CLASSIFY_SCHEDULE", "50 * * * *")


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    """환경변수를 정수로 읽는다 — 못 읽거나 범위를 벗어나면 기본값으로 되돌린다.

    ⚠️ **범위 검사를 빼면 설정 실수가 조용히 진행된다**: 처리 한도가 음수면 DB 가 거부해 태스크가
    죽고, 0이면 매번 0건 처리하며 아무 소리 없이 정체한다.

    Args:
        name: 환경변수 이름.
        default: 못 읽었을 때 쓸 값.
        minimum: 이 값 미만이면 기본값으로 되돌린다. **경고 로그를 남긴다** — 조용히 바꿔치우면
            왜 기대와 다르게 도는지 알 수 없다.

    Returns:
        검증된 정수.
    """
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


def classify_assets(**_context) -> dict[str, int]:
    """분류 배치를 한 번 돌리는 껍데기 — 스킬 로드·판정·저장·색인은 전부 러너가 한다.

    **DB에 쓴다**(판정 라벨 행) **· 검색 색인에도 쓴다**(``mm_skill_labels`` 부분 갱신).

    Returns:
        판정·실패·색인 건수 dict. ⚠️ **개수만** 돌려준다 — 태스크 사이 값 전달 저장소에는
        직렬화 가능한 작은 값만 담아야 한다(스킬별 라벨 분포까지 넣으면 불필요하게 커진다).
    """
    from processing.app.run_mm_classify import run_batch
    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    init_settings(os.environ.get("META_ENV", _DEFAULT_ENV))
    db = PostgresUtil()
    with db:
        report = run_batch(db, limit=_int_env("DAG_MM_CLASSIFY_LIMIT", 50))
    summary = {
        "judged": report.get("judged", 0),
        "failed": report.get("failed", 0),
        "indexed": report.get("indexed", 0),
    }
    _LOG.info("dag_mm_classify 완료: %s (skipped=%s)", summary, report.get("skipped"))
    return summary


def _has_more_mm_classify(**context) -> bool:
    """이번에 판정한 것이 있으면 참 — 자기 자신을 다시 깨워 남은 것을 이어서 소화한다.

    대상이 바닥나면 다음 실행이 0건이 되어 이 게이트가 차단한다(빈 실행 한 번으로 끝).

    Returns:
        판정에 성공한 자산이 있으면 참.
    """
    summary = context["ti"].xcom_pull(task_ids="classify_assets") or {}
    return summary.get("judged", 0) > 0


with DAG(
    dag_id="dag_mm_classify",
    description="활성 분류 스킬 × registered 자산 → asset_mm_skill_label · OS 패싯 부분 갱신",
    schedule=_SCHEDULE,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,            # 과거분 보충 금지 — 무엇이 남았는지는 DB 상태(판정 행)가 정본이다
    max_active_runs=1,        # 동시 1개 — 둘이 돌면 같은 자산을 두 번 판정하고 LLM 만 두 배로 쓴다
    tags=["pipeline", "mm_classify"],
):
    classify = PythonOperator(task_id="classify_assets", python_callable=classify_assets)
    # 연속 드레인 — 이번에 뭔가 판정했으면 자기 자신을 다시 깨워 남은 것을 끝까지 소화한다.
    gate_more = ShortCircuitOperator(
        task_id="gate_more_mm_classify", python_callable=_has_more_mm_classify
    )
    trigger_more = TriggerDagRunOperator(task_id="trigger_more", trigger_dag_id="dag_mm_classify")
    classify >> gate_more >> trigger_more
