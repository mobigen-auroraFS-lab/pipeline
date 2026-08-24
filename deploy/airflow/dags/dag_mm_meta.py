"""요약·키워드가 있는 자산을 훑어 **멀티모달 메타 소속**을 만드는 DAG(084).

**흐름에서의 위치**: 적재·관계 파이프라인과 **나란히 도는 별도 배치**다. 기존 경로에 손대지 않고
(적재 스테이지 없음 · spec §1) 이미 저장된 요약·키워드만 읽어 "이 파일은 어떤 개체를 말하는가"를
판정하고 소속 엣지를 만든다. 그래서 이 DAG 가 멈춰도 수집·적재·검색은 그대로 돈다.

⚠️ **배치 함수를 직접 조립하지 않는다.** 색인 조립(배치당 1회)·자산별 트랜잭션·설명 단계 순서는
러너의 ``run_batch`` 한 곳에 있다 — 여기에 복사하면 한쪽만 고쳐져 갈라진다(대표 사고: 색인을
자산마다 만들어 자산 수 × 개체 수로 발산).

**연속 드레인**: 한 번에 정해진 개수만 처리하므로, 판정한 것이 있으면 자기 자신을 다시 깨워
이어서 소화한다. ⚠️ 조건은 "이번에 판정했는가"다 — 전량 실패로 정체한 상태에서 다시 깨우면
진전 없이 빙빙 돈다(그때는 정기 실행 안전망에 맡긴다 · 관계 DAG 와 같은 규율).

⚠️ 모듈 최상위는 가볍게 — 설정·DB·배치 러너 import 는 태스크 함수 **안에서** 한다(스케줄러가
DAG 파일을 자주 파싱한다).
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

_LOG = logging.getLogger("meta_extract.dag_mm_meta")

# ── 운영 설정(env 기본값 — 모듈 최상위는 경량) ───────────────────────────────────
_DEFAULT_ENV = "prod"
# 정기 실행은 안전망이자 정상 경로다(이 배치를 밀어 주는 트리거가 없다 — 적재와 독립이므로).
# 관계 DAG(30분)와 **시각을 어긋나게** 둔다: 같은 시각에 겹치면 LLM 을 두 배치가 동시에 두드린다.
_SCHEDULE = os.environ.get("DAG_MM_META_SCHEDULE", "45 * * * *")


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


def bind_mm_meta(**_context) -> dict[str, int]:
    """소속 배치를 한 번 돌리는 껍데기 — 대상 선별·판정·저장·설명은 전부 러너가 한다.

    **DB에 쓴다**(개체 노드·소속 엣지·판정 이력·메타 설명).

    발굴 모드는 환경변수 ``MM_META_DISCOVERY_MODE`` 로 정한다(기본 ``propose`` — 등록된 메타에만
    붙이고 미등록 개체는 후보 리포트로만 낸다). ⚠️ 어휘 밖 값이면 **태스크가 실패한다** —
    조용히 기본값으로 흡수하면 "왜 메타가 안 생기나"를 며칠 추적하게 된다.

    Returns:
        판정·실패·소속 자산 건수 dict. ⚠️ **개수만** 돌려준다 — 태스크 사이 값 전달 저장소에는
        직렬화 가능한 작은 값만 담아야 한다(후보 목록·자산 id 까지 넣으면 불필요하게 커진다).
    """
    from processing.app.run_mm_meta_binding import (
        DISCOVERY_MODE_ENV,
        resolve_discovery_mode,
        run_batch,
    )
    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    init_settings(os.environ.get("META_ENV", _DEFAULT_ENV))
    mode = resolve_discovery_mode(os.environ.get(DISCOVERY_MODE_ENV))
    db = PostgresUtil()
    with db:
        result = run_batch(db, mode=mode, limit=_int_env("DAG_MM_META_LIMIT", 50))
    # 토글(MM_META_BINDING_ENABLED=0)이 꺼져 있으면 binding 이 None 이다 — 그 위에서 터지지 않게.
    binding = result.get("binding") or {}
    summary = {
        "judged": binding.get("judged_ok", 0),
        "failed": binding.get("judged_failed", 0),
        "bound": binding.get("assets_bound", 0),
    }
    _LOG.info("dag_mm_meta 완료: %s (mode=%s)", summary, mode)
    return summary


def _has_more_mm_meta(**context) -> bool:
    """이번에 판정한 것이 있으면 참 — 자기 자신을 다시 깨워 남은 것을 이어서 소화한다.

    대상이 바닥나면 다음 실행이 0건이 되어 이 게이트가 차단한다(빈 실행 한 번으로 끝).

    Returns:
        판정에 성공한 자산이 있으면 참.
    """
    summary = context["ti"].xcom_pull(task_ids="bind_mm_meta") or {}
    return summary.get("judged", 0) > 0


with DAG(
    dag_id="dag_mm_meta",
    description="registered·키워드 보유 자산 → 개체 판정 → mm_member 소속 엣지·메타 설명",
    schedule=_SCHEDULE,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,            # 과거분 보충 금지 — 무엇이 남았는지는 DB 상태(판정 이력)가 정본이다
    max_active_runs=1,        # 동시 1개 — 둘이 돌면 같은 자산을 두 번 판정하고 LLM 만 두 배로 쓴다
    tags=["pipeline", "mm_meta"],
):
    bind = PythonOperator(task_id="bind_mm_meta", python_callable=bind_mm_meta)
    # 연속 드레인 — 이번에 뭔가 판정했으면 자기 자신을 다시 깨워 남은 것을 끝까지 소화한다.
    gate_more = ShortCircuitOperator(
        task_id="gate_more_mm_meta", python_callable=_has_more_mm_meta
    )
    trigger_more = TriggerDagRunOperator(task_id="trigger_more", trigger_dag_id="dag_mm_meta")
    bind >> gate_more >> trigger_more
