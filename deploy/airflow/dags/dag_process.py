"""대기 자산과 멈춘 자산을 한 배치로 처리해 등록까지 끌고 가는 DAG(모델을 쓰는 무거운 단계).

**흐름에서의 위치**: 수집 DAG 가 만든 대기 자산을 받아 분류·추출·임베딩·저장까지 끝낸다.
처리 로직은 전부 배치 함수 안에 있고, 이 파일은 "언제·몇 개씩·어떤 제약으로" 부를지만 정한다.

⚠️ **모델을 올리는 프로세스가 절대 둘이 되지 않게 한다.** 동시 실행 1개 제한과 크기 1 슬롯
풀을 **이중으로** 걸었다. 하나만 걸어 두면 배포 설정이 바뀔 때 조용히 풀리고, 그러면 GPU
메모리가 터진다. 한 프로세스로 묶으므로 모델은 배치 전체에서 **한 번만** 올라간다.

**연속 드레인**: 한 번에 정해진 개수만 처리하므로, 처리할 것이 남았으면 자기 자신을 다시
깨워 정기 실행을 기다리지 않고 이어서 소화한다. ⚠️ 재트리거 조건은 "이번에 뭔가 처리했는가"다 —
아무것도 못 했을 때 다시 깨우면 **아무 진전 없이 빙빙 도는 루프**가 된다. 그때는 정기 실행과
멈춘 자산 되돌리기 안전망에 맡긴다.

정기 실행은 **안전망**이다(정상 경로는 수집 DAG 의 트리거). 그래서 뜸하게 둔다 — 모델을 올리는
배치라 헛돌면 비싸고, 트리거가 놓친 잔여만 주기적으로 회수하면 충분하다.

⚠️ 모듈 최상위는 가볍게 — 설정·DB·배치 함수 import 는 태스크 함수 **안에서** 한다.
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
# 정기 실행은 안전망 — 정상 경로는 수집 DAG 의 트리거다. 모델을 올리는 배치라 헛돌면 비싸므로
# 뜸하게 두고, 트리거가 놓친 잔여(크래시·재시도분)만 주기적으로 회수한다.
_SCHEDULE = os.environ.get("DAG_PROCESS_SCHEDULE", "0 * * * *")
# 크기 1 슬롯 풀 — 모델을 올리는 프로세스가 둘이 되지 않게 묶는다(풀 자체는 배포에서 만든다).
_POOL = os.environ.get("DAG_PROCESS_POOL", "gpu")


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    """환경변수를 정수로 읽는다 — 못 읽거나 범위를 벗어나면 기본값으로 되돌린다.

    ⚠️ **범위 검사를 빼면 설정 실수가 조용히 진행된다**: 배치 한도가 음수면 DB 가 거부해 태스크가
    죽고, 0이면 매번 0건 처리하며 아무 소리 없이 정체한다. 재시도 한계가 0이면 첫 실패에서
    바로 격리된다. 세 소비처 모두 1 이상이 유효값이라 최솟값 기본이 1이다.

    Args:
        name: 환경변수 이름.
        default: 못 읽었을 때 쓸 값.
        minimum: 이 값 미만이면 기본값으로 되돌린다. **경고 로그를 남긴다** — 조용히
            바꿔치우면 왜 기대와 다르게 도는지 알 수 없다.

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


def process_batch(**_context) -> dict[str, int]:
    """대기 자산(+멈춘 자산 되돌리기)을 배치 함수로 한 번 처리하는 껍데기.

    **DB에 쓴다**(자산 상태·메타·임베딩·기록).

    Returns:
        갈래별 건수 dict. ⚠️ **원본 보고 객체가 아니라 개수만** 돌려준다 — 태스크 사이 값 전달
        저장소는 직렬화 가능한 값만 담을 수 있고, 자산 id 목록까지 넣으면 불필요하게 커진다.
    """
    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil
    from processing.ingest.batch_runner import process_received_batch

    # ⚠️ 설정을 **명시적으로 넘긴다** — 배치가 이것으로 색인기를 배선한다. 안 넘기면 색인 설정을
    # 못 읽어 색인이 말없이 꺼진다.
    settings = init_settings(os.environ.get("META_ENV", _DEFAULT_ENV))
    db = PostgresUtil()
    with db:
        report = process_received_batch(
            db,
            limit=_int_env("DAG_PROCESS_LIMIT", 50),                 # 한 번에 처리할 최대 자산 수
            max_failures=_int_env("DAG_PROCESS_MAX_FAILURES", 3),    # 이 횟수 이상 실패하면 격리
            older_than_s=_int_env("DAG_PROCESS_STUCK_OLDER_THAN_S", 900),  # 이보다 오래 멈추면 되돌린다
            settings=settings,                                        # 색인기 배선에 필요(위 주석 참조)
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
    """이번에 새로 등록한 자산이 있을 때만 참 — 없으면 아래 단계를 건너뛴다.

    두 곳에서 쓴다: 관계 DAG 를 깨울지, 그리고 **자기 자신을 다시 깨울지**.
    ⚠️ 후자가 중요하다 — 아무것도 처리하지 못했는데 다시 깨우면 진전 없이 빙빙 도는 루프가 된다.

    건너뛰어도 아무것도 잃지 않는다 — 관계 DAG 의 정기 실행이 살아 있어 다음 주기에 회수된다.

    Returns:
        새로 등록한 자산이 있으면 참.
    """
    summary = context["ti"].xcom_pull(task_ids="process_batch") or {}
    return summary.get("registered", 0) > 0


def archive_processed(**_context) -> int:
    """처리가 끝난 자산의 파일을 인입에서 아카이브로 옮기고 경로를 갱신하는 꼬리 태스크.

    **DB에 쓴다**(경로 갱신) **그리고 파일을 옮긴다**.

    **여러 번 돌려도 안전하다** — 옮긴 파일은 인입 밖에 있어 다음 스윕에서 자연히 빠진다.

    ⚠️ **옮긴 뒤에 경로를 갱신하는 순서**다. 중간에 죽어 갱신이 빠져도, 목적 경로가 자산 id 로
    정해지므로 다음 스윕이 같은 곳을 가리켜 이동은 무동작이 되고 갱신만 다시 시도된다.
    (반대 순서면 경로는 갱신됐는데 파일은 그대로인 상태가 남는다.)

    모델을 쓰지 않으므로 슬롯 풀에 묶지 않는다.

    Returns:
        옮긴 파일 수.

    Raises:
        RuntimeError: 인입·아카이브 경로 환경변수가 없을 때.
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
    description="received(+고착) 자산 배치 처리 → registered",
    schedule=_SCHEDULE,
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,            # 과거분 보충 금지 — 무엇이 남았는지는 DB 상태가 정본이다
    max_active_runs=1,        # 동시 1개 — 모델을 올리는 프로세스가 둘이 되면 GPU 가 터진다
    tags=["pipeline", "process"],
):
    # 슬롯 풀로 한 번 더 묶는다 — 위 동시 실행 제한과 **이중 가드**(하나가 풀려도 남는다).
    process = PythonOperator(task_id="process_batch", python_callable=process_batch, pool=_POOL)
    # 새로 등록한 것이 있을 때만 통과 — 게이트·트리거는 모델을 쓰지 않아 슬롯 풀에 묶지 않는다.
    gate = ShortCircuitOperator(task_id="gate_new_registered", python_callable=_has_new_registered)
    # 등록 직후 관계 DAG 를 바로 깨운다(정기 실행까지 기다리지 않게). 정기 실행은 안전망으로 유지.
    trigger_relations = TriggerDagRunOperator(task_id="trigger_relations", trigger_dag_id="dag_relations")
    # 연속 드레인 — 이번에 뭔가 처리했으면 **자기 자신을 다시 깨워** 남은 대기 자산을 이어서
    #   소화한다(한 번에 정해진 개수만 처리하므로 정기 실행을 기다리면 오래 걸린다).
    #   대기 자산이 바닥나면 다음 실행이 0건이 되어 이 게이트가 차단한다(빈 실행 한 번으로 끝).
    # ⚠️ 조건이 "뭔가 처리했는가"인 것이 핵심이다 — 전량 실패로 정체된 상황에서 다시 깨우면
    #   진전 없이 빙빙 돈다. 그때는 정기 실행과 되돌리기 안전망에 맡긴다.
    #   동시 실행이 1개라 재트리거된 실행은 지금 실행이 **끝난 뒤** 순차로 돈다(GPU 보장 불변).
    gate_more = ShortCircuitOperator(task_id="gate_more_received", python_callable=_has_new_registered)
    trigger_more = TriggerDagRunOperator(task_id="trigger_more", trigger_dag_id="dag_process")
    # 처리완료 파일 아카이브 — 처리 뒤 독립 꼬리(게이트·트리거와 나란히 돌고 슬롯 풀을 쓰지 않는다).
    archive = PythonOperator(task_id="archive_processed", python_callable=archive_processed)
    process >> gate >> trigger_relations
    process >> gate_more >> trigger_more
    process >> archive
