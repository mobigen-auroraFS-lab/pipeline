"""F-1.4 자산 처리 상태 머신.

``asset.status`` 전이 규칙과 DB 갱신 헬퍼. 모델 A(조기 INSERT) 전제:
오케스트레이터가 파일 픽업 시 ``asset`` 행을 ``received`` 로 만들고,
단계가 진행될 때마다 ``set_status`` 로 전이한다. 임의 단계에서 실패하면
``mark_failed`` 로 ``failed`` + 사유를 남기고 다음 파일로 넘어간다.

디스패처 단일 권위: 미지원 modality 등은 사전 차단하지 않고 흐름에 태운 뒤,
``dispatch_extract`` 의 예외를 오케스트레이터가 잡아 ``mark_failed`` 로 흡수한다.

상태 헬퍼는 psycopg ``Connection`` 을 받아 오케스트레이터의 트랜잭션 안에서 조합된다
(``src/relations/*`` 와 동일 패턴). 전이 검증 로직(``validate_transition``)은 DB 없이도 순수 호출 가능.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from src.domain.status_vocab import AssetStatus as _AssetStatus

# ⚠️ 이 이름은 이제 코어의 ``StrEnum`` 을 **그대로 재수출**한다 — 그래서 ``str(AssetStatus.RECEIVED)``
#    는 ``"received"`` 다(옛 ``(str, Enum)`` 시절의 ``"AssetStatus.RECEIVED"`` 가 아니다). 로그·
#    f-string 에 그대로 찍히는 값이라 이관 때 소비처를 전수 확인했다 — 이 레포는 어디서나 ``.value``
#    또는 이미 문자열인 변수만 쓰므로 영향이 없다(리뷰 2026-09-09 · ``tests/test_status.py`` 가 봉인).
# 🔴 **값 목록은 코어 정본을 쓴다**(2026-09-02). 전에는 이 파일이 값까지 소유했는데, 그러면
#    백엔드가 가져다 쓸 길이 없어(3레포 구조상 service 는 pipeline 을 의존하지 않는다) 상태
#    문자열을 17곳에 직접 타이핑하게 된다. **값은 읽는 쪽도 쓰므로 공유 코어**에 두고,
#    **전이 규칙(FSM)은 상태를 바꾸는 이 레포**가 계속 소유한다.
#    아래 재수출 덕분에 기존 ``from processing.ingest.status import AssetStatus`` 는 그대로 돈다.
AssetStatus = _AssetStatus


# 불변식: TERMINAL 상태는 ALLOWED_TRANSITIONS 에서 빈 집합을 가져야 한다.
TERMINAL: frozenset[AssetStatus] = frozenset(
    {AssetStatus.REGISTERED, AssetStatus.FAILED, AssetStatus.DEFERRED}
)

# 전이 규약: status 는 해당 단계 '진입 전'(처리 시작 시점)에 set_status 로 찍는 진행형 마커이며,
# 단계 완료는 따로 찍지 않고 '다음 전이'로 암시된다(예: classifying→extracting = 분류 완료).
# 결과/종착 상태(registered/deferred/failed)만 처리 '후'에 찍는다.
# routing 은 route_file 이 asset 행 생성(received) 전에 끝나는 즉시 작업이라 사후 마커이고,
# run_ingest 에서 classifying 과 한 트랜잭션에 묶여 단독으로는 관측되지 않는다(provenance 는 lineage 가 보존).
# 정상 진행 경로 + 임의 비종료 단계에서 failed 로 전이 가능.
# classifying → deferred: 의료 표준 포맷 감지 시 추출 보류.
# classifying → extracting: 일반 도메인 및 의료 비표준 포맷(현 stopgap 경로).
ALLOWED_TRANSITIONS: dict[AssetStatus, frozenset[AssetStatus]] = {
    AssetStatus.RECEIVED: frozenset({AssetStatus.ROUTING, AssetStatus.FAILED}),
    AssetStatus.ROUTING: frozenset({AssetStatus.CLASSIFYING, AssetStatus.FAILED}),
    AssetStatus.CLASSIFYING: frozenset({AssetStatus.EXTRACTING, AssetStatus.DEFERRED, AssetStatus.FAILED}),
    AssetStatus.EXTRACTING: frozenset({AssetStatus.REGISTERED, AssetStatus.FAILED}),
    AssetStatus.REGISTERED: frozenset(),
    AssetStatus.FAILED: frozenset(),
    AssetStatus.DEFERRED: frozenset(),
}


class InvalidTransitionError(RuntimeError):
    """허용되지 않은 상태 전이."""


class ConcurrentTransitionError(InvalidTransitionError):
    """동시 전이 충돌 — 조건부 UPDATE 가 0행을 갱신(다른 워커가 먼저 상태를 바꿈, lost update 거부).

    ``InvalidTransitionError`` 를 상속한다(의도적): run_ingest 의 fresh 트랜잭션
    실패-기록 격리부가 이미 ``except InvalidTransitionError`` 로 종료 상태 전이를 흡수하므로,
    충돌도 같은 경로로 자연 흡수되어 배치가 멈추지 않는다(헌법 8조 — 호출부 시그니처 무변경).
    한 번에 하나씩 처리하는 경로에서는 애초에 발생하지 않는다.
    """


def validate_transition(current: AssetStatus | str, target: AssetStatus | str) -> None:
    """이 전이가 허용된 것인지 확인한다.

    Args:
        current: 지금 상태.
        target: 바꾸려는 상태.

    Raises:
        InvalidTransitionError: 허용 표에 없는 전이일 때. **조용히 무시하지 않는다** —
            허용 밖 전이는 흐름이 꼬였다는 신호라 그 자리에서 드러나야 한다.
    """
    cur = AssetStatus(current)
    tgt = AssetStatus(target)
    if tgt not in ALLOWED_TRANSITIONS.get(cur, frozenset()):
        raise InvalidTransitionError(f"{cur.value} → {tgt.value} 전이는 허용되지 않습니다.")


def fetch_status(conn: Connection[Any], asset_id: uuid.UUID) -> AssetStatus:
    """``asset.status`` 조회. 없으면 ``LookupError``."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT status FROM asset WHERE asset_id = %s", (asset_id,))
        row = cur.fetchone()
    if row is None:
        raise LookupError(f"asset 없음: asset_id={asset_id}")
    return AssetStatus(row["status"])


def set_status(
    conn: Connection[Any],
    asset_id: uuid.UUID,
    target: AssetStatus | str,
    *,
    reason: str | None = None,
) -> None:
    """현재 상태를 읽어 전이를 검증한 뒤 ``asset.status`` 를 **조건부**로 갱신한다.

    **DB에 쓴다.** 검사와 갱신 사이에 남이 끼어들 수 있으므로, 갱신에도 "지금 이 상태일
    때만"이라는 조건을 건다 — 둘이 동시에 통과해도 실제로 바뀌는 쪽은 하나다.

    Args:
        conn: DB 연결.
        asset_id: 대상 자산.
        target: 바꿀 상태.
        reason: 사유. **주지 않으면 이전 사유가 지워진다** — 정상 전이는 사유를 남길 일이
            없으므로 이것이 기본 동작이다.

    Raises:
        InvalidTransitionError: 허용 밖 전이일 때.
        ConcurrentTransitionError: 그사이 남이 상태를 바꿔 갱신이 0행일 때. 실제 상태를
            다시 읽어 메시지에 담는다.
    """
    tgt = AssetStatus(target)
    current = fetch_status(conn, asset_id)
    validate_transition(current, tgt)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE asset SET status = %s, status_reason = %s, updated_at = now() "
            "WHERE asset_id = %s AND status = %s",
            (tgt.value, reason, asset_id, current.value),
        )
        if cur.rowcount == 0:
            # 조건부 UPDATE 0행 = 그사이 다른 워커가 기대 현재상태를 바꿈(또는 행 소멸).
            # 실제 상태를 재조회해 진단 메시지에 담는다(없으면 unknown).
            cur.execute("SELECT status FROM asset WHERE asset_id = %s", (asset_id,))
            row = cur.fetchone()
            observed = row[0] if row is not None else "unknown"
            raise ConcurrentTransitionError(
                f"동시 전이 충돌: asset_id={asset_id} 가 {current.value}→{tgt.value} 를 기대했으나 "
                f"현재 상태는 {observed} 입니다(다른 워커가 먼저 전이)."
            )


def mark_failed(conn: Connection[Any], asset_id: uuid.UUID, reason: str) -> None:
    """현재 단계에서 ``failed`` 로 전이하고 사유를 남긴다(디스패처 예외 등 흡수용)."""
    set_status(conn, asset_id, AssetStatus.FAILED, reason=reason)
