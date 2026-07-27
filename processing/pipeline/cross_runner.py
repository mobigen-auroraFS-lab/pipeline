"""자산 사이(cross-asset) 단계를 실행하는 **공용 러너** — 도메인을 코드로 분기하지 않는다.

**흐름에서의 위치**: 팩이 고른 네 전략을 받아 계약 순서대로 부른다.

    후보 추리기 → 점수 매기기 → 이을지 판정 → 엣지 저장

**여기에 ``if 도메인 == …`` 을 절대 두지 않는다**(헌법 4조). 도메인 차이는 *어떤 전략이
꽂혔는가*로만 드러난다. 그래서 새 도메인이 생겨도 이 파일은 한 줄도 바뀌지 않는다 —
여기에 분기를 하나 넣는 순간 도메인이 늘 때마다 이 파일을 고치게 되고, 조합형 구조의
이점이 사라진다.

러너 자신은 입력을 만지지 않고 그대로 넘기며 난수도 분기도 없다 — 꽂힌 전략이 결정적이면
러너의 출력도 결정적이다(헌법 3조).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from psycopg import Connection


def run_cross_asset(
    resolved: dict[str, Callable[..., Any]],
    conn: Connection[Any],
    source_asset_id: str,
) -> int:
    """네 전략을 계약 순서대로 실행하고 이어진 엣지 수를 돌려준다.

    Args:
        resolved: 슬롯 이름 → 전략 함수. 키는 ``candidates``·``score``·``decide``·
            ``persist_edges`` 넷이다. 러너는 그 전략이 어느 도메인 것인지 **알지 못하며
            알 필요도 없다**.
        conn: DB 연결. 연결을 받는 세 슬롯에 그대로 넘긴다 — 판정 슬롯만은 순수 함수라
            받지 않는다.
        source_asset_id: 후보를 찾을 기준 자산.

    Returns:
        '잇기로 판정한' 쌍의 수. 저장 슬롯이 아무것도 돌려주지 않는 계약이라 러너가 직접
        센다. ⚠️ **실제 저장된 수와 다를 수 있다** — 저장 구현이 자기 자신을 가리키는 엣지나
        후보 밖 대상을 건너뛸 수 있기 때문이다. 정확한 저장 수가 필요해지면 저장 슬롯
        계약을 개수 반환으로 넓혀야 한다. 후보나 판정이 비면 0.
    """
    # contracts.py Protocol 순서대로 실행 — 도메인 분기 없이 4 Callable 만 호출한다.
    cands = resolved["candidates"](conn, source_asset_id)
    scored = resolved["score"](conn, cands)
    decisions = resolved["decide"](scored)
    # persist_edges 는 None 을 반환하므로(부수효과 전용), 적재 엣지 수는 러너가 센다.
    resolved["persist_edges"](conn, decisions)
    # 적재 엣지 수 = 'match' 결정 수. 빈 입력이면 자연히 0.
    return sum(1 for d in decisions if d.verdict == "match")
