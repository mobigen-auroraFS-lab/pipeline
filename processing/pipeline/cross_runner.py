"""제네릭 cross_asset 러너 — 팩이 resolve 한 슬롯 Callable 을 계약 순서대로 실행.

목적(spec 016, plan D-1)
    008 이 만든 cross_asset 슬롯 resolve seam 과 ``contracts.py`` 에 정의돼 있으나 한 번도
    실행된 적 없는 cross_asset 계약 4종(Candidate/Score/Decide/EdgePersist)을, 도메인을
    코드로 분기하지 않고 **한 곳에서 제네릭하게** 실행한다. G3 에서 ``run_relations`` 의
    ``else`` 분기(현재 NotImplementedError)를 이 러너 호출로 교체한다.

헌법 4조 — 이 모듈의 본질(팩 선택으로만 갈리고 core 분기 0)
    러너는 **순수 오케스트레이션**이다. ``resolved`` dict 의 4개 Callable 만 보고
    ``contracts.py`` Protocol 순서대로 호출할 뿐, ``if domain == ...`` / ``if pack.name == ...``
    같은 도메인 코드 분기를 절대 두지 않는다. 도메인 차이는 *어떤 전략이 resolve 됐는가*
    (팩 선택)로만 드러나며, core 실행 경로는 모든 도메인에 대해 동일하다. 덕분에 새 도메인을
    추가해도 이 러너는 한 줄도 바뀌지 않는다.

계약 순서(contracts.py Protocol)
    candidates(conn, source_asset_id) -> list[Candidate]
      → score(conn, pairs)            -> list[ScoredPair]
      → decide(scored)                -> list[Decision]      # conn 안 받음(순수)
      → persist_edges(conn, decisions)-> None

결정성(헌법 3조)
    러너 자체는 입력을 변형하지 않고 그대로 전달하며 분기·난수가 없으므로, 주입된 전략이
    결정적이면 러너 출력(적재 엣지 수)도 동일 입력 2회 동일하다.
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
    """resolve 된 4슬롯 전략을 계약 순서대로 실행하고 적재 엣지 수를 돌려준다.

    인자
        resolved: 슬롯명→Callable 매핑. 키는 ``candidates``/``score``/``decide``/
            ``persist_edges`` 4종(팩이 resolve 한 전략). 러너는 전략의 실체(어느 도메인·
            어느 구현인지)에 비의존하며 이 4 Callable 만 호출한다.
        conn: DB 연결(불투명). conn 을 받는 슬롯(candidates/score/persist_edges)에 그대로
            전달한다. decide 는 순수 함수라 conn 을 받지 않는다.
        source_asset_id: 후보 탐색의 기준 자산.

    반환
        적재 의도된 엣지 수 = verdict=='match' 결정 수. ``persist_edges`` 는 계약상 None 을
        반환하므로(EdgePersistStage), 러너가 decisions 에서 match 수를 직접 센다.
        ⚠️ 엄밀히는 "match 결정 수"다 — persist 구현(예 sync_graph_edges)이 자기참조·후보집합
        밖 타깃·inactive relation_kind 를 skip 할 수 있어 실제 upsert 수와 다를 여지가 있다(샘플
        정상 경로에선 일치). 정확한 적재 수가 필요하면 EdgePersistStage 계약을 int 반환으로
        확장해야 한다(후속). 빈 후보·빈 결정이면 0 을 돌려준다.
    """
    # contracts.py Protocol 순서대로 실행 — 도메인 분기 없이 4 Callable 만 호출한다.
    cands = resolved["candidates"](conn, source_asset_id)
    scored = resolved["score"](conn, cands)
    decisions = resolved["decide"](scored)
    # persist_edges 는 None 을 반환하므로(부수효과 전용), 적재 엣지 수는 러너가 센다.
    resolved["persist_edges"](conn, decisions)
    # 적재 엣지 수 = 'match' 결정 수. 빈 입력이면 자연히 0.
    return sum(1 for d in decisions if d.verdict == "match")
