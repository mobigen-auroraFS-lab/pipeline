"""파이프라인 **슬롯 계약** — 각 단계에 끼울 함수가 어떤 모양이어야 하는지 정의한다.

**흐름에서의 위치**: 도메인 팩이 슬롯마다 전략(구현)을 고르고, 레지스트리가 이름으로 찾아
실행한다. 이 파일은 그 전략들이 지켜야 할 **시그니처**만 선언한다 — 구현은 없다.

런타임에 검사하지 않는다(``isinstance`` 로 확인하지 않음). 사람이 읽는 계약이자 타입
검사기가 보는 기준이며, 어긋난 전략은 배선 시점에 드러난다.

단계는 데이터 모양으로 둘로 갈린다 — **자산 하나를 처리**(분류·추출·임베딩·저장)와
**자산 사이를 잇는 처리**(후보·점수·판정·엣지 저장).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from psycopg import Connection

    from processing.classify.types import ClassificationResult
    from processing.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
    from processing.pipeline.cross_types import Candidate, Decision, ScoredPair


# ── per-asset ────────────────────────────────────────────────────────────
class ClassifyStage(Protocol):
    """자산이 어느 도메인·종류인지 판정한다 — 뒤 단계가 쓸 전략을 고르는 근거가 된다."""

    def __call__(self, ctx: ExtractContext) -> ClassificationResult: ...


class ExtractStage(Protocol):
    """파일에서 메타·태그를 뽑아 한 자산 레코드로 만든다(임베딩은 다음 단계 몫)."""

    def __call__(self, ctx: ExtractContext) -> AssetRecord: ...


class EmbedStage(Protocol):
    """추출 결과를 벡터로 만든다. 자산 하나가 여러 청크·키프레임을 낼 수 있어 목록을 돌려준다.

    Args:
        ctx: 파일 경로·모달리티 등 처리 문맥.
        rec: 추출 단계가 만든 자산 레코드. **여기서 벡터로 만들 원문·프레임이 들어 있다** —
            파일을 다시 읽지 않는다.
    """

    def __call__(self, ctx: ExtractContext, rec: AssetRecord) -> list[EmbeddingItem]: ...


class PersistStage(Protocol):
    """앞 단계 결과를 DB 에 쓴다. 트랜잭션 경계는 호출자(러너)가 잡는다.

    Args:
        conn: 호출자가 연 연결. **여기서 커밋하지 않는다** — 한 자산의 여러 저장이
            반쯤 남는 것을 막으려면 경계를 밖에서 잡아야 한다.
        asset_id: 저장 대상 자산.
        rec: 저장할 내용.
    """

    def __call__(self, conn: Connection, asset_id, rec: AssetRecord) -> None: ...


# ── cross-asset ──────────────────────────────────────────────────────────
class CandidateStage(Protocol):
    """이 자산과 이어질 만한 상대를 추린다 — **뒤 단계는 여기서 나온 것만** 본다."""

    def __call__(self, conn: Connection, source_asset_id: str) -> list[Candidate]: ...


class ScoreStage(Protocol):
    """후보 쌍마다 점수를 매긴다(어떤 근거로 매길지는 전략이 정한다).

    Args:
        conn: DB 연결. 점수 근거를 조회하는 전략이 쓴다.
        pairs: 앞 단계가 추린 후보. **모든 쌍의 출발 자산이 같다**는 전제로 만들어진
            목록이라, 전략이 조회를 한 번으로 묶을 수 있다.
    """

    def __call__(self, conn: Connection, pairs: list[Candidate]) -> list[ScoredPair]: ...


class DecideStage(Protocol):
    """점수를 보고 실제로 이을지 정한다. **DB 를 보지 않는다** — 순수 판정이라 단위 검증이 쉽다.

    Args:
        scored: 점수가 매겨진 쌍. 이 정보만으로 판정이 끝나야 한다 — 여기서 DB 를 더
            읽으면 같은 입력이 상황에 따라 다른 결정을 내게 된다.
    """

    def __call__(self, scored: list[ScoredPair]) -> list[Decision]: ...


class EdgePersistStage(Protocol):
    """판정 결과를 그래프에 저장한다.

    Args:
        conn: 호출자가 연 연결(트랜잭션 경계는 밖).
        decisions: 판정 목록. **잇지 않기로 한 것까지 넘어온다** — 무엇을 저장할지는
            구현이 판정값을 보고 고른다.
    """

    def __call__(self, conn: Connection, decisions: list[Decision]) -> None: ...
