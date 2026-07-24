"""v2 스테이지 슬롯 계약(Protocol).

현 코드의 통일 계약(ExtractContext → AssetRecord, finalize_asset)을 슬롯 인터페이스로
명문화한다. 도메인 팩(단계 B)이 슬롯마다 이 계약을 만족하는 전략을 선택한다.
런타임 isinstance 가 아니라 문서·테스트 계약 용도의 구조적 계약이다(운영 시그니처 배선 시 mypy 구조검사 가능).
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
    def __call__(self, ctx: ExtractContext) -> ClassificationResult: ...


class ExtractStage(Protocol):
    def __call__(self, ctx: ExtractContext) -> AssetRecord: ...  # 메타/태그 (embeddings 비움·037 OS 전용 전환으로 FTS 경로 제거)


class EmbedStage(Protocol):
    def __call__(self, ctx: ExtractContext, rec: AssetRecord) -> list[EmbeddingItem]: ...


class PersistStage(Protocol):
    def __call__(self, conn: Connection, asset_id, rec: AssetRecord) -> None: ...


# ── cross-asset ──────────────────────────────────────────────────────────
class CandidateStage(Protocol):
    def __call__(self, conn: Connection, source_asset_id: str) -> list[Candidate]: ...


class ScoreStage(Protocol):
    def __call__(self, conn: Connection, pairs: list[Candidate]) -> list[ScoredPair]: ...


class DecideStage(Protocol):
    def __call__(self, scored: list[ScoredPair]) -> list[Decision]: ...


class EdgePersistStage(Protocol):
    def __call__(self, conn: Connection, decisions: list[Decision]) -> None: ...
