"""도메인 분류 프로파일 — 도메인 지식(시그니처·어휘·LLM 라벨)을 데이터로 분리.

cascade 엔진은 등록 메커니즘이 아니라 ``DomainProfileProvider`` 추상 seam 에만
의존한다. A 단계는 코드 등록(``RegistryProvider``), B 단계(후속)는 DB provider 로
교체하되 엔진·테스트는 불변. 순수 데이터(lexicon/llm_label)는 후일 DB 이전 대상,
signatures 는 능력 모듈(코드)로 유지(설계 원칙 2).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SigHit:
    """시그니처 매칭 결과. signature=판별 종류, detail=부가 정보(예: resourceType).

    주의: cascade 가 s1_scores 에 {"signature": hit.signature, **hit.detail} 로 펼치므로
    detail 에 "signature" 키가 있으면 충돌 덮어쓰기가 발생한다 — 관례상 피해야 한다.
    """

    signature: str
    detail: dict[str, Any] = field(default_factory=dict)


# 첫 head 바이트를 받아 매칭 시 SigHit, 아니면 None. 텍스트 디코딩은 규칙 내부 책임.
# 고신뢰 바이너리 시그니처(DICOM 매직, HL7 세그먼트 등)만 여기에 등록한다.
SignatureRule = Callable[[bytes], "SigHit | None"]


@dataclass(frozen=True)
class DomainProfile:
    """한 도메인의 분류 지식. domain=asset.domain_label 값.

    설계 원칙:
    - lexicon·llm_label 은 순수 데이터 → 후일 DB 이전 대상.
    - signatures 는 능력 모듈(코드) → 도메인 팩과 함께 코드로 유지.
    새 도메인 추가 시 도메인 프로파일 모듈만 추가하면 cascade 코드 수정 없음.
    """

    domain: str
    lexicon: frozenset[str] = frozenset()      # stage2: 어휘(순수 데이터)
    llm_label: str = ""                         # stage3 LLM 허용 라벨용 예비 필드(순수 데이터). 현재
    #   stage3 배선은 이 필드를 소비하지 않는다 — 의료 stage3 는 3년차 이연(2026-07-06)이라 미배선(P3-6).
    signatures: tuple[SignatureRule, ...] = ()  # stage1: 능력 모듈(코드)


class DomainProfileProvider(Protocol):
    """cascade 엔진이 프로파일을 얻는 추상 seam.

    A 단계는 코드 등록(RegistryProvider), B 단계에서 DB provider 로 교체해도
    cascade 엔진·테스트는 수정 불필요.
    """

    def all_profiles(self) -> list[DomainProfile]:
        """등록된 프로파일 전체를 돌려준다."""
        ...


# 프로세스 전역 등록(import 부수효과; pipeline.builtins.register_defaults 패턴과 동형).
# cascade 가 `from processing.classify import domains` 로 이 딕셔너리를 간접 채운다.
DOMAIN_PROFILES: dict[str, DomainProfile] = {}


def register_profile(profile: DomainProfile) -> None:
    """도메인 프로파일을 전역 목록에 등록한다.

    각 도메인 모듈을 import 하는 것만으로 등록이 일어난다(import 부수효과) — 그래서
    프로파일을 추가하려면 **그 모듈이 import 되는지**부터 확인해야 한다.
    """
    # 같은 domain 키로 재등록하면 덮어쓴다 — 테스트에서 mock 프로파일 주입 시 활용 가능.
    DOMAIN_PROFILES[profile.domain] = profile


class RegistryProvider:
    """A 단계 provider — 코드로 등록된 프로파일 반환. B 단계에서 DB provider 로 교체."""

    def all_profiles(self) -> list[DomainProfile]:
        """코드로 등록된 프로파일을 그대로 돌려준다(추후 DB 조회로 교체할 자리)."""
        return list(DOMAIN_PROFILES.values())
