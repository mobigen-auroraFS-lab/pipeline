"""v2 정책 엔진 — 도메인 팩을 컴포지션 시점에 검증한다.

**설계 의도**: 정책은 팩이 선택한 전략의 capability 태그를 검사해 "이 조합이 도메인 규정을
준수하는가"를 컴포지션 시점(run_ingest 진입부)에 판별한다. 규정 위반이 런타임 깊숙이
전파되기 전에 조기 실패(fast-fail) 시킨다.

constraint 는 팩이 고른 전략의 capability 태그를 검사한다. 단계 B는 NoExternalLLM
(외부 LLM 전략 금지)만 의료에 적용한다. PHI 선행·결정성 스코어러·Negative Override 등
cross-asset/PHI 관련 constraint 는 단계 C에서 추가한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from processing.pipeline.packs import DomainPack
    from processing.pipeline.registry import StrategyRegistry


class PolicyViolation(Exception):
    """팩이 정책 constraint 를 위반."""


class Constraint(Protocol):
    def check(self, pack: DomainPack, registry: StrategyRegistry) -> str | None:
        """정책 위반 여부를 판정한다.

        Returns:
            위반이면 **사람이 읽을 사유 문자열**, 통과면 ``None``.
        """
        ...


@dataclass(frozen=True)
class ForbidTag:
    """팩의 어느 슬롯 전략도 이 태그를 가지면 안 된다.

    **알려진 한계(단계 D=3년차 이연·2026-07-06 이전)**: ``check`` 는 ``pack.per_asset`` 슬롯만 순회한다.
    ``pack.cross_asset`` 슬롯은 현재 검사하지 않는다. 따라서 단계 D 에서 의료 cross_asset 에
    'external_llm' 태그를 가진 전략을 배선하더라도 medical_strict 의 ForbidTag 가 통과해버린다.
    단계 D 착수 전에 cross_asset 슬롯도 순회하도록 보완해야 한다.
    """

    tag: str

    def check(self, pack, registry) -> str | None:
        """자산 단위 슬롯 전략들이 금지 태그를 갖고 있지 않은지 확인한다.

        ⚠️ **자산 사이(cross-asset) 슬롯은 아직 보지 않는다** — 클래스 docstring 참조.
        """
        for slot, name in pack.per_asset.items():
            if self.tag in registry.tags(slot, name):
                return f"{slot}={name} 전략이 금지 태그 '{self.tag}' 보유"
        return None


@dataclass(frozen=True)
class RequireTag:
    """특정 슬롯 전략이 이 태그를 반드시 가져야 한다.

    교차 검증 예: RequireTag("score", "onprem_llm") → score 슬롯 전략이 반드시
    온프레미스 LLM 기반이어야 한다는 조건을 선언적으로 강제한다.
    """

    slot: str
    tag: str

    def check(self, pack, registry) -> str | None:
        """지정 슬롯의 전략이 필수 태그를 갖고 있는지 확인한다(슬롯 자체가 없어도 위반)."""
        name = pack.per_asset.get(self.slot)
        if name is None:
            return f"슬롯 '{self.slot}' 미정의"
        if self.tag not in registry.tags(self.slot, name):
            return f"{self.slot}={name} 전략에 필수 태그 '{self.tag}' 없음"
        return None


@dataclass(frozen=True)
class DomainPolicy:
    """constraint 목록을 묶는 컨테이너. constraints 가 빈 튜플이면 무제약."""
    name: str
    constraints: tuple[Constraint, ...] = ()


POLICIES: dict[str, DomainPolicy] = {
    # 일반 도메인: 외부 LLM 포함 모든 전략 허용.
    "general_default": DomainPolicy("general_default", ()),
    # 의료: 과제 정책 — 외부 LLM API 완전 금지(온프레미스 LLM 만 허용, PHI 보호).
    "medical_strict": DomainPolicy("medical_strict", (ForbidTag("external_llm"),)),
}


def validate(pack: DomainPack, registry: StrategyRegistry) -> None:
    """팩의 정책을 검증한다. 위반 시 PolicyViolation.

    **호출 시점**: run_ingest 진입부(policy_validate 스테이지)에서 파일당 1회 호출된다.
    팩 구성이 사실상 고정(frozen dataclass·얕은 동결이나 변조 금지 관례)이므로 중복 검증 비용은 무시할 수준이다.
    """
    policy = POLICIES.get(pack.policy)
    if policy is None:
        raise PolicyViolation(f"미등록 정책: {pack.policy!r}")
    violations = [msg for c in policy.constraints if (msg := c.check(pack, registry)) is not None]
    if violations:
        raise PolicyViolation(f"정책 '{policy.name}' 위반: " + "; ".join(violations))
