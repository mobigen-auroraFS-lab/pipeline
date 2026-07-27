"""v2 정책 엔진 — 도메인 팩을 컴포지션 시점에 검증한다.

**흐름에서의 위치**: 파이프라인이 파일을 처리하기 **직전**에 한 번 돈다. 팩이 고른 전략들이
어떤 성질(태그)을 갖는지 보고, 그 도메인이 허용하지 않는 조합이면 그 자리에서 멈춘다.

**왜 실행 전에 보는가**: 위반이 처리 도중에 드러나면 이미 절반쯤 쓴 상태에서 멈춘다.
규정 위반은 되돌리기가 특히 어려우므로(예: 외부로 나간 호출은 취소할 수 없다) 배선을
확인하는 단계에서 걸러 낸다.

지금 걸린 제약은 의료의 **외부 LLM 금지** 하나다. 자산 사이 단계·비식별화 관련 제약은
그 팩을 실제로 배선할 때 함께 추가한다.
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

        Args:
            pack: 검사할 도메인 팩(어느 슬롯에 어느 전략을 쓰기로 했는지).
            registry: 전략 성질을 조회할 레지스트리. 팩은 이름만 갖고 있어서, 그 이름이
                가리키는 전략의 태그는 여기서 찾아야 한다.

        Returns:
            위반이면 **사람이 읽을 사유 문자열**, 통과면 ``None``. 예외가 아니라 문자열인
            이유는 여러 제약의 위반 사유를 모아 한 번에 보여 주기 위해서다.
        """
        ...


@dataclass(frozen=True)
class ForbidTag:
    """팩의 어느 슬롯 전략도 이 태그를 가지면 안 된다.

    ⚠️ **알려진 구멍**: 자산 하나를 처리하는 슬롯만 본다. 자산 사이를 잇는 슬롯은 검사하지
    않으므로, 거기에 금지 태그를 가진 전략을 배선해도 **통과해 버린다**. 의료 관계 전략을
    실제로 배선하기 전에 반드시 자산 사이 슬롯까지 순회하도록 고쳐야 한다.
    """

    tag: str

    def check(self, pack, registry) -> str | None:
        """자산 단위 슬롯 전략들이 금지 태그를 갖고 있지 않은지 확인한다.

        ⚠️ **자산 사이 슬롯은 아직 보지 않는다** — 클래스 설명의 구멍 참조.

        Args:
            pack: 검사할 팩.
            registry: 전략 이름 → 태그를 찾을 곳.

        Returns:
            금지 태그를 가진 슬롯이 있으면 그 사유, 없으면 ``None``.
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
        """지정 슬롯의 전략이 필수 태그를 갖고 있는지 확인한다.

        Args:
            pack: 검사할 팩.
            registry: 전략 이름 → 태그를 찾을 곳.

        Returns:
            사유 문자열, 또는 통과면 ``None``. **슬롯 자체가 없어도 위반**이다 —
            필수 조건을 건 슬롯이 비어 있으면 조건이 지켜졌다고 볼 수 없다.
        """
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
    """팩의 정책을 검증한다 — 위반이면 처리를 시작하지 않는다.

    파일마다 한 번 도는데, 팩 구성은 사실상 고정이라 같은 검사를 반복하는 비용은 무시할
    수준이다. 그보다 **어떤 파일도 검사 없이 지나가지 않는 것**이 중요하다.

    Args:
        pack: 검사할 도메인 팩.
        registry: 전략 성질을 조회할 레지스트리.

    Raises:
        PolicyViolation: 등록되지 않은 정책 이름이거나, 제약을 하나라도 어겼을 때.
            **위반 사유를 모아서** 한 번에 알린다 — 하나 고치고 다시 돌리는 왕복을 줄인다.
    """
    policy = POLICIES.get(pack.policy)
    if policy is None:
        raise PolicyViolation(f"미등록 정책: {pack.policy!r}")
    violations = [msg for c in policy.constraints if (msg := c.check(pack, registry)) is not None]
    if violations:
        raise PolicyViolation(f"정책 '{policy.name}' 위반: " + "; ".join(violations))
