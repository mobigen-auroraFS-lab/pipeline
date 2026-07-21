"""v2 전략 레지스트리 — 슬롯별 (이름 → 구현 + capability 태그).

**태그 규약** (policy.py 의 ForbidTag/RequireTag 가 이 태그를 기준으로 정책 검증):
- 'onprem_llm'   : 온프레미스 LLM 을 사용하는 전략(의료 정책 허용 범위).
- 'deterministic': 동일 입력 → 동일 출력이 보장되는 전략(재현성 100% 대상).
- 'external_llm' : 외부 LLM API 를 호출하는 전략(medical_strict 에서 금지).

태그는 다중 부여 가능(예: {'onprem_llm', 'deterministic'}).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class _Entry:
    fn: Callable
    tags: frozenset[str]


class StrategyRegistry:
    """슬롯(classify/extract/embed/persist/…)별로 이름 붙은 전략을 보관한다.

    **불변식**: 같은 (slot, name) 쌍을 두 번 register 하면 나중 것이 덮어쓴다.
    테스트에서 mock 전략으로 교체할 때 이 성질을 활용할 수 있다.
    """

    def __init__(self) -> None:
        self._slots: dict[str, dict[str, _Entry]] = {}

    def register(self, slot: str, name: str, fn: Callable, *, tags: Iterable[str] = ()) -> None:
        """전략 등록. 동일 (slot, name) 이 있으면 덮어쓴다."""
        self._slots.setdefault(slot, {})[name] = _Entry(fn, frozenset(tags))

    def resolve(self, slot: str, name: str) -> Callable:
        """등록된 전략 Callable 반환. 미등록이면 KeyError."""
        try:
            return self._slots[slot][name].fn
        except KeyError as e:
            raise KeyError(f"미등록 전략: slot={slot!r} name={name!r}") from e

    def tags(self, slot: str, name: str) -> frozenset[str]:
        """전략의 capability 태그 집합 반환. 미등록이면 KeyError.

        policy.py 의 Constraint.check 가 이 메서드로 태그를 조회해 정책 위반 여부를 판단한다.
        """
        try:
            return self._slots[slot][name].tags
        except KeyError as e:
            raise KeyError(f"미등록 전략: slot={slot!r} name={name!r}") from e


# 프로세스 전역 기본 레지스트리.
# builtins.py 를 import 하면 register_defaults(DEFAULT_REGISTRY) 가 실행된다(부수효과 등록).
# 테스트에서 격리가 필요할 경우 별도 StrategyRegistry() 인스턴스를 생성해 사용한다.
DEFAULT_REGISTRY = StrategyRegistry()
