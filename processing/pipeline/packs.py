"""v2 도메인 팩 — 슬롯별 전략 선택 + 정책.

**설계 의도**: 도메인별 분기를 코드 if/else 로 두지 않고, 팩이 "각 슬롯에 어떤 전략 이름을 쓸지"만
선언한다. 실제 구현은 StrategyRegistry(registry.py)에 등록된 Callable 이 담당하므로,
신규 도메인 추가 = 팩 정의 + 레지스트리 등록만으로 완결된다(core 파이프라인 수정 불요).

**슬롯 구분**:
- per_asset  : 파일 1건 처리(classify/extract/embed/persist). run_ingest 가 순서대로 호출.
- cross_asset: 자산 간 관계 제안(candidates/score/decide/persist_edges). run_relations 가 위임.

현재 일반·의료는 **같은 전략을 쓰되 정책이 다르다** — 의료 전용 전략은 그 팩을 실제로
운용할 때 갈아 끼운다.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class DomainPack:
    """도메인 하나의 전략 배선표 + 정책 이름.

    **여기 적는 값은 이름뿐이다** — 등록되지 않은 이름을 적어도 이 시점에는 아무 일이
    없고, 실제로 그 슬롯을 쓰려는 순간 터진다.

    ⚠️ **매핑 내용까지 읽기 전용으로 만든다.** dataclass 동결은 필드를 다시 대입하는 것만
    막지, 안에 든 dict 를 고치는 것은 막지 못한다. 그대로 두면 어딘가에서 슬롯 하나를
    바꿔치기해 **라우팅이 말없이 달라진다** — 추적이 거의 불가능한 종류의 사고다.

    관계의 방향·대칭 같은 성질은 여기 있지 않다 — 그것은 관계 종류 카탈로그(DB) 소관이다.
    팩은 "어느 전략을 쓸지"만 고른다.
    """
    name: str
    per_asset: Mapping[str, str]    # 슬롯 → 전략 이름(생성 후 읽기전용 MappingProxyType)
    cross_asset: Mapping[str, str]  # 슬롯 → 전략 이름 (candidates/score/decide/persist_edges)
    policy: str                     # POLICIES 키(policy.py 참조)

    def __post_init__(self) -> None:
        """생성 직후 슬롯 구성이 온전한지 확인한다 — 빠진 슬롯은 실행 도중이 아니라 여기서 드러난다."""
        # 얕은 동결 해소: 입력 매핑을 복사 후 읽기전용 뷰로 재바인딩(frozen 이라 object.__setattr__).
        object.__setattr__(self, "per_asset", MappingProxyType(dict(self.per_asset)))
        object.__setattr__(self, "cross_asset", MappingProxyType(dict(self.cross_asset)))


# 일반 도메인이 자산 사이 단계에 쓰는 전략 묶음.
# ⚠️ 'decide' 는 등록된 함수가 없다 — 신뢰도 임계 판정이 점수 전략 안에서 함께 처리된다.
#    슬롯 이름만 선언해 두는 이유는, 나중에 판정을 떼어낼 때 배선표를 그대로 쓰기 위해서다.
_GENERAL_CROSS = {
    "candidates": "embedding_topk",
    "score": "llm_propose",
    "decide": "confidence",
    "persist_edges": "graph_upsert",
}

GENERAL_PACK = DomainPack(
    name="general",
    per_asset={"classify": "cascade_v1", "extract": "by_modality", "embed": "by_modality", "persist": "asset_upsert"},
    cross_asset=dict(_GENERAL_CROSS),
    policy="general_default",
)
MEDICAL_PACK = DomainPack(
    name="medical",
    # per_asset: 단계 D 이전에는 일반 전략(by_modality)으로 stopgap 처리.
    # DICOM 등 의료 포맷은 run_ingest 에서 status='deferred' 로 보류되므로
    # 실질적으로 일반 임베딩이 적재되는 자산은 비의료 파일에 한정된다.
    per_asset={"classify": "cascade_v1", "extract": "by_modality", "embed": "by_modality", "persist": "asset_upsert"},
    cross_asset=dict(_GENERAL_CROSS),   # 의료 전략은 단계 D 에서 교체
    policy="medical_strict",            # ForbidTag("external_llm") — 온프레미스 LLM 만 허용
)

# 샘플 도메인 — 조합형 구조가 실제로 도는지 보여 주는 데모 팩(결정적·LLM 미사용).
# ⚠️ **이 묶음이 일반 팩과 달라야** 관계 실행이 공용 러너 쪽으로 간다. 실행 경로를 도메인
#    이름이 아니라 **배선표 비교**로 고르기 때문이다(헌법 4조 — 이름 분기 금지).
# 저장 슬롯의 등록명과 함수명이 다른 점 주의 — 배선표에 적힌 이름 쪽이 기준이다.
_SAMPLE_CROSS = {
    "candidates": "sample_candidates",
    "score": "sample_score",
    "decide": "sample_decide",
    "persist_edges": "sample_graph_upsert",
}

SAMPLE_PACK = DomainPack(
    name="sample",
    per_asset=dict(GENERAL_PACK.per_asset),  # per_asset 은 일반과 동일
    cross_asset=dict(_SAMPLE_CROSS),
    policy="general_default",
)

_PACKS: dict[str, DomainPack] = {
    "general": GENERAL_PACK,
    "medical": MEDICAL_PACK,
    "sample": SAMPLE_PACK,
}


def for_domain(label: str) -> DomainPack:
    """분류 라벨에 맞는 도메인 팩을 고른다.

    Args:
        label: 분류 결과 라벨. 분류가 아직 안 끝났거나 모르는 값일 수 있다.

    Returns:
        해당 팩. **모르는 라벨은 일반 팩으로 접는다** — 라벨 하나 때문에 배치가 멈추는
        것보다 낫다. 특별 취급이 필요한 팩은 라벨이 정확히 와야 선택된다.
    """
    return _PACKS.get(label, GENERAL_PACK)
