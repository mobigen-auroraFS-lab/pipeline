"""F-5.1 도메인 분류 결과 데이터클래스."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# asset.domain_label / asset_classification.final_label 과 동일 값.
# "review" 는 엔진이 확정하지 못해 사람 검토(HITL)가 필요한 상태.
DOMAIN_MEDICAL = "medical"
DOMAIN_GENERAL = "general"
DOMAIN_REVIEW = "review"

# 분류 정책 버전. asset_classification 행에 기록해 규칙 변경 이력 추적에 쓴다.
POLICY_VERSION = "classify.v1"


@dataclass
class ClassificationResult:
    """3-stage cascade 분류 결과. asset_classification 한 행에 대응.

    불변식:
    - final_label 은 등록 도메인 값 또는 DOMAIN_GENERAL/DOMAIN_REVIEW 중 하나.
    - decided_stage 가 1 이면 confidence=1.0(시그니처 확정), 2 이면 0.7–0.9,
      3 이면 0.6(LLM 판정) 또는 0.0(review — LLM 실패·라벨 불명).
    - stage{1,2,3}_scores 는 디버깅·감사용 부가 정보이며 판정 로직에 영향 없음.
    """

    final_label: str  # medical | general | review
    confidence: float
    decided_stage: int  # 1 | 2 | 3
    # stage1_scores: {domain: {"signature": ..., **detail}} — run_ingest 가 deferred 판별에 참조
    stage1_scores: dict[str, Any] = field(default_factory=dict)
    stage2_scores: dict[str, Any] = field(default_factory=dict)
    stage3_scores: dict[str, Any] = field(default_factory=dict)
    policy_version: str = POLICY_VERSION
