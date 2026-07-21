"""도메인-불가지 분류 cascade — Stage 1(시그니처) → 2(어휘) → 3(온프레미스 LLM).

도메인 지식은 ``DomainProfileProvider`` 가 주는 ``DomainProfile`` 목록에 산다(코드 분기 없음).
앞 단계가 확정하면 즉시 반환. general 은 '어느 도메인도 충분치 않음'의 엔진 내장 폴백.
라우팅 직후·추출 전에 호출(per-asset 두 번째 스테이지, 도메인 팩 선택 분기점).
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from processing.classify import domains as _domains  # noqa: F401 — 기본 도메인 프로파일 등록
from processing.classify import stage2_zeroshot, stage3_gemma
from processing.classify.profiles import DomainProfileProvider, RegistryProvider
from processing.classify.text_probe import extract_text_for_classification
from processing.classify.types import (
    DOMAIN_GENERAL,
    POLICY_VERSION,
    ClassificationResult,
)

_HEAD_BYTES = 8192  # 시그니처 판별에 충분한 선두 바이트(DICOM 프리앰블 132B 포함)
_HIT_THRESHOLD = 2  # top 도메인 확정 최소 hit — 1개 어휘만으론 일반 문서 오탐 가능성↑
_MARGIN = 1         # top 과 2위의 최소 격차 — 어휘 동점 시 모호→stage3 으로 위임

# 신뢰도 계층: stage1 > stage2_확정 > stage2_general > stage3.
# run_ingest 의 deferred 판별은 신뢰도를 직접 비교하지 않고 decided_stage+final_label 사용.
_CONF_S1 = 1.0            # Stage 1 시그니처 확정 — 결정론적, 재현성 100%
_CONF_S2_CONFIRMED = 0.9  # Stage 2 어휘 단독 확정
_CONF_S2_GENERAL = 0.7    # Stage 2 hit 0 → general 폴백
_CONF_S3 = 0.6            # Stage 3 LLM 판정(온프레미스, temperature=0)

_log = logging.getLogger(__name__)


def _read_head(file_path: str, n: int = _HEAD_BYTES) -> bytes:
    # 실패(권한·존재 없음)는 빈 bytes — stage1 은 모두 miss → stage2 로 자연스럽게 이행.
    try:
        with open(file_path, "rb") as f:
            return f.read(n)
    except OSError:
        return b""


def _build_scan_text(file_path: str, modality: str) -> str:
    """Stage 2 어휘 스캔 입력: 파일명 + 결정적 추출 텍스트(LLM 없음).

    파일명을 포함하는 이유: 확장자·명명 규칙이 도메인 힌트를 담기도 함
    (예: 'report_dicom_20240101.json' 에 'dicom' 노출).
    """
    parts = [Path(file_path).name, extract_text_for_classification(file_path, modality)]
    return "\n".join(p for p in parts if p)


def classify(
    file_path: str,
    modality: str,
    *,
    provider: DomainProfileProvider | None = None,
    _llm_classify: Callable = stage3_gemma.classify,
) -> ClassificationResult:
    """프로파일 기반 도메인-불가지 3-stage cascade."""
    provider = provider or RegistryProvider()
    profiles = provider.all_profiles()

    # Stage 1 — 시그니처: 정확히 한 도메인만 매칭하면 확정, 0/충돌은 Stage 2 로.
    # s1_scores 구조: {domain: {"signature": <종류>, **detail}}
    # detail 에 "signature" 키가 없어야 충돌 없음 — SigHit.detail 은 관례상 그러하나 주의.
    # run_ingest 가 이 구조를 직접 참조해 deferred 자산을 판별한다.
    head = _read_head(file_path)
    s1_scores: dict[str, dict] = {}
    for p in profiles:
        for rule in p.signatures:
            hit = rule(head)
            if hit is not None:
                # 같은 도메인에 여러 규칙이 있어도 첫 번째 매칭만 기록(break).
                s1_scores[p.domain] = {"signature": hit.signature, **hit.detail}
                break
    if len(s1_scores) == 1:
        domain = next(iter(s1_scores))
        return ClassificationResult(
            final_label=domain, confidence=_CONF_S1, decided_stage=1, stage1_scores=s1_scores
        )
    if len(s1_scores) > 1:
        # 다도메인 시그니처 충돌 — 실 파일이 복합 포맷이거나 시그니처 규칙 설계 오류일 수 있음.
        _log.warning("Stage 1 시그니처 충돌 — 다수 도메인 매칭, Stage 2 로 위임: %s", sorted(s1_scores))

    # Stage 2 — 어휘: 도메인별 hit, top hit≥임계 & margin → 확정 / 0 → general / 그 외 → Stage 3.
    scan = _build_scan_text(file_path, modality)
    s2_scores: dict[str, dict] = {}
    for p in profiles:
        n, terms = stage2_zeroshot.count_hits(scan, p.lexicon)
        s2_scores[p.domain] = {"hits": n, "terms": terms}

    ranked = sorted(profiles, key=lambda p: s2_scores[p.domain]["hits"], reverse=True)
    top_hits = s2_scores[ranked[0].domain]["hits"] if ranked else 0
    second_hits = s2_scores[ranked[1].domain]["hits"] if len(ranked) > 1 else 0

    if top_hits == 0:
        # 어떤 도메인 어휘도 없음 → 특정 도메인 소속이 아님으로 판단해 general 확정.
        return ClassificationResult(
            final_label=DOMAIN_GENERAL, confidence=_CONF_S2_GENERAL, decided_stage=2,
            stage1_scores=s1_scores, stage2_scores=s2_scores,
        )
    if top_hits >= _HIT_THRESHOLD and (top_hits - second_hits) >= _MARGIN:
        # top 도메인이 충분한 hit + 2위와의 격차 확보 → 어휘만으로 안전하게 확정.
        return ClassificationResult(
            final_label=ranked[0].domain, confidence=_CONF_S2_CONFIRMED, decided_stage=2,
            stage1_scores=s1_scores, stage2_scores=s2_scores,
        )

    # Stage 3 — 온프레미스 LLM(모호 케이스만), 라벨 = 등록 도메인 + general.
    # labels 는 소문자 계약. stage3_gemma.classify 가 응답을 .lower() 후 이 집합과 비교한다.
    # LLM 실패·라벨 불명 → "review"(HITL) + confidence=0.0 으로 반환.
    labels = [p.domain for p in profiles] + [DOMAIN_GENERAL]
    s3_label, s3 = _llm_classify(scan, labels)
    conf = _CONF_S3 if s3_label != "review" else 0.0
    return ClassificationResult(
        final_label=s3_label, confidence=conf, decided_stage=3,
        stage1_scores=s1_scores, stage2_scores=s2_scores, stage3_scores=s3,
        policy_version=POLICY_VERSION,
    )
