"""의료 도메인 프로파일 — 시그니처 능력(DICOM/HL7/FHIR) + 어휘 + 등록.

(구 stage1_magic 의 시그니처 판별을 능력 모듈로 이관. 어휘는 medical_terms 에서 재사용.)
"""
from __future__ import annotations

import re

from processing.classify.medical_terms import MEDICAL_TERMS
from processing.classify.profiles import DomainProfile, SigHit, register_profile
from processing.classify.types import DOMAIN_MEDICAL

# FHIR R4 주요 리소스 타입(부분 집합).
# 너무 많은 리소스 타입을 추가하면 일반 JSON 문서 오탐 위험 — 의료 고유 리소스만 유지.
_FHIR_RESOURCES = frozenset(
    {
        "Patient", "Encounter", "Observation", "ImagingStudy", "DiagnosticReport",
        "Condition", "Procedure", "MedicationRequest", "Bundle", "DocumentReference",
        "ServiceRequest", "Specimen",
    }
)
# '"resourceType"' 빠른 존재 확인 후 정규식으로 값 추출 — 두 단계로 불필요한 regex 실행 방지.
_FHIR_RE = re.compile(r'"resourceType"\s*:\s*"([A-Za-z]+)"')


def _decode(head: bytes) -> str:
    """바이트 앞부분을 텍스트로 푼다(깨지는 글자는 대체 문자로 넘긴다).

    시그니처 검사용이라 완벽한 디코딩이 목적이 아니다 — 예외로 멈추지 않는 것이 중요하다.
    """
    # BOM(﻿) 제거 후 앞 공백 트리밍 — HL7 'MSH|' startswith 판별을 위해 필요.
    return head.decode("utf-8", errors="ignore").lstrip("﻿").lstrip()


def _dicom(head: bytes) -> SigHit | None:
    """DICOM: 128바이트 프리앰블 뒤 'DICM' 매직(DICOM PS3.10 §7.1).

    _HEAD_BYTES=8192 이므로 132바이트 판별에 충분하다.
    """
    if len(head) >= 132 and head[128:132] == b"DICM":
        return SigHit("dicom")
    return None


def _hl7v2(head: bytes) -> SigHit | None:
    """HL7 v2: 'MSH|' 로 시작 — 메시지 헤더 세그먼트(필드구분자 '|').

    stage2 어휘에 'hl7'도 있으나 \b 단어경계로 'HL7v2' 에서 'hl7' 매칭 방지됨.
    시그니처 매칭이 어휘 의존 없이 확정적으로 판별하는 근거.
    """
    if _decode(head).startswith("MSH|"):
        return SigHit("hl7v2")
    return None


def _fhir(head: bytes) -> SigHit | None:
    """FHIR: resourceType 이 알려진 리소스.

    SigHit.detail 에 resourceType 을 담아 run_ingest 의 deferred 판별 시
    어떤 FHIR 리소스인지 추적 가능.
    """
    text = _decode(head)
    if '"resourceType"' in text:
        m = _FHIR_RE.search(text)
        if m and m.group(1) in _FHIR_RESOURCES:
            return SigHit("fhir", {"resourceType": m.group(1)})
    return None


# 의료 프로파일 등록 — import 부수효과로 DOMAIN_PROFILES["medical"] 에 삽입.
# cascade 의 `from processing.classify import domains` 가 이 모듈을 간접 로드해 등록을 완료한다.
MEDICAL_PROFILE = DomainProfile(
    domain=DOMAIN_MEDICAL,
    lexicon=MEDICAL_TERMS,
    llm_label=DOMAIN_MEDICAL,
    signatures=(_dicom, _hl7v2, _fhir),
)
register_profile(MEDICAL_PROFILE)
