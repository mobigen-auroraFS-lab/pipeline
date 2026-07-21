"""Stage 3 — 온프레미스 LLM zero-shot 도메인 판정(동적 라벨).

**모든 LLM 은 온프레미스 전용**(`cfg.openai_*`, OpenAI 호환 온프레미스 서버). 외부 LLM 미사용.
허용 라벨 집합은 등록 도메인 + general 로 호출부(cascade)가 동적으로 만들어 넘긴다.
``complete`` 주입으로 네트워크 없이 테스트 가능. 실패/미초기화 → 'review'(HITL).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from processing.classify.types import DOMAIN_REVIEW

_MAX_TEXT = 4000  # 프롬프트에 포함할 텍스트 최대 길이 — LLM 컨텍스트 창 여유분 확보


def _build_prompt(labels: list[str]) -> str:
    # 프롬프트 끝에 '\n\n텍스트:\n' 를 두어 classify() 가 text 를 직접 이어 붙인다.
    opts = " 또는 ".join(labels)
    return (
        f"다음 텍스트가 어느 도메인인지 판정해라. 후보: {opts}. "
        '반드시 JSON 만 출력: {"label": "<후보 중 하나>"}\n\n텍스트:\n'
    )


def _default_complete(prompt: str) -> str:
    """온프레미스 LLM 호출(공통 seam). 원문 문자열 반환.

    지연 임포트 이유: src.llm.client 는 LLM 설정 초기화가 필요하므로 모듈 로드 시가 아닌
    실제 호출 시점에 임포트해 미초기화 상태에서도 모듈 import 가 성공하게 한다.
    complete= 주입 시 이 함수가 호출되지 않아 네트워크 없이 테스트 가능.
    """
    from src.llm.client import complete_text

    return complete_text(prompt)


def classify(
    text: str,
    labels: list[str],
    *,
    complete: Callable[[str], str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """온프레미스 LLM 으로 labels 중 하나 판정. 허용 외/실패 → 'review'.

    계약:
    - labels 는 소문자 문자열 집합이어야 한다. LLM 응답을 .lower() 한 뒤 이 집합과 비교.
    - 의료 데이터의 경우 외부 LLM 호출 금지 정책이 complete= 주입으로 강제된다
      (NoExternalLLM 정책 — pipeline.policy.medical_strict).
    - temperature=0 으로 호출해 결정 재현성 100% 를 보장(src.llm.client 내부 설정).
    """
    complete = complete or _default_complete
    try:
        raw = complete(_build_prompt(labels) + text[:_MAX_TEXT])
        data = json.loads(raw) if raw else {}
        label = str(data.get("label", "")).strip().lower()
        if label in labels:
            return label, {"stage3": "llm", "label": label}
        # LLM 이 허용 라벨 외의 값을 반환한 경우 — HITL 검토 트리거.
        return DOMAIN_REVIEW, {"stage3": "llm_unclear", "raw": raw[:200]}
    except Exception as exc:  # noqa: BLE001 — 미초기화/호출 실패는 review 로 흡수
        return DOMAIN_REVIEW, {"stage3": "error", "error": str(exc)}
