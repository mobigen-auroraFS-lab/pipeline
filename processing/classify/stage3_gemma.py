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
    """후보 라벨을 넣어 분류 프롬프트를 만든다.

    ⚠️ 끝을 "텍스트:" 로 열어 둔다 — 호출부가 본문을 **그대로 이어 붙이는** 구조라,
    끝 문구를 바꾸면 프롬프트가 어긋난다.

    Args:
        labels: 후보 라벨. 목록 밖 답이 와도 호출부가 버린다.

    Returns:
        본문을 이어 붙일 수 있는 프롬프트 앞부분.
    """
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
    """LLM 에게 후보 중 하나를 고르게 한다 — 못 고르면 사람 검토로 넘긴다.

    Args:
        text: 판정할 본문. 길면 앞부분만 쓴다(컨텍스트 여유분 확보).
        labels: 후보 라벨. **소문자여야 한다** — 응답을 소문자로 낮춘 뒤 비교하므로,
            대문자가 섞이면 맞는 답도 후보 밖으로 판정된다.
        complete: LLM 호출 함수. 주입하면 네트워크 없이 검증된다. 외부 LLM 금지 정책은
            **이 자리에 무엇을 꽂느냐로** 강제된다.

    Returns:
        ``(라벨, 근거 dict)``. 후보 밖 답이거나 호출이 실패하면 **검토 라벨**을 돌려준다 —
        틀린 도메인으로 확정하는 것보다 사람이 보게 두는 편이 낫다.
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
