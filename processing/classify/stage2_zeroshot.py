"""Stage 2 — 어휘 사전 기반 zero-shot 점수(도메인-불가지).

주어진 어휘 사전(lexicon)에 대해 텍스트의 hit 수를 센다(학습 없음). 도메인 판정
(임계·margin)은 cascade 엔진이 담당하고, 본 모듈은 순수 카운팅만 한다.

매칭 규칙
    - 영문/ASCII 어휘는 **단어 경계 매칭**(예: 'contact' 의 'ct' 오탐 방지).
    - 한글 어휘는 부분 문자열 매칭(어절 경계 모호).
"""
from __future__ import annotations

import re
from functools import lru_cache


@lru_cache(maxsize=64)
def _compiled(lexicon: frozenset[str]):
    """어휘 사전 → (라틴 단어경계 정규식 | None, 한글 용어 집합). 사전별 캐시.

    latin 을 길이 내림차순으로 정렬하는 이유: 정규식 교대(|) 는 첫 매칭에서 멈추므로
    긴 패턴을 앞에 두어야 짧은 부분 패턴이 먼저 소비되는 문제를 막는다.
    frozenset 인수는 해시 가능 → lru_cache 가 프로파일당 한 번만 컴파일.
    """
    latin = sorted({t for t in lexicon if t.isascii()}, key=len, reverse=True)
    korean = frozenset(t for t in lexicon if not t.isascii())
    latin_re = (
        re.compile(r"\b(?:" + "|".join(re.escape(t) for t in latin) + r")\b", re.IGNORECASE)
        if latin
        else None
    )
    return latin_re, korean


def count_hits(text: str, lexicon: frozenset[str]) -> tuple[int, list[str]]:
    """(hit 수, 매칭 용어 ≤20). 어휘 사전당 단어경계(ASCII)/부분문자열(한글) 매칭.

    반환 용어 수를 20개로 제한하는 이유: 디버그·로그 크기 제어용이며 hit 수 자체는
    집합 크기(len(hits))로 정확히 계산된다.
    hit 는 중복 없는 set — 같은 어휘가 여러 번 등장해도 1로 센다(존재 여부).
    """
    text = text or ""
    latin_re, korean = _compiled(lexicon)
    hits: set[str] = set()
    if latin_re is not None:
        # .lower() 로 정규화해 대소문자 달리 쓰인 동일 어휘를 하나로 취급.
        hits |= {m.group(0).lower() for m in latin_re.finditer(text)}
    hits |= {t for t in korean if t in text}
    return len(hits), sorted(hits)[:20]
