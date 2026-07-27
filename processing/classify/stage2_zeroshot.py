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
    """어휘 사전을 매칭용 형태로 컴파일한다(사전별로 캐시).

    ⚠️ **긴 것부터 정렬해야 한다.** 정규식 교대는 먼저 맞는 것에서 멈추므로, 짧은 패턴이
    앞에 있으면 긴 용어의 앞부분만 먹고 끝난다.

    Args:
        lexicon: 어휘 집합. **변경 불가 집합이어야** 캐시 키로 쓸 수 있다 — 그래서 사전마다
            딱 한 번만 컴파일된다.

    Returns:
        ``(라틴 문자용 정규식 또는 None, 한글 용어 집합)``. 라틴 용어가 없으면 정규식은 ``None``.
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
    """텍스트에 사전 용어가 몇 개나 나오는지 센다.

    **같은 용어가 여러 번 나와도 1로 센다** — 재료는 "몇 종류가 나오는가"이지 빈도가 아니다.
    라틴 문자는 단어 경계로, 한글은 부분 문자열로 찾는다(한글은 조사가 붙어 경계가 흐리다).

    Args:
        text: 스캔할 텍스트. ``None`` 이어도 안전하다.
        lexicon: 어휘 집합.

    Returns:
        ``(맞은 용어 종류 수, 맞은 용어 목록)``. **목록만 20개로 자른다** — 로그가 비대해지는
        것을 막기 위해서이고, 앞의 개수는 자르지 않은 정확한 값이다.
    """
    text = text or ""
    latin_re, korean = _compiled(lexicon)
    hits: set[str] = set()
    if latin_re is not None:
        # .lower() 로 정규화해 대소문자 달리 쓰인 동일 어휘를 하나로 취급.
        hits |= {m.group(0).lower() for m in latin_re.finditer(text)}
    hits |= {t for t in korean if t in text}
    return len(hits), sorted(hits)[:20]
