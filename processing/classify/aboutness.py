"""자산이 "무엇에 관한" 것인지 개체를 뽑아 저장한다 — 적재 시점에 한 번.

"이 자산은 무엇**에 관한** 것인가"를 적재 시점에 LLM 1회로 확정해 ``asset_metadata.ext_meta['about']``
에 동결 저장한다(자산당 1회 — 검색 시점 LLM 0·QPS 무관). 검색은 저장값과의 글자 비교만 한다
(``src/search/about_filter.py``).

왜 키워드로는 안 되는가:
    우리 요약·keywords 는 백과사전식이라 **배경·유래로 스치듯 언급된 개체**가 섞인다 — 씨름 문서에
    "고구려 고분벽화", 불고기 문서에 "고구려 맥적(유래)". 그래서 어휘 존재 확인(P1)은 "언급≠주제"로
    실패했다. aboutness 프롬프트는 **배경 언급을 제외하라**고 명시해 LLM 의 독해로 주제 개체만 남긴다.

헌법 정합: ``complete_json`` 단일 seam(temp=0)·온프레미스 gemma. 출력은 적재 시 1회 DB 동결이라
검색 결정성에 영향이 없다(주제 분류와 같은 패턴 — LLM 은 쓰기 경로에만).
"""

from __future__ import annotations

import json
import logging
from typing import Any

_LOG = logging.getLogger(__name__)

# 개체 수 상한 — 늘리면 스치듯 언급된 개체가 다시 섞여 들어온다.
_ABOUT_MAX = 3

# ⚠️ "배경·유래·비유는 제외"가 이 프롬프트의 핵심이다 — 빼면 "언급된 것"이 "주제"로 섞인다.
_ABOUT_PROMPT = """다음 자산 요약을 읽고, 이 자산이 "무엇에 관한(주제)" 것인지 핵심 개체·주제어를 1~3개 명사로 뽑아라.
- 배경·유래·비유·부수 설명으로 스치듯 언급된 개체는 제외(예: 씨름 자산에 "고구려 고분벽화" 언급 → 고구려 제외).
요약: {s}
JSON 하나만: {{"about": ["명사1", "명사2"]}}"""

# ext_meta 병합 upsert(멱등) — 기존 키 보존·'about' 만 덮어씀(jsonb || 연산). 마이그레이션 0.
_PERSIST_SQL = "UPDATE asset_metadata SET ext_meta = coalesce(ext_meta, '{}'::jsonb) || %s::jsonb WHERE asset_id = %s"


def extract_about(summary: str | None, *, client: Any | None = None) -> list[str]:
    """요약을 읽고 주제 개체를 뽑는다(최대 3개 · 실패는 빈 목록으로 흡수).

    Args:
        summary: 자산 요약. **비어 있으면 LLM 을 부르지 않는다** — 무내용 자산에 비용을
            쓰지 않는다.
        client: LLM 클라이언트(주입 가능 — 네트워크 없이 검증된다).

    Returns:
        개체 목록(최대 3개). **응답 형식이 어긋나면 빈 목록**으로 접는다 — 이 추출이
        실패해도 적재와 검색은 계속돼야 한다.
    """
    if not summary or not summary.strip():
        return []
    from src.llm.client import complete_json

    out = complete_json(_ABOUT_PROMPT.format(s=summary.strip()[:200]), client=client)
    about = out.get("about") if isinstance(out, dict) else None
    if not isinstance(about, list):
        return []
    cleaned = [str(x).strip() for x in about if str(x).strip()]
    return cleaned[:_ABOUT_MAX]


def persist_about(conn: Any, asset_id: Any, about: list[str]) -> None:
    """확장 메타에 개체 목록을 **병합** 저장한다.

    ⚠️ 통째로 덮어쓰지 않고 이 키만 갈아 끼운다 — 다른 확장 메타를 날리지 않기 위해서다.

    Args:
        conn: DB 연결.
        asset_id: 대상 자산.
        about: 저장할 개체 목록. 빈 목록도 저장한다.
    """
    payload = json.dumps({"about": [str(x) for x in about]}, ensure_ascii=False)
    with conn.cursor() as cur:
        cur.execute(_PERSIST_SQL, (payload, asset_id))


def extract_and_persist_about(
    conn: Any, asset_id: Any, *, summary: str | None, client: Any | None = None
) -> list[str]:
    """추출→저장 편의 함수(적재 훅·백필 공용). 추출 결과(빈 리스트 포함)를 저장하고 반환한다.

    ⚠️ **결과가 비어도 저장한다.** 키가 없는 것과 빈 목록인 것은 다르다 — 저장하지 않으면
    "아직 안 해 본 자산"으로 보여 백필이 같은 자산을 영원히 다시 시도한다.

    Args:
        conn: DB 연결.
        asset_id: 대상 자산.
        summary: 자산 요약.
        client: LLM 클라이언트(주입 가능).

    Returns:
        추출한 개체 목록(빈 목록일 수 있다).
    """
    about = extract_about(summary, client=client)
    persist_about(conn, asset_id, about)
    return about


__all__ = ["extract_about", "persist_about", "extract_and_persist_about"]
