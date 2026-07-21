"""자산 aboutness 개체 추출·저장 (spec 073 — 검색 무관 L1 적재시점 층).

"이 자산은 무엇**에 관한** 것인가"를 적재 시점에 LLM 1회로 확정해 ``asset_metadata.ext_meta['about']``
에 동결 저장한다(자산당 1회 — 검색 시점 LLM 0·QPS 무관). 검색은 저장값과의 글자 비교만 한다
(``src/search/about_filter.py``).

왜 keywords 로는 안 되고 aboutness 가 따로 필요한가(073 측정의 핵심 발견):
    우리 요약·keywords 는 백과사전식이라 **배경·유래로 스치듯 언급된 개체**가 섞인다 — 씨름 문서에
    "고구려 고분벽화", 불고기 문서에 "고구려 맥적(유래)". 그래서 어휘 존재 확인(P1)은 "언급≠주제"로
    실패했다. aboutness 프롬프트는 **배경 언급을 제외하라**고 명시해 LLM 의 독해로 주제 개체만 남긴다.

헌법 정합: ``complete_json`` 단일 seam(temp=0)·온프레미스 gemma. 출력은 적재 시 1회 DB 동결이라
검색 결정성에 영향이 없다(065 asset_topic 과 동일 패턴 — 쓰기 경로 LLM).
"""

from __future__ import annotations

import json
import logging
from typing import Any

_LOG = logging.getLogger(__name__)

# 개체 수 상한 — 073 측정에서 1~3개가 희소성/정밀 균형(더 많으면 언급 오염이 되돌아옴).
_ABOUT_MAX = 3

# 073 측정 프롬프트 그대로(측정-구현 일치). "배경·유래·비유 제외"가 언급≠주제를 차단하는 핵심 지시.
_ABOUT_PROMPT = """다음 자산 요약을 읽고, 이 자산이 "무엇에 관한(주제)" 것인지 핵심 개체·주제어를 1~3개 명사로 뽑아라.
- 배경·유래·비유·부수 설명으로 스치듯 언급된 개체는 제외(예: 씨름 자산에 "고구려 고분벽화" 언급 → 고구려 제외).
요약: {s}
JSON 하나만: {{"about": ["명사1", "명사2"]}}"""

# ext_meta 병합 upsert(멱등) — 기존 키 보존·'about' 만 덮어씀(jsonb || 연산). 마이그레이션 0.
_PERSIST_SQL = "UPDATE asset_metadata SET ext_meta = coalesce(ext_meta, '{}'::jsonb) || %s::jsonb WHERE asset_id = %s"


def extract_about(summary: str | None, *, client: Any | None = None) -> list[str]:
    """summary 를 읽고 aboutness 개체 리스트(최대 3)를 뽑는다(FR-001·fail-safe []).

    - 빈/공백 summary 는 LLM 미호출·``[]``(무내용 자산은 부여할 주제가 없음).
    - LLM 응답이 스키마 위반(비-dict·비-list·비-str 원소)이면 ``[]`` 폴백 — 추출 실패가
      적재·검색을 깨지 않는다(FR-204 동형). ``client`` 주입 시 네트워크 없이 단위 검증.
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
    """ext_meta 에 ``about`` 을 병합 저장한다(FR-002·멱등 — 재실행 시 같은 값 덮어씀)."""
    payload = json.dumps({"about": [str(x) for x in about]}, ensure_ascii=False)
    with conn.cursor() as cur:
        cur.execute(_PERSIST_SQL, (payload, asset_id))


def extract_and_persist_about(
    conn: Any, asset_id: Any, *, summary: str | None, client: Any | None = None
) -> list[str]:
    """추출→저장 편의 함수(적재 훅·백필 공용). 추출 결과(빈 리스트 포함)를 저장하고 반환한다.

    빈 ``[]`` 도 저장한다 — "추출을 시도했음"이 기록돼 백필 ``--only-missing``(about 키 부재)이
    같은 자산을 무한 재시도하지 않는다(멱등 스킵 근거).
    """
    about = extract_about(summary, client=client)
    persist_about(conn, asset_id, about)
    return about


__all__ = ["extract_about", "persist_about", "extract_and_persist_about"]
