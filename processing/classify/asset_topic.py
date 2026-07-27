"""자산의 주제를 **자기 내용만 보고** 한 번 확정한다 — 이웃이 아니라 자기 자신에서.

**흐름에서의 위치**: 적재 막바지에 자산당 한 번 돈다. 확정된 주제는 전용 테이블에 남고,
같은-주제 묶기·주제 패싯은 전부 그 테이블을 조인해 파생한다.

**왜 이웃이 아니라 자기 내용인가**
    예전에는 관계 엣지에 붙은 라벨을 자산 주제로 끌어다 썼다. 그 라벨은 두 자산을 견주며
    붙은 것이라 **상대 쪽으로 기울고**, 어떤 이웃이 활성 임계를 넘느냐는 운에 좌우된다 —
    농구 영상이 축구·배드민턴으로 노출되는 식이다. "이것은 무엇인가"와 "무엇과 이어지나"는
    다른 질문이므로 분리했다.

판정 흐름: 자기 텍스트 구성 → **닫힌 대분류 전체**를 후보로 조회 → LLM 이 그중 하나 확정 →
부모 아래 **닫힌 소분류 시드**에서 다시 하나 선택 → 영문 라벨은 레지스트리에서 조회 → 저장.

지켜야 할 것
    - **후보를 좁히지 않는다.** 대분류가 스물 몇 개뿐이라 프롬프트에 전부 담을 수 있는데,
      유사도로 추리면 정답이 후보에서 빠져 "해당 없음"이 나온다(미부여의 주된 원인이었다).
    - **후보 밖 답은 받지 않는다.** 한 번 다시 묻고, 그래도 밖이면 **미부여**로 둔다 —
      억지로 가장 가까운 것에 붙이면 틀린 주제가 조용히 정본이 된다.
    - **catch-all 라벨은 후보에서 뺀다.** 남겨 두면 애매한 자산이 전부 그리로 몰린다.
    - **예외를 삼키지 않는다.** 호출부가 자산은 등록된 채로 두고 주제만 미부여로 격리한다.
    - 같은 입력이면 같은 결과다 — 온도 0 + 닫힌 후보 + 멱등 저장(헌법 3조).
"""
from __future__ import annotations

import logging
from typing import Any

from psycopg.rows import dict_row

# 영문 라벨 조회는 코어 정본을 그대로 쓴다(같은 조회를 두 벌 두면 한쪽이 낡는다).
# 모듈 상단에서 import 하는 것은 테스트가 여기를 바꿔 끼우는 지점이기도 하다.
from src.relations.topic_canonicalize import _lookup_topic_en

logger = logging.getLogger(__name__)

# 분류 정책 버전 — 프롬프트나 후보 구성이 바뀌면 올린다. 저장된 결과가 어느 규칙으로 나온
# 것인지 되짚을 유일한 단서다.
POLICY_VERSION = "asset_topic.v1"

# ⚠️ 후보를 좁히지 않으므로 이 값은 **쓰이지 않는다**. 호출부 시그니처를 깨지 않으려고
# 기본값만 남겨 둔 것이다.
_DEFAULT_TOPIC_K = 5

# 어디에도 안 맞는 것을 담아 두는 라벨. **주제 후보에서는 뺀다** — 남겨 두면 무내용·애매한
# 자산이 전부 이리로 몰려 분류가 무의미해진다. 관계 쪽은 이 라벨을 계속 쓰므로 시드에서
# 지우지는 않는다. 여기 한 곳에만 문자열을 두어 조회와 판정이 어긋나지 않게 한다.
_UNCLASSIFIED_LABEL = "미분류"


def build_self_text(
    summary: str | None, keywords: list | None, labels: list | None = None
) -> str:
    """LLM 에 보여 줄 "이 자산의 내용" 한 덩어리를 만든다(순수 함수).

    **붙이는 순서를 고정한다** — 순서가 바뀌면 같은 자산에 다른 프롬프트가 만들어져
    결과가 흔들린다(헌법 3조).

    Args:
        summary: 요약. 공백뿐이면 뺀다.
        keywords: 키워드 목록. 빈 원소는 걸러 낸다.
        labels: 이미지·영상 라벨. ``{label, score}`` 형태를 가정하되 **문자열 원소도 받는다**
            (형태가 섞여 들어와도 죽지 않게). 점수는 쓰지 않고 **들어온 순서를 그대로** 쓴다.

    Returns:
        이어 붙인 문자열. 쓸 내용이 하나도 없으면 빈 문자열 — 호출부가 이것으로
        "분류할 것이 없다"를 판단해 LLM 을 아예 부르지 않는다.
    """
    parts: list[str] = []

    if summary and str(summary).strip():
        parts.append(str(summary).strip())

    if keywords:
        kw = [str(k).strip() for k in keywords if k and str(k).strip()]
        if kw:
            parts.append(" ".join(kw))

    if labels:
        labs: list[str] = []
        for item in labels:
            # dict 형([{label, score}])이면 label 만, 문자열이면 그대로(방어적).
            label = item.get("label") if isinstance(item, dict) else item
            if label and str(label).strip():
                labs.append(str(label).strip())
        if labs:
            parts.append(" ".join(labs))

    return " ".join(parts).strip()


def topic_candidates_for_self_text(
    conn, self_text: str | None, *, k: int = _DEFAULT_TOPIC_K
) -> list[str]:
    """자기 텍스트가 있으면 **닫힌 대분류 전체**를 후보로 돌려준다.

    **후보를 유사도로 추리지 않는다.** 대분류가 스물 몇 개뿐이라 프롬프트에 전부 담을 수
    있는데, 유사도 상위만 남기면 정답이 후보에서 빠져 LLM 이 정확히 "해당 없음"을 낸다 —
    미부여의 주된 원인이었다.

    Args:
        conn: DB 연결.
        self_text: 자산 자기 텍스트. **비어 있으면 빈 목록**을 돌려줘 호출부가 LLM 을
            아예 부르지 않게 한다(무내용 자산에 비용을 쓰지 않는다).
        k: ⚠️ **쓰이지 않는다.** 후보를 좁히지 않으므로 상한 개념 자체가 없다 —
            호출부 시그니처를 깨지 않으려고 남겨 둔 인자다.

    Returns:
        닫힌 대분류 라벨 전체(이름순 고정). catch-all 라벨은 빠진다. 레지스트리가 비어
        있으면 빈 목록 — 그때는 분류 자체를 건너뛴다.
    """
    if not self_text or not str(self_text).strip():
        return []
    # 닫힌 대분류(parent NULL·taxonomy) 전체를 topic_ko 오름차순으로. '미분류'(catch-all)는 조회에서
    # 배제한다(문자열은 한 곳에서만 관리). 시드 자체는 그대로 두고 **분류 후보만** 좁힌다.
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT topic_ko
            FROM topic_registry
            WHERE parent_topic IS NULL AND source = 'taxonomy' AND topic_ko <> %s
            ORDER BY topic_ko ASC
            """,
            (_UNCLASSIFIED_LABEL,),
        )
        rows = cur.fetchall()
    return [str(r["topic_ko"]) for r in rows]


def fetch_closed_subtopics(conn, topic_ko: str) -> list[str]:
    """부모 대분류 아래의 **닫힌 소분류 시드 목록**을 돌려준다. 시드가 없으면 빈 목록.

    **소분류도 만들어 내지 않고 고르게 한다.** 자유롭게 생성시켰더니 뜻이 겹치는 라벨이
    난립하고 한 라벨이 절반 이상을 삼켰다 — 변별력이 사라진다. 대분류와 같은 방식으로
    닫힌 목록에서 고르게 해 결정성과 변별력을 함께 얻는다.

    Args:
        conn: DB 연결.
        topic_ko: 이미 확정된 부모 대분류.

    Returns:
        그 부모 아래 소분류 시드 전체(이름순 고정). **시드가 없으면 빈 목록**이고,
        그때는 소분류를 붙이지 않는다(없는 라벨을 지어내지 않는다).
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT topic_ko
            FROM topic_registry
            WHERE parent_topic = %s AND source = 'taxonomy'
            ORDER BY topic_ko ASC
            """,
            (topic_ko,),
        )
        rows = cur.fetchall()
    return [str(r["topic_ko"]) for r in rows]


# 대분류 판정 프롬프트(온도 0) — 자기 텍스트와 후보 목록을 주고 그중 하나를 고르게 한다.
# ⚠️ "목록에 없는 것을 지어내지 마라"를 규칙으로 못박는다 — 안 박으면 그럴듯한 라벨을 만들어 낸다.
_CLASSIFY_PROMPT = """너는 자산의 자기 내용(요약·키워드·라벨)을 읽고 그 자산의 **대표 주제(topic)**를
아래 "후보 주제 목록" 중 **정확히 하나**로 고르고, 자산의 구체적인 **하위주제(subtopic)**를 만드는
분류기다.

규칙:
- topic 은 반드시 "후보 주제 목록"에 있는 라벨 하나만 고른다. 목록에 없는 라벨을 지어내지 않는다.
- 어느 후보 주제도 이 자산에 맞지 않으면(억지로 고르지 말고) topic_ko 를 정확히 "none" 으로 출력한다.
- subtopic_ko 는 **특정 개체·고유명사(예: 파리, 남산타워, 양자역학) 금지**. 여러 자산이 공유할 수 있는
  재사용 가능한 **일반 카테고리(예: 도시여행, 전통건축, 물리학)** 수준으로 만든다. 자신이 없으면 null.
- topic_en/subtopic_en 은 대응 영문(없으면 null).
- confidence 는 판정 확신도 0~1 실수.
- JSON 객체 하나만 출력한다. 코드블록·설명 문장 금지.
- 형식: {{"topic_ko":"<후보 중 하나 또는 none>","topic_en":"...","subtopic_ko":"...","subtopic_en":"...","confidence":0.0}}

자산 자기 내용:
{self_text}

후보 주제 목록:
{candidates}

출력: {{"topic_ko":"...","topic_en":"...","subtopic_ko":"...","subtopic_en":"...","confidence":0.0}}"""

# 후보 밖 답이 왔을 때 덧붙이는 경고. ⚠️ 재질의에도 **"해당 없음" 도피처를 남긴다** —
# 없애면 LLM 이 억지로 가장 가까운 것을 골라 틀린 주제가 정본이 된다.
_RETRY_SUFFIX = """

경고: 직전 응답의 topic_ko 가 후보 목록에 없었다. 아래 후보 중 하나의 정확한 라벨만 고르되, 정말 어느 후보도 맞지 않으면 "none" 으로 답하라(억지로 고르지 마라): {candidates}"""


def _pick_topic_via_llm(self_text: str, candidates: list[str], *, client) -> dict | None:
    """닫힌 후보 중 하나로 대분류를 확정한다(LLM 호출·온도 0).

    후보 밖 답이 오면 경고 문구를 붙여 **한 번만** 다시 묻는다. 재질의에도 "해당 없음"
    도피처를 남긴다 — 없애면 LLM 이 억지로 가장 가까운 것을 골라 틀린 주제가 정본이 된다.

    Args:
        self_text: 자산 자기 텍스트.
        candidates: 닫힌 후보 목록.
        client: LLM 클라이언트. **주입할 수 있게 열어 둔 자리**라 네트워크 없이 검증된다.

    Returns:
        LLM 응답 dict(대분류가 후보 안임이 확인된 상태), 또는 두 번 다 실패하면 ``None``
        (미부여). 소분류·영문·신뢰도는 호출부가 이 dict 에서 읽는다.
    """
    from src.llm.client import complete_json

    prompt = _CLASSIFY_PROMPT.format(
        self_text=self_text, candidates="\n".join(f"- {c}" for c in candidates)
    )
    out = complete_json(prompt, client=client)
    if _topic_in_candidates(out, candidates):
        return out

    # 후보 밖 → 경고 문구를 덧붙여 1회 재질의(강제 매핑 금지·환각 차단).
    retry_prompt = prompt + _RETRY_SUFFIX.format(candidates=", ".join(candidates))
    out2 = complete_json(retry_prompt, client=client)
    if _topic_in_candidates(out2, candidates):
        return out2
    return None


def _topic_in_candidates(out: dict, candidates: list[str]) -> bool:
    """LLM 응답의 대분류가 닫힌 후보 안에 있는지 확인한다.

    Args:
        out: LLM 응답. **dict 가 아닐 수도 있다**(형식 위반)— 그 경우도 거짓이다.
        candidates: 닫힌 후보 목록.

    Returns:
        후보 안이면 참.
    """
    topic_ko = out.get("topic_ko") if isinstance(out, dict) else None
    return isinstance(topic_ko, str) and topic_ko in candidates


# 소분류 선택 프롬프트(온도 0) — 이미 확정된 부모 대분류를 명시하고 그 아래
# 의 닫힌 소분류(subtopic) 시드 목록만 후보로 제시해 그중 정확히 하나를 고르게 한다. topic 선택
# (_CLASSIFY_PROMPT)과 대칭이되 후보가 부모 스코프 시드 목록이라는 점만 다르다. 후보 밖 라벨을 지어내지
# 못하도록 규칙을 못박고, 어느 후보도 안 맞으면 "해당 없음"으로 미부여시킨다(강제 매핑
# 금지). 여기서는 개체·고유명사 유도 문구가 불필요하다 — 후보 자체가 큐레이션된 재사용 카테고리다.
_SUBTOPIC_PROMPT = """너는 자산의 자기 내용(요약·키워드·라벨)을 읽고, 이미 정해진 대표 주제(topic)
"{topic_ko}" 아래의 **하위주제(subtopic)**를 아래 "후보 하위주제 목록" 중 **정확히 하나**로 고르는
분류기다.

규칙:
- subtopic 은 반드시 "후보 하위주제 목록"에 있는 라벨 하나만 고른다. 목록에 없는 라벨을 지어내지 않는다.
- 어느 후보 하위주제도 이 자산에 맞지 않으면(억지로 고르지 말고) subtopic_ko 를 정확히 "none" 으로 출력한다.
- JSON 객체 하나만 출력한다. 코드블록·설명 문장 금지.
- 형식: {{"subtopic_ko":"<후보 중 하나 또는 none>"}}

대표 주제(topic): {topic_ko}

자산 자기 내용:
{self_text}

후보 하위주제 목록:
{candidates}

출력: {{"subtopic_ko":"..."}}"""

# 소분류 재질의 경고 — 원 프롬프트 뒤에 덧붙인다. "해당 없음" 도피처를
# 재질의에도 유지해 "억지 최근접 배정"(강제 매핑)을 유발하지 않는다(정말 안 맞으면 none 으로).
_SUBTOPIC_RETRY_SUFFIX = """

경고: 직전 응답의 subtopic_ko 가 후보 목록에 없었다. 아래 후보 중 하나의 정확한 라벨만 고르되, 정말 어느 후보도 맞지 않으면 "none" 으로 답하라(억지로 고르지 마라): {candidates}"""


def _subtopic_in_candidates(out: dict, candidates: list[str]) -> bool:
    """LLM 응답의 소분류가 닫힌 후보 안에 있는지 확인한다(대분류 쪽과 대칭·응답 키만 다르다).

    Args:
        out: LLM 응답.
        candidates: 닫힌 후보 목록.

    Returns:
        후보 안이면 참. "해당 없음"·후보 밖·키 누락은 **모두 거짓**이라 재질의나 미부여
        경로로 흘러, 억지 배정이 생기지 않는다.
    """
    sub = out.get("subtopic_ko") if isinstance(out, dict) else None
    return isinstance(sub, str) and sub in candidates


def _pick_subtopic_via_llm(
    self_text: str, topic_ko: str, candidates: list[str], *, client
) -> str | None:
    """닫힌 시드 후보 중 하나로 소분류를 고른다(대분류 쪽과 같은 구조).

    Args:
        self_text: 자산 자기 텍스트.
        topic_ko: 이미 확정된 부모 대분류. 프롬프트에 명시해 범위를 좁힌다.
        candidates: 그 부모 아래 닫힌 시드. **비어 있으면 호출부가 애초에 부르지 않는다**.
        client: LLM 클라이언트(주입 가능).

    Returns:
        고른 소분류 라벨, 또는 두 번 다 실패하거나 "해당 없음"이면 ``None``(미부여).
    """
    from src.llm.client import complete_json

    prompt = _SUBTOPIC_PROMPT.format(
        self_text=self_text,
        topic_ko=topic_ko,
        candidates="\n".join(f"- {c}" for c in candidates),
    )
    out = complete_json(prompt, client=client)
    if _subtopic_in_candidates(out, candidates):
        return out["subtopic_ko"]

    # 후보 밖 → 경고 문구를 덧붙여 1회 재질의(강제 매핑 금지·환각 차단).
    retry_prompt = prompt + _SUBTOPIC_RETRY_SUFFIX.format(candidates=", ".join(candidates))
    out2 = complete_json(retry_prompt, client=client)
    if _subtopic_in_candidates(out2, candidates):
        return out2["subtopic_ko"]
    return None


def _lookup_subtopic_en(conn, topic_ko: str, subtopic_ko: str) -> str | None:
    """부모 스코프 subtopic 의 정본 영문(``topic_registry.topic_en``·닫힌 시드 en). 없으면 None.

    ⚠️ **부모까지 조건에 넣어야 한다** — 같은 이름의 소분류가 다른 부모 아래에도 있을 수
    있어, 이름만으로 찾으면 엉뚱한 영문 라벨이 붙는다.

    Args:
        conn: DB 연결.
        topic_ko: 부모 대분류.
        subtopic_ko: 소분류 라벨.

    Returns:
        영문 라벨, 또는 없으면 ``None``.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT topic_en FROM topic_registry WHERE parent_topic = %s AND topic_ko = %s",
            (topic_ko, subtopic_ko),
        )
        row = cur.fetchone()
    if row is None:
        return None
    en = row["topic_en"]
    return str(en) if en is not None else None


def _load_self_meta(conn, asset_id) -> tuple[str | None, list | None, list | None]:
    """``asset_metadata.ext_meta`` 에서 summary/keywords/labels 로드(자기 텍스트 소스).

    Args:
        conn: DB 연결.
        asset_id: 대상 자산.

    Returns:
        ``(요약, 키워드, 라벨)``. **메타 행이 없으면 셋 다 ``None``** — 예외가 아니다
        (아직 적재 중이거나 메타가 없는 자산은 정상 상태다).
    """
    # ⚠️ ``->>`` 와 ``->`` 를 구분해 쓴다: 요약은 문자열로 꺼내고(``->>``), 키워드·라벨은
    # JSON 배열째 꺼내야(``->``) 파이썬 list 로 디코드된다.
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT m.ext_meta->>'summary' AS summary,
                   m.ext_meta->'keywords' AS keywords,
                   m.ext_meta->'labels'   AS labels
            FROM asset_metadata m
            WHERE m.asset_id = %s
            """,
            (asset_id,),
        )
        row = cur.fetchone()
    if row is None:
        return None, None, None
    return row.get("summary"), row.get("keywords"), row.get("labels")


def _coerce_confidence(value: Any) -> float | None:
    """confidence 를 float 로 강제(파싱 실패·부재 → None)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _upsert_asset_topic(
    conn,
    asset_id,
    topic_ko: str,
    topic_en: str | None,
    subtopic_ko: str | None,
    subtopic_en: str | None,
    confidence: float | None,
) -> None:
    """자산 주제를 저장한다 — 자산당 한 행(다시 분류하면 덮어쓴다).

    **DB에 쓴다.** 같은 자산을 다시 분류해도 행이 늘지 않고 덮어써진다.

    분류 정책 버전을 함께 남긴다 — 프롬프트나 후보 구성이 바뀌었을 때 "이 결과가 어느
    규칙으로 나온 것인지"를 되짚을 유일한 단서다.

    Args:
        conn: DB 연결.
        asset_id: 대상 자산.
        topic_ko: 확정된 대분류(필수).
        topic_en: 대분류 영문. 레지스트리에 없으면 ``None``.
        subtopic_ko: 소분류. 미부여면 ``None``.
        subtopic_en: 소분류 영문. 미부여·미등록이면 ``None``.
        confidence: LLM 이 준 신뢰도. 파싱 실패면 ``None``.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO asset_topic
                (asset_id, topic_ko, topic_en, subtopic_ko, subtopic_en,
                 confidence, decided_by, policy_version)
            VALUES (%s, %s, %s, %s, %s, %s, 'hybrid', %s)
            ON CONFLICT (asset_id) DO UPDATE SET
                topic_ko = EXCLUDED.topic_ko,
                topic_en = EXCLUDED.topic_en,
                subtopic_ko = EXCLUDED.subtopic_ko,
                subtopic_en = EXCLUDED.subtopic_en,
                confidence = EXCLUDED.confidence,
                decided_by = EXCLUDED.decided_by,
                policy_version = EXCLUDED.policy_version,
                updated_at = now()
            """,
            (
                asset_id,
                topic_ko,
                topic_en,
                subtopic_ko,
                subtopic_en,
                confidence,
                POLICY_VERSION,
            ),
        )


def classify_asset_topic(
    conn, asset_id, *, self_text: str | None = None, settings=None, client=None
) -> dict | None:
    """자산 주제를 판정해 저장한다 — 대분류·소분류 모두 닫힌 후보에서 고른다. 미부여면 None.

    반환 ``{topic_ko, topic_en, subtopic_ko, subtopic_en, confidence, decided_by:'hybrid'}`` 또는
    None(미부여·topic 미확정). 미부여 경로: 자기 텍스트 없음(LLM 미호출) · 전체-27 후보 없음(레지스트리
    미시드) · 후보 밖 답이 두 번 연속 온 경우. 소분류만 미부여면 대분류는 그대로 저장한다.
    **예외는 삼키지 않고 올린다** — 호출부가 자산을 등록된 채 두고 주제만 미부여로 격리한다.

    Args:
        self_text: (선택) 이미 구성된 자기 텍스트. None 이면 ``asset_metadata`` 에서 로드해 구성한다.
        settings: (예약) 후보수 등 정책 파라미터 주입용(현재는 기본값 사용).
        client: LLM 클라이언트 주입 seam(미주입=운영 클라이언트·temp=0).
    """
    # ① 자기 텍스트 확보(미주입 시 메타 로드 후 구성). 비면 미부여(LLM 미호출).
    if self_text is None:
        summary, keywords, labels = _load_self_meta(conn, asset_id)
        self_text = build_self_text(summary, keywords, labels)
    if not self_text or not str(self_text).strip():
        return None

    # ② 닫힌 대분류 전체를 후보로. 비어 있으면 미부여(레지스트리가 시드되지 않은 상태).
    candidates = topic_candidates_for_self_text(conn, self_text, k=_DEFAULT_TOPIC_K)
    if not candidates:
        logger.info("자기주제 분류 스킵 — 전체-27 후보 없음(레지스트리 미시드): asset_id=%s", asset_id)
        return None

    # ③ LLM 닫힌 확정(후보 밖이면 1회 재질의). 재실패 → 미부여(강제 매핑 금지).
    picked = _pick_topic_via_llm(self_text, candidates, client=client)
    if picked is None:
        logger.info("자기주제 분류 미부여 — 닫힌집합 검증 2회 실패: asset_id=%s", asset_id)
        return None

    topic_ko = picked["topic_ko"]
    # ⚠️ 대분류 판정이 곁들여 만들어 준 소분류는 **쓰지 않는다** —
    # 소분류도 대분류처럼 부모의 닫힌 시드에서 고르게 한다(자유 생성은 라벨이 난립한다). 대분류
    # 콜의 subtopic 생성 지시(_CLASSIFY_PROMPT)는 무해하게 잔존하며 여기서 무시한다(제거는 후속 이연).

    # ④ 소분류를 부모 대분류의 **닫힌 시드 목록**에서 고른다.
    #    왜 코어의 정규화 함수를 쓰지 않는가: 그쪽은 '어휘를 자유롭게 만들어 내고 부모 범위에서
    #    재사용'하는 방식이라 뜻이 겹치는 라벨이 난립하고 한 라벨이 절반 이상을 삼켰다. 여기서는
    #    소분류 시드에서 고르게 해(fetch_closed_subtopics → _pick_subtopic_via_llm) 결정성·변별력을 얻는다.
    #    코어 함수 자체는 그대로 둔다 — 관계 경로가 계속 쓰고 있다.
    #    시드 미존재(subcands 빈)면 subtopic 미부여(None)·강제 생성 금지.
    subcands = fetch_closed_subtopics(conn, topic_ko)
    subtopic_ko = (
        _pick_subtopic_via_llm(self_text, topic_ko, subcands, client=client)
        if subcands
        else None
    )
    # subtopic_en 은 registry 정본(부모 스코프) 조회 — 닫힌 시드라 정본 en 이 존재. 미부여면 None.
    subtopic_en = _lookup_subtopic_en(conn, topic_ko, subtopic_ko) if subtopic_ko else None

    # ⑤ 영문 라벨은 레지스트리 정본을 먼저 쓰고, 없으면 LLM 응답 값으로 채운다.
    topic_en = _lookup_topic_en(conn, topic_ko) or picked.get("topic_en")

    confidence = _coerce_confidence(picked.get("confidence"))

    # ⑥ 멱등 upsert(policy_version 기록).
    _upsert_asset_topic(
        conn, asset_id, topic_ko, topic_en, subtopic_ko, subtopic_en, confidence
    )

    return {
        "topic_ko": topic_ko,
        "topic_en": topic_en,
        "subtopic_ko": subtopic_ko,
        "subtopic_en": subtopic_en,
        "confidence": confidence,
        "decided_by": "hybrid",
    }


# 자기주제 정본 **조회(read)**·주제 패싯·같은주제·미분류 함수(fetch_asset_topic·find_same_topic_groups·
# list_topics·assets_in_topic·assets_unclassified)는 코어 ``src/topic/asset_topic_query.py``
# 로 이관됐다(파이프라인=분류·write / 백엔드=read 런타임 소유 분리). 이 모듈은 classify·write 만 담는다.
