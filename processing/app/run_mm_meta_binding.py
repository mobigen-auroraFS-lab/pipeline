"""084 멀티모달 메타 소속 배치 CLI — 저장된 요약·키워드에서 개체를 판정해 ``mm_member`` 엣지를 만든다.

**흐름에서의 위치**: 기존 파이프라인(수집→분류→추출→임베딩→적재→색인)과 **완전히 분리된 별도
배치**다(spec §1 · 2026-08-24 개정). 적재 경로는 한 글자도 바뀌지 않고, 이 배치는 이미 저장된
산출물(요약·키워드)만 읽는다 — 그래서 "신규 자산 처리"와 "소급 백필"이 같은 배치 하나다.

한 자산의 처리 순서(한 단계도 건너뛰면 안 된다)::

    판정(LLM 1회/자산) → apply_rules(광역 제외·스톱패턴·접미 병합) →
    resolve_registered_aliases(등록 메타 별칭 합류) → 발굴 모드 분기 → upsert_entity_edges(자산 스코프 교체)

⚠️ **병합 색인·별칭 색인은 배치 시작에 한 번만 만든다.** 자산마다 만들면 자산 수 × 개체 수로
발산한다(자산 10만 × 개체 10만 = 100억 · 코어 T017 이 고친 결함). 이 러너는 색인을 **찾아보기만**
하고 훑거나 복사하지 않는다 — 단위 테스트가 "훑으면 터지는 가짜 색인"으로 그것을 봉인한다.

**타입 어휘(정의문)도 배치 시작에 한 번 읽는다**(F05 · spec §10). 색인 2벌과 **같은 읽기
트랜잭션**에서 ``fetch_meta_type_vocab`` 으로 읽어 판정(``type_defs=``)에 싣고, 그 어휘로부터
``prompt_version_for`` 로 **판(pv)**을 정한다. 🔴 그 한 값이 두 곳에 동시에 쓰인다 —
**재선별 술어**(``fetch_binding_targets``)와 **저장 스탬프**(``upsert_entity_edges``). 둘이 갈리면
매 배치가 같은 자산을 영원히 다시 집거나(술어가 더 새 판을 찾음), 문안은 v1 인데 스탬프만 v2 로
찍혀 정의문 효과 확인·재판정 범위 산정이 불가능해진다.

**발굴 모드**(spec §1-1 · 설정 ``MM_META_DISCOVERY_MODE``)
    - ``propose``(기본) — **이미 존재하는 메타**(수동 선등록 + 지난 배치 발굴분)의 표기·별칭과
      일치하는 판정만 소속시킨다. 미등록 개체는 노드·엣지를 만들지 않고 **후보 리포트**로만 낸다.
      사람이 ``scripts/register_mm_meta.py`` 로 승인하면 그때부터 채워진다.
    - ``auto`` — 미등록 개체도 자동 생성(전량 사전검증이 이 모드의 근거).

⚠️ **후보 승인 뒤의 재소속은 ``--rejudge``** 다. 판정에 성공한 자산은 소속이 0건이어도 판정 이력을
남긴다(그러지 않으면 매 배치가 같은 자산에 LLM 을 다시 부른다 · spec §9-1). 그래서 후보를 등록한
직후에는 그 후보가 나온 자산들을 ``--rejudge --assets <id…>`` 로 다시 돌린다(후보 리포트가 자산
id 를 함께 낸다). 판정은 temperature=0 이라 같은 답이 나오고, 이번에는 등록 메타에 붙는다.

IO 경계를 나눠 뒀다: 조립부(``run_binding``·``run_describe``)는 DB·LLM 을 **아예 모르고** 주입된
함수만 부른다. 실제 커넥션·클라이언트는 ``main`` 만 만든다(``run_opensearch_resync`` 선례).

사용법::

    python -m processing.app.run_mm_meta_binding --env dev --dry-run          # 미리보기(쓰기 0)
    python -m processing.app.run_mm_meta_binding --env dev --limit 200        # 앞 200건만
    python -m processing.app.run_mm_meta_binding --env dev --discovery-mode auto
    python -m processing.app.run_mm_meta_binding --env dev --rejudge <asset_id> …
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from psycopg.rows import dict_row

from src.mm_meta import (
    DESC_PROMPT_VERSION,
    ENTITY_TYPE_DEFS,
    LINEAGE_ACTIVITY,
    MM_MEMBER_KIND_CODE,
    MM_META_VISIBLE_STATUSES,
    PROMPT_VERSION,
    RULE_VERSION,
    EntityTypeDef,
    ExtractedEntity,
    apply_rules,
    describe_meta,
    ensure_mm_member_kind,
    fetch_meta_description_targets,
    fetch_meta_members,
    fetch_meta_type_vocab,
    fetch_official_name_index,
    fetch_registered_alias_index,
    judge_asset_entities,
    prompt_version_for,
    resolve_registered_aliases,
    upsert_entity_edges,
    upsert_meta_description,
)

_LOG = logging.getLogger("meta_extract.run_mm_meta_binding")

# ── 발굴 모드(spec §1-1) ────────────────────────────────────────────────────────
DISCOVERY_PROPOSE = "propose"
DISCOVERY_AUTO = "auto"
DISCOVERY_MODES: tuple[str, ...] = (DISCOVERY_PROPOSE, DISCOVERY_AUTO)
# 코어 설정(``src.config.settings``)에는 이 키가 없다 — 발굴 모드는 파이프 배치의 운영 손잡이라
# spec §구현 확정 G3 이 "T006(파이프) 소관"으로 정했다. 그래서 env 를 직접 읽는다.
DISCOVERY_MODE_ENV = "MM_META_DISCOVERY_MODE"

# ── 타입 어휘 출처(F05 · spec §10) ──────────────────────────────────────────────
# 운영자가 리포트만 보고 "등록한 정의문이 적용됐나"를 알 수 있게 세 갈래로 적어 준다.
#   registered  — ``mm_skill``(skill_code=mm_meta_type) 등록 행에서 읽었다.
#   code_preset — 등록 행이 없거나 비활성이라 코드 프리셋으로 폴백했다(코어가 WARNING 도 남긴다).
#   none        — 어휘를 아예 주입받지 않았다(정의문 없는 옛 문안 · 조립부를 직접 부른 경우).
VOCAB_SOURCE_REGISTERED = "registered"
VOCAB_SOURCE_PRESET = "code_preset"
VOCAB_SOURCE_NONE = "none"
# 리포트 한 줄에 붙일 사람 말. 출처를 코드 값으로만 찍으면 운영자가 다시 물어보게 된다.
_VOCAB_SOURCE_LABELS = {
    VOCAB_SOURCE_REGISTERED: "등록 행",
    VOCAB_SOURCE_PRESET: "코드 프리셋(등록 행 없음 → 폴백)",
    VOCAB_SOURCE_NONE: "미주입(정의문 없이 판정)",
}

# 리포트에 담는 상한 — 콘솔·XCom 이 넘치지 않게 자른다(전체 수는 ``*_total`` 로 함께 보고한다).
_MAX_CANDIDATES = 200
_MAX_TOP_BUNDLES = 10
_MAX_EXAMPLE_CHARS = 60
_MAX_CANDIDATE_ASSETS = 3
_MAX_ORPHANS = 50
_MAX_DESC_SAMPLES = 10


# ── 대상 선별(spec §6) ──────────────────────────────────────────────────────────
# registered · 키워드 보유 · (판정 이력 없음 또는 버전 낡음).
#   ⚠️ **도메인 경계 없음**(의료 포함) — 2026-08-24 사용자 지시로 spec §1 의 '의료 제외'는 폐기됐다
#      (2026-07-23 도메인 제외 전면 제거 결정과 정합 · 헌법 4조).
#   ⚠️ ``CASE`` 로 감싼 이유: ``jsonb_array_length`` 는 배열이 아닌 값에 예외를 낸다. PostgreSQL 은
#      ``AND`` 의 단축평가를 **보장하지 않으므로**(옵티마이저가 순서를 바꾼다) 앞에 타입 검사를
#      나란히 두는 것으로는 못 막는다 — 손 SQL·구버전이 남긴 행 하나가 배치 전체를 죽인다.
_TARGET_BASE_SQL = """
SELECT a.asset_id,
       m.ext_meta->>'summary' AS summary,
       m.ext_meta->'keywords' AS keywords
FROM asset a
JOIN asset_metadata m ON m.asset_id = a.asset_id
WHERE a.status = 'registered'
  AND CASE WHEN jsonb_typeof(m.ext_meta->'keywords') = 'array'
           THEN jsonb_array_length(m.ext_meta->'keywords') ELSE 0 END > 0
"""

# 판정 이력 필터 — "현행 문안·현행 규칙으로 이미 판정된 자산"만 제외한다.
#   ``NOT EXISTS`` 인 이유는 085 대상 선별과 같다: 한 자산에 계보 행이 여러 개라 LEFT JOIN 이면
#   같은 자산이 여러 번 나온다(중복 판정 호출).
#   선별 술어 = **rv(순서 비교) OR pv(동일성)**(구현 확정 3). 위 두 축 중 하나라도 어긋나면 재선별.
#   ⚠️ ``rule_version`` 캐스팅도 ``CASE`` 로 감싼다 — 숫자가 아닌 값이 한 행이라도 있으면
#      ``::int`` 가 질의 전체를 터뜨린다(코어가 낡음 판정을 파이썬으로 옮긴 것과 같은 이유).
_TARGET_HISTORY_SQL = """
  AND NOT EXISTS (
      SELECT 1 FROM asset_lineage l
       WHERE l.asset_id = a.asset_id
         AND l.activity = %s
         AND l.payload->>'prompt_version' = %s
         AND CASE WHEN l.payload->>'rule_version' ~ '^-?[0-9]+$'
                  THEN (l.payload->>'rule_version')::int ELSE -1 END >= %s
  )
"""

# 고아 메타 — 노출 대상 상태의 소속이 하나도 없는 개체 노드. 수동 선등록한 빈 메타는 **정상**이라
# ``source`` 를 함께 읽어 리포트가 구분해 보여 준다(spec §6-1 — 빈 메타 허용).
_ORPHAN_METAS_SQL = """
SELECT en.entity_type,
       en.entity_uid,
       COALESCE(NULLIF(en.canonical->>'name', ''), en.entity_uid) AS name,
       COALESCE(en.canonical->>'source', 'auto') AS source
FROM node en
WHERE en.node_kind = 'entity'
  AND NOT EXISTS (
      SELECT 1
      FROM graph_edge ge
      JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id
      WHERE ge.dst_node = en.node_id
        AND rk.kind_code = %s
        AND ge.status = ANY(%s)
  )
ORDER BY en.entity_type, en.entity_uid
LIMIT %s
"""


def resolve_discovery_mode(raw: str | None) -> str:
    """발굴 모드 문자열을 해소한다(순수 · spec §1-1).

    Args:
        raw: 설정·명령행에서 온 값. ``None``·빈 문자열이면 기본값 ``propose`` 다 — 잡동사니 메타를
            만들지 않는 쪽이 기본이어야 한다("묶인 것은 확실하다" 원칙). 대소문자·앞뒤 공백은 흡수한다.

    Returns:
        ``propose`` 또는 ``auto``.

    Raises:
        ValueError: 어휘 밖 값일 때. **조용히 기본값으로 흡수하지 않는다** — 오타 하나로 "왜 메타가
            안 생기나"(또는 그 반대)를 며칠 추적하게 된다.
    """
    text = (raw or "").strip().lower()
    if not text:
        return DISCOVERY_PROPOSE
    if text not in DISCOVERY_MODES:
        raise ValueError(f"발굴 모드가 어휘 밖이다: {raw!r} (허용 {DISCOVERY_MODES})")
    return text


def vocab_source_of(type_defs: Sequence[EntityTypeDef] | None) -> str:
    """이 어휘가 **어디서 왔는지** 판정한다(순수 · 리포트 표시용).

    코어 계약을 그대로 읽는 것이다: ``fetch_meta_type_vocab`` 은 등록 행이면 **새로 만든** 튜플을,
    행이 없거나 비활성이면 코드 프리셋 상수 ``ENTITY_TYPE_DEFS`` **그 객체**를 돌려준다. 그래서
    객체 동일성(``is``)만으로 두 경로가 갈린다 — 내용 비교로는 "등록 행이 프리셋과 똑같은 문안"인
    경우를 폴백으로 잘못 적게 된다(등록 CLI 가 프리셋을 그대로 올리므로 **정상 등록 직후가 바로 그
    상태**다. 운영자가 알고 싶은 것은 "내 등록이 살아 있나"다).

    ⚠️ **표시 전용**이다 — 문안도 스탬프도 이 값에 좌우되지 않는다(둘 다 ``type_defs`` 자체와
    ``prompt_version_for`` 가 정한다). 그래서 호출부가 어휘를 다른 자료형으로 감싸 넘겨(``list(…)``)
    출처가 잘못 적히더라도 판정·저장은 어긋나지 않는다.

    Args:
        type_defs: 판정에 실은 타입 정의문. ``None``·빈 목록이면 어휘를 주입받지 않은 것이다.

    Returns:
        ``registered``·``code_preset``·``none`` 중 하나.
    """
    if not tuple(type_defs or ()):
        return VOCAB_SOURCE_NONE
    return VOCAB_SOURCE_PRESET if type_defs is ENTITY_TYPE_DEFS else VOCAB_SOURCE_REGISTERED


def split_by_registration(
    entities: Sequence[ExtractedEntity],
    alias_index: Mapping[tuple[str, str], str],
) -> tuple[tuple[ExtractedEntity, ...], tuple[ExtractedEntity, ...]]:
    """판정을 **등록된 메타에 일치하는 것**과 **미등록 후보**로 가른다(순수 · propose 모드 판단 한 곳).

    "등록된 메타" = 이미 ``node`` 에 있는 개체(수동 선등록분 + 지난 배치 발굴분)의 **표기·별칭**이다
    (``fetch_registered_alias_index`` 가 만든 색인의 키가 정확히 그것이다). 별칭 치환
    (``resolve_registered_aliases``)을 **먼저** 돌린 뒤에 이 함수를 부른다 — 그래야 별칭으로 판정된
    개체도 등록 메타로 인정된다.

    Args:
        entities: 규칙·별칭까지 통과한 판정 목록(보통 자산 하나 분량). 순서를 보존한다.
        alias_index: 등록 메타 색인. **찾아보기만 한다**(훑거나 복사하지 않는다 — 배치 시작에 한 번
            만든 것을 자산 루프 내내 그대로 쓴다).

    Returns:
        ``(등록분, 미등록 후보)`` 두 튜플. 색인이 비어 있으면 전부 후보다.
    """
    matched: list[ExtractedEntity] = []
    withheld: list[ExtractedEntity] = []
    for entity in entities:
        target = matched if (entity.entity_type, entity.uid) in alias_index else withheld
        target.append(entity)
    return tuple(matched), tuple(withheld)


def _failure_key(exc: Exception) -> str:
    """예외를 리포트 집계 키로 바꾼다 — **타입명만** 담는다.

    예외 메시지·파일 경로에는 식별 가능한 내용이 섞일 수 있다(헌법 10조 · ``run_relations`` 의
    큐 사유 규율과 같다).

    Args:
        exc: 잡은 예외.

    Returns:
        ``exception:<타입명>`` 형식의 집계 키.
    """
    return f"exception:{type(exc).__name__}"


def run_binding(
    materials: Sequence[Mapping[str, Any]],
    *,
    mode: str = DISCOVERY_PROPOSE,
    official_index: Mapping[tuple[str, str], str] | None = None,
    alias_index: Mapping[tuple[str, str], str] | None = None,
    summary_max_chars: int | None = None,
    type_defs: Sequence[EntityTypeDef] | None = None,
    dry_run: bool = False,
    judge_fn: Callable[..., Any] | None = None,
    persist_fn: Callable[[str, Sequence[ExtractedEntity]], Mapping[str, Any]] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """자산 재료를 돌며 판정→규칙→별칭→발굴 모드→저장을 수행하고 diff 리포트를 만든다.

    **이 함수는 DB·LLM 을 모른다** — 판정도 저장도 주입된 함수가 한다. 그래서 실 DB·실 LLM 없이
    순서·분기·집계를 통째로 단위 검증할 수 있고, 실행 계층(``main``)은 커넥션·트랜잭션만 책임진다.

    자산 하나의 실패가 배치를 멈추지 않는다(자산 단위 ``try/except`` 격리 · 구현 확정 5 — 코어
    ``judge`` 는 전송 계층 예외를 그대로 올린다). 🔴 **판정 실패는 저장하지 않는다** — 저장하면
    "판정 완료"로 굳어 재시도가 영구히 멈춘다(spec §2).

    Args:
        materials: ``[{asset_id, summary, keywords}]`` — ``fetch_binding_targets`` 결과 모양.
        mode: 발굴 모드(``propose``·``auto`` · spec §1-1). 어휘 밖이면 **즉시** ``ValueError``.
        official_index: 접미 병합용 공식 표기 색인(``fetch_official_name_index`` 결과).
            ``None``(기본)이면 병합은 자산 안 동시 등장에서만 일어난다(코어 기존 동작).
            🔴 배치 시작에 한 번 만든 것을 넘긴다 — 이 함수는 찾아보기만 한다.
        alias_index: 등록 메타 별칭 색인(``fetch_registered_alias_index`` 결과). ``None``(기본)이면
            치환하지 않고, ``propose`` 모드에서는 **일치하는 등록 메타가 없다**는 뜻이 되어 전부
            후보가 된다.
        summary_max_chars: 판정 프롬프트에 싣는 요약 상한(설정 ``MM_META_JUDGE_SUMMARY_CHARS``).
            ``None``(기본)이면 코어 문안 기본값 250 을 쓴다.
        type_defs: 개체 타입 **정의문** 목록(F05 · ``fetch_meta_type_vocab`` 결과). ``None``(기본)이면
            정의문 없이 **기존 문안 그대로** 판정한다(코어 하위호환). 이 값이 곧 리포트의
            ``prompt_version``(``prompt_version_for``)을 정한다 — 🔴 문안과 스탬프가 갈리면 정의문
            효과 확인도, 재판정 대상 산정도 못 한다. 🔴 배치 시작에 한 번 읽은 것을 넘긴다.
        dry_run: 참이면 **아무 것도 쓰지 않고** 무엇이 바뀔지만 보고한다(판정 LLM 호출은 한다 —
            무엇이 저장될지 알려면 판정이 필요하다).
        judge_fn: 개체 판정 함수. ``None``(기본)이면 코어 ``judge_asset_entities``(LLM 단일
            seam·temp=0)를 **호출 시점에 이름으로** 찾는다 — def 기본값으로 묶어 두면
            ``mock.patch.object(모듈, "judge_asset_entities", …)`` 가 안 먹어 배선 테스트가
            네트워크를 타게 된다(``run_relations._domain_fn`` 과 같은 관례).
        persist_fn: ``(asset_id, entities) -> 저장 결과 dict`` 저장부. 미주입이면 ``dry_run`` 일 때만
            허용되고, 쓰기 모드에서는 ``ValueError`` — 저장부 없는 쓰기 배치는 "조용한 0건"이 된다.
        client: 판정에 쓸 LLM 클라이언트. ``None``(기본)이면 코어 seam 이 운영 온프레미스 클라이언트를
            쓴다(temperature 는 seam 기본값 0 · 헌법 3조).

    Returns:
        diff 리포트 dict — 대상·판정 성공/실패(사유별)·소속 엣지 증감·신규 메타·상위 묶음·
        **미등록 후보 목록**(propose 모드)·**타입 어휘 출처와 실제 사용한 문안 판**. 같은 입력이면
        같은 리포트가 나온다(결정적 정렬).

    Raises:
        ValueError: ``mode`` 가 어휘 밖이거나, 쓰기 모드인데 ``persist_fn`` 이 없을 때.
    """
    if mode not in DISCOVERY_MODES:
        raise ValueError(f"발굴 모드가 어휘 밖이다: {mode!r} (허용 {DISCOVERY_MODES})")
    if not dry_run and persist_fn is None:
        raise ValueError("쓰기 모드인데 저장부(persist_fn)가 없다 — 조용한 0건 배치를 막는다")

    aliases: Mapping[tuple[str, str], str] = alias_index if alias_index is not None else {}
    # seam 기본값 해소: None 이면 모듈 수준 이름을 **호출 시점에** 잡는다(def 기본값으로 묶으면
    # 테스트가 바꿔 끼운 이름이 무시된다 · run_relations 의 _domain_fn 과 같은 이유).
    judge = judge_fn if judge_fn is not None else judge_asset_entities
    report: dict[str, Any] = {
        "mode": mode,
        "dry_run": bool(dry_run),
        # 어휘 출처·문안 판은 **판정 전에** 확정된다(자산 0건이어도 리포트에 남는다) — 운영자가
        # "이번 배치가 무슨 문안으로 돌았나"를 결과 유무와 무관하게 확인할 수 있어야 한다.
        "vocab_source": vocab_source_of(type_defs),
        "type_defs": len(tuple(type_defs or ())),
        "prompt_version": prompt_version_for(type_defs),
        "targets": len(materials),
        "judged_ok": 0,
        "judged_failed": 0,
        "assets_bound": 0,
        "assets_empty": 0,
        "assets_shrunk": 0,
        "edges_inserted": 0,
        "edges_deleted": 0,
        "withheld": 0,
    }
    failures: dict[str, int] = {}
    # 배치 안에서 센 묶음(= 이번에 소속이 생긴 자산 수). 실제 묶음 크기는 DB 가 정본이고, 여기 값은
    # "이번 배치가 무엇을 많이 채웠나"다 — 리포트 이름(top_bundles)에 그 뜻을 담는다.
    bundles: dict[tuple[str, str], dict[str, Any]] = {}
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    new_metas: list[dict[str, str]] = []
    seen_new: set[tuple[str, str]] = set()

    for item in materials:
        asset_id = str(item.get("asset_id"))
        summary = item.get("summary") or ""
        keywords = list(item.get("keywords") or [])
        try:
            judgement = judge(
                summary, keywords, client=client, summary_max_chars=summary_max_chars,
                # 정의문을 실은 문안으로 판정한다 — 스탬프(report["prompt_version"])는 **같은 값**
                # 에서 나왔다(prompt_version_for). 둘을 따로 정하면 문안과 판이 갈린다.
                type_defs=type_defs,
            )
            if not judgement.ok:
                # 실패는 이력을 남기지 않는다 → 다음 배치가 이 자산을 다시 집는다(spec §2).
                report["judged_failed"] += 1
                key = str(judgement.failure.value if judgement.failure else "unknown")
                failures[key] = failures.get(key, 0) + 1
                continue

            entities = apply_rules(judgement.entities, official_index=official_index)
            entities = resolve_registered_aliases(entities, aliases)
            if mode == DISCOVERY_PROPOSE:
                bound, withheld = split_by_registration(entities, aliases)
            else:
                bound, withheld = tuple(entities), ()

            for entity in withheld:
                slot = candidates.setdefault(
                    (entity.entity_type, entity.uid),
                    {"entity_type": entity.entity_type, "name": entity.name, "assets": 0,
                     "keywords": [], "asset_ids": [], "example": summary[:_MAX_EXAMPLE_CHARS]},
                )
                slot["assets"] += 1
                if entity.keyword not in slot["keywords"] and len(slot["keywords"]) < 3:
                    slot["keywords"].append(entity.keyword)
                if len(slot["asset_ids"]) < _MAX_CANDIDATE_ASSETS:
                    slot["asset_ids"].append(asset_id)
            report["withheld"] += len(withheld)

            if dry_run:
                deleted, inserted = 0, len(bound)
            else:
                # persist_fn 은 자산 하나를 **한 트랜잭션**으로 저장한다(자산 스코프 교체 + 계보).
                result = persist_fn(asset_id, bound)  # type: ignore[misc] — 위에서 None 을 걸렀다
                deleted = int(result.get("edges_deleted", 0))
                inserted = int(result.get("edges_inserted", len(bound)))

            report["judged_ok"] += 1
            report["edges_inserted"] += inserted
            report["edges_deleted"] += deleted
            if deleted > inserted:
                report["assets_shrunk"] += 1
            if bound:
                report["assets_bound"] += 1
            else:
                report["assets_empty"] += 1

            for entity in bound:
                key = (entity.entity_type, entity.uid)
                slot = bundles.setdefault(
                    key, {"entity_type": entity.entity_type, "name": entity.name, "assets": 0}
                )
                slot["assets"] += 1
                if key not in aliases and key not in seen_new:
                    seen_new.add(key)
                    new_metas.append({"entity_type": entity.entity_type, "name": entity.name})
            _LOG.info(
                "mm_meta %s: bound=%d withheld=%d deleted=%d", asset_id, inserted,
                len(withheld), deleted,
            )
        except Exception as exc:  # noqa: BLE001 — 자산 단위 격리(한 건 실패가 배치를 멈추지 않는다)
            report["judged_failed"] += 1
            key = _failure_key(exc)
            failures[key] = failures.get(key, 0) + 1
            _LOG.warning("mm_meta failed %s: %s", asset_id, type(exc).__name__)

    report["failures"] = dict(sorted(failures.items()))
    report["new_metas"] = new_metas
    report["top_bundles"] = sorted(
        bundles.values(), key=lambda b: (-b["assets"], b["entity_type"], b["name"])
    )[:_MAX_TOP_BUNDLES]
    ordered = sorted(
        candidates.values(), key=lambda c: (-c["assets"], c["entity_type"], c["name"])
    )
    report["candidates_total"] = len(ordered)
    report["candidates"] = ordered[:_MAX_CANDIDATES]
    return report


def run_describe(
    targets: Sequence[Mapping[str, Any]],
    *,
    members_fn: Callable[[str, str], Any],
    describe_fn: Callable[[str, str, Any], Any] | None = None,
    save_fn: Callable[..., Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """설명이 없거나 낡은 메타에 한 문장 설명을 만들어 저장한다(T015 · spec §8-1).

    **판정 루프와 분리된 단계**다 — 설명은 "이 묶음에 무엇이 들어 있나"라서 묶음이 확정된 **뒤**에
    만들어야 한다. 소속 배치 도중에 만들면 그 자산이 아직 안 들어온 묶음을 설명하게 된다.

    🔴 **성공한 설명만 저장한다.** 실패를 저장하면 그 메타는 다음 배치에서 "설명 있음"으로 보여
    영구히 다시 만들어지지 않는다(코어 ``upsert_meta_description`` 계약).

    Args:
        targets: ``fetch_meta_description_targets`` 결과(``entity_type``·``entity_uid``·``name``·
            ``bundle_size``·``description``·``reason``).
        members_fn: ``(entity_type, entity_uid) -> [(모달리티, 요약)]`` 재료 조회부.
        describe_fn: ``(name, entity_type, members) -> MetaDescription`` 설명 생성부.
            ``None``(기본)이면 코어 ``describe_meta``(LLM 단일 seam·temp=0)를 **호출 시점에
            이름으로** 찾는다(``judge_fn`` 과 같은 이유 — 테스트가 바꿔 끼울 수 있게).
        save_fn: 저장부 — ``entity_type``·``entity_uid``·``description``·``member_count``·
            ``prompt_version`` 을 **키워드로** 받는다. ``None``(기본)이면 ``dry_run`` 일 때만 허용되고
            쓰기 모드에서는 ``ValueError``(조용한 0건 방지).
        dry_run: 참이면 생성만 하고 저장하지 않는다(무엇이 어떻게 바뀔지 리포트로만 본다).

    Returns:
        설명 단계 리포트 — 대상 수·성공/실패(사유별)·대상 사유 분포·60자 초과 건수·표본(앞뒤 문장).

    Raises:
        ValueError: 쓰기 모드인데 ``save_fn`` 이 없을 때.
    """
    if not dry_run and save_fn is None:
        raise ValueError("쓰기 모드인데 저장부(save_fn)가 없다 — 조용한 0건 배치를 막는다")

    describe = describe_fn if describe_fn is not None else describe_meta
    report: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "targets": len(targets),
        "described": 0,
        "failed": 0,
        "over_target": 0,
    }
    failures: dict[str, int] = {}
    reasons: dict[str, int] = {}
    samples: list[dict[str, str]] = []

    for target in targets:
        entity_type = str(target.get("entity_type", ""))
        entity_uid = str(target.get("entity_uid", ""))
        name = str(target.get("name") or entity_uid)
        reason = str(target.get("reason", ""))
        reasons[reason] = reasons.get(reason, 0) + 1
        try:
            members = members_fn(entity_type, entity_uid)
            result = describe(name, entity_type, members)
            if not result.ok:
                report["failed"] += 1
                key = str(result.failure.value if result.failure else "unknown")
                failures[key] = failures.get(key, 0) + 1
                continue
            if not dry_run:
                save_fn(  # type: ignore[misc] — 위에서 None 을 걸렀다
                    entity_type=entity_type,
                    entity_uid=entity_uid,
                    description=result.description,
                    # 🔴 저장 건수는 **묶음 크기**다(재료 줄 수가 아니다) — 요약 없는 자산이 재료에서
                    #    빠져도 이 값이 묶음 크기여야 낡음 판정이 성립한다(무한 재생성 방지).
                    member_count=int(target.get("bundle_size", 0)),
                    prompt_version=DESC_PROMPT_VERSION,
                )
            report["described"] += 1
            if getattr(result, "exceeds_target", False):
                report["over_target"] += 1
            if len(samples) < _MAX_DESC_SAMPLES:
                samples.append({"name": name, "entity_type": entity_type,
                                "before": str(target.get("description") or ""),
                                "after": result.description})
        except Exception as exc:  # noqa: BLE001 — 메타 단위 격리
            report["failed"] += 1
            key = _failure_key(exc)
            failures[key] = failures.get(key, 0) + 1
            _LOG.warning("mm_meta describe failed %s/%s: %s", entity_type, entity_uid,
                         type(exc).__name__)

    report["failures"] = dict(sorted(failures.items()))
    report["reasons"] = dict(sorted(reasons.items()))
    report["samples"] = samples
    return report


def fetch_binding_targets(
    conn: Any,
    *,
    asset_ids: Sequence[str] | None = None,
    rejudge: bool = False,
    limit: int | None = None,
    prompt_version: str = PROMPT_VERSION,
) -> list[dict[str, Any]]:
    """판정 대상 자산과 그 재료(요약·키워드)를 한 번에 읽는다(조회 전용·결정적 정렬 · spec §6).

    선별 조건은 **registered · 키워드 보유 · (판정 이력 없음 또는 버전 낡음)** 이다. 버전 낡음은
    ``rv``(순서 비교) 또는 ``pv``(동일성) 어느 한쪽만 어긋나도 성립한다(구현 확정 3) — 규칙·문안을
    고치면 그 판정분이 자동으로 재선별된다. **도메인 경계는 없다**(의료 포함 · 2026-08-24 지시).

    Args:
        conn: DB 커넥션.
        asset_ids: 대상을 이 자산들로 좁힌다. ``None``(기본)이면 조건에 맞는 전체.
        rejudge: 참이면 **판정 이력 필터를 빼고** 다시 판정한다. 후보를 등록(승인)한 직후 그 후보가
            나왔던 자산을 다시 붙이는 경로다 — 이력이 남아 있어 평소에는 재선별되지 않기 때문이다.
        limit: 한 번에 가져올 상한. ``None``(기본)이면 전량(배치를 나눠 돌 때만 준다).
        prompt_version: 재선별 술어의 ``pv`` 축 — "이 판으로 이미 판정된 자산"을 제외한다.
            기본값은 현행 문안 판이지만, 🔴 **배치는 반드시 명시로 넘긴다**: 이번 판정이 실제로 쓸
            문안 판(``prompt_version_for(type_defs)``)과 같은 값이어야 한다. 다르면 선별과 저장이
            엇갈려 같은 자산을 매 배치 다시 집는다(무한 재판정) — 술어가 저장 스탬프보다 **새
            판**을 찾으면 그 자산의 이력은 영원히 조건을 만족하지 못한다.

    Returns:
        ``[{asset_id(str), summary(str), keywords(list[str])}]`` — asset_id 오름차순. 요약·키워드는
        문자열로 눌러 담는다(``None`` 이 프롬프트에 "None" 으로 새지 않게).
    """
    sql = _TARGET_BASE_SQL
    params: list[Any] = []
    if asset_ids is not None:
        sql += "  AND a.asset_id = ANY(%s)\n"
        params.append([str(a) for a in asset_ids])
    if not rejudge:
        sql += _TARGET_HISTORY_SQL
        params.extend([LINEAGE_ACTIVITY, prompt_version, RULE_VERSION])
    sql += "ORDER BY a.asset_id\n"
    if limit is not None:
        sql += "LIMIT %s\n"
        params.append(int(limit))

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    return [
        {
            "asset_id": str(r["asset_id"]),
            "summary": str(r["summary"] or ""),
            "keywords": [str(k) for k in (r["keywords"] or []) if k is not None],
        }
        for r in rows
    ]


def fetch_orphan_metas(conn: Any, *, limit: int = _MAX_ORPHANS) -> list[dict[str, str]]:
    """소속이 하나도 없는 메타(고아 노드)를 읽는다(조회 전용 · diff 리포트용).

    규칙이 강화되면 옛 소속이 사라지면서 메타만 남을 수 있다. 수동 선등록한 빈 메타는 **정상**이므로
    (spec §6-1) ``source`` 를 함께 돌려줘 리포트가 둘을 구분해 보여 준다.

    Args:
        conn: DB 커넥션.
        limit: 최대 행수. 콘솔이 흐르지 않게 자른다.

    Returns:
        ``[{entity_type, entity_uid, name, source}]`` — ``(타입, 표기 키)`` 오름차순.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_ORPHAN_METAS_SQL,
                    (MM_MEMBER_KIND_CODE, list(MM_META_VISIBLE_STATUSES), int(limit)))
        rows = cur.fetchall()
    return [
        {"entity_type": str(r["entity_type"]), "entity_uid": str(r["entity_uid"]),
         "name": str(r["name"] or r["entity_uid"]), "source": str(r["source"] or "auto")}
        for r in rows
    ]


def format_report(
    report: Mapping[str, Any],
    *,
    describe: Mapping[str, Any] | None = None,
    orphans: Sequence[Mapping[str, str]] | None = None,
) -> str:
    """배치 결과를 사람이 읽는 여러 줄로 만든다(순수 함수).

    Args:
        report: ``run_binding`` 이 돌려준 diff 리포트.
        describe: ``run_describe`` 리포트. ``None`` 이면 그 절을 빼고 찍는다(설명 단계를 껐을 때).
        orphans: ``fetch_orphan_metas`` 결과. ``None`` 이면 그 절을 빼고 찍는다.

    Returns:
        요약 문자열. 후보·상위 묶음은 **앞 몇 건만** 덧붙인다 — 전부 찍으면 콘솔이 넘쳐 정작
        상태를 못 본다(자세한 목록은 반환 dict 에 있다).
    """
    dry = " (dry-run · 쓰기 0)" if report.get("dry_run") else ""
    # 타입 어휘 줄 — "등록한 정의문이 이번 배치에 적용됐나"와 "무슨 판으로 스탬프됐나"를 한 줄에.
    source = str(report.get("vocab_source", VOCAB_SOURCE_NONE))
    label = _VOCAB_SOURCE_LABELS.get(source, source)
    count = int(report.get("type_defs", 0) or 0)
    vocab_line = f"  타입 어휘: {label}"
    if count:
        vocab_line += f" {count}종"
    vocab_line += f" · pv={report.get('prompt_version', '?')}"
    lines = [
        f"[멀티모달 메타 소속] mode={report.get('mode')}{dry}",
        vocab_line,
        f"  대상 {report.get('targets', 0)}건 | 판정 성공 {report.get('judged_ok', 0)} · "
        f"실패 {report.get('judged_failed', 0)}",
        f"  소속 자산 {report.get('assets_bound', 0)} · 소속 0건 {report.get('assets_empty', 0)} | "
        f"엣지 +{report.get('edges_inserted', 0)} / -{report.get('edges_deleted', 0)}"
        f" (줄어든 자산 {report.get('assets_shrunk', 0)})",
        f"  신규 메타 {len(report.get('new_metas', []))}건",
    ]
    if report.get("failures"):
        lines.append(f"  ⚠️ 실패 사유: {report['failures']}")
    top = report.get("top_bundles") or []
    if top:
        joined = ", ".join(f"{b['name']}({b['assets']})" for b in top[:5])
        lines.append(f"  이번 배치 상위 묶음: {joined}")
    cands = report.get("candidates") or []
    if cands:
        joined = ", ".join(f"{c['name']}[{c['entity_type']}]({c['assets']})" for c in cands[:10])
        lines.append(
            f"  미등록 후보 {report.get('candidates_total', len(cands))}종 — "
            "승인=scripts/register_mm_meta.py 등록 후 --rejudge 로 재소속: "
            f"{joined}"
        )
    if describe is not None:
        lines.append(
            f"  설명: 대상 {describe.get('targets', 0)} · 생성 {describe.get('described', 0)} · "
            f"실패 {describe.get('failed', 0)} · 60자 초과 {describe.get('over_target', 0)}"
        )
    if orphans:
        user_owned = sum(1 for o in orphans if o.get("source") == "user")
        lines.append(
            f"  고아 메타 {len(orphans)}건(그중 수동 등록 {user_owned}건은 정상 — 빈 메타 허용)"
        )
    return "\n".join(lines)


def run_batch(
    db: Any,
    *,
    mode: str = DISCOVERY_PROPOSE,
    asset_ids: Sequence[str] | None = None,
    rejudge: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
    describe: bool = True,
) -> dict[str, Any]:
    """배치 한 판을 배선한다 — 어휘·색인 조립 1회 → 자산 루프 → 설명 단계 → 고아 리포트.

    ``main``(명령행)과 Airflow DAG 가 **같은 이 함수**를 부른다. DAG 에 배선을 복사하면 한쪽만
    고쳐져 갈라진다(예: DAG 쪽에서 색인을 자산마다 만드는 실수).

    🔴 **문안 판(pv)은 여기서 한 번만 정해진다**(``_load`` 안 · 타입 어휘로부터). 그 값이 재선별
    술어와 저장 스탬프 양쪽에 그대로 흘러간다 — 두 곳에서 따로 정하면 갈릴 수 있고, 갈리면 매
    배치가 같은 자산을 다시 판정한다(LLM 비용이 자산 수만큼 반복된다).

    트랜잭션 경계가 이 함수의 핵심 책임이다:
        - 타입 어휘 + 색인 2벌 + 대상 목록 = **읽기 트랜잭션 한 번**(전부 배치당 1회가 계약이다).
        - 자산 하나의 저장 = **fresh 트랜잭션**(한 건 실패가 다른 자산을 롤백시키지 않는다).
        - 설명 저장도 건마다 별도 트랜잭션(묶음이 확정된 뒤 도는 별도 단계).

    Args:
        db: 트랜잭션을 열 수 있는 DB 핸들(``PostgresUtil``). 풀 수명은 호출부가 관리한다.
        mode: 발굴 모드(``propose``·``auto`` · spec §1-1).
        asset_ids: 대상을 이 자산들로 좁힌다. ``None``(기본)이면 조건에 맞는 전체.
        rejudge: 참이면 판정 이력이 있어도 다시 판정한다(후보 승인 뒤 재소속 경로).
        limit: 이번 실행에서 처리할 자산 수 상한. ``None``(기본)이면 전량.
        dry_run: 참이면 **아무 것도 쓰지 않는다**(카탈로그 보장도 쓰기라 건너뛴다).
        describe: 참(기본)이면 설명 단계를 함께 돈다. 설정 ``MM_META_DESCRIBE_ENABLED`` 가 꺼져
            있으면 이 값과 무관하게 돌지 않는다(둘 중 하나라도 꺼져 있으면 안 돈다).

    Returns:
        ``{"binding": diff 리포트, "describe": 설명 리포트|None, "orphans": [...]}``.
        토글(``MM_META_BINDING_ENABLED``)이 꺼져 있으면 ``{"skipped": "disabled", …}`` 를 돌려주고
        질의도 LLM 호출도 하지 않는다. ``binding`` 리포트에는 이번 배치가 쓴 **타입 어휘 출처**
        (``vocab_source``)와 **문안 판**(``prompt_version``)이 함께 담긴다.

    Raises:
        MmMetaPersistError: 타입 어휘 등록 행이 닫힌 5종과 어긋날 때(코어 fail-fast). 어휘가 어긋난
            채 돌면 그 배치의 판정 전체가 의도와 다른 문안으로 나가므로 **첫 읽기에서** 멈춘다.
            등록 행이 아예 없거나 비활성인 것은 예외가 아니라 코드 프리셋 폴백이다(경고만 남는다).
    """
    from src.config.settings import get_current_settings

    cfg = get_current_settings()
    if not cfg.mm_meta.binding_enabled:
        # 끄기 ≠ 삭제 — 이미 쌓인 소속·묶음은 그대로 두고 **새 판정만** 멈춘다.
        _LOG.info("mm_meta binding disabled(MM_META_BINDING_ENABLED=0)")
        return {"skipped": "disabled", "binding": None, "describe": None, "orphans": []}

    if not dry_run:
        # 카탈로그 행이 없거나 비활성이면 저장이 fail-fast 로 거부된다(조용한 0건 금지).
        db.execute_in_transaction(ensure_mm_member_kind, idempotent=True)

    def _load(conn: Any) -> tuple[tuple[EntityTypeDef, ...], str, Any, Any, list[dict[str, Any]]]:
        """읽기 트랜잭션 한 번에 **타입 어휘 · 색인 2벌 · 대상 목록**을 만든다(전부 배치당 1회).

        어휘를 **가장 먼저** 읽는다 — 그것이 이번 배치의 문안 판(``pv``)을 정하고, 바로 아래 대상
        선별이 그 판으로 "이미 판정된 자산"을 걸러야 하기 때문이다. 한 트랜잭션 안이라 어휘와 대상이
        서로 다른 시점의 DB 를 보는 일이 없다(배치 도중 어휘가 등록돼도 이번 판은 일관된다).

        Args:
            conn: 읽기 트랜잭션의 커넥션(호출부가 연다).

        Returns:
            ``(타입 정의문, 문안 판, 공식 표기 색인, 별칭 색인, 대상 재료)``.
        """
        type_defs = fetch_meta_type_vocab(conn)
        # 🔴 이 한 값이 재선별 술어와 저장 스탬프 **양쪽**에 쓰인다(따로 계산하지 않는다).
        prompt_version = prompt_version_for(type_defs)
        official = fetch_official_name_index(conn)
        aliases = fetch_registered_alias_index(conn)
        targets = fetch_binding_targets(
            conn, asset_ids=list(asset_ids) if asset_ids else None, rejudge=rejudge, limit=limit,
            prompt_version=prompt_version,
        )
        return type_defs, prompt_version, official, aliases, targets

    type_defs, prompt_version, official_index, alias_index, materials = db.execute_in_transaction(
        _load, idempotent=True
    )
    if mode == DISCOVERY_PROPOSE and materials and not alias_index:
        # 등록 메타 0 은 **정상 상태**다(등록 전) — 다만 이번 배치는 소속을 하나도 만들지 못하고
        # 후보 리포트만 낸다. 조용히 지나가면 "왜 묶음이 안 생기나"를 코드에서 찾게 된다.
        # ⚠️ 이 검사는 여기(색인을 만든 자리)에서만 한다 — 조립부는 색인을 **찾아보기만** 해야
        #    하고, 길이를 재는 것조차 그 계약 밖이다(가짜 색인 테스트가 그것을 막는다).
        _LOG.warning(
            "등록된 메타가 0건이다 — propose 모드라 이번 배치는 소속을 만들지 않고 "
            "미등록 후보만 보고한다(승인=scripts/register_mm_meta.py → --rejudge)"
        )

    def _persist(asset_id: str, entities: Sequence[ExtractedEntity]) -> Mapping[str, Any]:
        """자산 하나의 소속을 **한 트랜잭션**으로 교체한다(엣지 + 계보가 전부 아니면 전무).

        Args:
            asset_id: 저장 대상 자산.
            entities: 규칙·별칭·발굴 모드까지 통과한 판정 목록. **빈 목록도 저장한다**(성공·개체
                0 — 기존 소속을 지우고 이력을 남겨 다음 배치가 다시 판정하지 않게).

        Returns:
            코어 저장 결과 dict(삭제·삽입 엣지 수 포함).
        """
        return db.execute_in_transaction(
            # 🔴 ``prompt_version`` 을 **명시**한다. 코어 기본값은 모듈 상수(현행 최신 판)라, 정의문
            #    없이 나간 판정도 최신 판으로 찍힌다 — 위 ``_load`` 가 정한 값(= 재선별 술어와 같은
            #    값)만 스탬프에 남긴다. 그래야 문안과 판이 항상 일치한다.
            lambda conn, _aid=asset_id, _ents=entities: upsert_entity_edges(
                conn, _aid, _ents, prompt_version=prompt_version
            ),
            idempotent=False,
        )

    report = run_binding(
        materials,
        mode=mode,
        official_index=official_index,
        alias_index=alias_index,
        summary_max_chars=cfg.mm_meta.judge_summary_chars,
        type_defs=type_defs,
        dry_run=dry_run,
        persist_fn=None if dry_run else _persist,
    )

    describe_report: dict[str, Any] | None = None
    if describe and cfg.mm_meta.describe_enabled:
        desc_targets = db.execute_in_transaction(
            lambda conn: fetch_meta_description_targets(conn, prompt_version=DESC_PROMPT_VERSION),
            idempotent=True,
        )

        def _members(entity_type: str, entity_uid: str) -> Any:
            """설명 재료(모달리티·요약)를 읽는다 — 읽기 전용 트랜잭션."""
            return db.execute_in_transaction(
                lambda conn, _t=entity_type, _u=entity_uid: fetch_meta_members(conn, _t, _u),
                idempotent=True,
            )

        def _save(**kwargs: Any) -> Any:
            """설명 한 건을 ``node.canonical`` 에 병합 저장한다(기존 키 보존)."""
            return db.execute_in_transaction(
                lambda conn: upsert_meta_description(
                    conn,
                    kwargs["entity_type"],
                    kwargs["entity_uid"],
                    description=kwargs["description"],
                    member_count=kwargs["member_count"],
                    prompt_version=kwargs["prompt_version"],
                ),
                idempotent=False,
            )

        describe_report = run_describe(
            desc_targets,
            members_fn=_members,
            save_fn=None if dry_run else _save,
            dry_run=dry_run,
        )

    orphans = db.execute_in_transaction(fetch_orphan_metas, idempotent=True)
    return {"binding": report, "describe": describe_report, "orphans": orphans}


def _build_parser():
    """명령행 옵션을 정의한다(환경·미리보기·발굴 모드·재판정·상한·설명 단계).

    Returns:
        구성된 ``argparse.ArgumentParser``.
    """
    import argparse

    p = argparse.ArgumentParser(
        description="멀티모달 메타 소속 배치 (판정→규칙→별칭→발굴 모드→mm_member 엣지)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    p.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="쓰기 0 — 무엇이 바뀔지만 보고한다(판정 LLM 호출은 한다)",
    )
    p.add_argument(
        "--discovery-mode", dest="discovery_mode", choices=list(DISCOVERY_MODES), default=None,
        help=f"발굴 모드(미지정=환경변수 {DISCOVERY_MODE_ENV}, 그것도 없으면 {DISCOVERY_PROPOSE})",
    )
    p.add_argument(
        "--rejudge", action="store_true",
        help="판정 이력이 있어도 다시 판정한다(후보를 등록·승인한 뒤 재소속시킬 때)",
    )
    p.add_argument("--limit", type=int, default=None, help="이번 실행에서 처리할 자산 수 상한")
    p.add_argument(
        "--no-describe", dest="no_describe", action="store_true",
        help="메타 설명 생성 단계를 건너뛴다(소속만 갱신)",
    )
    p.add_argument("asset_ids", nargs="*", metavar="ASSET_ID", default=[])
    return p


# ── 초기 설정(부트스트랩) 절차 ────────────────────────────────────────────────
# [런타임·main() 안·순서 중요]
#   1) bootstrap_env(env): load_dotenv + init_settings(필수 env 검증·frozen 설정)
#   2) PostgresUtil() + `with db:`: 연결 풀 + PG17 검증
#   3) run_batch: 토글 확인 → 카탈로그 보장 → 색인 1회 → 자산 루프 → 설명 → 고아 리포트
#   4) 결과 출력. 배선 자체는 run_batch 한 곳뿐이다 — DAG 도 같은 함수를 부른다.
def main() -> int:
    """멀티모달 메타 소속 배치를 실행한다(명령행 진입점).

    Returns:
        0=성공(실패 자산 0), 1=실패한 자산이 있음.
    """
    args = _build_parser().parse_args()

    from src.config.bootstrap import bootstrap_env
    from src.database.postgres_util import PostgresUtil

    bootstrap_env(args.env)
    mode = resolve_discovery_mode(args.discovery_mode or os.environ.get(DISCOVERY_MODE_ENV))

    db = PostgresUtil()
    with db:
        result = run_batch(
            db,
            mode=mode,
            asset_ids=list(args.asset_ids) or None,
            rejudge=args.rejudge,
            limit=args.limit,
            dry_run=args.dry_run,
            describe=not args.no_describe,
        )

    if result.get("skipped"):
        print("[멀티모달 메타 소속] MM_META_BINDING_ENABLED=0 — 아무 것도 하지 않는다")
        return 0
    report = result["binding"]
    print(format_report(report, describe=result["describe"], orphans=result["orphans"]))
    return 1 if report["judged_failed"] else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
