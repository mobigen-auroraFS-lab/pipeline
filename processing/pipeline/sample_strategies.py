"""샘플 도메인 팩 cross_asset 전략(결정적·비학습·무LLM).

**흐름에서의 위치**: 샘플 팩이 자산 사이 네 슬롯에 끼우는 전략들이다. 실제 관계 품질이
목적이 아니라, **조합형 구조가 끝까지 도는지 보여 주는 것**이 목적이다.

지켜야 할 것
    - 학습하지 않는다 — 사전 계산된 임베딩과 고정 규칙만 쓴다(헌법 1조).
    - LLM 을 부르지 않는다 — 데모가 모델 가용성에 매이면 구조 검증이 안 된다(헌법 2조).
    - 순서를 못 박는다 — 동점이면 대상 자산 id 순으로 갈라 매번 같은 결과를 낸다(헌법 3조).
    - 스키마를 늘리지 않는다 — 기존 엣지 테이블과 기존 관계 어휘만 재사용한다(헌법 6조).
"""
from __future__ import annotations

import math
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from processing.pipeline.cross_types import Candidate, Decision, Evidence, ScoredPair
from src.relations.graph_persist import sync_graph_edges

# ── 상수 ────────────────────────────────────────────────────────────────────
# 샘플 후보 top-k. 데모이므로 작게 고정한다.
SAMPLE_TOP_K = 10
# 결정 임계 τ — score 가 이 값 이상이면 'match'. 경계는 >= 로 match 쪽에 포함(결정적).
SAMPLE_DECIDE_TAU = 0.5
# 샘플 데모가 쓰는 기존 relation_kind 어휘(시드 active). 의미 약결합 데모용.
SAMPLE_RELATION_KIND = "same_series"
# 샘플 격리 마커: 이 경로 조각을 가진 자산만 샘플 후보로 본다.
# 전역 임베딩 top-k(find_embedding_candidates)는 운영 임베딩 수천 건에 묻혀 샘플 픽스처를
# 못 찾고, zero-norm 임베딩의 NaN 코사인이 정렬 최상위를 점유한다(2026-06-05 e2e 진단).
# → 샘플은 경로 마커로 후보를 격리해 결정적·자기완결적으로 동작한다.
_SAMPLE_PATH_MARKER = "%/sample_pack/%"
_BLOCK_KEY = "sample_pack"       # Candidate.block_key — 경로 마커 기반 블로킹(추적)
_METHOD = "sample"
_EVIDENCE_FIELD = "embedding"    # Evidence.field — 비교 대상(임베딩 코사인)
_COMPARATOR = "embedding_cosine"


def sample_candidates(conn: Connection[Any], source_asset_id: str) -> list[Candidate]:
    """샘플 경로 표식을 가진 자산만 후보로 추린다 — 운영 데이터와 섞이지 않게.

    일반 후보 탐색(임베딩 유사도 상위)을 그대로 쓰면 데모가 운영 데이터 수천 건에 묻혀
    아무것도 못 찾는다. 게다가 벡터 길이가 0 인 임베딩이 섞이면 코사인이 NaN 이 되어
    정렬 최상단을 차지해 버린다. 그래서 **경로 표식으로 후보를 격리**한다.

    Args:
        conn: DB 연결.
        source_asset_id: 기준 자산. 자기 자신은 후보에서 뺀다.

    Returns:
        후보 목록. 대상 자산 id 오름차순으로 순서가 고정된다(헌법 3조).
    """
    sql = """
        SELECT a.asset_id::text AS id
        FROM asset a
        WHERE a.fs_path LIKE %s
          AND a.status = 'registered'
          AND a.asset_id::text <> %s
        ORDER BY a.asset_id
        LIMIT %s
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (_SAMPLE_PATH_MARKER, source_asset_id, SAMPLE_TOP_K))
        rows = cur.fetchall()
    cands = [
        Candidate(
            source_id=source_asset_id,
            target_id=str(r["id"]),
            block_key=_BLOCK_KEY,
            method=_METHOD,
        )
        for r in rows
    ]
    # 결정성: target asset_id 오름차순 고정(SQL ORDER BY 와 일치, 이중 보장).
    cands.sort(key=lambda c: c.target_id)
    return cands


def sample_score(conn: Connection[Any], pairs: list[Candidate]) -> list[ScoredPair]:
    """후보마다 기준 자산과의 임베딩 유사도를 점수로 매긴다(LLM 미호출).

    **후보 전부를 한 번의 조회로 채점한다** — 후보마다 따로 물으면 왕복이 후보 수만큼
    늘고, 조회 시점이 달라 결과가 흔들릴 여지도 생긴다.

    Args:
        conn: DB 연결.
        pairs: 채점할 후보. **모든 쌍의 기준 자산이 같다는 전제**로 첫 항목에서 기준을
            읽는다 — 섞어 넣으면 엉뚱한 자산과의 유사도가 매겨진다.

    Returns:
        점수와 근거가 붙은 쌍 목록(입력 순서 보존). 임베딩이 없거나 값이 NaN·무한대면
        **0.0 으로 접는다** — 이상값이 상위를 차지하는 것을 막는다.
    """
    if not pairs:
        return []
    source_id = pairs[0].source_id
    target_ids = [c.target_id for c in pairs]
    # source×target 청크 임베딩을 **채널별로 자기조인**한다 — 자산당 청크가 여럿이면 카티전 곱(팬아웃)이
    # 되므로 GROUP BY target + MAX 로 (동일 채널) 청크쌍의 최대 코사인만 남긴다(자산쌍 1개 점수로 축약).
    sql = """
        SELECT ta.asset_id::text AS id,
               MAX(1 - (sa.embedding <=> ta.embedding)) AS sim
        FROM asset_embedding sa
        JOIN asset_embedding ta ON ta.channel = sa.channel
        WHERE sa.asset_id::text = %s
          AND ta.asset_id::text = ANY(%s)
        GROUP BY ta.asset_id
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (source_id, target_ids))
        rows = cur.fetchall()
    sim_by_target: dict[str, float] = {}
    for r in rows:
        try:
            sim = float(r["sim"])
        except (TypeError, ValueError):
            sim = 0.0
        if not math.isfinite(sim):  # NaN/inf(zero-norm 등) → 0.0 보수 처리
            sim = 0.0
        sim_by_target[str(r["id"])] = sim
    out: list[ScoredPair] = []
    for c in pairs:
        score = sim_by_target.get(c.target_id, 0.0)
        ev = Evidence(
            field=_EVIDENCE_FIELD,
            comparator=_COMPARATOR,
            similarity=score,
            weight=score,
        )
        out.append(ScoredPair(candidate=c, score=score, evidence=[ev]))
    return out


def sample_decide(scored: list[ScoredPair]) -> list[Decision]:
    """점수가 임계 이상이면 잇기로 판정한다(순수 함수 — DB 를 보지 않는다).

    경계값은 **잇는 쪽**에 포함한다. 어느 쪽이든 상관없지만, 한쪽으로 못 박아 두지 않으면
    같은 점수가 실행마다 다르게 갈릴 여지가 생긴다(헌법 3조).

    Args:
        scored: 점수가 매겨진 쌍.

    Returns:
        판정 목록. **입력 순서를 그대로 보존한다** — 잇지 않기로 한 것도 함께 담는다.
    """
    out: list[Decision] = []
    for sp in scored:
        verdict = "match" if sp.score >= SAMPLE_DECIDE_TAU else "non_match"
        out.append(
            Decision(candidate=sp.candidate, verdict=verdict, score=sp.score)
        )
    return out


def sample_persist_edges(conn: Connection[Any], decisions: list[Decision]) -> None:
    """잇기로 판정한 것만 기존 엣지 테이블에 저장한다(스키마를 늘리지 않는다·헌법 6조).

    저장은 **공용 헬퍼 한 곳**으로만 한다 — 직접 INSERT 를 쓰면 그쪽에 있는 안전장치
    (자기참조 거부·후보 밖 대상 거부)를 통째로 우회하게 된다. 관계 종류도 이미 시드된
    어휘 하나만 쓴다(데모라 의미를 느슨하게 붙인다).

    Args:
        conn: 호출자가 연 연결(트랜잭션 경계는 밖).
        decisions: 판정 목록. **잇지 않기로 한 것도 섞여 온다** — 여기서 걸러 낸다.
            모든 판정의 기준 자산이 같다는 전제로 첫 건에서 기준을 읽는다.

    저장할 것이 하나도 없으면 **호출 자체를 생략한다** — 빈 호출도 노드 보장 같은
    부수 작업을 일으키기 때문이다.
    """
    matches = [d for d in decisions if d.verdict == "match"]
    if not matches:
        # 적재할 엣지가 없으면 sync_graph_edges 를 부르지 않는다(불필요한 노드 보장 회피).
        return None
    source_id = matches[0].candidate.source_id
    allowed_target_ids = frozenset(d.candidate.target_id for d in matches)
    edges: list[dict[str, Any]] = [
        {
            "target_media_item_id": d.candidate.target_id,
            "relation_type_code": SAMPLE_RELATION_KIND,
            "confidence": d.score,
            "reason": "샘플 팩 결정적 cross_asset 전략(데모)",
        }
        for d in matches
    ]
    # 계보를 남기지 않는다 — 데모가 만든 엣지가 운영 계보에 섞이면 "언제 무엇이 이어졌나"를
    # 되짚을 때 잡음이 된다. 계보는 운영 관계 제안 경로만 남긴다.
    sync_graph_edges(
        conn,
        source_asset_id=source_id,
        edges=edges,
        allowed_target_ids=allowed_target_ids,
    )
    return None
