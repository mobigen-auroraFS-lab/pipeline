"""샘플 도메인 팩 cross_asset 전략(결정적·비학습·무LLM).

목적(spec 016)
    008 이 만든 cross_asset 슬롯 resolve seam 의 **성공 경로**를, ``contracts.py`` 에
    정의돼 있으나 한 번도 실행된 적 없는 cross_asset 계약 4종으로 처음 실행해 증명한다.
    이 모듈은 그 계약을 만족하는 **트리비얼·결정적 데모 전략**이다.

헌법 준수
    - 1조(비학습): 학습/파인튜닝/.fit 없음 — 사전계산된 임베딩 유사도와 고정 규칙만 사용.
    - 2조(LLM 단일 seam): LLM 을 전혀 호출하지 않는다(샘플은 결정적 규칙 점수).
    - 3조(결정성 100%): 정렬 tiebreak 는 target asset_id 오름차순으로 고정한다.
    - 6조(스키마 불변): persist 는 기존 ``graph_edge`` + 기존 ``relation_kind`` 어휘만
      재사용한다. 신규 테이블·컬럼·마이그레이션·relation_kind 카탈로그 등록 없음.

계약(contracts.py Protocol)
    - ``sample_candidates(conn, source_asset_id) -> list[Candidate]``      # CandidateStage
    - ``sample_score(conn, pairs) -> list[ScoredPair]``                     # ScoreStage
    - ``sample_decide(scored) -> list[Decision]``                          # DecideStage
    - ``sample_persist_edges(conn, decisions) -> None``                    # EdgePersistStage
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
# 샘플 픽스처 격리 마커(FR-004): 이 경로 조각을 가진 registered 자산만 샘플 후보로 본다.
# 전역 임베딩 top-k(find_embedding_candidates)는 운영 임베딩 수천 건에 묻혀 샘플 픽스처를
# 못 찾고, zero-norm 임베딩의 NaN 코사인이 정렬 최상위를 점유한다(2026-06-05 e2e 진단).
# → 샘플은 경로 마커로 후보를 격리해 결정적·자기완결적으로 동작한다.
_SAMPLE_PATH_MARKER = "%/sample_pack/%"
_BLOCK_KEY = "sample_pack"       # Candidate.block_key — 경로 마커 기반 블로킹(추적)
_METHOD = "sample"
_EVIDENCE_FIELD = "embedding"    # Evidence.field — 비교 대상(임베딩 코사인)
_COMPARATOR = "embedding_cosine"


def sample_candidates(conn: Connection[Any], source_asset_id: str) -> list[Candidate]:
    """``/sample_pack/`` 마커 자산을 결정적 이웃 후보로 선별(전역 임베딩 검색 비의존).

    spec 016 e2e 격리 요구(2026-06-05 진단): 전역 ``find_embedding_candidates`` 는 운영
    임베딩(수천 건)에 묻혀 샘플 픽스처를 못 찾고, zero-norm 임베딩의 NaN 코사인이 PG
    ``ORDER BY DESC`` 최상위를 점유한다. 그래서 샘플 후보는 ``/sample_pack/`` 경로 마커를
    가진 ``registered`` 자산으로 한정해(FR-004) 운영 데이터와 격리하고, **target asset_id
    오름차순**으로 결정적 정렬한다(헌법 3조). spec FR-001 이 허용한 "결정적 픽스처 이웃" 방식.
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
    """후보를 **소스↔후보 쌍 코사인**으로 채점(결정적·LLM 미호출).

    헌법 2조: LLM seam 을 전혀 호출하지 않는다. 점수는 소스와 각 후보의 같은 채널 임베딩
    코사인 유사도(사전계산 벡터)라는 고정 규칙이다.

    설계(전역 랭킹 비의존)
        모든 pair 의 source_id 가 동일하다는 계약 전제 하에, 소스와 **후보 집합 한정**으로
        같은 채널 임베딩 쌍의 MAX 코사인을 한 번에 조회해 ``{target_id: sim}`` 맵을 만든다.
        전역 top-k 가 아니라 주어진 후보만 채점하므로 운영 데이터·NaN 정렬에 영향받지 않는다.
        zero-norm 등으로 sim 이 비유한(NaN/inf)이면 0.0 으로 보수 처리한다(결정성·안전).
        후보가 맵에 없으면(임베딩 부재 등) 0.0. ``Evidence(field='embedding',
        comparator='embedding_cosine')`` 부착. 조회 1회를 공유하므로 2회 동일 출력(결정성).
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
    """결정적 임계로 verdict 결정(순수 함수, conn 불필요).

    규칙: ``score >= SAMPLE_DECIDE_TAU`` 이면 'match', 아니면 'non_match'.
    경계값(score == τ)은 ``>=`` 로 match 쪽에 결정적으로 포함한다(헌법 3조).
    입력 순서를 보존하므로 동점·경계 입력이라도 동일 입력 2회 동일 출력이다.
    """
    out: list[Decision] = []
    for sp in scored:
        verdict = "match" if sp.score >= SAMPLE_DECIDE_TAU else "non_match"
        out.append(
            Decision(candidate=sp.candidate, verdict=verdict, score=sp.score)
        )
    return out


def sample_persist_edges(conn: Connection[Any], decisions: list[Decision]) -> None:
    """verdict=='match' 결정만 기존 ``graph_edge`` 로 upsert(헌법 6조: 스키마 불변).

    재사용
        기존 ``graph_persist.sync_graph_edges`` 헬퍼 한 곳으로만 적재한다. 신규 테이블·
        컬럼·마이그레이션·relation_kind 카탈로그 등록은 일절 하지 않는다. relation_type_code
        는 시드 active 어휘 ``SAMPLE_RELATION_KIND``('same_series')만 사용한다(데모 의미
        약결합). ``allowed_target_ids`` 는 match 타깃 집합으로 두어 sync_graph_edges 의
        환각 방지 게이트(후보 집합 밖 타깃 거부)와 정합시킨다.

    설계
        Decision('match') 한 건 → sync_graph_edges 가 받는 edge dict 한 건으로 매핑한다
        (target_media_item_id / relation_type_code / confidence=score / reason). 모든
        결정의 source_id 가 동일하다는 계약 전제 하에 첫 match 의 source 를 쓴다.
        'non_match' 는 건너뛰며, match 가 하나도 없으면 적재 호출 자체를 생략한다.
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
    # 013: 샘플 팩은 슬롯 조합 데모 전용 — 계보(asset_lineage) 기록 대상 외라 collect 미전달.
    # 프로덕션 관계 제안(asset_entry.propose_relations_for_asset)만 generated.edges 로 쌍을 남긴다.
    sync_graph_edges(
        conn,
        source_asset_id=source_id,
        edges=edges,
        allowed_target_ids=allowed_target_ids,
    )
    return None
