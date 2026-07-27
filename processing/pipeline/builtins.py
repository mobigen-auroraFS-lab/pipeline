"""v2 기본 전략을 DEFAULT_REGISTRY 에 등록(import 시 부수효과).

⚠️ **import 하는 것만으로 등록이 일어난다**(파일 맨 아래 한 줄). 실행 진입점이 이 모듈을
가져오는 것으로 레지스트리가 채워지므로, 그 import 를 "안 쓰는 것 같다"고 지우면 팩이
전략을 찾지 못해 터진다.

테스트는 별도 레지스트리를 만들어 등록 함수를 직접 불러 격리할 수 있다.
"""
from __future__ import annotations

from processing.classify import cascade
from processing.dispatch.dispatcher import dispatch_embed, dispatch_extract_meta
from processing.dispatch.types import ExtractContext
from processing.ingest.asset_persist import finalize_asset
from processing.pipeline.registry import DEFAULT_REGISTRY, StrategyRegistry
from processing.pipeline.sample_strategies import (
    sample_candidates,
    sample_decide,
    sample_persist_edges,
    sample_score,
)
from src.relations.asset_candidates import find_embedding_candidates
from src.relations.graph_persist import sync_graph_edges
from src.relations.llm_propose import propose_edges_json


def _classify_cascade_v1(ctx: ExtractContext):
    """분류 슬롯 계약에 맞추는 얇은 어댑터 — 문맥에서 필요한 값만 꺼내 분류기에 넘긴다.

    분류기 자체는 도메인을 모른다(도메인별 규칙은 그쪽 레지스트리에 있다). 그래서 새 도메인이
    생겨도 이 어댑터는 그대로 둔다.

    Args:
        ctx: 처리 문맥. 파일 경로와 모달리티만 쓴다.

    Returns:
        분류 결과.
    """
    return cascade.classify(ctx.file_path, ctx.modality)


def register_defaults(registry: StrategyRegistry) -> None:
    """기본 전략을 레지스트리에 등록한다.

    태그가 정책 검사의 근거가 되므로 **등록할 때 정확히 붙여야 한다** — 외부 LLM 을 쓰는
    전략에 그 태그를 빠뜨리면 의료 정책이 통과시켜 버린다.

    Args:
        registry: 등록 대상. 보통 프로세스 전역 레지스트리이고, 테스트는 격리된 것을 준다.
    """
    # per-asset
    registry.register("classify", "cascade_v1", _classify_cascade_v1, tags={"onprem_llm"})
    registry.register("extract", "by_modality", dispatch_extract_meta, tags={"onprem_llm"})
    registry.register("embed", "by_modality", dispatch_embed, tags={"deterministic"})
    registry.register("persist", "asset_upsert", finalize_asset)

    # 자산 사이 전략 — 지금은 일반 도메인 실행이 묶음 함수 하나에 위임돼 있어, 여기 등록된
    # 것들은 배선표가 가리키는 대상으로만 쓰인다. 'decide' 는 임계 판정이 점수 쪽에 들어
    # 있어 등록하지 않는다(팩에는 슬롯 이름만 있다).
    registry.register("candidates", "embedding_topk", find_embedding_candidates, tags={"deterministic"})
    registry.register("score", "llm_propose", propose_edges_json, tags={"onprem_llm"})
    registry.register("persist_edges", "graph_upsert", sync_graph_edges)

    # 샘플 도메인 — 조합형 구조가 도는지 보여 주는 데모(결정적·LLM 미사용).
    # ⚠️ 저장 전략은 **등록명과 함수명이 다르다**. 배선표에 적힌 이름에 맞춘 것이라, 함수명을
    #    보고 등록명을 고치면 팩이 전략을 못 찾는다.
    registry.register("candidates", "sample_candidates", sample_candidates, tags={"deterministic"})
    registry.register("score", "sample_score", sample_score, tags={"deterministic"})
    registry.register("decide", "sample_decide", sample_decide, tags={"deterministic"})
    registry.register("persist_edges", "sample_graph_upsert", sample_persist_edges)


register_defaults(DEFAULT_REGISTRY)
