"""v2 기본 전략을 DEFAULT_REGISTRY 에 등록(import 시 부수효과).

**등록 패턴**: 이 모듈을 import 하는 순간 register_defaults(DEFAULT_REGISTRY) 가 실행된다.
run_ingest/run_relations 진입부가 builtins 를 import 함으로써 레지스트리를 초기화한다.
테스트에서는 별도 StrategyRegistry 에 register_defaults 를 호출해 격리할 수 있다.

태그 규약: 'onprem_llm'=온프레미스 LLM 사용, 'deterministic'=결정적.
by_modality 는 외부 LLM 을 쓰지 않으므로 'external_llm' 태그가 없다(의료 정책 통과).
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
    """ClassifyStage 어댑터 — cascade.classify(file_path, modality) 를 ctx 기반으로 감쌈.

    **교차 참조**: cascade.classify 는 도메인-불가지 cascade 엔진(src/classify/domains/ 하위
    DomainProfile 레지스트리 기반)으로, 새 도메인 추가 시 이 어댑터 코드는 수정하지 않아도 된다.
    """
    return cascade.classify(ctx.file_path, ctx.modality)


def register_defaults(registry: StrategyRegistry) -> None:
    """per-asset 및 cross-asset 기본 전략을 레지스트리에 등록한다.

    **per-asset 전략**:
    - classify/cascade_v1  : 도메인-불가지 cascade 분류기(onprem_llm).
    - extract/by_modality  : 모달리티(텍스트·이미지·영상·오디오)별 메타 추출(onprem_llm).
    - embed/by_modality    : 모달리티별 임베딩(SentenceTransformer/CLIP; deterministic).
    - persist/asset_upsert : asset + 관련 테이블 upsert(태그 없음 — DB IO 만).

    **cross-asset 전략** (현재 run_relations 에서 슬롯별 resolve 하지 않고 묶음 위임):
    - candidates/embedding_topk : 코사인 유사도 top-k 후보 선별(deterministic).
    - score/llm_propose         : LLM JSON 엣지 생성(onprem_llm). propose_edges_json 으로 매핑.
    - persist_edges/graph_upsert: graph_edge + 카탈로그 upsert.
    - decide/confidence         : propose_relations_for_asset 내부 auto_approve 임계로 처리되므로
                                  별도 Callable 미등록(슬롯 이름만 팩에 선언).

    단계 D(3년차 이연·2026-07-06): 의료 전용 cross_asset 전략(예: blocking_5keys)이 추가될 때 이 함수에서 함께 등록한다.
    그 시점에 run_relations 는 팩별 슬롯 resolve 로 전환해야 한다(현재 묶음 위임과의 전환점).
    """
    # per-asset
    registry.register("classify", "cascade_v1", _classify_cascade_v1, tags={"onprem_llm"})
    registry.register("extract", "by_modality", dispatch_extract_meta, tags={"onprem_llm"})
    registry.register("embed", "by_modality", dispatch_embed, tags={"deterministic"})
    registry.register("persist", "asset_upsert", finalize_asset)

    # cross-asset(관계) 전략 — 팩 seam 의 resolve 대상으로 등록만 한다.
    # 현재 run_relations 는 propose_relations_for_asset 로 묶음 위임하고, 슬롯별 개별 resolve 는
    # 단계 D 의료 전략 분기에서 도입한다. 'decide'(confidence) 는 propose 내부 auto_approve 임계로 처리.
    registry.register("candidates", "embedding_topk", find_embedding_candidates, tags={"deterministic"})
    registry.register("score", "llm_propose", propose_edges_json, tags={"onprem_llm"})
    registry.register("persist_edges", "graph_upsert", sync_graph_edges)

    # 샘플 도메인 cross-asset 전략(spec 016) — 결정적·무LLM 데모. 008 슬롯 resolve seam +
    # contracts.py cross_asset 계약 4종을 일반과 다른 전략으로 처음 배선해 제네릭 러너로 실행한다.
    # persist_edges 의 등록명('sample_graph_upsert')과 함수명(sample_persist_edges)이 다른 점 주의 —
    # 팩의 슬롯명(SAMPLE_PACK.cross_asset['persist_edges']='sample_graph_upsert')과 일치시킨 것이다.
    registry.register("candidates", "sample_candidates", sample_candidates, tags={"deterministic"})
    registry.register("score", "sample_score", sample_score, tags={"deterministic"})
    registry.register("decide", "sample_decide", sample_decide, tags={"deterministic"})
    registry.register("persist_edges", "sample_graph_upsert", sample_persist_edges)


register_defaults(DEFAULT_REGISTRY)
