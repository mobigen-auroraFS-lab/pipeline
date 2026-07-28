"""F-3.1 텍스트/문서 추출 함수(디스패처가 호출).

기존 추출/요약/임베딩 함수를 재사용해 ``AssetRecord`` 로 매핑한다
(구 monolith ``run_extract_meta.py`` 의 텍스트 분기와 동일 로직, 출력만 AssetRecord).
"""

from __future__ import annotations

from processing.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
from processing.skills.meta_split import split_core_ext
from src.config.settings import active_embed_channel, active_embed_model, get_current_settings


def _extract_text_meta(ctx: ExtractContext) -> AssetRecord:
    """텍스트 자산의 추출 슬롯 구현 — 레지스트리에 이 이름으로 등록된다."""
    # 무거운 import(추출/요약)는 함수 내부 — 디스패처 import 시 미로딩.
    # 모든 LLM 은 설정된 단일 온프레미스 엔드포인트를 사용한다(외부 LLM 미사용).
    from processing.extractors.text_meta_extractor import extract_text_meta
    from src.llm.text_summarizer import summarize_and_extract_keywords

    cfg = ctx.settings or get_current_settings()
    file_kind = ctx.modality
    meta = extract_text_meta(
        file_path=ctx.file_path,
        file_kind=file_kind,
        encoding=cfg.encoding,
        chunk_size=cfg.embed.chunk_size,
        # 토큰 수는 **활성 임베딩 모델**(active_embed_model)의 토크나이저로 센다 — _embed_text 와 동일
        # 모델이라 정합. ⚠️ 설정에서 모델명을 직접 읽으면 채널을 바꿨을 때 라벨만 옛 모델로 남는다.
        embedding_model_name=active_embed_model(cfg),
    )
    # | 연산자로 dict 머지 — 오른쪽(요약)이 같은 키를 덮어쓴다.
    meta = meta | summarize_and_extract_keywords(file_path=ctx.file_path, file_kind=file_kind)
    core_meta, ext_meta = split_core_ext(meta)
    # embeddings=[] — embed 슬롯은 _embed_text 가 별도로 채운다(분리 설계).
    return AssetRecord(core_meta=core_meta, ext_meta=ext_meta, tags=[], embeddings=[])


def _embed_text(ctx: ExtractContext, rec: AssetRecord) -> list[EmbeddingItem]:
    """텍스트 문서를 청크 단위로 임베딩해 EmbeddingItem 목록을 반환한다.

    다른 모달리티와 달리 **추출 단계의 중간 산출물을 넘겨받지 않는다** — 텍스트는 파일에서
    바로 다시 쪼개도 같은 결과가 나오고, 추출 단계에 공유할 계산이 없다.

    어느 채널·어느 모델을 쓸지는 활성 임베딩 프로파일 하나가 정한다 — 적재·검색·관계가
    같은 출처를 봐야 질의와 저장이 같은 공간에서 만난다.

    Args:
        ctx: 처리 문맥(파일 경로·설정).
        rec: 추출 레코드. **이 함수는 읽지 않는다** — 슬롯 계약을 맞추려고 받는 인자다
            (텍스트는 파일에서 직접 청크를 만든다).

    Returns:
        청크마다 한 항목. 청크 순번은 0부터이며, 채널과 함께 청크를 식별한다.
    """
    from src.embedders.text_embedder import embedding_text_chunks

    cfg = ctx.settings or get_current_settings()
    channel = active_embed_channel(cfg)
    model = active_embed_model(cfg)
    chunks = embedding_text_chunks(
        ctx.file_path,
        file_kind=ctx.modality,
        encoding=cfg.encoding,
        chunk_size=cfg.embed.chunk_size,
        embedding_model_name=model,
        normalize_embeddings=cfg.embed.normalize,
        channel=channel,   # 채널이 로컬 모델이냐 원격 API 냐를 정한다.
        settings=cfg,
    )
    return [
        EmbeddingItem(
            channel=channel,
            vector=c["embedding_vector"],
            model_name=model,
            chunk_index=int(c["chunk_index"]),
        )
        for c in chunks
    ]
