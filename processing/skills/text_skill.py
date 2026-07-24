"""F-3.1 텍스트/문서 추출 함수(디스패처가 호출).

기존 추출/요약/임베딩 함수를 재사용해 ``AssetRecord`` 로 매핑한다
(구 monolith ``run_extract_meta.py`` 의 텍스트 분기와 동일 로직, 출력만 AssetRecord).
"""

from __future__ import annotations

from src.config.settings import active_embed_channel, active_embed_model, get_current_settings
from processing.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
from processing.skills.meta_split import split_core_ext


def _extract_text_meta(ctx: ExtractContext) -> AssetRecord:
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
        # 모델이라 정합(종전 cfg.embed.model=KoSimCSE 하드코딩은 st_api=bge-m3 와 불일치했음).
        embedding_model_name=active_embed_model(cfg),
    )
    # | 연산자로 dict 머지 — 오른쪽(요약)이 같은 키를 덮어쓴다.
    meta = meta | summarize_and_extract_keywords(file_path=ctx.file_path, file_kind=file_kind)
    core_meta, ext_meta = split_core_ext(meta)
    # embeddings=[] — embed 슬롯은 _embed_text 가 별도로 채운다(분리 설계).
    return AssetRecord(core_meta=core_meta, ext_meta=ext_meta, tags=[], embeddings=[])


def _embed_text(ctx: ExtractContext, rec: AssetRecord) -> list[EmbeddingItem]:
    """텍스트 문서를 청크 단위로 임베딩해 EmbeddingItem 목록을 반환한다.

    텍스트는 미디어(이미지/영상/오디오)와 달리 scratch 핸드오프 없이
    파일에서 직접 청크를 재생성한다 — extract 단계에서 공유 계산이 없기 때문이다.
    chunk_index 는 0-based 순번으로, DB persist 시 활성 채널(기본 'st')과 함께 청크를 식별한다.

    채널·모델은 활성 임베딩 프로파일(018)로 결정한다 — 적재·검색·관계가 공유하는 단일 출처.
    기본 active='st' → channel='st'·KoSimCSE 로 기존 적재와 동치(회귀 0).
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
        channel=channel,   # 062: 채널 백엔드(로컬/API)로 라우팅. 기본 st=로컬(동작 불변).
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
