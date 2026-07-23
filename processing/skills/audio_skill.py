"""F-3.2 오디오 추출 함수(디스패처가 호출).

구 monolith ``run_extract_meta.py`` 의 오디오 분기를 이식(STT → 요약 → 청크 임베딩). 출력만 ``AssetRecord``.
무거운 import(faster-whisper/임베더)는 함수 내부에 둔다.
"""

from __future__ import annotations

from src.config.settings import active_embed_channel, active_embed_model, get_current_settings
from processing.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
from processing.skills.meta_split import split_core_ext


def _extract_audio_meta(ctx: ExtractContext) -> AssetRecord:
    """오디오 파일의 메타데이터를 추출하고 STT 전사 텍스트를 scratch 에 저장한다.

    처리 순서: STT(faster-whisper) → 오디오 속성 메타 → LLM 요약·키워드.
    STT 결과는 요약 LLM 입력으로 쓰이고, 동시에 ctx.scratch["stt_text"] 에 보존해
    _embed_audio 가 whisper 를 재실행하지 않고 청크 임베딩에 재사용한다.
    계약: _embed_audio 는 반드시 같은 ctx 로 이 함수 실행 후 호출되어야 한다.
    """
    from processing.extractors.audio_meta_extractor import extract_audio_meta
    from src.llm.text_summarizer import summarize_and_extract_keywords_from_audio
    from processing.preprocess.stt import transcribe_audio_local

    file = ctx.file_path
    stt_result = transcribe_audio_local(file_path=file)
    meta = extract_audio_meta(file_path=file)
    # 무내용 가드(spec 065 FR-701): STT 전사가 비었거나 얇으면 요약기가 LLM 을 호출하지 않고
    # summary='' 를 돌려준다(기악 오디오 등 placeholder 요약 원천 차단). 빈 summary → 자기주제 미부여.
    meta = meta | summarize_and_extract_keywords_from_audio(text=stt_result["text"])

    ctx.scratch["stt_text"] = stt_result["text"]  # 임베딩 슬롯 재사용(whisper 재실행 방지)

    core_meta, ext_meta = split_core_ext(meta)
    return AssetRecord(core_meta=core_meta, ext_meta=ext_meta, tags=[], embeddings=[])


def _embed_audio(ctx: ExtractContext, rec: AssetRecord) -> list[EmbeddingItem]:
    """STT 전사 텍스트를 청크 단위로 임베딩해 EmbeddingItem 목록을 반환한다.

    텍스트 skill 과 동일한 ST(SentenceTransformer) 채널 단일 임베딩 방식이다.
    채널·모델은 활성 임베딩 프로파일(018)로 결정한다(기본 active='st'·KoSimCSE → 회귀 0).
    CLIP 채널이 없는 이유: 오디오는 시각 정보가 없으므로 이미지/영상과 달리 ST 만 생성한다.
    STT 전사 텍스트는 ctx.scratch["stt_text"] 에서 꺼내므로 whisper 를 재실행하지 않는다.
    계약 위반(extract 없이 단독 호출) 시 RuntimeError 로 즉시 탐지된다.
    """
    from src.embedders.text_embedder import embedding_plain_text_chunks

    cfg = ctx.settings or get_current_settings()
    channel = active_embed_channel(cfg)
    model = active_embed_model(cfg)
    # 계약 위반 즉시 탐지: extract 없이 embed 만 단독 호출하면 RuntimeError.
    stt_text = ctx.scratch.get("stt_text")
    if stt_text is None:
        raise RuntimeError("_embed_audio: ctx.scratch['stt_text'] 없음 — _extract_audio_meta 를 같은 ctx 로 먼저 실행해야 합니다.")
    chunks = embedding_plain_text_chunks(
        stt_text,
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
