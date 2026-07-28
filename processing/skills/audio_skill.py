"""F-3.2 오디오 추출 함수(디스패처가 호출).

구 monolith ``run_extract_meta.py`` 의 오디오 분기를 이식(STT → 요약 → 청크 임베딩). 출력만 ``AssetRecord``.
무거운 import(faster-whisper/임베더)는 함수 내부에 둔다.
"""

from __future__ import annotations

from processing.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
from processing.skills.meta_split import split_core_ext
from src.config.settings import active_embed_channel, active_embed_model, get_current_settings


def _extract_audio_meta(ctx: ExtractContext) -> AssetRecord:
    """오디오 파일의 메타데이터를 추출하고 STT 전사 텍스트를 scratch 에 저장한다.

    처리 순서: STT(faster-whisper) → 오디오 속성 메타 → LLM 요약·키워드.
    STT 결과는 요약 LLM 입력으로 쓰이고, 동시에 ctx.scratch["stt_text"] 에 보존해
    _embed_audio 가 whisper 를 재실행하지 않고 청크 임베딩에 재사용한다.
    계약: _embed_audio 는 반드시 같은 ctx 로 이 함수 실행 후 호출되어야 한다.
    """
    from processing.extractors.audio_meta_extractor import extract_audio_meta
    from processing.preprocess.stt import transcribe_audio_local
    from src.llm.text_summarizer import summarize_and_extract_keywords_from_audio

    file = ctx.file_path
    stt_result = transcribe_audio_local(file_path=file)
    meta = extract_audio_meta(file_path=file)
    # 무내용 가드: 전사가 비었거나 너무 짧으면 요약기가 LLM 을 부르지 않고
    # summary='' 를 돌려준다(기악 오디오 등 placeholder 요약 원천 차단). 빈 summary → 자기주제 미부여.
    meta = meta | summarize_and_extract_keywords_from_audio(text=stt_result["text"])

    ctx.scratch["stt_text"] = stt_result["text"]  # 임베딩 슬롯 재사용(whisper 재실행 방지)

    core_meta, ext_meta = split_core_ext(meta)
    return AssetRecord(core_meta=core_meta, ext_meta=ext_meta, tags=[], embeddings=[])


def _embed_audio(ctx: ExtractContext, rec: AssetRecord) -> list[EmbeddingItem]:
    """STT 전사 텍스트를 청크 단위로 임베딩해 EmbeddingItem 목록을 반환한다.

    시각 채널을 만들지 않는다 — 오디오에는 볼 것이 없다. 텍스트 채널 하나만 쓴다.

    Args:
        ctx: 처리 문맥. ⚠️ **추출 단계가 남긴 전사 텍스트가 실려 있어야 한다** — 음성
            인식을 다시 돌리지 않기 위해 넘겨받는 구조이고, 없으면 예외로 즉시 알린다.
        rec: 추출 레코드. 이 함수는 읽지 않는다(슬롯 계약을 맞추려는 인자).

    Returns:
        청크마다 한 항목.

    Raises:
        RuntimeError: 같은 문맥으로 추출을 먼저 돌리지 않았을 때.
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
