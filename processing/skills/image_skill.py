"""F-3.2 이미지 추출 함수(디스패처가 호출).

구 monolith ``run_extract_meta.py`` 의 이미지 분기를 이식. 출력만 ``AssetRecord``.
무거운 import(torch/CLIP/VLM)는 함수 내부에 둔다 — 텍스트 전용 실행 시 미로딩.
"""

from __future__ import annotations

from processing.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
from processing.skills.meta_split import split_core_ext
from processing.skills.title_context import summarizer_kwargs, title_for_summary
from src.config.settings import active_embed_channel, active_embed_model, get_current_settings

_CHANNEL_CLIP = "clip"


def _extract_image_meta(ctx: ExtractContext) -> AssetRecord:
    """이미지 파일의 메타데이터를 추출하고 CLIP 벡터를 scratch 에 저장한다.

    처리 순서:
    1) 이미지 속성 메타(width·height·format — `extract_image_meta`. EXIF 는 추출하지 않는다·2026-07-15 P3-11)
    2) VLM 캡션·키워드·객체  3) CLIP 제로샷 라벨
    objects(VLM 출력)는 CLIP 제로샷의 한국어 후보 레이블로 활용된 뒤 최종 메타에서 제거된다.
    CLIP 이미지 벡터를 ``ctx.scratch["clip_vec"]`` 에 저장해 _embed_image 에서 재사용한다
    — CLIP 추론을 두 번 실행하지 않기 위한 핸드오프 계약이다.
    """
    from processing.extractors.image_meta_extractor import extract_image_meta
    from src.embedders.image_embedder import (
        clip_zero_shot_ko_meta_items,
        zero_shot_tag_image_korean_clip,
    )
    from src.llm.image_summarizer import summarize_image_caption_keywords_objects

    cfg = ctx.settings or get_current_settings()
    file = ctx.file_path

    # 1) 파일·이미지 속성 메타  2) VLM 캡션·키워드·객체
    meta = extract_image_meta(file_path=file)
    # 제목 참고(spec 114) — 같은 호출에서 제목이 그림과 맞는지(title_match)도 판단한다.
    summary = summarize_image_caption_keywords_objects(file_path=file, **summarizer_kwargs(title_for_summary(file, cfg)))
    objects = summary.get("objects") or []
    meta = meta | summary

    # 3) CLIP 이미지 임베딩 + 한글 라벨 제로샷
    # VLM 이 추출한 objects 를 CLIP 제로샷 후보로 전달 — 빈 리스트면 제로샷 미수행.
    obj_list = [str(o) for o in objects] if isinstance(objects, list) else []
    zs = zero_shot_tag_image_korean_clip(file_path=file, korean_labels=obj_list)
    if zs["label_scores"]:
        labels_all = clip_zero_shot_ko_meta_items(zs["label_scores"])
        # score 하한 필터 + top-k — 설정값 초과 레이블은 메타에 포함하지 않는다.
        labels = [
            it for it in labels_all if float(it.get("score") or 0.0) >= cfg.vlm.labels_score_min
        ][: cfg.vlm.image_labels_meta_top_k]
        meta = meta | {"labels": labels}

    meta.pop("objects", None)  # objects 는 CLIP 후보용일 뿐 최종 meta 에서 제외

    # 계약: _embed_image 는 반드시 같은 ctx 로 이 함수 실행 후 호출되어야 한다.
    ctx.scratch["clip_vec"] = zs["clip_image_embedding"]

    core_meta, ext_meta = split_core_ext(meta)
    return AssetRecord(core_meta=core_meta, ext_meta=ext_meta, tags=[], embeddings=[])


def image_text_embedding(meta: dict, cfg) -> EmbeddingItem:
    """그림 설명 · 키워드 · 라벨을 글로 묶어 텍스트 채널 임베딩 하나를 만든다.

    적재(``_embed_image``)와 재요약 배치(``run_resummarize`` — 사진 요약을 다시 만들면 이 벡터도 다시 만들어야
    한다)가 **같은 규칙**을 쓰도록 한 곳에 둔다. 규칙이 갈리면 같은 사진이 적재 시점에 따라 다른 벡터를 갖는다.

    Args:
        meta: 요약 · 키워드 · 라벨이 든 메타(키가 없어도 된다).
        cfg: 설정 — 활성 임베딩 채널 · 모델 · 정규화 여부를 여기서 읽는다.

    Returns:
        활성 채널의 임베딩 항목(``chunk_index=0`` — 사진은 단일 청크). 설명이 텅 비면 공백 한 칸으로 만든다
        — 저장은 성공해야 하기 때문이다.
    """
    from processing.preprocess.vlm_text_for_embedding import build_image_vlm_text_for_embedding
    from src.embedders.text_embedder import embed_texts_for, pad_embedding_to_storage_dim

    channel = active_embed_channel(cfg)
    model = active_embed_model(cfg)  # EmbeddingItem.model_name 라벨(채널의 모델명)
    chunk_content = build_image_vlm_text_for_embedding(meta)
    if not chunk_content.strip():
        chunk_content = " "
    # 캡션 임베딩도 채널이 정한 백엔드(로컬 모델/원격 API)로 보낸다 — 적재와 질의가 같은 모델을 써야
    #   (텍스트·오디오와 동일). 어느 모델을 쓸지는 임베딩 함수가 채널로 해소한다.
    st_raw = embed_texts_for(
        [chunk_content],
        channel=channel,
        settings=cfg,
        normalize_embeddings=cfg.embed.normalize,
    )[0]
    # chunk_index=0: 이미지는 단일 청크(비텍스트 미디어 공통).
    return EmbeddingItem(channel=channel, vector=pad_embedding_to_storage_dim(st_raw), model_name=model, chunk_index=0)


def _embed_image(ctx: ExtractContext, rec: AssetRecord) -> list[EmbeddingItem]:
    """이미지 임베딩을 만든다 — 기본은 텍스트·시각 2채널, 시각 채널을 끄면 텍스트 하나만.

    두 채널을 만든다: 그림 설명을 **글로 옮겨** 만든 텍스트 벡터와, 그림 자체의 시각 벡터.
    글로 옮긴 쪽이 있어야 "김치 담그는 사진" 같은 한국어 질의가 걸린다.

    Args:
        ctx: 처리 문맥. ⚠️ **추출 단계가 남긴 시각 벡터가 실려 있어야 한다** — 무거운 시각
            모델을 두 번 돌리지 않기 위해 넘겨받는 구조이고, 없으면 예외로 즉시 알린다.
        rec: 추출 레코드. 여기서 캡션·키워드·라벨을 읽어 텍스트 채널 입력을 만든다.

    Returns:
        임베딩 항목. 시각 채널을 끄면 텍스트 하나만 나온다(라벨·계약은 그대로).
        설명이 텅 비면 공백 한 칸을 넣는다 — 저장은 성공해야 하기 때문이다.

    Raises:
        RuntimeError: 같은 문맥으로 추출을 먼저 돌리지 않았을 때.
    """
    from src.config.embedding_constants import DEFAULT_CLIP_MODEL_NAME

    cfg = ctx.settings or get_current_settings()
    text_item = image_text_embedding(dict(rec.core_meta) | dict(rec.ext_meta), cfg)
    # 계약 위반 즉시 탐지: extract 없이 embed 만 단독 호출하면 RuntimeError.
    clip_vec = ctx.scratch.get("clip_vec")
    if clip_vec is None:
        raise RuntimeError("_embed_image: ctx.scratch['clip_vec'] 없음 — _extract_image_meta 를 같은 ctx 로 먼저 실행해야 합니다.")
    items = [text_item]
    # 시각 채널 토글. 끄면 그 항목만 빠지고 나머지 계약은 그대로다.
    if cfg.embed.enable_clip:
        items.append(
            EmbeddingItem(channel=_CHANNEL_CLIP, vector=clip_vec, model_name=DEFAULT_CLIP_MODEL_NAME, chunk_index=0)
        )
    return items
