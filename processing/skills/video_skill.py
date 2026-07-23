"""F-3.2 영상 추출 함수(디스패처가 호출).

구 monolith ``run_extract_meta.py`` 의 영상 분기를 이식(키프레임 → VLM 요약 → 키프레임별 ST/CLIP 임베딩 쌍).
출력만 ``AssetRecord``. 무거운 import(scenedetect/CLIP/VLM)는 함수 내부에 둔다.
"""

from __future__ import annotations

import logging

from src.config.settings import active_embed_channel, active_embed_model, get_current_settings
from processing.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
from processing.skills.meta_split import split_core_ext

_CHANNEL_CLIP = "clip"
_LOG = logging.getLogger("meta_extract.video_skill")


def _extract_video_meta(ctx: ExtractContext) -> AssetRecord:
    """영상 메타데이터를 추출하고 키프레임 임베딩 입력을 scratch 에 저장한다.

    처리 흐름:
    1) 대표 키프레임 JPEG 추출(scenedetect) → 2) 키프레임별 VLM 캡션·객체 추출 →
    3) 영상 전체 요약(video_summarizer) → 4) 키프레임별 CLIP 임베딩·제로샷 라벨 →
    5) 라벨 score 필터 + top-k → 6) ctx.scratch["keyframes"] 에 임베딩 입력 stash.

    jpeg_bytes 는 CLIP/VLM 계산에만 쓰이며, meta 와 stash 에서는 제거한다(메모리 절약).
    ``clip_image_embedding`` 은 meta 의 keyframes 항목에서도 제거하고 stash 에만 보존한다.
    계약: _embed_video 는 반드시 같은 ctx 로 이 함수 실행 후 호출되어야 한다.
    """
    from src.embedders.video_embedder import embed_video_keyframes_clip
    from src.llm.image_summarizer import summarize_image_caption_keywords_objects_from_jpeg_bytes
    from src.llm.video_summarizer import summarize_video_from_scene_results
    from processing.preprocess.keyframe_dedup import KeyframeDedupConfig
    from processing.preprocess.video_keyframes import (
        extract_video_basic_meta,
        extract_video_representative_frame_bytes,
    )

    cfg = ctx.settings or get_current_settings()
    file = ctx.file_path

    # 048: 키프레임 추출 직후·VLM 루프 이전 near-dup 제거(설정 단일 출처 주입).
    # enabled=False 면 extract 가 현행 경로 그대로 → 결과 바이트 동일(FR-103).
    dedup_config = KeyframeDedupConfig(
        enabled=cfg.video.dedup_enabled,
        hash_max=cfg.video.dedup_hash_max,
        ssim_min=cfg.video.dedup_ssim_min,
        ssim_gray_lo=cfg.video.dedup_ssim_gray_lo,
        hist_min=cfg.video.dedup_hist_min,
        compare_mode=cfg.video.dedup_compare_mode,
        recent_window=cfg.video.dedup_recent_window,
    )
    frame_items = extract_video_representative_frame_bytes(
        video_path=file, max_frames=cfg.video.max_keyframes, dedup=dedup_config
    )
    # 069 B9(P2-10): 키프레임 0장 관측만(현행 동작 유지 — failed 전환·zero-vector 통일 금지).
    # 코덱 미지원·손상 등으로 대표 프레임을 못 뽑으면 이 영상은 시각 임베딩·키프레임 라벨이 비어
    # 검색에서 사실상 누락되는데, 지금까지 조용했다. 근본 처방(ffmpeg 폴백)은 064 몫이고 여기선
    # 관측 공백만 메운다 — 상태·임베딩 계약은 그대로 두어 US-B "국소" 원칙을 지킨다.
    if not frame_items:
        _LOG.warning("키프레임 0장 — 시각 임베딩·라벨 없이 진행(현행 유지·관측): video=%s", file)
    korean_labels_per_frame: list[list[str]] = []
    result: list[dict] = []
    for item in frame_items:
        summ = summarize_image_caption_keywords_objects_from_jpeg_bytes(item["jpeg_bytes"])
        objects = summ.get("objects") or []
        # VLM objects → CLIP 제로샷 후보 레이블(이미지 skill 과 동일한 패턴)
        obj_list = [str(o) for o in objects] if isinstance(objects, list) else []
        korean_labels_per_frame.append(obj_list)
        result.append(
            {
                "scene_index": item["scene_index"],
                "start_sec": item["start_sec"],
                "end_sec": item["end_sec"],
                "frame_sec": item["frame_sec"],
                "jpeg_bytes": item["jpeg_bytes"],
                "summary": summ,
            }
        )

    meta = extract_video_basic_meta(file_path=file)
    meta = meta | summarize_video_from_scene_results(result)
    clip_ve = embed_video_keyframes_clip(result, korean_labels_per_frame=korean_labels_per_frame)

    # 키프레임 라벨 score 하한 + top-k
    for kf in clip_ve.get("keyframes") or []:
        labels_all = kf.get("labels") or []
        kf["labels"] = [
            it for it in labels_all if float(it.get("score") or 0.0) >= cfg.vlm.labels_score_min
        ][: cfg.video.labels_meta_top_k]
    # jpeg_bytes 는 이후 불필요 — result 에서 제거해 메모리를 돌려준다.
    for _it in result:
        _it.pop("jpeg_bytes", None)

    # clip_image_embedding 은 DB 에 직접 저장하지 않고 stash 를 통해 embed 슬롯으로 전달한다.
    meta["keyframes"] = [
        {k: v for k, v in kf.items() if k != "clip_image_embedding"} for kf in clip_ve["keyframes"]
    ]

    # 키프레임별 임베딩 입력을 stash(키프레임/CLIP/VLM 재실행 방지)
    # stash 순서 = clip_ve["keyframes"] 순서 = scene 순서 → _embed_video 가 chunk_index 로 활용.
    stash: list[dict] = []
    for kf in clip_ve["keyframes"]:
        summ = kf.get("summary") if isinstance(kf.get("summary"), dict) else {}
        stash.append(
            {
                "clip_vec": kf["clip_image_embedding"],
                "summary": str(summ.get("summary", "") or ""),
                "keywords": summ["keywords"] if isinstance(summ.get("keywords"), list) else [],
                "labels": kf.get("labels") or [],
            }
        )
    ctx.scratch["keyframes"] = stash

    core_meta, ext_meta = split_core_ext(meta)
    return AssetRecord(core_meta=core_meta, ext_meta=ext_meta, tags=[], embeddings=[])


def _embed_video(ctx: ExtractContext, rec: AssetRecord) -> list[EmbeddingItem]:
    """키프레임별 임베딩을 생성해 반환한다 — 기본 ST/CLIP 쌍, 063 ``embed_enable_clip=False`` 시 ST만.

    키프레임 n 개 → EmbeddingItem 2n 개(ST·CLIP 쌍; clip off 시 n 개=ST만). chunk_index 는 키프레임 순번(0-based).
    같은 chunk_index 를 공유하는 ST/CLIP 쌍이 하이브리드 검색에서 동일 시점 프레임을 나타낸다.
    텍스트 채널·모델은 활성 임베딩 프로파일(018)로 결정한다(기본 active='st'·KoSimCSE → 회귀 0).
    CLIP 벡터는 ctx.scratch["keyframes"] 에서 꺼내므로 CLIP 추론을 재실행하지 않는다(시각 채널은 무변경).
    계약 위반(extract 없이 단독 호출) 시 RuntimeError 로 즉시 탐지된다.
    """
    from src.config.embedding_constants import DEFAULT_CLIP_MODEL_NAME
    from src.embedders.text_embedder import embed_texts_for, pad_embedding_to_storage_dim
    from processing.preprocess.vlm_text_for_embedding import build_image_vlm_text_for_embedding

    cfg = ctx.settings or get_current_settings()
    channel = active_embed_channel(cfg)
    model = active_embed_model(cfg)  # EmbeddingItem.model_name 라벨(채널의 모델명)
    # 계약 위반 즉시 탐지: extract 없이 embed 만 단독 호출하면 RuntimeError.
    keyframes = ctx.scratch.get("keyframes")
    if keyframes is None:
        raise RuntimeError("_embed_video: ctx.scratch['keyframes'] 없음 — _extract_video_meta 를 같은 ctx 로 먼저 실행해야 합니다.")
    embeddings: list[EmbeddingItem] = []
    for i, kf in enumerate(keyframes):
        frame_meta = {"summary": kf["summary"], "keywords": kf["keywords"], "labels": kf["labels"]}
        chunk_content = build_image_vlm_text_for_embedding(frame_meta)
        if not chunk_content.strip():
            chunk_content = " "
        # 062: 키프레임 VLM 캡션 ST 임베딩도 채널 백엔드(로컬/API)로 라우팅(적재=질의 정합·st_api).
        st_raw = embed_texts_for(
            [chunk_content],
            channel=channel,
            settings=cfg,
            normalize_embeddings=cfg.embed.normalize,
        )[0]
        st_vec = pad_embedding_to_storage_dim(st_raw)
        # 키프레임당 ST(+CLIP) 항목(같은 chunk_index, 채널로 구분)
        embeddings.append(EmbeddingItem(channel=channel, vector=st_vec, model_name=model, chunk_index=i))
        # 063: clip 임베딩 토글(기본 True=기존 동치). off면 키프레임 clip 항목만 스킵.
        if cfg.embed.enable_clip:
            embeddings.append(EmbeddingItem(
                channel=_CHANNEL_CLIP, vector=kf["clip_vec"],
                model_name=DEFAULT_CLIP_MODEL_NAME, chunk_index=i,
            ))
    return embeddings
