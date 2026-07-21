"""F-3.4 디스패처 — 단순 if/elif 분기.

레지스트리·Protocol·데코레이터·훅 없이 ``modality`` 기준으로 적절한 추출 함수를 호출한다.
출력은 항상 ``AssetRecord`` 로 통일되어 영속화가 단일 경로로 처리한다.
새 modality 는 분기 한 줄 추가. 확장이 커지면 그때 레지스트리로 리팩터한다.

**extract/embed 분리 이유**: 미디어 skill 은 CLIP 벡터·STT 텍스트·키프레임 같은 고비용 중간 산출물을
``ExtractContext.scratch`` 에 남긴다. ``dispatch_embed`` 는 같은 ctx 를 받아 scratch 에서 꺼내
재계산 없이 임베딩에 재사용한다. 두 함수는 항상 같은 ctx 인스턴스로 쌍으로 호출해야 한다.

의료 도메인(``ctx.domain == 'medical'``)도 현 단계에서는 modality 기준으로 동일 분기한다.
의료 표준 포맷(DICOM/HL7/FHIR) 전용 추출은 후속(단계 D·3년차 이연·2026-07-06)에서 분기를 추가한다.
"""

from __future__ import annotations

from processing.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext
from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind
from processing.skills.audio_skill import _embed_audio, _extract_audio_meta
from processing.skills.image_skill import _embed_image, _extract_image_meta
from processing.skills.text_skill import _embed_text, _extract_text_meta
from processing.skills.video_skill import _embed_video, _extract_video_meta


class UnsupportedModalityError(ValueError):
    """디스패처가 처리할 수 없는 modality."""


def dispatch_extract_meta(ctx: ExtractContext) -> AssetRecord:
    """``ctx.modality`` 에 맞는 메타 추출 함수(_extract_*_meta)를 호출(임베딩 제외)."""
    modality = ctx.modality
    if modality in ALLOWED_TEXT_META_FILE_KINDS:
        return _extract_text_meta(ctx)
    if modality == MediaKind.IMAGE.value:
        return _extract_image_meta(ctx)
    if modality == MediaKind.VIDEO.value:
        return _extract_video_meta(ctx)
    if modality == MediaKind.AUDIO.value:
        return _extract_audio_meta(ctx)
    raise UnsupportedModalityError(f"지원하지 않는 modality: {modality!r}")


def dispatch_embed(ctx: ExtractContext, rec: AssetRecord) -> list[EmbeddingItem]:
    """``ctx.modality`` 에 맞는 임베딩 함수(_embed_*)를 호출."""
    modality = ctx.modality
    if modality in ALLOWED_TEXT_META_FILE_KINDS:
        return _embed_text(ctx, rec)
    if modality == MediaKind.IMAGE.value:
        return _embed_image(ctx, rec)
    if modality == MediaKind.VIDEO.value:
        return _embed_video(ctx, rec)
    if modality == MediaKind.AUDIO.value:
        return _embed_audio(ctx, rec)
    raise UnsupportedModalityError(f"지원하지 않는 modality: {modality!r}")
