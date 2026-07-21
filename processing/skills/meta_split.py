"""추출 메타 dict 를 core_meta(파일/시스템) vs ext_meta(도메인 신호)로 무손실 분리.

데이터 마이그레이션(scripts/migrate_media_to_asset.py)과 동일한 키 기준을 공유한다.
"""

from __future__ import annotations

from typing import Any

# 도메인 신호(ext_meta)로 보낼 키. 나머지는 core_meta.
EXT_META_KEYS: frozenset[str] = frozenset(
    {"summary", "keywords", "labels", "objects", "keyframes", "stt", "caption"}
)


def split_core_ext(meta: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """meta 를 (core_meta, ext_meta) 로 분리. 키 합집합은 원본과 동일(무손실)."""
    core: dict[str, Any] = {}
    ext: dict[str, Any] = {}
    for k, v in meta.items():
        (ext if k in EXT_META_KEYS else core)[k] = v
    return core, ext
