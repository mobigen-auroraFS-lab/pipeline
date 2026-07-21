"""VLM 산출 메타를 SentenceTransformer 임베딩 입력용 텍스트로 만든다."""

from __future__ import annotations

from typing import Any

from src.embedders.text_embedding_normalize import normalize_text_for_embedding


def build_image_vlm_text_for_embedding(meta: dict[str, Any]) -> str:
    """VLM 요약·키워드·제로샷 라벨을 한 덩어리로 묶어 ST 임베딩 입력에 쓴다."""
    summary_txt = str(meta.get("summary", "") or "").strip()
    kws = meta.get("keywords") or []
    kw_line = (
        " ".join(str(k).strip() for k in kws if str(k).strip())
        if isinstance(kws, list)
        else ""
    )
    lab_parts: list[str] = []
    for item in meta.get("labels") or []:
        if isinstance(item, dict):
            lab = item.get("label")
            if lab:
                lab_parts.append(str(lab).strip())
        elif isinstance(item, str) and item.strip():
            lab_parts.append(item.strip())
    label_line = " ".join(lab_parts)
    parts = [p for p in (summary_txt, kw_line, label_line) if p]
    raw = "\n".join(parts).strip() if parts else " "
    return normalize_text_for_embedding(raw) if raw.strip() else " "
