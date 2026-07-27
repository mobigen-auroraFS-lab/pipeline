"""VLM 산출 메타를 SentenceTransformer 임베딩 입력용 텍스트로 만든다."""

from __future__ import annotations

from typing import Any

from src.embedders.text_embedding_normalize import normalize_text_for_embedding


def build_image_vlm_text_for_embedding(meta: dict[str, Any]) -> str:
    """그림 설명·키워드·라벨을 한 덩어리 텍스트로 묶는다(순수 함수).

    Args:
        meta: 추출 메타. **키가 없거나 형태가 달라도 죽지 않는다** — 라벨은 dict 와 문자열
            둘 다 받고, 없는 항목은 조용히 건너뛴다.

    Returns:
        정규화된 텍스트. **쓸 내용이 없으면 공백 한 칸**을 돌려준다 — 빈 문자열을 임베딩에
        넣으면 실패하므로, 저장은 성공시키고 벡터가 무의미해지는 쪽을 택한다.
    """
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
