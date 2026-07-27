"""F-5.1 분류용 결정적 텍스트 추출 (LLM 없음).

분류 단계는 무거운 LLM(캡션/요약) 없이, 어휘 판정에 쓸 텍스트만 싸게 뽑는다.
- txt/json: 평문 읽기
- pdf/office(word/excel/powerpoint): 텍스트 레이어(`data_loader`)
- image: OCR(Tesseract, kor+eng)
- video/audio/unknown: 분류 단계 텍스트 없음(빈 문자열) → 시그니처/경로/stage3 가 처리
"""

from __future__ import annotations

from pathlib import Path

from src.file.file_type_defs import ALLOWED_TEXT_META_FILE_KINDS, MediaKind

_MAX_CHARS = 4000  # stage2 어휘 스캔용 — stage3 LLM 입력(_MAX_TEXT=4000)과 별도 관리
_PLAIN_KINDS = frozenset({MediaKind.TEXT.value, MediaKind.JSON.value})  # txt, json


def _resolve_encoding() -> str:
    """분류용 읽기 인코딩. 하드코딩 대신 설정 인코딩을 쓴다 —

    extract(``data_loader``)가 ``cfg.encoding`` 으로 읽으므로, 분류 stage2 도 같은 인코딩으로 읽어야
    cp949 문서의 어휘 매칭이 어긋나지 않는다(utf-8 강제 시 글자 깨짐). 설정 미초기화(순수 단위 등)면
    보수적으로 "utf-8" 폴백 — 지연 import 로 미초기화 오염을 막는다.
    """
    try:
        from src.config.settings import get_current_settings

        return get_current_settings().encoding
    except (RuntimeError, AttributeError):
        return "utf-8"


def extract_text_for_classification(file_path: str, modality: str, *, max_chars: int = _MAX_CHARS) -> str:
    """분류 단서로 쓸 텍스트를 뽑는다 — LLM 을 쓰지 않는다.

    여기서 캡션·요약을 만들면 분류 한 건마다 모델을 돌리게 된다. 앞 단계는 **싸고 결정적**
    이어야 하므로 파일에서 바로 읽을 수 있는 것만 쓴다.

    Args:
        file_path: 대상 파일.
        modality: 파일 종류. **영상·오디오·미상은 빈 문자열**을 돌려준다 — 그런 파일은
            시그니처·파일명·마지막 LLM 단계가 맡는다.
        max_chars: 읽을 최대 글자 수. 앞부분만 봐도 도메인 단서는 충분하다.

    Returns:
        추출한 텍스트. **실패는 빈 문자열로 흡수한다** — 글자를 못 읽는 것이 분류 실패로
        번지면 안 된다.
    """
    if modality in ALLOWED_TEXT_META_FILE_KINDS:
        return _document_text(file_path, modality, max_chars)
    if modality == MediaKind.IMAGE.value:
        return _ocr_image(file_path, max_chars)
    return ""


def _document_text(file_path: str, modality: str, max_chars: int) -> str:
    """분류 판단에 쓸 텍스트를 파일에서 뽑는다(앞부분만).

    추출 단계와 **같은 인코딩 규칙**을 쓴다 — 다르면 같은 문서인데 분류와 저장이 서로
    다른 글자를 보게 된다.

    Args:
        file_path: 대상 파일.
        modality: 파일 종류(어떻게 읽을지 정한다).
        max_chars: 읽을 최대 글자 수.

    Returns:
        추출된 텍스트. 읽기에 실패하면 빈 문자열(분류를 멈추지 않는다).
    """
    enc = _resolve_encoding()  # extract 와 동일 인코딩(B10) — cp949 문서 stage2 어휘 매칭 정합.
    if modality in _PLAIN_KINDS:
        try:
            with open(file_path, encoding=enc, errors="ignore") as f:
                return f.read(max_chars)
        except OSError:
            return ""
    # pdf / office → 텍스트 레이어(data_loader 지연 임포트: 무거운 패키지를 분류 불필요 경로에서 로드 방지)
    from src.file.data_loader import iter_document_chunks, normalize_file_kind

    kind = normalize_file_kind(modality)
    if kind is None:
        return ""
    try:
        out: list[str] = []
        total = 0
        for ch in iter_document_chunks(
            Path(file_path), file_kind=kind, encoding=enc,
            chunk_size=max_chars, overlap_size=0, max_input_chars=max_chars,
        ):
            out.append(ch)
            total += len(ch)
            if total >= max_chars:
                break
        return "".join(out)[:max_chars]
    except Exception:  # noqa: BLE001 — 분류용이므로 실패는 빈 텍스트로 흡수
        return ""


def _ocr_image(file_path: str, max_chars: int) -> str:
    """이미지에서 글자를 읽어 낸다(분류 단서용).

    Args:
        file_path: 이미지 경로.
        max_chars: 읽을 최대 글자 수.

    Returns:
        읽어 낸 글자. **어떤 실패도 빈 문자열로 흡수한다** — 글자 인식은 느리고 실패도
        잦은데, 그것이 분류 실패로 번지면 안 된다.
    """
    try:
        import pytesseract
        from PIL import Image

        with Image.open(file_path) as img:
            text = pytesseract.image_to_string(img, lang="kor+eng")
        return (text or "").strip()[:max_chars]
    except Exception:  # noqa: BLE001 — OCR 실패는 빈 텍스트
        return ""
