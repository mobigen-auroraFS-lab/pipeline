"""
음성 파일 → 텍스트(STT). 온프레미스 로컬 추론만 사용한다(외부 API 미호출).

``faster-whisper``(CTranslate2 백엔드)로 Whisper 모델을 로컬에서 돌린다. 기본은 한국어(``ko``),
``small`` 모델, CPU + ``int8`` 양자화이며, 인자로 GPU(``cuda``/``float16``)·모델 크기를 바꿀 수 있다.
``vad_filter=True`` 로 무음 구간을 걸러 환각을 줄이고, 모델이 낸 세그먼트 텍스트를 공백으로 이어
'전체 텍스트'만 돌려준다(타임스탬프·세그먼트 경계는 보존하지 않음).

audio_skill 이 이 전체 텍스트를 받아 요약·임베딩(media_chunks 의 STT 텍스트 청크)으로 잇는다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from faster_whisper import WhisperModel


class TranscriptionResult(TypedDict):
    """전사 결과 — 세그먼트를 이어 붙인 전체 텍스트만 담는다."""

    text: str


@lru_cache(maxsize=2)
def _get_whisper(model_size: str, device: str, compute_type: str) -> WhisperModel:
    """Whisper 모델 프로세스 캐시(069 P1-5) — 파일마다 재로드(수 GB 가중치·수 초)를 제거한다.

    같은 (model_size, device, compute_type) 조합은 1회만 로드해 재사용한다. maxsize=2 는
    CPU/GPU 조합 전환 여지만 남긴 보수값(배치는 사실상 단일 조합). 추론 전용이라 상태 오염 없음.
    트레이드오프(리뷰 🟡6): 로드된 모델은 명시 해제 API 가 없어 프로세스 종료까지 메모리 상주 —
    배치 프로세스(현 용도)엔 무해하나 장기 상주 서버로 재사용 시 상주 메모리를 감안할 것.
    """
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_audio_local(
    file_path: str | Path,
    *,
    model_size: str = "small",   # Whisper-small
    language: str | None = "ko",
    device: str = "cpu",         # GPU면 "cuda"
    compute_type: str = "int8",  # GPU면 "float16" 권장
) -> TranscriptionResult:
    """로컬 Whisper 로 ``file_path`` 를 전사해 전체 텍스트(``{"text": ...}``)를 반환한다."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    model = _get_whisper(model_size, device, compute_type)
    segments_iter, _info = model.transcribe(
        str(path),
        language=language,
        vad_filter=True,
        beam_size=5,
        # 069 B2(P2-2): faster-whisper 는 temperature 미지정 시 기본 폴백 래더
        # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] 로, 낮은 온도 결과가 compression/logprob 임계에
        # 걸리면 더 높은 온도(샘플링)로 재시도해 비결정 전사가 나온다. temperature=0.0 을
        # 명시해 그 래더를 끄고 재현성을 고정한다(헌법 3조).
        # 트레이드오프: 재시도 폴백이 없어져, 임계에 걸리는 일부 어려운 구간은 빈 세그먼트로
        # 남을 수 있다(환각 대신 누락). 결정성 요구가 우선이므로 이 손실을 수용한다.
        temperature=0.0,
    )
    texts: list[str] = []
    for seg in segments_iter:
        t = (seg.text or "").strip()
        texts.append(t)
    return {"text": " ".join(texts).strip()}
