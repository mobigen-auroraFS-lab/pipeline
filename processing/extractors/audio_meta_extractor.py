"""오디오 파일 속성 메타(길이·샘플레이트·채널) 추출 — ``src/skills/audio_skill.py`` 가 호출.

내용(전사)이 아닌 컨테이너 속성만 본다. STT 전사·요약은 skill 쪽에서 별도로 처리한다.

069 P1-6: ``sf.read``(전체 float 디코드 — 긴 오디오를 통째로 메모리에 올림) 대신 ``sf.info``
(헤더만)로 속성을 읽는다. duration 은 soundfile 이 헤더의 frames/samplerate 로 계산해 주는
값이라 결과 의미 동일·비용은 파일 길이와 무관해진다.
"""

from typing import TypedDict

import soundfile as sf


class AudioMeta(TypedDict):
    # P3-10 정정(069): 실반환값과 타입힌트 일치화 — duration 은 round(...,3)의 float,
    # channels 는 int. (기존 int/str 선언은 실값과 불일치하던 힌트 오류 — 소비처 동작 무영향)
    duration: float
    sample_rate: int
    channels: int


def extract_audio_meta(file_path: str) -> AudioMeta:
    # [확인 필요 사항·리뷰 🟡5] sf.info().duration 은 헤더 frame 수 기반 — WAV/FLAC 등 정적 컨테이너는
    # 실측(sf.read)과 동일하나, 헤더 frame 이 부정확할 수 있는 일부 스트리밍/VBR 포맷에선 미세 차이 가능.
    # 현 파이프라인 오디오는 정적 파일 위주라 실해 없다고 판단(포맷 확장 시 재검토).
    info = sf.info(file_path)  # 헤더만 읽음(전체 디코드 없음 — P1-6)
    return {
        "duration": round(float(info.duration), 3),
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
    }
