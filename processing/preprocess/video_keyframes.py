"""영상 → 장면 단위 대표 키프레임(메모리 JPEG) 추출.

PySceneDetect 의 ``ContentDetector`` 로 컷(장면 전환)을 찾아 각 장면의 중앙 시점 프레임을
OpenCV 로 읽어 JPEG bytes 로 인코딩한다. 파일로 저장하지 않고 메모리에서 바로 image_skill
(CLIP 라벨·VLM 요약)으로 넘겨 영상의 키프레임 임베딩·검색에 쓰기 위한 전처리다.
``extract_video_basic_meta`` 는 별도로 duration/fps/해상도만 뽑는다.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import cv2
from scenedetect import ContentDetector, detect

# 078: KeyframeBytesResult(공유 계약)를 코어(embedders.frame_types)로 승격 — 이 모듈(파이프라인)은
# 그 계약을 생산하고 embedders(코어)가 소비한다. 코어→파이프라인 역참조를 없애려 계약을 코어에 둔다.
from src.embedders.frame_types import KeyframeBytesResult

if TYPE_CHECKING:
    from processing.preprocess.keyframe_dedup import KeyframeDedupConfig

logger = logging.getLogger(__name__)


class VideoBasicMeta(TypedDict):
    duration: float
    frame_rate: float
    width: int
    height: int


def _to_seconds(timecode: object) -> float:
    """타임코드 객체·문자열·숫자를 초 단위 실수로 통일한다(라이브러리마다 형태가 다르다)."""
    if hasattr(timecode, "get_seconds"):
        return float(timecode.get_seconds())  # type: ignore[no-any-return]
    # scenedetect 버전 차이를 고려한 안전장치
    return float(timecode)  # type: ignore[arg-type]


def extract_video_basic_meta(
    file_path: str | Path,
) -> VideoBasicMeta:
    """영상 기본 메타(duration/fps/width/height)를 추출한다."""
    src = Path(file_path)
    if not src.is_file():
        raise FileNotFoundError(str(src))

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {src}")

    try:
        frame_rate = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()

    duration = (frame_count / frame_rate) if frame_rate > 0 else 0.0
    return {
        "duration": round(duration, 3),
        "frame_rate": round(frame_rate, 3),
        "width": width,
        "height": height,
    }


def _read_scene_mid_frame(
    cap: cv2.VideoCapture,
    *,
    start_sec: float,
    end_sec: float,
) -> tuple[float, object]:
    """장면 구간의 **가운데 프레임**을 읽는다.

    시작 프레임을 쓰지 않는 이유: 장면 전환 직후에는 화면이 흐리거나 자막만 있는 경우가
    많아 그 장면을 대표하지 못한다.

    Returns:
        ``(시각(초), 프레임)``.
    """
    frame_sec = start_sec + max(0.0, (end_sec - start_sec) / 2.0)
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_sec * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("대표 프레임을 읽지 못했습니다.")
    return frame_sec, frame


def _extract_representative_core(
    video_path: str | Path,
    *,
    threshold: float = 30.0,
    min_scene_len: int = 15,
    jpeg_quality: int = 85,
    max_frames: int | None = None,
    dedup: KeyframeDedupConfig | None = None,
) -> list[KeyframeBytesResult]:
    """
    영상에서 장면(Scene) 단위 대표 프레임(중앙 시점) JPEG bytes를 반환한다.

    파일로 저장하지 않고 메모리에서 바로 후속 요약 파이프라인에 연결할 때 사용한다.

    장면이 하나도 안 잡히면(단일 컷·아주 짧은 영상 등) 영상 중앙 1프레임만 ``scene_index=1`` 로
    돌려준다. ``max_frames`` 는 장면 수 상한(앞에서부터 자름). 프레임을 못 읽거나 JPEG 인코딩에
    실패한 장면은 건너뛴다(예외로 중단하지 않음).

    048: ``dedup`` 가 주어지고 ``enabled`` 면 **VLM 직전 near-dup 제거**를 적용한다(FR-101). 이 경우
    다중 장면 경로에서 ``max_frames`` pre-cap 을 **건너뛰고** 전 장면을 추출한 뒤 ``dedup_keyframes`` 로
    중복을 제거하고, 그 결과를 앞에서부터 ``max_frames`` 로 trim 한다(순서 = dedup → cap·FR-104).
    ``dedup`` 가 ``None`` 이거나 ``enabled=False`` 면 **현행 코드 경로 그대로**(pre-cap 유지)라 추출
    결과가 기존과 바이트 동일하다(FR-103·완전 no-op). 단일 프레임(no-scene) 경로는 1장이라 무변경.
    """
    src = Path(video_path)
    if not src.is_file():
        raise FileNotFoundError(str(src))

    # 048: dedup 활성 여부 — enabled 일 때만 "전 장면 추출 → dedup → cap" 경로를 탄다.
    _dedup_on = dedup is not None and dedup.enabled

    scenes = detect(str(src), ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    if not scenes:
        cap0 = cv2.VideoCapture(str(src))
        if not cap0.isOpened():
            raise RuntimeError(f"비디오를 열 수 없습니다: {src}")
        try:
            frame_rate = float(cap0.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = float(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            duration = (frame_count / frame_rate) if frame_rate > 0 else 0.0
            frame_sec = (duration / 2.0) if duration > 0 else 0.0
            cap0.set(cv2.CAP_PROP_POS_MSEC, frame_sec * 1000.0)
            ok0, frame0 = cap0.read()
            if not ok0 or frame0 is None:
                return []
            enc_ok, encoded0 = cv2.imencode(
                ".jpg",
                frame0,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            if not enc_ok:
                return []
            end_sec = duration if duration > 0 else 0.001
            return [
                {
                    "scene_index": 1,
                    "start_sec": 0.0,
                    "end_sec": round(end_sec, 3),
                    "frame_sec": round(frame_sec, 3),
                    "jpeg_bytes": encoded0.tobytes(),
                }
            ]
        finally:
            cap0.release()

    # 048: dedup off 면 현행 pre-cap(앞에서부터 자름) 유지 → 바이트 동일(FR-103). dedup on 이면
    # pre-cap 을 건너뛰고 전 장면을 추출한 뒤 아래에서 dedup → cap 순으로 trim 한다(FR-104).
    if not _dedup_on and max_frames is not None and max_frames > 0:
        scenes = scenes[:max_frames]

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {src}")

    results: list[KeyframeBytesResult] = []
    try:
        for i, (start_tc, end_tc) in enumerate(scenes, start=1):
            start_sec = _to_seconds(start_tc)
            end_sec = _to_seconds(end_tc)
            try:
                frame_sec, frame = _read_scene_mid_frame(cap, start_sec=start_sec, end_sec=end_sec)
            except RuntimeError:
                continue

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,  # type: ignore[arg-type]
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            if not ok:
                continue

            results.append(
                {
                    "scene_index": i,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "frame_sec": round(frame_sec, 3),
                    "jpeg_bytes": encoded.tobytes(),
                }
            )
    finally:
        cap.release()

    # 048: dedup on 이면 전 장면 추출 결과에 near-dup 제거를 적용한 뒤(dedup) max_frames 로 trim(cap).
    # 순서 = dedup → cap(FR-104). off 면 위에서 이미 pre-cap 했으므로 results 를 그대로 반환(FR-103).
    if _dedup_on and dedup is not None:
        from processing.preprocess.keyframe_dedup import dedup_keyframes

        kept, skips = dedup_keyframes(results, dedup)
        if skips:
            # FR-405·US4: skip 관측성 — 사유·개수(비용 절감 측정 SC-006 추적용). 디버그 레벨.
            logger.debug(
                "키프레임 dedup: %d/%d skip (mode=%s) — %s",
                len(skips),
                len(results),
                dedup.compare_mode,
                [{"scene": s["scene_index"], "reason": s["reason"]} for s in skips],
            )
        if max_frames is not None and max_frames > 0:
            kept = kept[:max_frames]
        return kept

    return results


def _ffprobe_has_video_stream(src: Path) -> bool:
    """ffprobe 로 파일에 **비디오 스트림**이 있는지(064·폴백 진입 판정). 부재/실패 시 False.

    core 가 빈 결과일 때 '진짜 영상인데 cv2 코덱 미지원'과 '오디오전용/비영상'을 구분한다 —
    후자에 트랜스코딩을 시도해봐야 헛일. ffprobe 미설치(FileNotFoundError)·비정상 종료·타임아웃은 False(graceful).
    """
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(src)],
            capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # FileNotFoundError(부재)·PermissionError·TimeoutExpired 등 전부 graceful False.
        return False
    return proc.returncode == 0 and "video" in proc.stdout


def _transcode_to_h264(src: Path) -> Path | None:
    """시스템 ffmpeg 로 ``src`` 를 임시 h264 mp4 로 트랜스코딩하고 경로를 돌려준다(064·폴백).

    cv2 번들 ffmpeg 가 못 푸는 코덱(AV1 등)을 시스템 ffmpeg(libdav1d/libaom 등 광범위 지원)로 정규화해
    이어서 cv2/scenedetect 로 재추출하게 한다. 키프레임만 필요하므로 오디오는 제외(``-an``). ffmpeg 미설치
    (FileNotFoundError)·권한(PermissionError)·비정상 종료·타임아웃 등은 None(graceful) — 호출부가 원래
    빈 결과를 유지한다. **재인코딩이라 프레임 경계/피처가 원본 native 디코드와 미세하게 다를 수 있다**
    (동일 ffmpeg 빌드·입력에서 결정적·AV1 은 원래 0프레임이라 개선만). 실패/예외 무관 임시파일은 정리한다.
    """
    fd, tmp_name = tempfile.mkstemp(prefix="kf_transcode_", suffix=".mp4")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src),
             "-c:v", "libx264", "-preset", "veryfast", "-an", "-f", "mp4", str(tmp)],
            capture_output=True, text=True, timeout=600, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # 부재·권한·타임아웃 등 전부 graceful — 임시파일 정리 후 None.
        logger.warning("키프레임 폴백 트랜스코딩 실패(%s): %r", src, exc)
        tmp.unlink(missing_ok=True)
        return None
    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        logger.warning("키프레임 폴백 트랜스코딩 비정상(rc=%s·%s)", proc.returncode, src)
        tmp.unlink(missing_ok=True)
        return None
    return tmp


def extract_video_representative_frame_bytes(
    video_path: str | Path,
    *,
    threshold: float = 30.0,
    min_scene_len: int = 15,
    jpeg_quality: int = 85,
    max_frames: int | None = None,
    dedup: KeyframeDedupConfig | None = None,
) -> list[KeyframeBytesResult]:
    """장면별 대표 키프레임(메모리 JPEG)을 추출한다 — cv2 실패 코덱은 시스템 ffmpeg 폴백(064).

    대부분(h264 등)은 ``_extract_representative_core``(cv2/scenedetect)로 바로 성공 → **그대로 반환**
    (happy-path·ffprobe/트랜스코딩 미진입·오버헤드 0·회귀 0). 결과가 **비어있고** ffprobe 상 비디오 스트림이
    있으면(cv2 번들 ffmpeg 코덱 미지원 추정·AV1 등), 시스템 ffmpeg 로 임시 h264 트랜스코딩 후 재추출한다.
    ffmpeg/ffprobe 부재·실패 시 graceful(원래 빈 결과·예외 없음). 임시파일은 finally 로 정리한다.
    """
    kwargs = {
        "threshold": threshold, "min_scene_len": min_scene_len,
        "jpeg_quality": jpeg_quality, "max_frames": max_frames, "dedup": dedup,
    }
    frames = _extract_representative_core(video_path, **kwargs)  # type: ignore[arg-type]
    src = Path(video_path)
    # happy-path: 프레임을 얻었거나(대부분) 애초에 비디오 스트림이 없으면(오디오전용) 폴백 불필요.
    if frames or not _ffprobe_has_video_stream(src):
        return frames
    # 064 폴백: cv2 코덱 미지원 추정 → 시스템 ffmpeg h264 정규화 후 재추출.
    tmp = _transcode_to_h264(src)
    if tmp is None:
        return frames  # ffmpeg 부재/실패 → 기존 빈 결과 유지(graceful)
    try:
        return _extract_representative_core(tmp, **kwargs)  # type: ignore[arg-type]
    except RuntimeError as exc:
        # 트랜스코딩본 재추출도 실패(cap 열기 실패 등 극단) → 원래 빈 결과 유지(graceful·무크래시).
        logger.warning("폴백 재추출 실패(%s): %r", src, exc)
        return frames
    finally:
        tmp.unlink(missing_ok=True)
