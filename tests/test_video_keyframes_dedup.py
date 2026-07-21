"""048 G2 — extract_video_representative_frame_bytes 의 dedup 통합 단위 테스트.

영상 fixture 없이 ``scenedetect.detect``·``cv2.VideoCapture``·``cv2.imencode`` 를 mock 해
추출 경로를 순수 단위로 검증한다:

  - T201(FR-103·SC-001): dedup=None ≡ dedup=KeyframeDedupConfig(enabled=False) → 동일 결과
    (off 경로는 현행 코드 그대로 — pre-cap 유지·바이트 동일).
  - T202(FR-104): dedup on → 전 장면 추출 → dedup_keyframes → [:max_frames] 순서
    (pre-cap 였으면 못 잡을 케이스를 잡는다).

실 OS·DB·LLM 0(순수 단위).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import processing.preprocess.video_keyframes as vk
from processing.preprocess.keyframe_dedup import KeyframeDedupConfig


def _jpeg(color: tuple[int, int, int], size: int = 64) -> bytes:
    img = np.full((size, size, 3), color, dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


class _FakeTimecode:
    """scenedetect timecode stub — get_seconds() 반환."""

    def __init__(self, sec: float) -> None:
        self._sec = sec

    def get_seconds(self) -> float:
        return self._sec


class _FakeCapture:
    """cv2.VideoCapture stub — scene 별로 결정적 BGR 프레임을 돌려준다.

    read() 호출 순서대로 ``frames`` 의 BGR 이미지를 반환한다(set(POS_MSEC)는 무시).
    """

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._i = 0

    def isOpened(self) -> bool:  # noqa: N802 (cv2 API 명명)
        return True

    def set(self, *_args: object) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._i >= len(self._frames):
            return False, None
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def get(self, *_args: object) -> float:
        return 0.0

    def release(self) -> None:
        return None


def _bgr(color: tuple[int, int, int], size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), color, dtype=np.uint8)


class TestExtractDedupIntegration(unittest.TestCase):
    def setUp(self) -> None:
        # is_file() 통과를 위해 실제 임시 파일(내용은 무의미 — detect/Capture 는 mock).
        self._tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        self._tmp.write(b"\x00")
        self._tmp.close()
        self.video_path = self._tmp.name

    def tearDown(self) -> None:
        Path(self._tmp.name).unlink(missing_ok=True)

    def _run(self, *, scenes_frames, dedup, max_frames=None):
        """주어진 (timecode쌍 목록, BGR프레임 목록)으로 detect·Capture·imencode 를 mock 해 추출."""
        scene_tcs = [(_FakeTimecode(s), _FakeTimecode(e)) for s, e, _ in scenes_frames]
        frames = [bgr for _, _, bgr in scenes_frames]

        def fake_detect(_path, _detector):
            return scene_tcs

        # imencode: 입력 BGR 프레임을 결정적 JPEG 로 인코딩(실제 cv2 사용 — 동일 입력 동일 출력).
        with (
            mock.patch.object(vk, "detect", fake_detect),
            mock.patch.object(vk, "ContentDetector", lambda **_kw: object()),
            mock.patch.object(vk.cv2, "VideoCapture", lambda _p: _FakeCapture(frames)),
        ):
            return vk.extract_video_representative_frame_bytes(
                self.video_path, max_frames=max_frames, dedup=dedup
            )

    def test_extract_dedup_off_identical(self) -> None:
        # FR-103·SC-001: dedup=None 과 enabled=False 가 동일 입력에서 동일 결과(바이트 포함).
        scenes = [
            (0.0, 1.0, _bgr((10, 10, 10))),
            (1.0, 2.0, _bgr((200, 50, 50))),
            (2.0, 3.0, _bgr((50, 200, 50))),
        ]
        none_res = self._run(scenes_frames=scenes, dedup=None)
        off_res = self._run(scenes_frames=scenes, dedup=KeyframeDedupConfig(enabled=False))
        self.assertEqual(len(none_res), 3)
        self.assertEqual(none_res, off_res)  # scene_index·시간·jpeg_bytes 전부 동일
        # 메타 보존: scene_index 1..3
        self.assertEqual([r["scene_index"] for r in none_res], [1, 2, 3])

    def test_off_keeps_precap_behavior(self) -> None:
        # FR-103: off 경로는 현행 pre-cap 유지 — max_frames=2 면 앞 2장(중복 제거 없이).
        # 동일한 3장(동일 색) 입력이라도 off 는 dedup 하지 않고 pre-cap 으로 2장만 추출.
        same = (5.0, 6.0, _bgr((77, 77, 77)))
        scenes = [(0.0, 1.0, _bgr((77, 77, 77))), (1.0, 2.0, _bgr((77, 77, 77))), same]
        off_res = self._run(
            scenes_frames=scenes, dedup=KeyframeDedupConfig(enabled=False), max_frames=2
        )
        none_res = self._run(scenes_frames=scenes, dedup=None, max_frames=2)
        self.assertEqual(len(off_res), 2)        # pre-cap → 2장(중복 그대로)
        self.assertEqual(off_res, none_res)

    def test_extract_dedup_on_then_cap(self) -> None:
        # FR-104: dedup on → 전 장면 추출 → dedup → [:max_frames].
        # 입력: 동일 burst 3장(A,A,A) + 다른 장면 1장(B). pre-cap(max=2)였다면 [A,A] → dedup 후 [A].
        # 올바른 순서(dedup→cap)면: 전 4장 추출 → dedup [A,B] → cap(2) → [A,B] (2장, 서로 다름).
        a = _bgr((30, 30, 30))
        b = _bgr((200, 80, 20))
        scenes = [
            (0.0, 1.0, a),
            (1.0, 2.0, a),
            (2.0, 3.0, a),
            (3.0, 4.0, b),
        ]
        on_res = self._run(
            scenes_frames=scenes,
            dedup=KeyframeDedupConfig(enabled=True),
            max_frames=2,
        )
        # dedup 후 A 1장 + B 1장 = 2장. pre-cap 이었으면 A burst 만 남아 B 를 놓쳤을 것.
        self.assertEqual(len(on_res), 2)
        idxs = [r["scene_index"] for r in on_res]
        self.assertEqual(idxs, [1, 4])  # A(첫 장면) + B(다른 장면) — 시간순 메타 보존

    def test_on_cap_trims_after_dedup(self) -> None:
        # FR-104: dedup 후에도 keep 이 max_frames 보다 많으면 앞에서부터 trim.
        # 서로 다른 4장 → dedup 0건 → keep 4 → cap(2) → 앞 2장.
        scenes = [
            (0.0, 1.0, _bgr((10, 10, 10))),
            (1.0, 2.0, _bgr((200, 50, 50))),
            (2.0, 3.0, _bgr((50, 200, 50))),
            (3.0, 4.0, _bgr((50, 50, 200))),
        ]
        on_res = self._run(
            scenes_frames=scenes, dedup=KeyframeDedupConfig(enabled=True), max_frames=2
        )
        self.assertEqual(len(on_res), 2)
        self.assertEqual([r["scene_index"] for r in on_res], [1, 2])

    def test_on_burst_dedup_no_cap(self) -> None:
        # dedup on·cap 없음: 동일 burst 5장 → keep 1.
        a = _bgr((90, 90, 90))
        scenes = [(float(i), i + 1.0, a) for i in range(5)]
        on_res = self._run(scenes_frames=scenes, dedup=KeyframeDedupConfig(enabled=True))
        self.assertEqual(len(on_res), 1)
        self.assertEqual(on_res[0]["scene_index"], 1)


if __name__ == "__main__":
    unittest.main()
