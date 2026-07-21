"""064 G1 — 영상 키프레임 코덱 폴백 단위 테스트 [FR-101~104·SC-01/02/03].

cv2/scenedetect·ffmpeg 실호출 없이 core·ffprobe·transcode 를 mock 해 래퍼 분기만 검증한다.
happy-path(프레임 있음)는 폴백 미진입(회귀 0), 빈 결과+비디오 스트림이면 트랜스코딩 재시도, ffmpeg
부재/실패는 graceful(원래 빈 결과·예외 없음).
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from processing.preprocess import video_keyframes as vk

_FRAME = {"scene_index": 1, "start_sec": 0.0, "end_sec": 1.0, "frame_sec": 0.5, "jpeg_bytes": b"x"}


class TestKeyframeCodecFallback(unittest.TestCase):
    def test_happy_path_no_fallback(self) -> None:
        # 프레임을 얻으면 ffprobe/transcode 미호출·core 결과 그대로(SC-01·회귀 0·오버헤드 0).
        with mock.patch.object(vk, "_extract_representative_core", return_value=[_FRAME]) as core, \
             mock.patch.object(vk, "_ffprobe_has_video_stream") as probe, \
             mock.patch.object(vk, "_transcode_to_h264") as trans:
            out = vk.extract_video_representative_frame_bytes("/x/a.mp4", max_frames=8)
        self.assertEqual(out, [_FRAME])
        probe.assert_not_called()
        trans.assert_not_called()
        self.assertEqual(core.call_count, 1)

    def test_fallback_transcode_and_retry(self) -> None:
        # 빈 결과 + 비디오 스트림 → transcode 후 temp 로 재추출(SC-02). core 2회(원본→temp).
        with mock.patch.object(vk, "_extract_representative_core", side_effect=[[], [_FRAME]]) as core, \
             mock.patch.object(vk, "_ffprobe_has_video_stream", return_value=True), \
             mock.patch.object(vk, "_transcode_to_h264", return_value=Path("/tmp/none.mp4")) as trans:
            out = vk.extract_video_representative_frame_bytes("/x/av1.mp4", max_frames=8)
        self.assertEqual(out, [_FRAME])
        trans.assert_called_once()
        self.assertEqual(core.call_count, 2)
        # 재시도가 원 인자(threshold·max_frames·dedup 등)를 그대로 전달(드리프트 가드).
        self.assertEqual(core.call_args_list[0].kwargs, core.call_args_list[1].kwargs)

    def test_no_video_stream_skips_fallback(self) -> None:
        # 빈 결과 + 비디오 스트림 없음(오디오전용 등) → 폴백 미진입·빈 결과(SC-03 일부).
        with mock.patch.object(vk, "_extract_representative_core", return_value=[]), \
             mock.patch.object(vk, "_ffprobe_has_video_stream", return_value=False), \
             mock.patch.object(vk, "_transcode_to_h264") as trans:
            out = vk.extract_video_representative_frame_bytes("/x/audio.mp4")
        self.assertEqual(out, [])
        trans.assert_not_called()

    def test_ffmpeg_missing_graceful(self) -> None:
        # 빈 결과 + 비디오 스트림 있으나 transcode None(ffmpeg 부재/실패) → 원래 빈 결과·예외 없음(SC-03).
        with mock.patch.object(vk, "_extract_representative_core", return_value=[]) as core, \
             mock.patch.object(vk, "_ffprobe_has_video_stream", return_value=True), \
             mock.patch.object(vk, "_transcode_to_h264", return_value=None):
            out = vk.extract_video_representative_frame_bytes("/x/av1.mp4")
        self.assertEqual(out, [])
        self.assertEqual(core.call_count, 1)  # 재시도 안 함(transcode 실패)


class TestFfprobeAndTranscodeGraceful(unittest.TestCase):
    def test_ffprobe_missing_returns_false(self) -> None:
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            self.assertFalse(vk._ffprobe_has_video_stream(Path("/x/a.mp4")))

    def test_transcode_missing_returns_none(self) -> None:
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(vk._transcode_to_h264(Path("/x/a.mp4")))


if __name__ == "__main__":
    unittest.main()
