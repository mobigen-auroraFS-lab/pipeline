"""069 T002(P1-6) — audio_meta 헤더-만 읽기(sf.info)·AudioMeta 타입 정정. 실오디오·디코드 0."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from processing.extractors import audio_meta_extractor as ame


class TestExtractAudioMeta(unittest.TestCase):
    def _fake_info(self, duration=12.3456, samplerate=44100, channels=2) -> MagicMock:
        info = MagicMock()
        info.duration = duration
        info.samplerate = samplerate
        info.channels = channels
        return info

    def test_uses_header_info_not_full_decode(self) -> None:
        # 핵심(P1-6): sf.info(헤더)만 호출·sf.read(전체 디코드) 미호출.
        with patch.object(ame.sf, "info", return_value=self._fake_info()) as m_info, \
             patch.object(ame.sf, "read") as m_read:
            out = ame.extract_audio_meta("/tmp/a.wav")
        m_info.assert_called_once_with("/tmp/a.wav")
        m_read.assert_not_called()
        self.assertEqual(out, {"duration": 12.346, "sample_rate": 44100, "channels": 2})

    def test_types_match_declaration(self) -> None:
        # P3-10 정정: duration=float(소수 3자리 반올림)·sample_rate/channels=int.
        with patch.object(ame.sf, "info", return_value=self._fake_info(duration=7, channels=1)):
            out = ame.extract_audio_meta("/tmp/mono.flac")
        self.assertIsInstance(out["duration"], float)
        self.assertIsInstance(out["sample_rate"], int)
        self.assertIsInstance(out["channels"], int)
        self.assertEqual(out["channels"], 1)


if __name__ == "__main__":
    unittest.main()
