"""069 T001(P1-5) — stt WhisperModel 프로세스 캐시(파일마다 재로드 제거). 실모델·네트워크 0."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestWhisperModelCache(unittest.TestCase):
    def setUp(self) -> None:
        # lru_cache 는 프로세스 전역 — 테스트 간 격리를 위해 매번 비운다.
        from processing.preprocess import stt

        stt._get_whisper.cache_clear()

    def _fake_model(self) -> MagicMock:
        m = MagicMock()

        def _fresh_transcribe(*a, **k):
            # 호출마다 새 이터레이터 — 캐시로 재사용되는 모델이 소진된 이터레이터를 받지 않게.
            seg = MagicMock()
            seg.text = "안녕"
            return (iter([seg]), MagicMock())

        m.transcribe.side_effect = _fresh_transcribe
        return m

    def test_same_config_loads_model_once(self) -> None:
        # 같은 (model_size, device, compute_type) 조합 → 파일 여러 개여도 WhisperModel 생성 1회.
        from processing.preprocess import stt

        with patch.object(stt, "WhisperModel", side_effect=lambda *a, **k: self._fake_model()) as mk:
            with patch.object(stt.Path, "is_file", return_value=True):
                out1 = stt.transcribe_audio_local("/tmp/a.mp3")
                out2 = stt.transcribe_audio_local("/tmp/b.mp3")
        self.assertEqual(mk.call_count, 1)  # 핵심: 재로드 없음(P1-5)
        self.assertEqual(out1, {"text": "안녕"})
        self.assertEqual(out2, {"text": "안녕"})

    def test_different_config_loads_separately(self) -> None:
        # 조합이 다르면(cpu/int8 vs cuda/float16) 캐시 키가 갈려 각각 로드된다.
        from processing.preprocess import stt

        with patch.object(stt, "WhisperModel", side_effect=lambda *a, **k: self._fake_model()) as mk:
            with patch.object(stt.Path, "is_file", return_value=True):
                stt.transcribe_audio_local("/tmp/a.mp3")
                stt.transcribe_audio_local("/tmp/a.mp3", device="cuda", compute_type="float16")
        self.assertEqual(mk.call_count, 2)

    def test_missing_file_raises_without_loading(self) -> None:
        # 파일 부재는 모델 로드 전에 FileNotFoundError(기존 계약 불변).
        from processing.preprocess import stt

        with patch.object(stt, "WhisperModel") as mk:
            with self.assertRaises(FileNotFoundError):
                stt.transcribe_audio_local("/no/such/file.mp3")
        mk.assert_not_called()

    def test_transcribe_passes_temperature_zero(self) -> None:
        # 069 B2(P2-2): faster-whisper 는 기본 폴백 래더([0.0,0.2,...]) 로 샘플링해 결정성을
        # 저해한다 — temperature=0.0 을 명시 전달해 재현성을 고정한다(헌법 3조).
        from processing.preprocess import stt

        with patch.object(stt, "WhisperModel", side_effect=lambda *a, **k: self._fake_model()):
            with patch.object(stt.Path, "is_file", return_value=True):
                stt.transcribe_audio_local("/tmp/a.mp3")
                model = stt._get_whisper("small", "cpu", "int8")
        self.assertEqual(model.transcribe.call_args.kwargs["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
