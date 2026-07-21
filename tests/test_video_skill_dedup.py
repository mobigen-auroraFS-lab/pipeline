"""048 G3 — video_skill 의 키프레임 dedup 배선 단위 테스트(FR-604).

검증 대상(순수 단위 — 실 영상·LLM·DB 0):
  - `_extract_video_meta` 가 settings 의 `video_keyframe_dedup_*` 로 ``KeyframeDedupConfig`` 를
    빌드해 `extract_video_representative_frame_bytes(dedup=...)` 로 주입한다.
  - VLM 캡션(`summarize_image_caption_...`)은 (dedup 후) extract 가 돌려준 프레임 수만큼만 호출된다.
  - `VIDEO_KEYFRAME_DEDUP_ENABLED=false` 면 주입 config 의 ``enabled=False``(off 배선).

video_skill 은 무거운 의존(extract·VLM·CLIP·video_summary)을 함수 내부에서 늦게 import 하므로
원천 모듈 속성을 patch 하면 호출 시 바인딩이 mock 을 가리킨다(test_skills_active_channel 과 동일 전략).
"""

from __future__ import annotations

import contextlib
import os
import unittest
from unittest import mock

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.config.settings import _build_settings
from processing.dispatch.types import ExtractContext
from processing.skills import video_skill

_REQUIRED_ENV = {
    "META_MODEL": "gemma",
    "OPENAI_BASE_URL": "http://localhost:1234/v1",
    "OPENAI_API_KEY": "sk-test",
    "SUMMARY_MAX_CHARS": "500",
    "TOP_K_KEYWORDS": "10",
    "CHUNK_SIZE": "1000",
    "OVERLAP_SIZE": "100",
    "ENCODING": "utf-8",
    "TEXT_EMBED_MODEL": "BM-K/KoSimCSE-roberta-multitask",
    "TEXT_EMBED_CHUNK_SIZE": "512",
    "TEXT_EMBED_NORMALIZE": "true",
}
_DEDUP_KEY = "VIDEO_KEYFRAME_DEDUP_ENABLED"


@contextlib.contextmanager
def _env(dedup_enabled: str | None = None):
    touched = list(_REQUIRED_ENV) + [_DEDUP_KEY]
    saved = {k: os.environ.get(k) for k in touched}
    try:
        os.environ.update(_REQUIRED_ENV)
        os.environ.pop(_DEDUP_KEY, None)
        if dedup_enabled is not None:
            os.environ[_DEDUP_KEY] = dedup_enabled
        yield
    finally:
        for k in touched:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


def _cfg(dedup_enabled: str | None = None):
    with _env(dedup_enabled):
        return _build_settings("dev")


def _frames(n: int) -> list[dict]:
    # extract 가 (dedup 후) 돌려주는 프레임 — jpeg_bytes 는 mock VLM 이라 내용 무관.
    return [
        {"scene_index": i + 1, "start_sec": float(i), "end_sec": i + 1.0,
         "frame_sec": i + 0.5, "jpeg_bytes": b"jpeg-%d" % i}
        for i in range(n)
    ]


class TestVideoSkillDedupWiring(unittest.TestCase):
    def _run(self, *, returned_frames: list[dict], dedup_enabled: str | None = None):
        cfg = _cfg(dedup_enabled)
        ctx = ExtractContext(file_path="/d/v.mp4", modality="video", settings=cfg)
        summ = {"summary": "장면 설명", "keywords": ["키"], "objects": ["객체"]}
        clip_kfs = [
            {"clip_image_embedding": [0.0] * FIX_EMBEDDING_DIMENSION, "summary": summ, "labels": []}
            for _ in returned_frames
        ]
        with (
            mock.patch(
                "processing.preprocess.video_keyframes.extract_video_representative_frame_bytes",
                return_value=returned_frames,
            ) as m_extract,
            mock.patch(
                "processing.preprocess.video_keyframes.extract_video_basic_meta",
                return_value={"duration": 1.0, "frame_rate": 30.0, "width": 10, "height": 10},
            ),
            mock.patch(
                "src.llm.image_summarizer.summarize_image_caption_keywords_objects_from_jpeg_bytes",
                return_value=summ,
            ) as m_vlm,
            mock.patch(
                "src.llm.video_summarizer.summarize_video_from_scene_results",
                return_value={"summary": "영상요약", "keywords": [], "objects": []},
            ),
            mock.patch(
                "src.embedders.video_embedder.embed_video_keyframes_clip",
                return_value={"keyframes": clip_kfs},
            ),
        ):
            video_skill._extract_video_meta(ctx)
        return m_extract, m_vlm

    def test_vlm_called_per_returned_frame_and_config_injected(self):
        m_extract, m_vlm = self._run(returned_frames=_frames(2))
        # FR-604: VLM 은 dedup 후 프레임 수만큼만 호출.
        self.assertEqual(m_vlm.call_count, 2)
        # G3 배선: settings 유래 KeyframeDedupConfig 가 extract 에 주입됐는가.
        cfg_passed = m_extract.call_args.kwargs["dedup"]
        self.assertTrue(cfg_passed.enabled)  # 기본 true
        self.assertEqual(cfg_passed.compare_mode, "recent")
        self.assertEqual(cfg_passed.recent_window, 4)
        self.assertEqual(cfg_passed.hash_max, 7)
        self.assertAlmostEqual(cfg_passed.ssim_min, 0.94)

    def test_disabled_env_injects_disabled_config(self):
        # off 배선: ENABLED=false → 주입 config.enabled=False (extract 가 현행 경로·FR-103).
        m_extract, _ = self._run(returned_frames=_frames(1), dedup_enabled="false")
        self.assertFalse(m_extract.call_args.kwargs["dedup"].enabled)


if __name__ == "__main__":
    unittest.main()
