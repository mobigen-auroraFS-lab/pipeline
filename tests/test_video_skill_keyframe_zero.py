"""069 US-B B9(P2-10) — 키프레임 0장 관측(logger.warning).

정책(plan 확정): 키프레임 0장은 **현행 동작 유지 + 관측만**(failed 전환·zero-vector 통일 금지 —
근본 처방은 064 ffmpeg 폴백). 여기서는 0장일 때 경고 로그가 남는지만 봉인한다. 실 영상·LLM·DB 0.
"""

from __future__ import annotations

import unittest
from unittest import mock

from processing.dispatch.types import ExtractContext
from processing.skills import video_skill
from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION


def _cfg() -> mock.Mock:
    # _extract_video_meta 가 읽는 설정 속성만 채운 대역(dedup config·상한·라벨 필터).
    return mock.Mock(
        video=mock.Mock(
            dedup_enabled=False,
            dedup_hash_max=7,
            dedup_ssim_min=0.94,
            dedup_ssim_gray_lo=0.0,
            dedup_hist_min=0.0,
            dedup_compare_mode="recent",
            dedup_recent_window=4,
            max_keyframes=8,
            labels_meta_top_k=5,
        ),
        vlm=mock.Mock(labels_score_min=0.0),
    )


class TestKeyframeZeroWarns(unittest.TestCase):
    def _run_zero_frames(self):
        ctx = ExtractContext(file_path="/d/v.mp4", modality="video", settings=_cfg())
        with (
            mock.patch(
                "processing.preprocess.video_keyframes.extract_video_representative_frame_bytes",
                return_value=[],  # 키프레임 0장
            ),
            mock.patch(
                "processing.preprocess.video_keyframes.extract_video_basic_meta",
                return_value={"duration": 1.0, "frame_rate": 30.0, "width": 10, "height": 10},
            ),
            mock.patch(
                "src.llm.video_summarizer.summarize_video_from_scene_results",
                return_value={"summary": "", "keywords": [], "objects": []},
            ),
            mock.patch(
                "src.embedders.video_embedder.embed_video_keyframes_clip",
                return_value={"keyframes": []},
            ),
        ):
            return video_skill._extract_video_meta(ctx)

    def test_zero_keyframes_logs_warning(self) -> None:
        # 0장이면 WARNING 로그가 남는다(관측 공백 메움). 현행 반환(빈 임베딩)은 그대로.
        with self.assertLogs("meta_extract.video_skill", level="WARNING") as cm:
            rec = self._run_zero_frames()
        self.assertTrue(any("키프레임" in m for m in cm.output))
        self.assertEqual(rec.embeddings, [])  # 동작 불변 — failed·zero-vector 아님
        self.assertEqual(rec.ext_meta.get("keyframes"), [])

    def test_nonzero_keyframes_no_warning(self) -> None:
        # 프레임이 있으면 경고 없음(정상 경로 오염 금지).
        ctx = ExtractContext(file_path="/d/v.mp4", modality="video", settings=_cfg())
        summ = {"summary": "s", "keywords": ["k"], "objects": ["o"]}
        frames = [{"scene_index": 1, "start_sec": 0.0, "end_sec": 1.0,
                   "frame_sec": 0.5, "jpeg_bytes": b"x"}]
        clip_kfs = [{"clip_image_embedding": [0.0] * FIX_EMBEDDING_DIMENSION,
                     "summary": summ, "labels": []}]
        with (
            mock.patch("processing.preprocess.video_keyframes.extract_video_representative_frame_bytes",
                       return_value=frames),
            mock.patch("processing.preprocess.video_keyframes.extract_video_basic_meta",
                       return_value={"duration": 1.0}),
            mock.patch("src.llm.image_summarizer.summarize_image_caption_keywords_objects_from_jpeg_bytes",
                       return_value=summ),
            mock.patch("src.llm.video_summarizer.summarize_video_from_scene_results",
                       return_value={"summary": "v", "keywords": [], "objects": []}),
            mock.patch("src.embedders.video_embedder.embed_video_keyframes_clip",
                       return_value={"keyframes": clip_kfs}),
        ):
            logger = __import__("logging").getLogger("meta_extract.video_skill")
            with mock.patch.object(logger, "warning") as m_warn:
                video_skill._extract_video_meta(ctx)
        m_warn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
