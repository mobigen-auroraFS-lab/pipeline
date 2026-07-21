"""v2 단계 A — skill 추출/임베딩 분해의 mock 기반 단위 테스트.

함수-로컬 import(`from src.x import y`)는 호출 시점에 소스 모듈 속성을 읽으므로
소스 모듈 속성을 patch 하면 가짜로 대체된다. 실제 모델/파일 없이 분해 구조를 검증한다.
"""
from __future__ import annotations

import unittest
from unittest import mock

from processing.dispatch.types import AssetRecord, EmbeddingItem, ExtractContext


def _ctx(modality="txt"):
    return ExtractContext(file_path="/d/a." + modality, modality=modality, settings=object())


class TestTextSplit(unittest.TestCase):
    def test_extract_meta_has_no_embeddings(self) -> None:
        from processing.skills import text_skill
        with mock.patch.multiple("processing.extractors.text_meta_extractor", extract_text_meta=mock.Mock(return_value={"size": 1})), \
             mock.patch.multiple("src.llm.text_summarizer", summarize_and_extract_keywords=mock.Mock(return_value={"summary": "s", "keywords": ["k"]})):
            ctx = _ctx()
            # active_embed_channel='st' → active_embed_model = text_embedding_model('m').
            # _extract_text_meta 가 토큰 수를 활성 임베딩 모델 토크나이저로 세도록 정합(count_tokens 정합 개선).
            ctx.settings = mock.Mock(encoding="utf-8", embed=mock.Mock(
                chunk_size=100, model="m", normalize=True, active_channel="st"))
            rec = text_skill._extract_text_meta(ctx)
        self.assertIsInstance(rec, AssetRecord)
        self.assertEqual(rec.embeddings, [])
        self.assertEqual(rec.ext_meta.get("summary"), "s")
        # 037: fts_plain 제거 — 더 이상 AssetRecord 에 존재하지 않는다.
        self.assertFalse(hasattr(rec, "fts_plain"))

    def test_embed_returns_items_and_wrapper_composes(self) -> None:
        from processing.skills import text_skill
        chunks = [{"embedding_vector": [0.1], "chunk_index": 0}, {"embedding_vector": [0.2], "chunk_index": 1}]
        ctx = _ctx()
        # active_embed_channel='st' → active_embed_model = text_embedding_model('m'). 기본 'st' 동치(018 G2).
        ctx.settings = mock.Mock(encoding="utf-8", embed=mock.Mock(
            chunk_size=100, model="m", normalize=True, active_channel="st"))
        with mock.patch.multiple("src.embedders.text_embedder", embedding_text_chunks=mock.Mock(return_value=chunks)):
            embs = text_skill._embed_text(ctx, AssetRecord())
        self.assertEqual([e.chunk_index for e in embs], [0, 1])
        self.assertTrue(all(isinstance(e, EmbeddingItem) and e.channel == "st" for e in embs))


class TestImageSplit(unittest.TestCase):
    def _cfg(self, embed_enable_clip=True):
        # active_embed_channel='st' → active_embed_model = text_embedding_model('m'). 기본 'st' 동치(018 G2).
        # 063: embed_enable_clip(기본 True) — clip 임베딩 토글.
        return mock.Mock(
            vlm=mock.Mock(labels_score_min=0.0, image_labels_meta_top_k=5),
            embed=mock.Mock(model="m", normalize=True, active_channel="st",
                            enable_clip=embed_enable_clip),
        )

    def test_extract_stashes_clip_vec_and_embed_reuses(self) -> None:
        from processing.skills import image_skill
        ctx = _ctx("image")
        ctx.settings = self._cfg()
        zs = {"label_scores": {"고양이": 0.9}, "clip_image_embedding": [0.5] * 4}
        with mock.patch.multiple("processing.extractors.image_meta_extractor", extract_image_meta=mock.Mock(return_value={"w": 10})), \
             mock.patch.multiple("src.llm.image_summarizer", summarize_image_caption_keywords_objects=mock.Mock(return_value={"summary": "s", "objects": ["고양이"]})), \
             mock.patch.multiple("src.embedders.image_embedder",
                                 zero_shot_tag_image_korean_clip=mock.Mock(return_value=zs),
                                 clip_zero_shot_ko_meta_items=mock.Mock(return_value=[{"label": "고양이", "score": 0.9}])):
            rec = image_skill._extract_image_meta(ctx)
        self.assertEqual(rec.embeddings, [])
        self.assertEqual(ctx.scratch["clip_vec"], [0.5] * 4)

        with mock.patch.multiple("src.embedders.text_embedder",
                                 embed_texts=mock.Mock(return_value=[[0.1, 0.2]]),
                                 pad_embedding_to_storage_dim=mock.Mock(side_effect=lambda v: v)), \
             mock.patch.multiple("processing.preprocess.vlm_text_for_embedding",
                                 build_image_vlm_text_for_embedding=mock.Mock(return_value="텍스트")):
            embs = image_skill._embed_image(ctx, rec)
        chans = sorted(e.channel for e in embs)
        self.assertEqual(chans, ["clip", "st"])
        clip = next(e for e in embs if e.channel == "clip")
        self.assertEqual(clip.vector, [0.5] * 4)

    def test_embed_skips_clip_when_disabled(self) -> None:
        # 063 SC-02: EMBED_ENABLE_CLIP off → clip 채널 항목 스킵·ST 캡션만(라벨·계약 불변).
        from processing.skills import image_skill
        ctx = _ctx("image")
        ctx.settings = self._cfg(embed_enable_clip=False)
        ctx.scratch["clip_vec"] = [0.5] * 4  # extract 는 여전히 clip_vec 계산(라벨용)·계약 유지
        rec = AssetRecord(core_meta={"summary": "s"})
        with mock.patch.multiple("src.embedders.text_embedder",
                                 embed_texts_for=mock.Mock(return_value=[[0.1, 0.2]]),
                                 pad_embedding_to_storage_dim=mock.Mock(side_effect=lambda v: v)), \
             mock.patch.multiple("processing.preprocess.vlm_text_for_embedding",
                                 build_image_vlm_text_for_embedding=mock.Mock(return_value="텍스트")):
            embs = image_skill._embed_image(ctx, rec)
        self.assertEqual([e.channel for e in embs], ["st"])  # clip 없음


class TestAudioSplit(unittest.TestCase):
    def test_extract_stashes_stt_text_and_embed_reuses(self) -> None:
        from processing.skills import audio_skill
        ctx = _ctx("audio")
        # active_embed_channel='st' → active_embed_model = text_embedding_model('m'). 기본 'st' 동치(018 G2).
        ctx.settings = mock.Mock(embed=mock.Mock(
            chunk_size=100, model="m", normalize=True, active_channel="st"))
        with mock.patch.multiple("processing.preprocess.stt", transcribe_audio_local=mock.Mock(return_value={"text": "안녕"})), \
             mock.patch.multiple("processing.extractors.audio_meta_extractor", extract_audio_meta=mock.Mock(return_value={"dur": 3})), \
             mock.patch.multiple("src.llm.text_summarizer", summarize_and_extract_keywords_from_audio=mock.Mock(return_value={"summary": "s"})):
            rec = audio_skill._extract_audio_meta(ctx)
        self.assertEqual(rec.embeddings, [])
        self.assertEqual(ctx.scratch["stt_text"], "안녕")

        with mock.patch.multiple("src.embedders.text_embedder",
                                 embedding_plain_text_chunks=mock.Mock(return_value=[{"embedding_vector": [0.1], "chunk_index": 0}])):
            embs = audio_skill._embed_audio(ctx, rec)
        self.assertEqual([e.channel for e in embs], ["st"])
        self.assertEqual(embs[0].chunk_index, 0)


class TestVideoSplit(unittest.TestCase):
    def test_extract_stashes_keyframes_and_embed_pairs(self) -> None:
        from processing.skills import video_skill
        ctx = _ctx("video")
        # active_embed_channel='st' → active_embed_model = text_embedding_model('m'). 기본 'st' 동치(018 G2).
        ctx.settings = mock.Mock(
            video=mock.Mock(
                max_keyframes=2, labels_meta_top_k=5,
                # 048: video_skill 이 KeyframeDedupConfig 를 빌드하므로 유효 dedup 설정 필요
                # (extract 는 mock 이라 enabled 무관 — __post_init__ 검증 통과용 유효값).
                dedup_enabled=False, dedup_hash_max=7, dedup_ssim_min=0.94,
                dedup_ssim_gray_lo=0.90, dedup_hist_min=0.97,
                dedup_compare_mode="recent", dedup_recent_window=4,
            ),
            vlm=mock.Mock(labels_score_min=0.0),
            embed=mock.Mock(model="m", normalize=True, active_channel="st", enable_clip=True),
        )  # 063: enable_clip 기본 True(기존 동치)
        frames = [{"scene_index": 0, "start_sec": 0, "end_sec": 1, "frame_sec": 0, "jpeg_bytes": b"x"}]
        clip_ve = {"keyframes": [{"clip_image_embedding": [0.7] * 4,
                                  "summary": {"summary": "s", "keywords": ["k"]},
                                  "labels": [{"label": "l", "score": 0.9}]}]}
        with mock.patch.multiple("processing.preprocess.video_keyframes",
                                 extract_video_representative_frame_bytes=mock.Mock(return_value=frames),
                                 extract_video_basic_meta=mock.Mock(return_value={"dur": 5})), \
             mock.patch.multiple("src.llm.image_summarizer",
                                 summarize_image_caption_keywords_objects_from_jpeg_bytes=mock.Mock(return_value={"summary": "s", "objects": ["k"]})), \
             mock.patch.multiple("src.llm.video_summarizer", summarize_video_from_scene_results=mock.Mock(return_value={"summary": "v"})), \
             mock.patch.multiple("src.embedders.video_embedder", embed_video_keyframes_clip=mock.Mock(return_value=clip_ve)):
            rec = video_skill._extract_video_meta(ctx)
        self.assertEqual(rec.embeddings, [])
        self.assertEqual(len(ctx.scratch["keyframes"]), 1)
        self.assertEqual(ctx.scratch["keyframes"][0]["clip_vec"], [0.7] * 4)

        with mock.patch.multiple("src.embedders.text_embedder",
                                 embed_texts=mock.Mock(return_value=[[0.1]]),
                                 pad_embedding_to_storage_dim=mock.Mock(side_effect=lambda v: v)), \
             mock.patch.multiple("processing.preprocess.vlm_text_for_embedding",
                                 build_image_vlm_text_for_embedding=mock.Mock(return_value="텍스트")):
            embs = video_skill._embed_video(ctx, rec)
        self.assertEqual(sorted(e.channel for e in embs), ["clip", "st"])
        self.assertTrue(all(e.chunk_index == 0 for e in embs))

        # 063 SC-02: 토글 off → 키프레임 clip 항목 스킵·ST 캡션만(같은 scratch 재사용).
        ctx.settings.embed.enable_clip = False
        with mock.patch.multiple("src.embedders.text_embedder",
                                 embed_texts=mock.Mock(return_value=[[0.1]]),
                                 pad_embedding_to_storage_dim=mock.Mock(side_effect=lambda v: v)), \
             mock.patch.multiple("processing.preprocess.vlm_text_for_embedding",
                                 build_image_vlm_text_for_embedding=mock.Mock(return_value="텍스트")):
            embs_off = video_skill._embed_video(ctx, rec)
        self.assertEqual([e.channel for e in embs_off], ["st"])  # clip 없음


if __name__ == "__main__":
    unittest.main()
