"""적재 skill 4개의 활성 임베딩 채널 프로파일 — 순수 단위 테스트(DB·모델 로딩 0).

018 G2: 적재 skill(text/image/video/audio)의 텍스트 임베딩 채널·모델을 'st' 하드코딩에서
**활성 프로파일 단일 출처**(`active_embed_channel(cfg)`·`active_embed_model(cfg)`)로 치환한다.

테스트 전략(docs/테스트_가이드.md §2~3)
  - **임베더 seam 모킹**: 무거운 임베딩(`embed_texts`/`embedding_*_chunks`)은 mock 으로 대체해
    DB·SentenceTransformer 로딩 없이 `EmbeddingItem` 채널·모델·임베더 호출 모델만 검증한다.
  - **cfg 주입**: `_build_settings("dev")` 로 `PipelineSettings` 를 직접 만들어(필수 env 임시 주입)
    `ExtractContext.settings` 로 주입한다 — `init_settings`/전역 상태 없이 순수 단위.
  - **회귀 가드(최우선)**: 기본 active='st' 시 채널 'st'·모델 KoSimCSE(=`text_embedding_model`)로
    기존 적재와 **완전 동치**임을 함께 검증한다(헌법 8조, SC-002).
  - **시각(CLIP) 무변경**: image/video 의 CLIP 채널(`channel='clip'`·CLIP 모델)은 active 와 무관하게
    불변임을 확인한다(텍스트 채널만 프로파일화).

임베더는 skill 함수 내부에서 `from src.embedders.text_embedder import ...` 로 늦게 import 하므로,
원천 모듈 속성(`src.embedders.text_embedder.*`)을 patch 하면 호출 시 바인딩이 mock 을 가리킨다.
"""

from __future__ import annotations

import contextlib
import os
import unittest
from unittest import mock

from src.config.embedding_constants import DEFAULT_CLIP_MODEL_NAME, FIX_EMBEDDING_DIMENSION
from src.config.settings import _build_settings
from processing.dispatch.types import AssetRecord, ExtractContext
from processing.skills import audio_skill, image_skill, text_skill, video_skill

# _build_settings 가 _require_env* 로 읽는 필수 env 최소 집합(값은 형식만 맞으면 됨).
# TEXT_EMBED_MODEL 은 모델 라우팅 검증을 위해 실제 KoSimCSE 값으로 고정한다.
_KOSIMCSE = "BM-K/KoSimCSE-roberta-multitask"
_BGE = "BAAI/bge-m3"
_REQUIRED_ENV = {
    "META_MODEL": "gemma",
    "OPENAI_BASE_URL": "http://localhost:1234/v1",
    "OPENAI_API_KEY": "sk-test",
    "SUMMARY_MAX_CHARS": "500",
    "TOP_K_KEYWORDS": "10",
    "CHUNK_SIZE": "1000",
    "OVERLAP_SIZE": "100",
    "ENCODING": "utf-8",
    "TEXT_EMBED_MODEL": _KOSIMCSE,
    "TEXT_EMBED_CHUNK_SIZE": "512",
    "TEXT_EMBED_NORMALIZE": "true",
}

_ACTIVE_KEY = "EMBED_ACTIVE_CHANNEL"
_TEXT_EMBED_MOD = "src.embedders.text_embedder"


@contextlib.contextmanager
def _env(active: str | None = None):
    """필수 env 를 임시로 채우고 ``EMBED_ACTIVE_CHANNEL`` 을 ``active`` 로 설정(None=미설정).

    본 테스트가 건드린 키만 정확히 원복한다(필수 env 가 원래 있었으면 보존).
    """
    touched = list(_REQUIRED_ENV) + [_ACTIVE_KEY]
    saved = {k: os.environ.get(k) for k in touched}
    try:
        os.environ.update(_REQUIRED_ENV)
        os.environ.pop(_ACTIVE_KEY, None)
        if active is not None:
            os.environ[_ACTIVE_KEY] = active
        yield
    finally:
        for k in touched:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]


def _cfg(active: str | None = None):
    """주어진 활성 채널로 ``PipelineSettings`` 를 만든다(env 는 빌드 후 즉시 원복).

    settings 객체가 active_embed_channel/text_embedding_model* 값을 캡처하므로,
    `_env` 컨텍스트 밖에서도 헬퍼(`active_embed_*`)가 cfg 만으로 동작한다.
    """
    with _env(active):
        return _build_settings("dev")


# 임베더 mock 반환(청크 임베딩: chunk_index + 1536D 벡터 / 텍스트 임베딩: 모델 출력 벡터 1개).
def _chunk_rows(n: int = 1) -> list[dict]:
    return [{"chunk_index": i, "embedding_vector": [0.1] * FIX_EMBEDDING_DIMENSION} for i in range(n)]


def _embed_texts_return() -> list[list[float]]:
    return [[0.1] * 1024]  # 모델 출력(미패딩) — skill 이 pad_embedding_to_storage_dim 으로 1536D 화


class TestTextSkillActiveChannel(unittest.TestCase):
    """text_skill._embed_text: 활성 채널·모델로 EmbeddingItem 생성 + 임베더 모델 라우팅."""

    def _run(self, active: str | None):
        cfg = _cfg(active)
        ctx = ExtractContext(file_path="/d/x.txt", modality="txt", settings=cfg)
        rec = AssetRecord()
        with mock.patch(
            f"{_TEXT_EMBED_MOD}.embedding_text_chunks", return_value=_chunk_rows(2)
        ) as m:
            items = text_skill._embed_text(ctx, rec)
        return items, m

    def test_st_bge_active_uses_bge_channel_and_model(self) -> None:
        items, m = self._run("st_bge")
        self.assertEqual(m.call_args.kwargs["embedding_model_name"], _BGE)
        self.assertEqual([it.channel for it in items], ["st_bge", "st_bge"])
        self.assertEqual([it.model_name for it in items], [_BGE, _BGE])

    def test_default_st_active_is_kosimcse_unchanged(self) -> None:
        # 회귀 가드: 기본 active='st' → channel='st'·KoSimCSE(기존 적재와 동치).
        items, m = self._run(None)
        self.assertEqual(m.call_args.kwargs["embedding_model_name"], _KOSIMCSE)
        self.assertTrue(all(it.channel == "st" for it in items))
        self.assertTrue(all(it.model_name == _KOSIMCSE for it in items))


class TestImageSkillActiveChannel(unittest.TestCase):
    """image_skill._embed_image: ST 채널만 프로파일화, CLIP 채널 불변."""

    def _run(self, active: str | None):
        cfg = _cfg(active)
        ctx = ExtractContext(file_path="/d/p.jpg", modality="image", settings=cfg)
        ctx.scratch["clip_vec"] = [0.0] * FIX_EMBEDDING_DIMENSION
        rec = AssetRecord(
            core_meta={"width": 100},
            ext_meta={"summary": "무선 충전기 제품 이미지", "keywords": ["충전기"], "labels": []},
        )
        with mock.patch(
            f"{_TEXT_EMBED_MOD}.embed_texts", return_value=_embed_texts_return()
        ) as m:
            items = image_skill._embed_image(ctx, rec)
        return items, m

    def test_st_bge_active_uses_bge_for_st_channel_clip_unchanged(self) -> None:
        items, m = self._run("st_bge")
        self.assertEqual(m.call_args.kwargs["model_name"], _BGE)
        st_items = [it for it in items if it.channel != "clip"]
        clip_items = [it for it in items if it.channel == "clip"]
        self.assertEqual([it.channel for it in st_items], ["st_bge"])
        self.assertEqual([it.model_name for it in st_items], [_BGE])
        # CLIP 채널은 active 와 무관하게 불변(텍스트 채널만 프로파일).
        self.assertEqual([it.channel for it in clip_items], ["clip"])
        self.assertEqual([it.model_name for it in clip_items], [DEFAULT_CLIP_MODEL_NAME])

    def test_default_st_active_unchanged(self) -> None:
        items, m = self._run(None)
        self.assertEqual(m.call_args.kwargs["model_name"], _KOSIMCSE)
        st_items = [it for it in items if it.channel != "clip"]
        clip_items = [it for it in items if it.channel == "clip"]
        self.assertEqual([it.channel for it in st_items], ["st"])
        self.assertEqual([it.model_name for it in st_items], [_KOSIMCSE])
        self.assertEqual([it.channel for it in clip_items], ["clip"])
        self.assertEqual([it.model_name for it in clip_items], [DEFAULT_CLIP_MODEL_NAME])


class TestVideoSkillActiveChannel(unittest.TestCase):
    """video_skill._embed_video: 키프레임별 ST 채널 프로파일화, CLIP 쌍 불변."""

    def _run(self, active: str | None):
        cfg = _cfg(active)
        ctx = ExtractContext(file_path="/d/v.mp4", modality="video", settings=cfg)
        ctx.scratch["keyframes"] = [
            {"clip_vec": [0.0] * FIX_EMBEDDING_DIMENSION, "summary": "프레임0", "keywords": ["a"], "labels": []},
            {"clip_vec": [0.0] * FIX_EMBEDDING_DIMENSION, "summary": "프레임1", "keywords": [], "labels": []},
        ]
        rec = AssetRecord()
        with mock.patch(
            f"{_TEXT_EMBED_MOD}.embed_texts", return_value=_embed_texts_return()
        ) as m:
            items = video_skill._embed_video(ctx, rec)
        return items, m

    def test_st_bge_active_uses_bge_for_st_clip_unchanged(self) -> None:
        items, m = self._run("st_bge")
        self.assertEqual(m.call_args.kwargs["model_name"], _BGE)
        st_items = [it for it in items if it.channel != "clip"]
        clip_items = [it for it in items if it.channel == "clip"]
        # 키프레임 2개 → ST 2 + CLIP 2.
        self.assertEqual([it.channel for it in st_items], ["st_bge", "st_bge"])
        self.assertEqual([it.model_name for it in st_items], [_BGE, _BGE])
        self.assertEqual([it.channel for it in clip_items], ["clip", "clip"])
        self.assertEqual([it.model_name for it in clip_items], [DEFAULT_CLIP_MODEL_NAME, DEFAULT_CLIP_MODEL_NAME])

    def test_default_st_active_unchanged(self) -> None:
        items, m = self._run(None)
        self.assertEqual(m.call_args.kwargs["model_name"], _KOSIMCSE)
        st_items = [it for it in items if it.channel != "clip"]
        clip_items = [it for it in items if it.channel == "clip"]
        self.assertEqual([it.channel for it in st_items], ["st", "st"])
        self.assertEqual([it.model_name for it in st_items], [_KOSIMCSE, _KOSIMCSE])
        self.assertEqual([it.channel for it in clip_items], ["clip", "clip"])


class TestAudioSkillActiveChannel(unittest.TestCase):
    """audio_skill._embed_audio: STT 청크 ST 채널 프로파일화(CLIP 없음)."""

    def _run(self, active: str | None):
        cfg = _cfg(active)
        ctx = ExtractContext(file_path="/d/a.mp3", modality="audio", settings=cfg)
        ctx.scratch["stt_text"] = "안녕하세요 테스트 전사 텍스트"
        rec = AssetRecord()
        with mock.patch(
            f"{_TEXT_EMBED_MOD}.embedding_plain_text_chunks", return_value=_chunk_rows(2)
        ) as m:
            items = audio_skill._embed_audio(ctx, rec)
        return items, m

    def test_st_bge_active_uses_bge_channel_and_model(self) -> None:
        items, m = self._run("st_bge")
        self.assertEqual(m.call_args.kwargs["embedding_model_name"], _BGE)
        self.assertEqual([it.channel for it in items], ["st_bge", "st_bge"])
        self.assertEqual([it.model_name for it in items], [_BGE, _BGE])

    def test_default_st_active_is_kosimcse_unchanged(self) -> None:
        items, m = self._run(None)
        self.assertEqual(m.call_args.kwargs["embedding_model_name"], _KOSIMCSE)
        self.assertTrue(all(it.channel == "st" for it in items))
        self.assertTrue(all(it.model_name == _KOSIMCSE for it in items))


if __name__ == "__main__":
    unittest.main()
