"""spec 114 — 파이프라인의 「제목 참고」 배선 단위 테스트(LLM · DB · 모델은 대역).

무엇을 봉인하나
① 설정(``vlm.summary_title_hint``)이 **정확히 True** 일 때만 켜진다 — 꺼져 있으면 요약기에 ``title`` 인자조차
   넘기지 않아 지금과 똑같이 동작한다(코드가 먼저 배포돼도 운영 적재가 바뀌지 않게).
② 스킬 4종(문서 · 음성 · 사진 · 영상)이 제목을 요약기에 넘기고, 요약기가 돌려준 ``title_match`` · ``title_only`` 가
   확장 메타(ext_meta)에 남는다.
③ 주제 판정은 **제목이 내용과 맞다고 확인된 경우(title_match == yes)만** 자기 텍스트 앞에 제목을 붙이고 판
   ``asset_topic.v2`` 로 남긴다. 그 밖에는 지금과 같다(v1).
④ 개체 판정은 제목이 내용과 다르다고 판단된 자산(title_match == no)에서 파일명 후보를 쓰지 않는다 — 틀린 파일명이
   개체를 오염시키던 경로를 막는다.
"""

from __future__ import annotations

import unittest
import uuid
from unittest import mock
from unittest.mock import MagicMock, patch

from processing.dispatch.types import ExtractContext
from processing.skills import title_context as TC
from processing.skills.meta_split import split_core_ext


def _cfg(on: bool) -> mock.Mock:
    cfg = mock.Mock()
    cfg.vlm.summary_title_hint = on
    cfg.vlm.labels_score_min = 0.1
    cfg.vlm.image_labels_meta_top_k = 5
    cfg.video.labels_meta_top_k = 5
    cfg.video.max_keyframes = 5
    # 영상 스킬이 키프레임 중복 제거 설정을 검사한다 — 꺼 두고 형식만 맞춘다(이 테스트의 관심사가 아니다).
    cfg.video.dedup_enabled = False
    cfg.video.dedup_hash_max = 6
    cfg.video.dedup_ssim_min = 0.9
    cfg.video.dedup_ssim_gray_lo = 0.1
    cfg.video.dedup_hist_min = 0.9
    cfg.video.dedup_compare_mode = "global"
    cfg.video.dedup_recent_window = 3
    cfg.encoding = "utf-8"
    return cfg


class TitleContextTest(unittest.TestCase):
    def test_정확히_True_일_때만_켜진다(self) -> None:
        self.assertTrue(TC.hint_enabled(_cfg(True)))
        self.assertFalse(TC.hint_enabled(_cfg(False)))
        self.assertFalse(TC.hint_enabled(mock.Mock()))   # Mock 속성은 참처럼 보이지만 True 가 아니다
        self.assertFalse(TC.hint_enabled(object()))
        self.assertFalse(TC.hint_enabled(None))

    def test_요약용_제목(self) -> None:
        self.assertEqual("경주 불국사 다보탑", TC.title_for_summary("/d/경주 불국사 다보탑.txt", _cfg(True)))
        self.assertIsNone(TC.title_for_summary("/d/경주 불국사 다보탑.txt", _cfg(False)))
        self.assertIsNone(TC.title_for_summary("/d/1612816.jpg", _cfg(True)))

    def test_확인된_제목은_yes_일_때만(self) -> None:
        path = "/d/조디악 리뷰.mp4"
        self.assertEqual("조디악 리뷰", TC.confirmed_title({"title_match": "yes"}, path, _cfg(True)))
        for tm in ("no", "unsure", None):
            with self.subTest(tm=tm):
                self.assertIsNone(TC.confirmed_title({"title_match": tm}, path, _cfg(True)))
        self.assertIsNone(TC.confirmed_title({"title_match": "yes"}, path, _cfg(False)))
        self.assertIsNone(TC.confirmed_title(None, path, _cfg(True)))

    def test_요약기_인자는_제목이_있을_때만(self) -> None:
        self.assertEqual({"title": "가나"}, TC.summarizer_kwargs("가나"))
        self.assertEqual({}, TC.summarizer_kwargs(None))


class TitleKeysGuardTest(unittest.TestCase):
    """제목 참고를 켰는데 ext_meta 키(v308)가 등록되지 않았으면 시작하자마자 멈춘다 — 순서가 꼬이면 적재가 「미등록 키」로
    전부 롤백된다(migration-reviewer 권고)."""

    def test_빠진_키를_도메인별로_알려_준다(self) -> None:
        allowed = {"general": {"summary", "title_match", "title_only"}, "medical": {"summary"}, "review": set()}
        with patch("src.registry.ext_meta_field_registry.fetch_allowed_ext_keys", side_effect=lambda c, d: allowed[d]):
            missing = TC.missing_title_keys(MagicMock(), ("general", "medical", "review"))
        self.assertEqual(["medical:title_match", "medical:title_only"], missing)

    def test_적재는_켜져_있고_키가_빠졌으면_시작하지_않는다(self) -> None:
        from processing.app import run_ingest as ri
        db = MagicMock()
        with patch.object(ri, "missing_title_keys", return_value=["general:title_match"]):
            with self.assertRaises(RuntimeError):
                ri.run_ingest([], db=db, settings=_cfg(True))

    def test_꺼져_있으면_확인하지_않는다(self) -> None:
        from processing.app import run_ingest as ri
        with patch.object(ri, "missing_title_keys") as m, patch.object(ri, "_make_opensearch_indexer", return_value=None):
            ri.run_ingest([], db=MagicMock(), settings=_cfg(False))
        m.assert_not_called()


class MetaSplitTest(unittest.TestCase):
    def test_제목_판단은_확장_메타로(self) -> None:
        core, ext = split_core_ext({"width": 10, "summary": "s", "title_match": "yes", "title_only": True})
        self.assertEqual({"width": 10}, core)
        self.assertEqual({"summary": "s", "title_match": "yes", "title_only": True}, ext)


class SkillWiringTest(unittest.TestCase):
    _PATH = "/d/[리뷰] 영화 조디악.mp4"
    _TITLE = "[리뷰] 영화 조디악"

    def test_문서_스킬(self) -> None:
        from processing.skills import text_skill
        for on in (True, False):
            with self.subTest(on=on):
                summ = mock.Mock(return_value={"summary": "s", "keywords": ["k"], "title_match": "yes"})
                cfg = _cfg(on)
                cfg.embed.chunk_size = 512
                with mock.patch.multiple("processing.extractors.text_meta_extractor", extract_text_meta=mock.Mock(return_value={})), \
                        mock.patch.multiple("src.llm.text_summarizer", summarize_and_extract_keywords=summ), \
                        mock.patch.object(text_skill, "active_embed_model", return_value="m"):
                    rec = text_skill._extract_text_meta(ExtractContext(file_path="/d/경주 불국사 다보탑.txt", modality="txt", settings=cfg))
                kwargs = summ.call_args.kwargs
                if on:
                    self.assertEqual("경주 불국사 다보탑", kwargs["title"])
                    self.assertEqual("yes", rec.ext_meta["title_match"])
                else:
                    self.assertNotIn("title", kwargs)

    def test_음성_스킬(self) -> None:
        from processing.skills import audio_skill
        for on in (True, False):
            with self.subTest(on=on):
                summ = mock.Mock(return_value={"summary": "", "keywords": [], "stt": "", "title_only": True, "title_match": "unsure"})
                with mock.patch.multiple("processing.extractors.audio_meta_extractor", extract_audio_meta=mock.Mock(return_value={})), \
                        mock.patch.multiple("processing.preprocess.stt", transcribe_audio_local=mock.Mock(return_value={"text": ""})), \
                        mock.patch.multiple("src.llm.text_summarizer", summarize_and_extract_keywords_from_audio=summ):
                    rec = audio_skill._extract_audio_meta(ExtractContext(file_path="/d/심수봉 - 부부행진곡.mp3", modality="mp3", settings=_cfg(on)))
                kwargs = summ.call_args.kwargs
                if on:
                    self.assertEqual("심수봉 - 부부행진곡", kwargs["title"])
                    self.assertTrue(rec.ext_meta["title_only"])
                else:
                    self.assertNotIn("title", kwargs)

    def test_사진_스킬(self) -> None:
        from processing.skills import image_skill
        summ = mock.Mock(return_value={"summary": "s", "keywords": ["k"], "objects": [], "title_match": "no"})
        with mock.patch.multiple("processing.extractors.image_meta_extractor", extract_image_meta=mock.Mock(return_value={})), \
                mock.patch.multiple("src.llm.image_summarizer", summarize_image_caption_keywords_objects=summ), \
                mock.patch.multiple("src.embedders.image_embedder",
                                    zero_shot_tag_image_korean_clip=mock.Mock(return_value={"label_scores": [], "clip_image_embedding": [0.0]}),
                                    clip_zero_shot_ko_meta_items=mock.Mock(return_value=[])):
            rec = image_skill._extract_image_meta(ExtractContext(file_path="/d/천마총 관모_02.jpg", modality="jpg", settings=_cfg(True)))
        self.assertEqual("천마총 관모_02", summ.call_args.kwargs["title"])
        self.assertEqual("no", rec.ext_meta["title_match"])

    def test_영상_스킬(self) -> None:
        from processing.skills import video_skill
        summ = mock.Mock(return_value={"summary": "s", "keywords": ["k"], "objects": [], "title_match": "yes"})
        with mock.patch.multiple("processing.preprocess.video_keyframes",
                                 extract_video_representative_frame_bytes=mock.Mock(return_value=[]),
                                 extract_video_basic_meta=mock.Mock(return_value={})), \
                mock.patch.multiple("src.llm.video_summarizer", summarize_video_from_scene_results=summ), \
                mock.patch.multiple("src.embedders.video_embedder", embed_video_keyframes_clip=mock.Mock(return_value={"keyframes": []})):
            rec = video_skill._extract_video_meta(ExtractContext(file_path=self._PATH, modality="mp4", settings=_cfg(True)))
        self.assertEqual(self._TITLE, summ.call_args.kwargs["title"])
        self.assertEqual("yes", rec.ext_meta["title_match"])


class TopicTest(unittest.TestCase):
    def _run(self, *, self_text, title=None, settings=None, load=None):
        from processing.classify import asset_topic as AT
        seen = {}

        def pick(st, cands, client=None):
            seen["self_text"] = st
            return {"topic_ko": "미디어·엔터테인먼트", "confidence": 0.9}
        upsert = MagicMock()
        patches = [patch.object(AT, "topic_candidates_for_self_text", return_value=["미디어·엔터테인먼트"]),
                   patch.object(AT, "_pick_topic_via_llm", side_effect=pick),
                   patch.object(AT, "fetch_closed_subtopics", return_value=[]),
                   patch.object(AT, "_lookup_topic_en", return_value="media"),
                   patch.object(AT, "_upsert_asset_topic", upsert)]
        if load:
            patches += [patch.object(AT, "_load_self_meta", return_value=load[0]),
                        patch.object(AT, "_load_title_context", return_value=load[1])]
        for p in patches:
            p.start()
        try:
            AT.classify_asset_topic(MagicMock(), "A1", self_text=self_text, title=title, settings=settings)
        finally:
            for p in patches:
                p.stop()
        return seen["self_text"], upsert.call_args

    def test_제목이_없으면_지금과_같다(self) -> None:
        from processing.classify import asset_topic as AT
        st, call = self._run(self_text="스튜디오 대화")
        self.assertEqual("스튜디오 대화", st)
        self.assertEqual(AT.POLICY_VERSION, call.kwargs.get("policy_version", AT.POLICY_VERSION))

    def test_확인된_제목이면_앞에_붙이고_판_v2(self) -> None:
        from processing.classify import asset_topic as AT
        st, call = self._run(self_text="스튜디오 대화", title="[리뷰] 영화 조디악")
        self.assertEqual("제목: [리뷰] 영화 조디악\n스튜디오 대화", st)
        self.assertEqual(AT.POLICY_VERSION_TITLE, call.kwargs["policy_version"])

    def test_DB_경로는_저장된_title_match_로_스스로_정한다(self) -> None:
        st, _ = self._run(self_text=None, settings=_cfg(True),
                          load=(("스튜디오 대화", ["대화"], None), ("yes", "/d/[리뷰] 영화 조디악.mp4")))
        self.assertTrue(st.startswith("제목: [리뷰] 영화 조디악\n"))
        st2, _ = self._run(self_text=None, settings=_cfg(True),
                           load=(("스튜디오 대화", ["대화"], None), ("no", "/d/[리뷰] 영화 조디악.mp4")))
        self.assertFalse(st2.startswith("제목:"))

    def test_DB_경로도_설정이_꺼져_있으면_제목_조회를_하지_않는다(self) -> None:
        from processing.classify import asset_topic as AT
        with patch.object(AT, "_load_self_meta", return_value=("스튜디오 대화", ["대화"], None)), \
                patch.object(AT, "_load_title_context", return_value=("yes", "/d/[리뷰] 영화 조디악.mp4")) as load_title:
            st, _ = self._run(self_text=None, settings=_cfg(False))
        self.assertFalse(st.startswith("제목:"))
        load_title.assert_not_called()


class BindingTest(unittest.TestCase):
    def test_title_match_가_no_면_파일명_후보를_쓰지_않는다(self) -> None:
        from processing.app import run_mm_meta_binding as rb
        from tests.test_run_mm_meta_binding import _FakeConn
        a1, a2 = "018f0000-0000-7000-8000-0000000001a1", "018f0000-0000-7000-8000-0000000001a2"
        conn = _FakeConn([
            {"asset_id": uuid.UUID(a1), "fs_path": "/d/[극한직업] 왜 안 웃기지.jpg", "summary": "s", "keywords": ["k"], "title_match": "no"},
            {"asset_id": uuid.UUID(a2), "fs_path": "/d/[극한직업] 왜 안 웃기지.jpg", "summary": "s", "keywords": ["k"], "title_match": "yes"},
        ])
        rows = rb.fetch_binding_targets(conn, rejudge=True)
        self.assertIn("title_match", conn.executed[0][0])
        self.assertIsNone(rows[0]["name_hint"])
        self.assertEqual((), rows[0]["name_candidates"])
        self.assertEqual("[극한직업] 왜 안 웃기지", rows[1]["name_hint"])
        self.assertIn("극한직업", rows[1]["name_candidates"])

    def test_판단이_없는_자산은_지금처럼_파일명_후보를_쓴다(self) -> None:
        # 설정을 켜기 전 적재분 · 꺼진 채 적재한 자산 — title_match 컬럼이 NULL 이면 지금과 같아야 한다.
        from processing.app import run_mm_meta_binding as rb
        from tests.test_run_mm_meta_binding import _FakeConn
        a1 = "018f0000-0000-7000-8000-0000000001a3"
        conn = _FakeConn([{"asset_id": uuid.UUID(a1), "fs_path": "/d/[극한직업] 왜 안 웃기지.jpg", "summary": "s",
                           "keywords": ["k"], "title_match": None}])
        rows = rb.fetch_binding_targets(conn, rejudge=True)
        self.assertEqual("[극한직업] 왜 안 웃기지", rows[0]["name_hint"])
        self.assertIn("극한직업", rows[0]["name_candidates"])


if __name__ == "__main__":
    unittest.main()
