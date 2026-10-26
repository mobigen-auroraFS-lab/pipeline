"""spec 114 — 재요약 배치(``run_resummarize``) 단위 테스트(LLM · DB · 검색 엔진은 대역).

무엇을 봉인하나
① 설정(``SUMMARY_TITLE_HINT``)이 꺼져 있으면 **시작하지 않는다** — 켜지 않고 돌리면 제목 없이 지금과 같은 요약을
   다시 만들 뿐이라 시간만 쓰고, 「재생성했다」는 착각을 남긴다.
② 저장된 재료를 재사용한다 — 영상은 저장된 장면 분석, 음성은 저장된 STT(재실행 없음), 문서 · 사진은 원본 파일.
   사진은 요약이 곧 임베딩 재료라 텍스트 임베딩도 다시 만든다(라벨은 그대로 · objects 는 메타에 남기지 않는다).
③ 확장 메타는 **요약 결과 키만** 바꾼다(about · labels · keyframes · stt 등 나머지는 그대로). 이번에 제목만
   요약이 아니면 옛 ``title_only`` 는 지운다.
④ 주제는 확인된 제목(title_match == yes)만 붙여 다시 판정하고, about 은 새 요약으로 다시 뽑는다.
⑤ 자산 하나가 실패해도 나머지는 계속한다 — 실패 목록 · 판단 분포를 보고한다.
⑥ 다시 만들 것이 없는 자산은 건너뛴다 — 제목 없는 자산(조회 단계) · 새 요약이 빈 자산(옛 요약을 빈 값으로 덮지 않는다).
⑦ 쓰기는 주제 판정(LLM) **뒤에** 몰아 둔다 — LLM 을 기다리는 동안 자산 행을 잠그지 않는다.
⑧ 색인만 실패하면 DB 는 새 값인 채 ``index_failed`` 로 따로 모은다(이어 하기가 건너뛰므로 ``--index-only`` 로 다시).
"""

from __future__ import annotations

import unittest
from unittest import mock

from processing.app import run_resummarize as RR


def _cfg(on: bool = True) -> mock.Mock:
    cfg = mock.Mock()
    cfg.vlm.summary_title_hint = on
    return cfg


def _asset(mod, **ext):
    paths = {"video": "/a/조디악 리뷰.mp4", "audio": "/a/심수봉 - 부부행진곡.mp3", "text": "/a/영화 해설 자막.txt",
             "image": "/a/천마총 관모_02.jpg"}
    return {"asset_id": "018f0000-0000-7000-8000-0000000000aa", "modality": mod, "fs_path": paths[mod],
            "domain_label": "general", "ext_meta": {"summary": "옛 요약", "keywords": ["옛"], "about": ["옛"], **ext}}


class FetchTargetsTest(unittest.TestCase):
    def _conn(self, rows):
        cur = mock.MagicMock()
        cur.fetchall.return_value = rows
        cur.__enter__.return_value = cur
        conn = mock.MagicMock()
        conn.cursor.return_value = cur
        return conn, cur

    def test_종류_상태_이어하기_조건(self) -> None:
        conn, cur = self._conn([])
        RR.fetch_targets(conn, ["video", "audio"], resume=True, limit=5)
        sql, params = cur.execute.call_args.args
        flat = " ".join(sql.split())
        self.assertIn("a.status = 'registered'", flat)
        self.assertIn("a.modality = ANY(%s)", flat)
        self.assertIn("NOT (m.ext_meta ? 'title_match')", flat)
        self.assertEqual((["video", "audio"], 5), params)

    def test_이어하기를_끄면_조건이_없다(self) -> None:
        conn, cur = self._conn([])
        RR.fetch_targets(conn, ["text"], resume=False)
        self.assertNotIn("title_match", cur.execute.call_args.args[0])

    def test_뜻_있는_제목이_없는_자산은_뺀다(self) -> None:
        rows = [{"asset_id": "a1", "modality": "audio", "fs_path": "/x/새 녹음 3.m4a", "domain_label": "general", "ext_meta": {}},
                {"asset_id": "a2", "modality": "audio", "fs_path": "/x/심수봉 - 부부행진곡.mp3", "domain_label": "general",
                 "ext_meta": {}}]
        conn, _ = self._conn(rows)
        self.assertEqual(["a2"], [r["asset_id"] for r in RR.fetch_targets(conn, ["audio"])])
        conn, _ = self._conn(rows)
        self.assertEqual(["a1", "a2"], [r["asset_id"] for r in RR.fetch_targets(conn, ["audio"], titled_only=False)])

    def test_제외_이름은_표시_파일명으로_거른다(self) -> None:
        rows = [{"asset_id": "a1", "modality": "text", "fs_path": "/x/018f0000-0000-7000-8000-000000000001__봉화 쌍벽당 종택.txt",
                 "domain_label": "general", "ext_meta": {}},
                {"asset_id": "a2", "modality": "text", "fs_path": "/x/영화 해설.txt", "domain_label": "general", "ext_meta": {}}]
        conn, _ = self._conn(rows)
        out = RR.fetch_targets(conn, ["text"], exclude_names={"봉화 쌍벽당 종택.txt"})
        self.assertEqual(["a2"], [r["asset_id"] for r in out])


    def test_제외_이름은_NFC_로_맞춰_비교한다(self) -> None:
        import unicodedata
        nfd = unicodedata.normalize("NFD", "봉화 쌍벽당 종택.txt")
        rows = [{"asset_id": "a1", "modality": "text", "fs_path": f"/x/{nfd}", "domain_label": "general", "ext_meta": {}}]
        conn, _ = self._conn(rows)
        self.assertEqual([], RR.fetch_targets(conn, ["text"], exclude_names={"봉화 쌍벽당 종택.txt"}))


class BuildUpdatesTest(unittest.TestCase):
    def test_설정이_꺼져_있으면_시작하지_않는다(self) -> None:
        with self.assertRaises(RR.TitleHintOffError):
            RR.build_updates(_asset("audio", stt="x"), _cfg(False))

    def test_영상은_저장된_장면으로(self) -> None:
        kf = [{"scene_index": 0, "summary": {"summary": "스튜디오"}}]
        summ = mock.Mock(return_value={"summary": "새", "keywords": ["k"], "objects": ["o"], "title_match": "yes"})
        with mock.patch.multiple("src.llm.video_summarizer", summarize_video_from_scene_results=summ):
            up = RR.build_updates(_asset("video", keyframes=kf), _cfg())
        self.assertEqual(kf, summ.call_args.args[0])
        self.assertEqual("조디악 리뷰", summ.call_args.kwargs["title"])
        self.assertEqual({"summary": "새", "keywords": ["k"], "objects": ["o"], "title_match": "yes"}, up)

    def test_음성은_저장된_STT_로_stt_는_바꾸지_않는다(self) -> None:
        summ = mock.Mock(return_value={"summary": "새", "keywords": ["k"], "stt": "", "title_only": True, "title_match": "unsure"})
        with mock.patch.multiple("src.llm.text_summarizer", summarize_and_extract_keywords_from_audio=summ):
            up = RR.build_updates(_asset("audio", stt=""), _cfg())
        self.assertEqual("", summ.call_args.args[0])
        self.assertNotIn("stt", up)
        self.assertTrue(up["title_only"])

    def test_문서는_원본_파일과_확장자(self) -> None:
        summ = mock.Mock(return_value={"summary": "새", "keywords": ["k"], "title_match": "no"})
        with mock.patch.multiple("src.llm.text_summarizer", summarize_and_extract_keywords=summ):
            up = RR.build_updates(_asset("text"), _cfg())
        self.assertEqual("/a/영화 해설 자막.txt", summ.call_args.args[0])
        self.assertEqual("txt", summ.call_args.kwargs["file_kind"])
        self.assertEqual("no", up["title_match"])

    def test_새_요약이_비면_건너뛴다(self) -> None:
        summ = mock.Mock(return_value={"summary": "", "keywords": [], "objects": []})
        with mock.patch.multiple("src.llm.video_summarizer", summarize_video_from_scene_results=summ):
            with self.assertRaises(RR.SkipAsset):
                RR.build_updates(_asset("video", keyframes=[]), _cfg())

    def test_사진은_원본으로_objects_는_남기지_않는다(self) -> None:
        summ = mock.Mock(return_value={"summary": "새", "keywords": ["관모"], "objects": ["관모"], "title_match": "yes"})
        with mock.patch.multiple("src.llm.image_summarizer", summarize_image_caption_keywords_objects=summ):
            up = RR.build_updates(_asset("image"), _cfg())
        self.assertEqual("/a/천마총 관모_02.jpg", summ.call_args.kwargs["file_path"])
        self.assertEqual("천마총 관모_02", summ.call_args.kwargs["title"])
        self.assertEqual({"summary": "새", "keywords": ["관모"], "title_match": "yes"}, up)


class ApplyUpdatesTest(unittest.TestCase):
    def _run(self, asset, updates, about=("조디악",)):
        conn = mock.MagicMock()
        validate, topic, persist = mock.Mock(), mock.Mock(), mock.Mock()
        RR.apply_updates(conn, asset, updates, _cfg(), about=list(about), validate_fn=validate, topic_fn=topic,
                         persist_about_fn=persist)
        return conn, validate, topic, persist

    def test_요약_키만_검사하고_합쳐_쓴다(self) -> None:
        a = _asset("audio", stt="전사", labels=[{"label": "x"}], title_only=True)
        up = {"summary": "새", "keywords": ["k"], "title_match": "yes"}
        conn, validate, topic, about = self._run(a, up)
        self.assertEqual(("general", up), (validate.call_args.args[1], validate.call_args.args[2]))
        sql, params = conn.execute.call_args.args
        flat = " ".join(sql.split())
        self.assertIn("ext_meta = (ext_meta || %s::jsonb) - %s::text[]", flat)
        self.assertEqual(["title_only"], params[1])   # 이번엔 제목만 요약이 아니다 → 옛 표시 지움

    def test_주제는_확인된_제목만_about_은_미리_뽑은_값을_저장(self) -> None:
        a = _asset("video")
        conn, _, topic, persist = self._run(a, {"summary": "새 요약", "keywords": ["k"], "title_match": "yes"})
        kw = topic.call_args.kwargs
        self.assertIn("새 요약", kw["self_text"])
        self.assertEqual("조디악 리뷰", kw["title"])
        self.assertEqual((a["asset_id"], ["조디악"]), persist.call_args.args[1:])
        conn2, _, topic2, _ = self._run(a, {"summary": "새 요약", "keywords": ["k"], "title_match": "no"})
        self.assertNotIn("title", topic2.call_args.kwargs)

    def test_쓰기는_주제_판정_뒤에(self) -> None:
        order = []
        conn = mock.MagicMock()
        conn.execute.side_effect = lambda *a, **k: order.append("update")
        RR.apply_updates(conn, _asset("video"), {"summary": "새", "keywords": ["k"], "title_match": "yes"}, _cfg(),
                         about=[], validate_fn=mock.Mock(), topic_fn=lambda *a, **k: order.append("topic"),
                         persist_about_fn=lambda *a: order.append("about"))
        self.assertEqual(["topic", "update", "about"], order)

    def test_사진_임베딩은_같은_트랜잭션에서_덮어쓴다(self) -> None:
        from processing.dispatch.types import EmbeddingItem
        item = EmbeddingItem(channel="st_api", vector=[0.1, 0.2], model_name="BAAI/bge-m3", chunk_index=0)
        conn = mock.MagicMock()
        RR.apply_updates(conn, _asset("image"), {"summary": "새", "keywords": ["k"], "title_match": "yes"}, _cfg(),
                         about=[], validate_fn=mock.Mock(), topic_fn=mock.Mock(), persist_about_fn=mock.Mock(),
                         embedding=item)
        sql, params = conn.execute.call_args_list[-1].args
        flat = " ".join(sql.split())
        self.assertIn("INSERT INTO asset_embedding", flat)
        self.assertIn("ON CONFLICT (asset_id, channel, chunk_index) DO UPDATE SET embedding = EXCLUDED.embedding", flat)
        self.assertEqual((_asset("image")["asset_id"], "st_api", 0, [0.1, 0.2], "BAAI/bge-m3", None), params)

    def test_임베딩이_없으면_임베딩_표를_건드리지_않는다(self) -> None:
        conn, *_ = self._run(_asset("video"), {"summary": "새", "keywords": ["k"], "title_match": "yes"})
        self.assertFalse(any("asset_embedding" in c.args[0] for c in conn.execute.call_args_list))


class ProcessAssetTest(unittest.TestCase):
    def _db(self):
        db = mock.MagicMock()
        db.transaction.return_value.__enter__.return_value = mock.MagicMock()
        return db

    def _run(self, *, dry_run=False, index_fn=None, asset=None, **extra):
        up = {"summary": "새", "keywords": ["k"], "title_match": "yes"}
        db, lines = self._db(), []
        fns = {"extract_about_fn": mock.Mock(return_value=["조디악"]), "validate_fn": mock.Mock(),
               "topic_fn": mock.Mock(), "persist_about_fn": mock.Mock(), **extra}
        with mock.patch.object(RR, "build_updates", return_value=up), \
                mock.patch.object(RR, "apply_updates") as apply:
            res = RR.process_asset(asset or _asset("video"), _cfg(), db=db, dry_run=dry_run, index_fn=index_fn,
                                   write_line=lines.append, **fns)
        fns["apply"] = apply
        return res, db, fns, lines

    def test_dry_run_은_DB_도_about_도_건드리지_않는다(self) -> None:
        index_fn = mock.Mock()
        res, db, fns, lines = self._run(dry_run=True, index_fn=index_fn)
        db.transaction.assert_not_called()
        fns["extract_about_fn"].assert_not_called()
        index_fn.assert_not_called()
        self.assertEqual("새", lines[0]["new"]["summary"])
        self.assertEqual("옛 요약", lines[0]["old"]["summary"])

    def test_색인만_실패하면_index_error_로_돌려준다(self) -> None:
        res, db, fns, lines = self._run(index_fn=mock.Mock(side_effect=RuntimeError("검색 엔진 끊김")))
        fns["apply"].assert_called_once()
        self.assertIn("검색 엔진 끊김", res["index_error"])
        self.assertIn("검색 엔진 끊김", lines[0]["index_error"])

    def test_정상이면_about_은_새_요약으로_뽑는다(self) -> None:
        res, _, fns, _ = self._run(index_fn=mock.Mock())
        self.assertEqual("새", fns["extract_about_fn"].call_args.args[0])
        self.assertIsNone(res["index_error"])
        self.assertIsNone(fns["apply"].call_args.kwargs["embedding"])   # 영상은 임베딩 무관

    def test_사진은_새_요약과_옛_라벨로_임베딩을_만들어_넘긴다(self) -> None:
        labels = [{"label": "관모", "score": 0.3}]
        embed = mock.Mock(return_value="임베딩")
        res, _, fns, _ = self._run(asset=_asset("image", labels=labels), image_embed_fn=embed)
        self.assertEqual({"summary": "새", "keywords": ["k"], "labels": labels}, embed.call_args.args[0])
        self.assertEqual("임베딩", fns["apply"].call_args.kwargs["embedding"])

    def test_사진인데_임베딩_함수가_없으면_기록하지_않는다(self) -> None:
        with self.assertRaises(RuntimeError):
            self._run(asset=_asset("image"))

    def test_사진_dry_run_은_임베딩도_만들지_않는다(self) -> None:
        embed = mock.Mock()
        self._run(asset=_asset("image"), image_embed_fn=embed, dry_run=True)
        embed.assert_not_called()


class RunTest(unittest.TestCase):
    def test_실패는_격리하고_분포를_보고한다(self) -> None:
        targets = [{"asset_id": f"a{i}", "modality": "audio"} for i in range(6)]

        def process(t):
            aid = t["asset_id"]
            if aid == "a2":
                raise RuntimeError("LLM 오류")
            if aid == "a4":
                raise RR.SkipAsset("새 요약이 비었다")
            return {"updates": {"title_match": "no" if aid == "a3" else "yes", "title_only": False},
                    "index_error": "끊김" if aid == "a5" else None}
        rep = RR.run_resummarize(targets, process_fn=process, workers=2)
        self.assertEqual(4, rep["ok"])
        self.assertEqual(["a2"], [f["asset_id"] for f in rep["failed"]])
        self.assertEqual(["a4"], [f["asset_id"] for f in rep["skipped"]])
        self.assertEqual(["a5"], [f["asset_id"] for f in rep["index_failed"]])
        self.assertEqual({"yes": 3, "no": 1}, rep["title_match"])

    def test_목록은_자산_id_순(self) -> None:
        targets = [{"asset_id": f"a{i}", "modality": "audio"} for i in (3, 1, 2)]

        def process(t):
            raise RuntimeError("x")
        rep = RR.run_resummarize(targets, process_fn=process, workers=3)
        self.assertEqual(["a1", "a2", "a3"], [f["asset_id"] for f in rep["failed"]])


class InvalidIdsTest(unittest.TestCase):
    def test_UUID_모양이_아닌_것만(self) -> None:
        good = "018f0000-0000-7000-8000-0000000000aa"
        self.assertEqual(["abc", "018f-bad"], RR.invalid_ids([good, "abc", "018f-bad"]))
        self.assertEqual([], RR.invalid_ids(None))


if __name__ == "__main__":
    unittest.main()
