"""006 검색 CLI 진입점(run_search) — 인자 파싱·매핑 단위 테스트(네트워크/DB 없음).

CLI 의 순수 부분(모달리티 파싱, 인자→search_hybrid 매핑)만 검증한다. 실제 검색은
``search_fn`` 주입으로 대체한다(실 동작은 T016 e2e).
"""

from __future__ import annotations

import argparse
import unittest

from processing.app import run_search


class TestResolveModalities(unittest.TestCase):
    """069 T301: run_search 는 공유 파서(parse_modalities_csv)로 파싱하고 유효값을 검증한다."""

    def test_none_and_blank_return_none(self) -> None:
        self.assertIsNone(run_search._resolve_modalities(None))
        self.assertIsNone(run_search._resolve_modalities(""))
        self.assertIsNone(run_search._resolve_modalities("   "))

    def test_comma_split_and_strip(self) -> None:
        self.assertEqual(run_search._resolve_modalities("text,image"), ["text", "image"])
        self.assertEqual(run_search._resolve_modalities(" text , , image "), ["text", "image"])

    def test_rejects_unknown_modality(self) -> None:
        # 069 P3-12: 오타 모달리티는 raw traceback 대신 명확한 ValueError 로 거부(main 이 parser.error 로 변환).
        with self.assertRaises(ValueError):
            run_search._resolve_modalities("text,bogus")

    def test_main_rejects_unknown_modality_cleanly(self) -> None:
        # CLI 진입점: 미지 모달리티는 argparse parser.error → SystemExit(2)(usage 출력·traceback 없음).
        # init_settings 는 DB 를 요구하므로 mock — 인자 검증 후 검색 이전에 거부됨을 확인.
        import sys
        from unittest.mock import patch

        argv = ["run_search", "--env", "dev", "--query", "x", "--modalities", "text,bogus"]
        with patch.object(sys, "argv", argv), patch(
            "src.config.settings.init_settings"
        ):
            with self.assertRaises(SystemExit) as ctx:
                run_search.main()
        self.assertEqual(ctx.exception.code, 2)


class TestRunMapsArgs(unittest.TestCase):
    def test_run_maps_args_to_search_kwargs(self) -> None:
        captured: dict[str, object] = {}

        def fake_search(query: str, **kw: object) -> dict[str, object]:
            captured["query"] = query
            captured.update(kw)
            return {"query": query, "results": {}, "meta": {}}

        ns = argparse.Namespace(query="질의", modalities="text,image", limit=5)
        out = run_search._run(ns, search_fn=fake_search)

        self.assertEqual(captured["query"], "질의")
        self.assertEqual(captured["modalities"], ["text", "image"])
        self.assertEqual(captured["limit_per_bucket"], 5)
        # 069 US-C: 037 로 no-op 였던 text_hybrid_alpha·min_scores 는 더 이상 전달하지 않는다.
        self.assertNotIn("text_hybrid_alpha", captured)
        self.assertNotIn("min_scores", captured)
        self.assertEqual(out["query"], "질의")

    def test_run_no_modalities_passes_none(self) -> None:
        captured: dict[str, object] = {}

        def fake_search(query: str, **kw: object) -> dict[str, object]:
            captured.update(kw)
            return {}

        ns = argparse.Namespace(query="질의", modalities=None, limit=20)
        run_search._run(ns, search_fn=fake_search)
        self.assertIsNone(captured["modalities"])


if __name__ == "__main__":
    unittest.main()
