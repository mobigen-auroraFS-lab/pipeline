"""run_opensearch_resync 복구 도구 단위 테스트 (spec 020 G4 — OS·DB 불필요).

이 CLI 는 정상 경로(증분 훅, G3)가 아니라 **복구/관리** 도구다 — 인덱스 손실·드리프트·
스키마 변경 시 PG 의 `registered` 자산 전체를 OpenSearch 로 재색인한다(US2). 실제 OS/DB
재색인은 G5(사람)이고, 본 그룹은 러너의 **조립 로직**까지만 단위로 덮는다.

설계 경계(docs/테스트_가이드.md §0 하이브리드) — 017/019 measure 러너와 동형:
  - **순수 조립부**(여기서 검증): `run_resync(client, conn, *, channel, index, recreate, sync_fn)`
    가 동기화 코어(`sync_all`)를 **주입 seam** 으로 올바른 인자(index·channel·recreate)로 호출하고
    결과(status·ok·errors)를 보고한다 — 가짜 client/conn/sync_fn 으로 OS·DB 없이.
  - **실행(IO) 부트스트랩**: `main()` 만 실제 get_client·PostgresUtil 을 잡는다(G5·사람). 단위는 미접촉.
"""

from __future__ import annotations

import inspect
import unittest
from unittest import mock

from processing.app import run_opensearch_resync as rr


class TestParser(unittest.TestCase):
    """인자 파서: --env·--channel(미지정=활성)·--index(미지정=설정)·--recreate."""

    def test_defaults(self) -> None:
        # 미지정 시 channel/index 는 None 으로 두고 main 이 활성 채널·설정 인덱스로 해소한다.
        args = rr._build_parser().parse_args([])
        self.assertEqual(args.env, "dev")
        self.assertIsNone(args.channel)
        self.assertIsNone(args.index)
        self.assertFalse(args.recreate)  # 기본은 비파괴 upsert

    def test_explicit(self) -> None:
        args = rr._build_parser().parse_args(
            ["--env", "prod", "--channel", "st_bge", "--index", "assets_bge", "--recreate"]
        )
        self.assertEqual(args.env, "prod")
        self.assertEqual(args.channel, "st_bge")
        self.assertEqual(args.index, "assets_bge")
        self.assertTrue(args.recreate)

    def test_pipeline_option_removed(self) -> None:
        # 027: 서버 융합 파이프라인이 클라이언트 융합으로 이동해 --ensure-pipeline 옵션은 제거됐다 —
        # 미지원 인자는 argparse 가 SystemExit 로 거부한다(잔존 참조 0의 행동 봉인).
        with self.assertRaises(SystemExit):
            rr._build_parser().parse_args(["--ensure-pipeline"])


class TestRunResync(unittest.TestCase):
    """순수 조립부 run_resync — sync_fn 주입으로 OS·DB 없이 인자 전달·결과 보고를 검증."""

    def test_calls_sync_fn_with_resolved_args_and_reports(self) -> None:
        client = mock.MagicMock(name="client")
        conn = mock.MagicMock(name="conn")
        sync_fn = mock.MagicMock(return_value=("created", 7, []))

        report = rr.run_resync(
            client, conn, channel="st", index="assets", recreate=True, sync_fn=sync_fn
        )

        # (client, conn) 위치인자 + index·channel·recreate 키워드로 정확히 1회 호출(조립 정확성).
        sync_fn.assert_called_once_with(
            client, conn, index="assets", channel="st", recreate=True,
            nori_user_words=None, noise_patterns=(),  # 026: settings 주입 패스스루(미지정=기본)
        )
        # 결과 보고: 상태·색인수(ok)·오류 + 색인 맥락(channel·index·recreate).
        self.assertEqual(report["status"], "created")
        self.assertEqual(report["ok"], 7)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["channel"], "st")
        self.assertEqual(report["index"], "assets")
        self.assertTrue(report["recreate"])

    def test_reports_errors_and_nondestructive_default(self) -> None:
        sync_fn = mock.MagicMock(return_value=("exists", 3, ["e1", "e2"]))
        report = rr.run_resync(
            mock.MagicMock(),
            mock.MagicMock(),
            channel="st",
            index="assets",
            recreate=False,
            sync_fn=sync_fn,
        )
        self.assertEqual(report["ok"], 3)
        self.assertEqual(report["errors"], ["e1", "e2"])
        # recreate=False(비파괴 기본)도 그대로 코어에 전달돼야 한다.
        self.assertFalse(sync_fn.call_args.kwargs["recreate"])

    def test_default_sync_fn_is_sync_all(self) -> None:
        # 복구=전체 재동기화 → sync_fn 미주입 시 기본값은 opensearch_sync.sync_all 이어야 한다.
        from src.search.opensearch_sync import sync_all

        self.assertIs(inspect.signature(rr.run_resync).parameters["sync_fn"].default, sync_all)


class TestFormatReport(unittest.TestCase):
    """결과 요약 포매터(순수) — 상태·색인수·오류수·채널·인덱스를 사람이 읽게 한 줄로."""

    def test_includes_counts_and_context(self) -> None:
        report = {
            "status": "created",
            "ok": 12,
            "errors": [],
            "channel": "st",
            "index": "assets",
            "recreate": False,
        }
        line = rr.format_report(report, doc_count=12)
        self.assertIn("created", line)
        self.assertIn("12", line)
        self.assertIn("assets", line)
        self.assertIn("st", line)

    def test_warns_when_analysis_stale(self) -> None:
        # 코어가 'analysis-stale'(분석기가 코드와 다름)을 주면 --recreate 안내를 덧붙인다 —
        # 이 한 줄이 없으면 운영자는 "재색인했는데 검색이 그대로" 를 설명할 길이 없다.
        report = {"status": "analysis-stale", "ok": 3, "errors": [], "channel": "st_bge",
                  "index": "assets", "recreate": False}
        line = rr.format_report(report)
        self.assertIn("analysis-stale", line)
        self.assertIn("--recreate", line)

    def test_appends_error_sample_when_errors(self) -> None:
        report = {
            "status": "exists",
            "ok": 1,
            "errors": ["boom1", "boom2", "boom3"],
            "channel": "st",
            "index": "assets",
            "recreate": False,
        }
        line = rr.format_report(report)
        # 오류가 있으면 샘플(상위 2건)을 덧붙여 사람이 즉시 원인 파악.
        self.assertIn("boom1", line)


if __name__ == "__main__":
    unittest.main()
