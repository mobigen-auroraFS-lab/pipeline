"""재색인 도구가 **검색 파이프라인도 보장**한다(2026-09-23 · spec 101 G2).

무엇이 문제였나: 파일 검색은 순위 섞기를 **검색 엔진에 맡긴다**(`assets-hybrid` 파이프라인).
그런데 그것을 **만드는 코드가 어디에도 없었다.** 027 이 이 도구에서 등록 기능을 지웠고
(그때는 쓰는 곳이 없었다) 그 뒤 파일 검색이 생기며 필요가 되살아났는데 **쓰는 쪽만 남았다.**

빈 환경에서 `/file-search` 가 `Pipeline assets-hybrid is not defined` 로 500 을 낸다
(2026-09-22 k8s 신규 환경 실측). 사람이 손으로 `PUT` 해서 넘겼고, **그 손을 없애는 것**이
이 변경이다. 색인 생성과 파이프라인 등록은 **같은 부트스트랩 시점**에 필요하므로 한 도구에 둔다.

🔴 **기본으로 켜져 있어야 한다** — 없으면 검색이 500 이므로 「옵션」이 아니다.
"""
from __future__ import annotations

import unittest
from typing import Any

from processing.app import run_opensearch_resync as mod


class TestRunEnsurePipeline:
    """자리표시 — 아래 TestCase 가 실제 검증이다."""


class _Calls:
    def __init__(self, result: str = "created") -> None:
        self.calls: list[tuple[Any, str, tuple[float, float]]] = []
        self._result = result

    def __call__(self, client: Any, name: str, *, weights: tuple[float, float]) -> str:
        self.calls.append((client, name, weights))
        return self._result


class TestEnsurePipelineAssembly(unittest.TestCase):
    """조립부는 **주입받은 함수만** 부른다 — 검색 엔진 없이 검증된다."""

    def test_이름과_가중치를_그대로_넘긴다(self) -> None:
        spy = _Calls()
        got = mod.run_ensure_pipeline(
            client="CLIENT", name="assets-hybrid", weights=(0.3, 0.7), ensure_fn=spy
        )
        self.assertEqual(got, "created")
        self.assertEqual(spy.calls, [("CLIENT", "assets-hybrid", (0.3, 0.7))])

    def test_결과를_그대로_돌려준다(self) -> None:
        """멱등 — 이미 있으면 ``exists``. 보고에 그대로 실린다."""
        spy = _Calls(result="exists")
        got = mod.run_ensure_pipeline(
            client="C", name="assets-hybrid", weights=(0.5, 0.5), ensure_fn=spy
        )
        self.assertEqual(got, "exists")


class TestReportIncludesPipeline(unittest.TestCase):
    """보고에 한 줄 실려야 사람이 **등록됐는지** 안다."""

    @staticmethod
    def _report(**over: Any) -> dict[str, Any]:
        base = {
            "status": "created", "ok": 3, "errors": [],
            "channel": "st_api", "index": "assets", "recreate": False,
        }
        base.update(over)
        return base

    def test_등록_결과가_보고에_나온다(self) -> None:
        line = mod.format_report(self._report(pipeline="created"), doc_count=3)
        self.assertIn("파이프라인", line)
        self.assertIn("created", line)

    def test_없으면_보고에_안_나온다(self) -> None:
        """끈 경우 — 종전 보고 모양을 흐리지 않는다."""
        line = mod.format_report(self._report(), doc_count=3)
        self.assertNotIn("파이프라인", line)


class TestWeightsComeFromSettings(unittest.TestCase):
    """🔴 등록값이 설정과 갈리면 **검색 순위가 조용히 어긋난다** — 오류가 안 나서 아무도 모른다."""

    def test_설정_가중치가_등록_본문까지_간다(self) -> None:
        from src.search.opensearch_search import search_pipeline_body

        weights = (0.3, 0.7)
        spy = _Calls()
        mod.run_ensure_pipeline(
            client="C", name="assets-hybrid", weights=weights, ensure_fn=spy
        )
        (_c, _n, passed), = spy.calls
        self.assertEqual(passed, weights, "조립부가 가중치를 바꾸지 않는다")

        body = search_pipeline_body(passed)
        got = body["phase_results_processors"][0]["normalization-processor"]
        self.assertEqual(got["combination"]["parameters"]["weights"], [0.3, 0.7])


class TestFlagDefault(unittest.TestCase):
    """🔴 기본이 **켜짐**이어야 한다 — 없으면 파일 검색이 500 이다."""

    def test_기본값은_켜짐이다(self) -> None:
        args = mod._build_parser().parse_args([])
        self.assertTrue(args.ensure_pipeline)

    def test_끌_수_있다(self) -> None:
        """이미 손으로 관리 중인 환경을 위해 빠져나갈 구멍은 둔다."""
        args = mod._build_parser().parse_args(["--no-ensure-pipeline"])
        self.assertFalse(args.ensure_pipeline)


if __name__ == "__main__":
    unittest.main()
