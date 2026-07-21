"""069 US-F 그룹3(T406·T407) — 사장 2파일 처분 봉인 테스트(순수·DB 불필요).

T406(㉠): ``run_relations_review`` CLI 삭제(052 API 상위호환) + 유일 소비 ``list_proposed_edges``
연쇄 삭제. portal 이 소비하는 ``approve_edge``/``reject_edge``/``promote_relation_kind`` 및
052 API(``list_edges_for_review``)는 **존치**.
T407(㉡): ``sample_search_api`` 삭제(디버그 3종 portal /search opt-in 이관). 진입점 지도 갱신.
"""
from __future__ import annotations

import importlib
import unittest


class TestRunRelationsReviewRemoved(unittest.TestCase):
    """T406 — run_relations_review CLI 모듈이 삭제되어 import 불가."""

    def test_module_absent(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("processing.app.run_relations_review")


class TestListProposedEdgesRemoved(unittest.TestCase):
    """T406 연쇄 — list_proposed_edges(유일 소비=삭제 CLI)는 제거, 나머지 review API 는 존치."""

    def test_list_proposed_edges_absent(self) -> None:
        import src.relations.review as review

        self.assertFalse(
            hasattr(review, "list_proposed_edges"),
            "list_proposed_edges 는 죽은 함수(유일 소비 run_relations_review 삭제) → 제거해야 함",
        )

    def test_portal_consumed_functions_retained(self) -> None:
        # portal 이 (직접/bulk_review 경유) 소비하는 함수 + 052 API 는 유지.
        import src.relations.review as review

        for name in (
            "approve_edge",
            "reject_edge",
            "promote_relation_kind",
            "bulk_review",
            "revise_edge",
            "list_edges_for_review",
            "list_relation_kinds",
        ):
            self.assertTrue(hasattr(review, name), f"{name} 는 존치해야 함(portal/052 소비)")


class TestSampleSearchApiRemoved(unittest.TestCase):
    """T407 — sample_search_api 삭제(디버그 3종은 portal /search opt-in 으로 이관)."""

    def test_module_absent(self) -> None:
        with self.assertRaises(ModuleNotFoundError):
            importlib.import_module("processing.app.sample_search_api")


class TestAppEntrypointMapUpdated(unittest.TestCase):
    """진입점 지도(processing.app.__init__ docstring)에서 삭제된 두 모듈 서술 제거."""

    def test_map_drops_deleted_modules(self) -> None:
        # 진입점 목록(bullet)에서 두 모듈이 빠진다. 삭제 사실을 서술하는 문장은 이름을 언급해도
        # 무방하므로, '· <모듈>' 목록 항목 형태만 부재를 검증한다.
        import processing.app as app_pkg

        doc = app_pkg.__doc__ or ""
        self.assertNotIn("· run_relations_review", doc)
        self.assertNotIn("· sample_search_api", doc)


if __name__ == "__main__":
    unittest.main()
