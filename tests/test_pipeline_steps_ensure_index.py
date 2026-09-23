"""적재 훅이 **색인 존재를 보장**한다(2026-09-23 · spec 101 G1).

무엇이 문제였나: 검색 엔진은 색인이 없는데 문서가 들어오면 **"없으면 내가 만들지"** 하고
자동으로 만든다. 이때 **값 모양만 보고 타입을 정한다** — 1536개짜리 벡터 배열을 보고
「소수점 숫자」로 판단해 ``float`` 으로 잡는다. 그 색인은 **벡터 검색이 전부 죽는다.**

코드에는 올바른 매핑으로 만드는 함수(``ensure_index``)가 **이미 있다.** 그런데 그것을 부르는
곳이 **전량 재색인 경로 한 군데뿐**이라, 새 환경에서 적재가 먼저 돌면 자동 생성이 일어났다.
실측(2026-09-22 · k8s 신규 환경)에서 정확히 그 일이 났다.

🔴 **여기서 지켜야 할 계약 셋**이 있고, 이 파일이 그것을 봉인한다.

1. **배치당 1회**만 부른다 — 자산마다 부르면 존재 확인이 자산 수만큼 날아간다.
2. **실패를 삼킨다** — 색인 문제로 적재가 되돌아가면 안 된다(기존 계약 · 복구 도구가 있다).
3. **토글이 꺼져 있으면 아무 것도 안 한다** — 검색 엔진 코드를 건드리지 않는다.
"""
from __future__ import annotations

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest import mock

from processing.ingest import pipeline_steps


class _Db:
    """``transaction()`` 만 있는 최소 대역."""

    @contextmanager
    def transaction(self):  # noqa: ANN201 - 대역
        yield object()


def _settings(*, enabled: bool = True) -> Any:
    return SimpleNamespace(
        opensearch=SimpleNamespace(
            sync_enabled=enabled,
            url="http://os:9200",
            index="assets",
            filename_noise_patterns=(),
        )
    )


class _Patches:
    """지연 import 되는 이름들을 한 번에 갈아끼운다."""

    def __init__(self, *, ensure_raises: bool = False) -> None:
        self.ensure_calls: list[tuple[Any, str]] = []
        self.index_calls: list[str] = []
        self._ensure_raises = ensure_raises

    def ensure_index(self, client: Any, index: str, **kwargs: Any) -> str:
        self.ensure_calls.append((client, index))
        if self._ensure_raises:
            raise RuntimeError("검색 엔진이 죽어 있다")
        return "created"

    def index_asset(self, client: Any, conn: Any, asset_id: str, **kwargs: Any) -> None:
        self.index_calls.append(asset_id)

    def get_client(self, url: str) -> Any:
        return SimpleNamespace(url=url)

    @contextmanager
    def applied(self):  # noqa: ANN201 - 대역
        with mock.patch(
            "src.search.opensearch_sync.ensure_index", self.ensure_index
        ), mock.patch(
            "src.search.opensearch_sync.index_asset", self.index_asset
        ), mock.patch(
            "src.search.opensearch_sync.get_client", self.get_client
        ), mock.patch(
            "src.config.settings.active_embed_channel", lambda _s: "st_api"
        ):
            yield


class TestEnsureIndexOnIngest(unittest.TestCase):
    def test_색인_존재를_보장한다(self) -> None:
        """🔴 이 테스트가 spec 101 G1 의 존재 이유다 — 없으면 엔진이 제멋대로 만든다."""
        p = _Patches()
        with p.applied():
            index = pipeline_steps._make_opensearch_indexer(db=_Db(), settings=_settings())
            index("asset-1")
        self.assertEqual(len(p.ensure_calls), 1)
        self.assertEqual(p.ensure_calls[0][1], "assets")

    def test_배치당_1회만_부른다(self) -> None:
        """성능 계약 — 자산마다 부르면 존재 확인이 자산 수만큼 날아간다."""
        p = _Patches()
        with p.applied():
            index = pipeline_steps._make_opensearch_indexer(db=_Db(), settings=_settings())
            for i in range(3):
                index(f"asset-{i}")
        self.assertEqual(len(p.ensure_calls), 1, "배치당 1회")
        self.assertEqual(len(p.index_calls), 3, "색인 자체는 자산마다")

    def test_보장에_실패해도_적재는_성공한다(self) -> None:
        """🔴 기존 계약 — 색인 문제로 적재를 되돌리면 안 된다(복구 도구가 있다)."""
        p = _Patches(ensure_raises=True)
        with p.applied():
            index = pipeline_steps._make_opensearch_indexer(db=_Db(), settings=_settings())
            index("asset-1")  # 예외가 밖으로 나오면 실패다

    def test_토글이_꺼져_있으면_아무것도_안_한다(self) -> None:
        """검색 엔진 코드를 아예 건드리지 않는다(opensearch-py 미설치 환경 보호)."""
        p = _Patches()
        with p.applied():
            index = pipeline_steps._make_opensearch_indexer(db=_Db(), settings=_settings(enabled=False))
            index("asset-1")
        self.assertEqual(p.ensure_calls, [])
        self.assertEqual(p.index_calls, [])

    def test_보장이_색인보다_먼저다(self) -> None:
        """순서가 뒤집히면 첫 문서가 자동 생성을 일으켜 이 spec 이 무의미해진다."""
        order: list[str] = []
        p = _Patches()
        orig_ensure, orig_index = p.ensure_index, p.index_asset
        p.ensure_index = lambda *a, **k: (order.append("ensure"), orig_ensure(*a, **k))[1]  # type: ignore[assignment]
        p.index_asset = lambda *a, **k: (order.append("index"), orig_index(*a, **k))[1]  # type: ignore[assignment]
        with p.applied():
            index = pipeline_steps._make_opensearch_indexer(db=_Db(), settings=_settings())
            index("asset-1")
        self.assertEqual(order, ["ensure", "index"])


if __name__ == "__main__":
    unittest.main()
