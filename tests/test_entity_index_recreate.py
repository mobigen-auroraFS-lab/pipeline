"""개체 색인 **재생성** 통로 검증 — 분석기를 바꿨을 때 실제로 갈아끼울 수 있는가.

왜 필요한가(2026-09-18 실측): 분석기에 소문자 필터를 넣었는데 자산 색인만 다시 만들어졌고
개체 색인은 옛 설정 그대로 남았다. 그래서 개체 검색에서 ``MBC`` 는 19건, ``mbc`` 는 0건이었다.
코드는 맞는데 **돌아가는 색인만 낡은** 상태였고, 파이프 배치에 재생성 통로가 없어서 그렇게 됐다.

분석기 변경은 **덮어쓰기로 반영되지 않는다** — 이미 색인된 문서가 옛 규칙으로 쪼개져 있어
설정만 바꿔서는 소용이 없다. 자산 쪽 복구 도구(``run_opensearch_resync --recreate``)와 같은
옵트인 통로를 개체 배치에도 둔다.
"""
from __future__ import annotations

import unittest

from processing.app.run_entity_embedding import _build_parser, sync_entity_index


class _Indices:
    """``indices`` 네임스페이스 흉내 — 무엇이 불렸는지만 기록한다."""

    def __init__(self, *, exists: bool = True) -> None:
        self._exists = exists
        self.deleted: list[str] = []
        self.created: list[str] = []

    def exists(self, *, index: str) -> bool:
        return self._exists

    def delete(self, *, index: str) -> None:
        self.deleted.append(index)
        self._exists = False

    def create(self, *, index: str, body: object) -> None:
        self.created.append(index)
        self._exists = True


class _Client:
    def __init__(self, *, exists: bool = True) -> None:
        self.indices = _Indices(exists=exists)

    def bulk(self, **_kw: object) -> dict[str, object]:
        return {"errors": False, "items": []}


class _Cursor:
    """``with conn.cursor() as cur`` 꼴을 받는 최소 대역 — 항상 빈 결과를 준다."""

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_a: object) -> None:
        return None

    def execute(self, *_a: object, **_k: object) -> _Cursor:
        return self

    def fetchall(self) -> list[object]:
        return []


class _Conn:
    """DB 는 빈 결과만 준다 — 여기서 보는 것은 색인 **준비 단계**(생성/재생성)뿐이다.

    호출부가 두 꼴을 섞어 쓴다(``conn.execute(...)`` 와 ``with conn.cursor() as cur``)므로
    둘 다 받는다. 대역이 한쪽만 흉내내면 테스트가 진짜 결함이 아닌 곳에서 넘어진다.
    """

    def execute(self, *_a: object, **_k: object) -> _Conn:
        return self

    def fetchall(self) -> list[object]:
        return []

    def cursor(self, *_a: object, **_k: object) -> _Cursor:
        return _Cursor()


class TestEntityIndexRecreate(unittest.TestCase):
    def test_기본은_지우지_않는다(self) -> None:
        """평소 배치는 덮어쓰기다 — 매번 지우면 도는 동안 검색이 비어 보인다."""
        client = _Client(exists=True)
        sync_entity_index(
            _Conn(), client=client, index="mm_entities", targets=[], model_name="m"
        )
        self.assertEqual(client.indices.deleted, [])

    def test_recreate_면_지우고_다시_만든다(self) -> None:
        """🔴 분석기를 바꾼 뒤의 유일한 반영 경로 — 덮어쓰기로는 옛 쪼개기가 남는다."""
        client = _Client(exists=True)
        sync_entity_index(
            _Conn(),
            client=client,
            index="mm_entities",
            targets=[],
            model_name="m",
            recreate=True,
        )
        self.assertEqual(client.indices.deleted, ["mm_entities"])
        self.assertEqual(client.indices.created, ["mm_entities"])

    def test_명령행에_recreate_가_있다(self) -> None:
        """사람이 부르는 통로가 없으면 옵션이 있어도 쓰이지 않는다(이번 누락의 실제 원인)."""
        args = _build_parser().parse_args(["--recreate"])
        self.assertTrue(args.recreate)

    def test_기본값은_거짓이다(self) -> None:
        """파괴적 동작은 **옵트인**이다 — 실수로 도는 배치가 색인을 비우면 안 된다."""
        self.assertFalse(_build_parser().parse_args([]).recreate)


if __name__ == "__main__":
    unittest.main()
