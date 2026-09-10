"""개체 색인 고아 정리 — DB(node)에 없는 개체의 색인 문서만 지운다(2026-09-10 · 비운 뒤 옛 개체 82건 잔존 결함)."""
from __future__ import annotations

import unittest

from processing.app.run_entity_embedding import purge_orphan_index_docs
from src.search.entity_index import entity_doc_id


class _Indices:
    def __init__(self, exists: bool) -> None:
        self._exists = exists

    def exists(self, index: str) -> bool:
        return self._exists


class _Client:
    """OpenSearch 대역 — 색인 문서 목록을 주고 삭제를 기록한다."""

    def __init__(self, docs: list[tuple[str, str]], exists: bool = True) -> None:
        self.indices = _Indices(exists)
        self._ids = {entity_doc_id(t, u): (t, u) for t, u in docs}
        self.deleted: list[str] = []

    def search(self, index: str, body: dict) -> dict:
        return {"hits": {"hits": [{"_id": i, "_source": {"entity_type": t, "entity_uid": u}}
                                  for i, (t, u) in self._ids.items()]}}

    def exists(self, index: str, id: str) -> bool:  # noqa: A002 — opensearch-py 시그니처
        return id in self._ids

    def delete(self, index: str, id: str) -> None:  # noqa: A002
        self.deleted.append(id)
        self._ids.pop(id, None)


class _Conn:
    def __init__(self, rows: list[tuple[str, str]]) -> None:
        self._rows = rows

    def execute(self, sql: str, params=None):
        rows = self._rows

        class _Cur:
            def fetchall(self_inner):
                return rows
        return _Cur()


class TestPurgeOrphanIndexDocs(unittest.TestCase):
    def test_DB에_없는_개체_문서만_지운다(self) -> None:
        client = _Client([("인물", "정은채"), ("인물", "옛개체"), ("장소", "숭례문")])
        removed = purge_orphan_index_docs(client, "mm_entities", _Conn([("인물", "정은채"), ("장소", "숭례문")]))
        self.assertEqual(removed, 1)
        self.assertEqual(client.deleted, [entity_doc_id("인물", "옛개체")])

    def test_모두_살아있으면_지우지_않는다(self) -> None:
        client = _Client([("인물", "정은채")])
        self.assertEqual(purge_orphan_index_docs(client, "mm_entities", _Conn([("인물", "정은채")])), 0)
        self.assertEqual(client.deleted, [])

    def test_인덱스가_없으면_0(self) -> None:
        client = _Client([], exists=False)
        self.assertEqual(purge_orphan_index_docs(client, "mm_entities", _Conn([])), 0)


if __name__ == "__main__":
    unittest.main()
