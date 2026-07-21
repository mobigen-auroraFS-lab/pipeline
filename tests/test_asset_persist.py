"""F-1.3 등록·적재(asset_persist) 단위 테스트.

psycopg Connection 을 mock 으로 대체해 SQL 배선·파라미터를 검증(실 DB 불필요).
식별자는 UUIDv7(앱 생성)이므로 uuid7 을 patch 해 결정적으로 검증한다.
실제 DB 적재는 T1-6 통합 수직 슬라이스에서 확인.
"""

from __future__ import annotations

import unittest
import uuid
from unittest import mock

from processing.dispatch.types import AssetRecord, EmbeddingItem
from processing.ingest import asset_persist
from processing.ingest.asset_persist import create_asset, finalize_asset

_FIXED = uuid.UUID("018f0000-0000-7000-8000-000000000001")


def _mock_conn(fetchone_value):
    conn = mock.MagicMock()
    cur = conn.cursor.return_value.__enter__.return_value
    cur.fetchone.return_value = fetchone_value
    return conn, cur


def _executes(cur):
    return cur.execute.call_args_list


class TestCreateAsset(unittest.TestCase):
    def _create_and_capture_insert(self, cur):
        """asset INSERT 실행 콜을 골라 반환(단일 INSERT 검증 포함)."""
        ins = [c for c in _executes(cur) if "INSERT INTO asset " in c.args[0]]
        self.assertEqual(len(ins), 1)
        return ins[0]

    def test_generates_uuid7_and_inserts_received(self) -> None:
        conn, cur = _mock_conn(None)
        with mock.patch.object(asset_persist, "uuid7", return_value=_FIXED):
            aid = create_asset(conn, fs_path="/d/a.txt", modality="txt")
        self.assertEqual(aid, _FIXED)
        ins = self._create_and_capture_insert(cur)
        # 053: 인자 modality 는 file_kind('txt'), 저장은 canonical('text').
        # params: (asset_id, modality, fs_path, file_hash, file_size, domain)
        self.assertEqual(ins.args[1], (_FIXED, "text", "/d/a.txt", None, None, "general"))
        self.assertIn("'received'", ins.args[0])

    def test_json_file_kind_stored_as_canonical_text(self) -> None:
        # 053(FR-201): file_kind 'json' 저장 경계에서 canonical 'text' 로 매핑.
        conn, cur = _mock_conn(None)
        with mock.patch.object(asset_persist, "uuid7", return_value=_FIXED):
            create_asset(conn, fs_path="/x/a.json", modality="json")
        ins = self._create_and_capture_insert(cur)
        self.assertEqual(ins.args[1][1], "text")

    def test_image_file_kind_unchanged(self) -> None:
        # image/video/audio 는 canonical 과 동일 — 저장값 무변경.
        conn, cur = _mock_conn(None)
        with mock.patch.object(asset_persist, "uuid7", return_value=_FIXED):
            create_asset(conn, fs_path="/x/a.png", modality="image")
        ins = self._create_and_capture_insert(cur)
        self.assertEqual(ins.args[1][1], "image")

    def test_unknown_modality_preserved(self) -> None:
        # unknown 은 격리표식 — 매핑 불변.
        conn, cur = _mock_conn(None)
        with mock.patch.object(asset_persist, "uuid7", return_value=_FIXED):
            create_asset(conn, fs_path="/x/scan.dcm", modality="unknown")
        ins = self._create_and_capture_insert(cur)
        self.assertEqual(ins.args[1][1], "unknown")


class TestFinalizeAsset(unittest.TestCase):
    def test_inserts_metadata_embeddings_and_registers(self) -> None:
        conn, cur = _mock_conn({"status": "extracting"})
        rec = AssetRecord(
            core_meta={"width": 10},
            ext_meta={"summary": "s"},
            tags=["t"],
            embeddings=[EmbeddingItem(channel="st", vector=[0.1, 0.2], model_name="legacy")],
        )
        finalize_asset(conn, _FIXED, rec)
        meta_ins = [c for c in _executes(cur) if "INSERT INTO asset_metadata" in c.args[0]]
        self.assertEqual(len(meta_ins), 1)
        self.assertEqual(meta_ins[0].args[1][0], _FIXED)
        self.assertEqual(meta_ins[0].args[1][3], ["t"])
        # 037: search_vector 컬럼·to_tsvector·fts_plain 파라미터 제거 — INSERT 는 4개 컬럼만.
        self.assertNotIn("search_vector", meta_ins[0].args[0])
        self.assertNotIn("to_tsvector", meta_ins[0].args[0])
        self.assertEqual(len(meta_ins[0].args[1]), 4)
        self.assertEqual(cur.executemany.call_count, 1)
        emb_rows = cur.executemany.call_args.args[1]
        self.assertEqual(emb_rows, [(_FIXED, "st", 0, [0.1, 0.2], "legacy", None)])
        upd = [c for c in _executes(cur) if "UPDATE asset SET status" in c.args[0]]
        self.assertEqual(len(upd), 1)
        # 009: finalize_asset 이 경유하는 set_status 가 조건부 UPDATE 로 원자화 →
        # 파라미터 끝에 기대 현재상태('extracting') WHERE 가드 추가.
        self.assertEqual(upd[0].args[1], ("registered", None, _FIXED, "extracting"))

    def test_no_embeddings_skips_executemany(self) -> None:
        conn, cur = _mock_conn({"status": "extracting"})
        finalize_asset(conn, _FIXED, AssetRecord())
        self.assertEqual(cur.executemany.call_count, 0)
        upd = [c for c in _executes(cur) if "UPDATE asset SET status" in c.args[0]]
        self.assertEqual(upd[0].args[1], ("registered", None, _FIXED, "extracting"))

    def test_rejects_non_extracting_before_any_insert(self) -> None:
        # 069 B7(P2-7): finalize_asset 서두 가드 — 상태가 extracting 이 아니면 메타·임베딩 INSERT
        # 이전에 InvalidTransitionError 로 차단한다. set_status 의 말미 검증만으로는 autocommit
        # 새 호출자에서 INSERT 가 먼저 커밋돼 중복/고아 행이 영속될 수 있어, 서두에서 막는다.
        from processing.ingest.status import InvalidTransitionError

        conn, cur = _mock_conn({"status": "received"})  # extracting 아님
        rec = AssetRecord(
            core_meta={"w": 1},
            embeddings=[EmbeddingItem(channel="st", vector=[0.1], model_name="m")],
        )
        with self.assertRaises(InvalidTransitionError):
            finalize_asset(conn, _FIXED, rec)
        # 가드가 서두에서 막으므로 metadata INSERT·임베딩 executemany 가 실행되지 않는다.
        meta_ins = [c for c in _executes(cur) if "INSERT INTO asset_metadata" in c.args[0]]
        self.assertEqual(len(meta_ins), 0)
        self.assertEqual(cur.executemany.call_count, 0)

    def test_happy_path_result_unchanged_with_guard(self) -> None:
        # 정상 경로(extracting)는 가드 추가 후에도 결과 동일 — metadata INSERT + registered 전이.
        conn, cur = _mock_conn({"status": "extracting"})
        finalize_asset(conn, _FIXED, AssetRecord(core_meta={"w": 1}))
        meta_ins = [c for c in _executes(cur) if "INSERT INTO asset_metadata" in c.args[0]]
        self.assertEqual(len(meta_ins), 1)
        upd = [c for c in _executes(cur) if "UPDATE asset SET status" in c.args[0]]
        self.assertEqual(upd[0].args[1], ("registered", None, _FIXED, "extracting"))


if __name__ == "__main__":
    unittest.main()
