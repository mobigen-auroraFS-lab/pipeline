"""061 G3 — archiver 실 DB/FS e2e [SC-02/03/04]. RUN_DB_E2E=1 일 때만 실행(사람 게이트).

실 PostgreSQL + 임시 인입/아카이브 디렉터리로 ``archive_registered_assets`` 를 검증한다:
  - registered 자산 파일 → archive 로 이동 + fs_path 갱신(SC-02)
  - received(비종료) 자산 파일 → 인입 잔류(SC-03)
  - 스윕 2회 멱등(SC-04)
DAG(``dag_process.archive_processed``)는 이 함수의 얇은 래퍼라 로직 동치(FR-011).
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path

from dotenv import load_dotenv

from processing.ingest import archiver

_RUN = os.getenv("RUN_DB_E2E") == "1"
_ENV = str(Path(__file__).resolve().parents[1] / ".env.dev")


def _write(dirpath: str, name: str) -> str:
    p = os.path.join(dirpath, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write("data-" + name)
    return p


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만")
class TestArchiverE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv(_ENV, override=False)
        from src.database.postgres_util import PostgresUtil

        cls.db = PostgresUtil()
        cls.db.__enter__()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.db.__exit__(None, None, None)

    def _cleanup(self, ids: list[str]) -> None:
        with self.db.transaction() as conn, conn.cursor() as cur:
            for aid in ids:
                cur.execute("DELETE FROM asset_lineage WHERE asset_id=%s", (aid,))
                cur.execute("DELETE FROM asset_embedding WHERE asset_id=%s", (aid,))
                cur.execute("DELETE FROM asset_metadata WHERE asset_id=%s", (aid,))
                cur.execute("DELETE FROM asset WHERE asset_id=%s", (aid,))

    def _registered(self, fs_path: str, ids: list[str]) -> str:
        from processing.dispatch.types import AssetRecord, EmbeddingItem
        from processing.ingest.asset_persist import create_asset, finalize_asset
        from processing.ingest.status import AssetStatus, set_status
        from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION

        v = [0.0] * FIX_EMBEDDING_DIMENSION
        v[0] = 0.5
        with self.db.transaction() as conn:
            aid = create_asset(conn, fs_path=fs_path, modality="txt", file_hash=uuid.uuid4().hex)
        ids.append(str(aid))
        with self.db.transaction() as conn:
            set_status(conn, aid, AssetStatus.ROUTING)
            set_status(conn, aid, AssetStatus.CLASSIFYING)
            set_status(conn, aid, AssetStatus.EXTRACTING)
        with self.db.transaction() as conn:
            finalize_asset(conn, aid, AssetRecord(
                embeddings=[EmbeddingItem(channel="st", vector=v, model_name="m")]))
        return str(aid)

    def _received(self, fs_path: str, ids: list[str]) -> str:
        from processing.ingest.asset_persist import create_asset

        with self.db.transaction() as conn:
            aid = create_asset(conn, fs_path=fs_path, modality="txt", file_hash=uuid.uuid4().hex)
        ids.append(str(aid))
        return str(aid)

    def _fs_path(self, aid: str) -> str:
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute("SELECT fs_path FROM asset WHERE asset_id=%s", (aid,))
            return cur.fetchone()[0]

    def test_registered_moved_received_retained_idempotent(self) -> None:
        ids: list[str] = []
        with tempfile.TemporaryDirectory() as inbox, tempfile.TemporaryDirectory() as archive:
            reg_file = _write(inbox, "reg.txt")
            rec_file = _write(inbox, "rec.txt")
            try:
                reg_id = self._registered(reg_file, ids)
                rec_id = self._received(rec_file, ids)

                moved = archiver.archive_registered_assets(
                    self.db, inbox_root=inbox, archive_root=archive
                )
                self.assertEqual(moved, 1)  # registered 1건만

                # SC-02: registered 파일 이동 + fs_path 갱신·이동 후 경로 유효
                self.assertFalse(os.path.exists(reg_file))
                new_path = self._fs_path(reg_id)
                self.assertTrue(archiver.is_under(new_path, archive))
                self.assertTrue(os.path.exists(new_path))
                self.assertTrue(new_path.endswith("reg.txt"))

                # SC-03: received 파일은 인입 잔류·fs_path 불변
                self.assertTrue(os.path.exists(rec_file))
                self.assertEqual(self._fs_path(rec_id), rec_file)

                # SC-04: 재스윕 멱등(registered 는 이미 인입 밖 → 0건·오류 없음)
                moved2 = archiver.archive_registered_assets(
                    self.db, inbox_root=inbox, archive_root=archive
                )
                self.assertEqual(moved2, 0)
                self.assertEqual(self._fs_path(reg_id), new_path)
            finally:
                self._cleanup(ids)


if __name__ == "__main__":
    unittest.main()
