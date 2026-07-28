"""F-1.x run_ingest 실 DB 통합(e2e) 골든 테스트 — v2 Strangler 회귀 안전망 (백로그 P0).

목적
    v2 전환(단계 A~)이 오케스트레이터의 **영속화·상태머신·트랜잭션·분기** 동작을
    바꾸지 않았음을 실제 PostgreSQL 에 대해 검증하는 골든 테스트. 단계 A 리팩터의
    합격선("기존 동작 정확 재현")을 기계적으로 확인하는 안전망이다.

설계
    run_ingest 의 무거운 경로(CLIP/VLM/whisper/LLM)는 주입 지점(extract_fn/classify_fn)에
    fake 를 넣어 제거한다. 즉 "실 DB + 가짜 추출/분류" 로 **영속화 계약만** 검증한다.
    (route_file/file_hash_and_size 는 실제 임시 파일로 그대로 태운다.)

실행 / 게이트
    기본 단위 테스트(python -m unittest discover)를 오염시키지 않도록 ``RUN_DB_E2E=1``
    일 때만 실행한다(미설정 시 전체 skip). DB 미접속·스키마 미적용 시에도 명확히 skip.

        conda activate AuroraFS
        RUN_DB_E2E=1 python -m unittest tests.test_run_ingest_e2e
"""

from __future__ import annotations

import os
import shutil
import tempfile
import types
import unittest
import uuid
from pathlib import Path

from dotenv import load_dotenv

from processing.classify.types import ClassificationResult
from processing.dispatch.types import AssetRecord, EmbeddingItem
from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION

_RUN = os.getenv("RUN_DB_E2E") == "1"
_ENV = Path(__file__).resolve().parents[1] / ".env.dev"


def _vec() -> list[float]:
    """1536D 더미 임베딩(asset_embedding.embedding VECTOR(1536) 적재 검증용)."""
    v = [0.0] * FIX_EMBEDDING_DIMENSION
    v[0], v[1], v[2] = 0.1, 0.2, 0.3
    return v


def _general(_path: str, _modality: str) -> ClassificationResult:
    return ClassificationResult(final_label="general", confidence=0.7, decided_stage=2)


def _medical_signature(_path: str, _modality: str) -> ClassificationResult:
    """의료 표준 포맷(시그니처) → run_ingest 가 deferred 분기를 타야 한다."""
    return ClassificationResult(
        final_label="medical", confidence=1.0, decided_stage=1,
        stage1_scores={"medical": {"signature": "dicom"}},  # cascade v2 중첩 구조 {domain: {signature}}
    )


def _record() -> AssetRecord:
    """고정 추출 결과 — DB 라운드트립을 검증할 알려진 값."""
    return AssetRecord(
        core_meta={"size": 12},
        ext_meta={"summary": "골든 요약"},
        tags=["golden"],
        embeddings=[EmbeddingItem(channel="st", vector=_vec(), model_name="fake-st", chunk_index=0)],
    )


@unittest.skipUnless(_RUN, "RUN_DB_E2E=1 일 때만 실행하는 실 DB e2e")
class TestRunIngestE2E(unittest.TestCase):
    db = None  # type: ignore[assignment]

    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv(_ENV, override=False)
        from src.database.postgres_util import PostgresUtil

        try:
            cls.db = PostgresUtil()
            cls.db.__enter__()  # 연결 풀 오픈
            with cls.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.asset')")
                if cur.fetchone()[0] is None:
                    raise unittest.SkipTest("asset 스키마 미적용")
        except unittest.SkipTest:
            raise
        except Exception as exc:  # noqa: BLE001 — 접속 불가 시 깔끔히 skip
            raise unittest.SkipTest(f"DB 미접속: {type(exc).__name__}: {exc}") from None

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.db is not None:
            cls.db.__exit__(None, None, None)

    def setUp(self) -> None:
        self._ids: list[uuid.UUID] = []
        self._dir = Path(tempfile.mkdtemp(prefix="ingest_e2e_"))

    def tearDown(self) -> None:
        # 생성한 asset 행 삭제 → CASCADE 로 metadata/embedding/classification 동반 삭제(테스트 멱등).
        ids = [i for i in self._ids if i is not None]
        if ids:
            with self.db.transaction() as conn, conn.cursor() as cur:
                cur.execute("DELETE FROM asset WHERE asset_id = ANY(%s)", (ids,))
        shutil.rmtree(self._dir, ignore_errors=True)

    # ── 헬퍼 ────────────────────────────────────────────────────────────────
    def _mkfile(self, name: str = "a.txt", content: str | None = None) -> str:
        p = self._dir / name
        p.write_text(content if content is not None else ("내용-" + uuid.uuid4().hex), encoding="utf-8")
        return str(p)

    def _run(self, files, *, extract_fn, classify_fn=_general):
        from processing.app.run_ingest import run_ingest

        # PR4b 중첩 settings: _make_opensearch_indexer 가 settings.opensearch.sync_enabled 를 직접 읽는다.
        # e2e 는 PG 영속화만 검증(OS 색인 off) → sync_enabled=False no-op settings 주입.
        settings = types.SimpleNamespace(opensearch=types.SimpleNamespace(sync_enabled=False))
        res = run_ingest(files, db=self.db, extract_fn=extract_fn, classify_fn=classify_fn, settings=settings)
        # 정리 대상 id 수집(registered/deferred + 실패 중 asset 생성된 것)
        self._ids += list(res["registered"]) + list(res["deferred"])
        self._ids += [i for (i, _) in res["failed"] if i is not None]
        return res

    def _q(self, sql: str, params):
        with self.db.transaction() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()

    # ── 테스트 ──────────────────────────────────────────────────────────────
    def test_success_persists_asset_metadata_embedding(self) -> None:
        res = self._run([self._mkfile()], extract_fn=lambda ctx: _record())
        self.assertEqual(len(res["registered"]), 1)
        aid = res["registered"][0]

        status, modality, domain = self._q(
            "SELECT status, modality, domain_label FROM asset WHERE asset_id=%s", (aid,)
        )
        self.assertEqual(status, "registered")
        self.assertEqual(modality, "text")  # 053: .txt 수집 → 저장은 canonical 'text'(추출은 file_kind)
        self.assertEqual(domain, "general")

        # 037: search_vector(PG FTS) 컬럼은 v270 에서 제거 — 적재 라운드트립은 core/ext/tags 만 검증.
        summary, tags = self._q(
            "SELECT ext_meta->>'summary', tags FROM asset_metadata WHERE asset_id=%s", (aid,)
        )
        self.assertEqual(summary, "골든 요약")
        self.assertIn("golden", tags)

        (n_emb,) = self._q("SELECT count(*) FROM asset_embedding WHERE asset_id=%s", (aid,))
        self.assertEqual(n_emb, 1)
        (channel,) = self._q("SELECT channel FROM asset_embedding WHERE asset_id=%s", (aid,))
        self.assertEqual(channel, "st")

        (label,) = self._q("SELECT final_label FROM asset_classification WHERE asset_id=%s", (aid,))
        self.assertEqual(label, "general")

    def test_duplicate_content_skipped(self) -> None:
        f = self._mkfile(content="dup-" + uuid.uuid4().hex)  # 동일 내용 → 동일 해시
        r1 = self._run([f], extract_fn=lambda ctx: _record())
        self.assertEqual(len(r1["registered"]), 1)
        r2 = self._run([f], extract_fn=lambda ctx: _record())
        self.assertEqual(r2["registered"], [])
        self.assertEqual(len(r2["skipped"]), 1)
        self.assertTrue(r2["skipped"][0][1].startswith("duplicate"))

    def test_medical_signature_deferred(self) -> None:
        called = {"extract": False}

        def extract(ctx):
            called["extract"] = True
            return _record()

        res = self._run([self._mkfile(name="scan.txt")], extract_fn=extract, classify_fn=_medical_signature)
        self.assertEqual(len(res["deferred"]), 1)
        self.assertEqual(res["registered"], [])
        self.assertFalse(called["extract"])  # 추출기 미호출

        aid = res["deferred"][0]
        (status,) = self._q("SELECT status FROM asset WHERE asset_id=%s", (aid,))
        self.assertEqual(status, "deferred")
        (n_meta,) = self._q("SELECT count(*) FROM asset_metadata WHERE asset_id=%s", (aid,))
        self.assertEqual(n_meta, 0)  # 추출 안 했으므로 메타 없음
        (label,) = self._q("SELECT final_label FROM asset_classification WHERE asset_id=%s", (aid,))
        self.assertEqual(label, "medical")  # 분류는 영속화됨

    def test_extract_failure_marks_failed_and_isolates_batch(self) -> None:
        f_bad = self._mkfile(name="bad.txt")
        f_ok = self._mkfile(name="ok.txt")

        def extract(ctx):
            if ctx.file_path.endswith("bad.txt"):
                raise RuntimeError("추출 실패")
            return _record()

        res = self._run([f_bad, f_ok], extract_fn=extract)
        self.assertEqual(len(res["registered"]), 1)  # 한 파일 실패해도 배치 계속
        self.assertEqual(len(res["failed"]), 1)

        bad_id = res["failed"][0][0]
        (status,) = self._q("SELECT status FROM asset WHERE asset_id=%s", (bad_id,))
        self.assertEqual(status, "failed")
        (n_meta,) = self._q("SELECT count(*) FROM asset_metadata WHERE asset_id=%s", (bad_id,))
        self.assertEqual(n_meta, 0)


if __name__ == "__main__":
    unittest.main()
