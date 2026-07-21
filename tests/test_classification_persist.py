"""F-5.1 분류 결과 영속화 단위 테스트(mock Connection)."""

from __future__ import annotations

import unittest
import uuid
from unittest import mock

from processing.classify.types import ClassificationResult
from processing.ingest.classification_persist import record_classification

_AID = uuid.UUID("018f0000-0000-7000-8000-000000000001")


def _conn():
    conn = mock.MagicMock()
    return conn, conn.cursor.return_value.__enter__.return_value


def _execs(cur):
    return [c.args[0] for c in cur.execute.call_args_list]


class TestRecordClassification(unittest.TestCase):
    def test_medical_inserts_and_updates_domain(self) -> None:
        conn, cur = _conn()
        r = ClassificationResult(final_label="medical", confidence=1.0, decided_stage=1, stage1_scores={"signature": "dicom"})
        record_classification(conn, _AID, r)
        sqls = _execs(cur)
        self.assertTrue(any("INSERT INTO asset_classification" in s for s in sqls))
        self.assertTrue(any("UPDATE asset SET domain_label" in s for s in sqls))
        # medical 은 unresolved_pool 미적재
        self.assertFalse(any("unresolved_pool" in s for s in sqls))

    def test_review_marks_domain_label_only(self) -> None:
        # FR-013: review 는 asset.domain_label 표식만 — 드롭된 unresolved_pool 에 INSERT 금지
        conn, cur = _conn()
        r = ClassificationResult(final_label="review", confidence=0.0, decided_stage=3)
        record_classification(conn, _AID, r)
        sqls = _execs(cur)
        self.assertFalse(any("unresolved_pool" in s for s in sqls))
        self.assertTrue(any("UPDATE asset SET domain_label" in s for s in sqls))

    def test_general_no_unresolved_pool(self) -> None:
        conn, cur = _conn()
        r = ClassificationResult(final_label="general", confidence=0.7, decided_stage=2)
        record_classification(conn, _AID, r)
        self.assertFalse(any("unresolved_pool" in s for s in _execs(cur)))

    def test_reclassification_upserts_latest(self) -> None:
        # 재분류(동일 asset_id 재실행)는 ON CONFLICT 로 최신 결과 덮어쓰기 — 멱등 재실행 보장
        conn, cur = _conn()
        r = ClassificationResult(final_label="general", confidence=0.9, decided_stage=1)
        record_classification(conn, _AID, r)
        insert_sql = next(s for s in _execs(cur) if "INSERT INTO asset_classification" in s)
        self.assertIn("ON CONFLICT (asset_id) DO UPDATE", insert_sql)

    def test_update_propagates_label_confidence(self) -> None:
        # UPDATE asset 파라미터에 final_label·confidence·asset_id 가 그대로 전달된다
        conn, cur = _conn()
        r = ClassificationResult(final_label="general", confidence=0.7, decided_stage=2)
        record_classification(conn, _AID, r)
        update_call = next(
            c for c in cur.execute.call_args_list if "UPDATE asset SET domain_label" in c.args[0]
        )
        self.assertEqual(update_call.args[1], ("general", 0.7, _AID))


if __name__ == "__main__":
    unittest.main()
