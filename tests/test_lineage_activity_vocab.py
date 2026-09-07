"""계보 활동명을 코어 정본(``LineageActivity``)에서 가져와도 **저장되는 문자열은 전과 같다**. DB 없음.

093 1단계에서 파이프의 리터럴 8종(``"ingest.received.v1"`` …)을 코어 상수로 바꿨다. 활동명은 계보
테이블에 그대로 저장되고 백엔드가 그 문자열로 집계하므로(``relations.proposed.v1`` 로 5버킷 판별),
한 글자라도 달라지면 **과거 계보와 새 계보가 다른 활동으로 갈린다**. 여기서 옛 리터럴과 값을 직접 대조한다.
"""
from __future__ import annotations

import json
import unittest

from processing.ingest import batch_runner
from src.database.lineage_activity import LineageActivity

# 2026-09-07 치환 전 파이프에 박혀 있던 리터럴(정확히 이 글자였다).
_LEGACY = {
    "INGEST_RECEIVED": "ingest.received.v1",
    "INGEST_ROUTING": "ingest.routing.v1",
    "INGEST_CLASSIFYING": "ingest.classifying.v1",
    "INGEST_CLASSIFIED": "ingest.classified.v1",
    "INGEST_DEFERRED": "ingest.deferred.v1",
    "INGEST_EXTRACTING": "ingest.extracting.v1",
    "INGEST_REGISTERED": "ingest.registered.v1",
    "INGEST_FAILED": "ingest.failed.v1",
    "INGEST_RESET": "ingest.reset.v1",
}


class TestLineageActivityParity(unittest.TestCase):
    """값·타입·직렬화가 리터럴과 구별되지 않는다."""

    def test_값이_옛_리터럴과_같다(self) -> None:
        for name, legacy in _LEGACY.items():
            with self.subTest(name=name):
                member = getattr(LineageActivity, name)
                self.assertEqual(member, legacy)          # 비교
                self.assertEqual(str(member), legacy)     # f-string·str()
                self.assertEqual(member.value, legacy)    # .value

    def test_문자열처럼_동작한다(self) -> None:
        # psycopg 파라미터·json·문자열 연산 어디서든 리터럴과 같게 보여야 한다.
        member = LineageActivity.INGEST_FAILED
        self.assertIsInstance(member, str)
        self.assertEqual(json.dumps(member), json.dumps("ingest.failed.v1"))
        self.assertEqual("x " + member, "x ingest.failed.v1")

    def test_batch_runner_상수가_정본을_가리킨다(self) -> None:
        self.assertIs(batch_runner.FAILED_ACTIVITY, LineageActivity.INGEST_FAILED)
        self.assertIs(batch_runner.RESET_ACTIVITY, LineageActivity.INGEST_RESET)
        self.assertEqual(batch_runner.FAILED_ACTIVITY, "ingest.failed.v1")
        self.assertEqual(batch_runner.RESET_ACTIVITY, "ingest.reset.v1")


if __name__ == "__main__":
    unittest.main()
