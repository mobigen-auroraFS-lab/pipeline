"""061 G1 — 인입 파일 아카이브 순수/IO 헬퍼 단위 테스트 [FR-101~104·SC-06].

archiver 의 경로 계산·인입-하위 판정·이동 계획은 **순수·결정적**(시각 `when` 주입)이며 DB·Airflow
없이 검증한다. IO(`execute_move`)만 임시디렉터리로 실제 이동·멱등을 확인한다.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import date

from processing.ingest import archiver

_WHEN = date(2026, 7, 8)


class TestArchiveDest(unittest.TestCase):
    def test_basic(self) -> None:
        self.assertEqual(
            archiver.archive_dest("/arc", "x.txt", when=_WHEN, exists=lambda _p: False),
            os.path.join("/arc", "20260708", "x.txt"),
        )

    def test_subdir(self) -> None:
        self.assertEqual(
            archiver.archive_dest("/arc", "x.txt", when=_WHEN, subdir="dup", exists=lambda _p: False),
            os.path.join("/arc", "dup", "20260708", "x.txt"),
        )

    def test_collision_appends_counter(self) -> None:
        base = os.path.join("/arc", "20260708")
        taken = {os.path.join(base, "x.txt"), os.path.join(base, "x_1.txt")}
        self.assertEqual(
            archiver.archive_dest("/arc", "x.txt", when=_WHEN, exists=lambda p: p in taken),
            os.path.join(base, "x_2.txt"),
        )


class TestIsUnder(unittest.TestCase):
    def test_under_and_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inbox = os.path.join(d, "inbox")
            sibling = os.path.join(d, "inbox2")  # prefix 오탐 방지 경계 케이스
            os.makedirs(inbox)
            os.makedirs(sibling)
            self.assertTrue(archiver.is_under(os.path.join(inbox, "a.txt"), inbox))
            self.assertFalse(archiver.is_under(os.path.join(sibling, "a.txt"), inbox))
            self.assertFalse(archiver.is_under(inbox, inbox))  # 자기 자신은 하위 아님


class TestRegisteredDest(unittest.TestCase):
    def test_asset_id_keyed_and_reproducible(self) -> None:
        # asset_id 키라 전역 유일 + 재호출(복구 재스윕) 시 동일 경로 재현(C4 안전)
        d1 = archiver.registered_dest("/arc", "id1", "/inbox/a.txt", when=_WHEN)
        self.assertEqual(d1, os.path.join("/arc", "20260708", "id1__a.txt"))
        self.assertEqual(archiver.registered_dest("/arc", "id1", "/inbox/a.txt", when=_WHEN), d1)


class TestPlanArchiveMoves(unittest.TestCase):
    def test_only_under_inbox_deterministic(self) -> None:
        rows = [("id1", "/inbox/a.txt", _WHEN), ("id2", "/arc/20260101/b.txt", _WHEN)]  # id2 인입 밖
        moves = archiver.plan_archive_moves(rows, inbox_root="/inbox", archive_root="/arc")
        self.assertEqual(
            moves, [("id1", "/inbox/a.txt", os.path.join("/arc", "20260708", "id1__a.txt"))]
        )

    def test_uses_created_at_day(self) -> None:
        # 날짜 subdir 은 자산 created_at 기준(재스윕 날짜 무관·복구 시 동일)
        rows = [("id1", "/inbox/a.txt", date(2026, 1, 2))]
        moves = archiver.plan_archive_moves(rows, inbox_root="/inbox", archive_root="/arc")
        self.assertEqual(moves[0][2], os.path.join("/arc", "20260102", "id1__a.txt"))

    def test_same_name_no_collision_via_asset_id(self) -> None:
        # 동명 파일 두 개라도 asset_id 로 유일 → 충돌 카운터 불필요·결정적
        rows = [("id1", "/inbox/x.txt", _WHEN), ("id2", "/inbox/sub/x.txt", _WHEN)]
        dests = [m[2] for m in archiver.plan_archive_moves(rows, inbox_root="/inbox", archive_root="/arc")]
        self.assertEqual(dests, [
            os.path.join("/arc", "20260708", "id1__x.txt"),
            os.path.join("/arc", "20260708", "id2__x.txt"),
        ])


class TestAssertArchiveSeparate(unittest.TestCase):
    def test_raises_when_archive_under_inbox(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inbox = os.path.join(d, "inbox")
            os.makedirs(os.path.join(inbox, "arc"))
            with self.assertRaises(ValueError):  # archive 가 inbox 하위 → 자기수렴 불변식 위반
                archiver.assert_archive_separate(inbox, os.path.join(inbox, "arc"))

    def test_ok_when_separate(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            inbox = os.path.join(d, "inbox")
            archive = os.path.join(d, "archive")
            os.makedirs(inbox)
            os.makedirs(archive)
            archiver.assert_archive_separate(inbox, archive)  # 분리 → 예외 없음


class TestExecuteMove(unittest.TestCase):
    def test_move_creates_parents(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in", "a.txt")
            os.makedirs(os.path.dirname(src))
            with open(src, "w") as f:
                f.write("hi")
            dest = os.path.join(d, "arc", "20260708", "a.txt")
            archiver.execute_move(src, dest)
            self.assertFalse(os.path.exists(src))
            self.assertTrue(os.path.exists(dest))
            with open(dest) as f:
                self.assertEqual(f.read(), "hi")

    def test_idempotent_when_already_moved(self) -> None:
        # 부분 실패 복구: src 없고 dest 있으면 no-op(예외 없음·SC-04)
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "arc", "a.txt")
            os.makedirs(os.path.dirname(dest))
            with open(dest, "w") as f:
                f.write("x")
            archiver.execute_move(os.path.join(d, "in", "a.txt"), dest)  # src 없음
            self.assertTrue(os.path.exists(dest))

    def test_missing_both_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                archiver.execute_move(os.path.join(d, "nope.txt"), os.path.join(d, "arc", "nope.txt"))


if __name__ == "__main__":
    unittest.main()
