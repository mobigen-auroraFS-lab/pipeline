"""F-1.1 로컬 파일 수집(collector) 단위 테스트. DB·LLM 불필요(임시 디렉터리만)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from processing.ingest.collector import collect_files


class CollectorTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _touch(self, rel: str) -> Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
        return p


class TestDirectoryCollection(CollectorTestBase):
    def test_collects_files_in_dir(self) -> None:
        self._touch("a.txt")
        self._touch("b.mp4")
        out = collect_files(input_dir=self.root)
        self.assertEqual(len(out), 2)
        self.assertTrue(all(Path(p).is_file() for p in out))

    def test_recursive_toggle(self) -> None:
        self._touch("top.txt")
        self._touch("sub/deep.txt")
        rec = collect_files(input_dir=self.root, recursive=True)
        non = collect_files(input_dir=self.root, recursive=False)
        self.assertEqual(len(rec), 2)
        self.assertEqual(len(non), 1)
        self.assertTrue(any(p.endswith("top.txt") for p in non))

    def test_hidden_excluded_by_default(self) -> None:
        # 069 US-F: include_hidden 옵션은 제거됐고 숨김 항목은 항상 제외(종전 기본 동작).
        self._touch("visible.txt")
        self._touch(".hidden.txt")
        default = collect_files(input_dir=self.root)
        self.assertEqual(len(default), 1)
        self.assertTrue(default[0].endswith("visible.txt"))


class TestFileList(CollectorTestBase):
    def test_reads_file_list_skipping_blanks(self) -> None:
        f1 = self._touch("one.txt")
        f2 = self._touch("two.txt")
        listing = self.root / "list.txt"
        listing.write_text(f"{f1}\n\n   \n{f2}\n", encoding="utf-8")
        out = collect_files(file_list=listing)
        self.assertEqual(out, [str(f1), str(f2)])

    def test_missing_file_list_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            collect_files(file_list=self.root / "nope.txt")


class TestCombineAndErrors(CollectorTestBase):
    def test_explicit_paths(self) -> None:
        out = collect_files(paths=["/a/x.txt", "/a/y.txt"])
        self.assertEqual(out, ["/a/x.txt", "/a/y.txt"])

    def test_exact_duplicate_removed_preserving_order(self) -> None:
        out = collect_files(paths=["/a/x.txt", "/a/y.txt", "/a/x.txt"])
        self.assertEqual(out, ["/a/x.txt", "/a/y.txt"])

    def test_no_source_raises(self) -> None:
        with self.assertRaises(ValueError):
            collect_files()

    def test_bad_input_dir_raises(self) -> None:
        with self.assertRaises(NotADirectoryError):
            collect_files(input_dir=self.root / "no_such_dir")

    def test_empty_dir_returns_empty(self) -> None:
        self.assertEqual(collect_files(input_dir=self.root), [])


if __name__ == "__main__":
    unittest.main()
