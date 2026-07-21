"""v2 단계 B — 전략 레지스트리 단위 테스트."""
from __future__ import annotations

import unittest

from processing.pipeline.registry import StrategyRegistry


class TestStrategyRegistry(unittest.TestCase):
    def test_register_resolve_tags(self) -> None:
        r = StrategyRegistry()
        r.register("extract", "fake", lambda ctx: "X", tags={"deterministic"})
        self.assertEqual(r.resolve("extract", "fake")(None), "X")
        self.assertEqual(r.tags("extract", "fake"), frozenset({"deterministic"}))

    def test_resolve_missing_raises(self) -> None:
        r = StrategyRegistry()
        with self.assertRaises(KeyError):
            r.resolve("extract", "nope")

    def test_default_tags_empty(self) -> None:
        r = StrategyRegistry()
        r.register("persist", "p", lambda *a: None)
        self.assertEqual(r.tags("persist", "p"), frozenset())


if __name__ == "__main__":
    unittest.main()
