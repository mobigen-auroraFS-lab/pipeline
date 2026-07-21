"""v2 단계 B — 정책 엔진 단위 테스트."""
from __future__ import annotations

import unittest

from processing.pipeline.packs import DomainPack
from processing.pipeline.policy import PolicyViolation, validate
from processing.pipeline.registry import StrategyRegistry


def _reg(extract_tags=()):
    r = StrategyRegistry()
    r.register("classify", "c", lambda ctx: None)
    r.register("extract", "x", lambda ctx: None, tags=extract_tags)
    r.register("embed", "e", lambda ctx, rec: [])
    r.register("persist", "p", lambda *a: None)
    return r


def _pack(policy):
    return DomainPack("d", {"classify": "c", "extract": "x", "embed": "e", "persist": "p"}, {}, policy)


class TestPolicy(unittest.TestCase):
    def test_general_default_passes(self) -> None:
        validate(_pack("general_default"), _reg())  # 예외 없음

    def test_medical_strict_rejects_external_llm(self) -> None:
        with self.assertRaises(PolicyViolation):
            validate(_pack("medical_strict"), _reg(extract_tags={"external_llm"}))

    def test_medical_strict_passes_when_no_external_llm(self) -> None:
        validate(_pack("medical_strict"), _reg())  # 예외 없음

    def test_unknown_policy_raises(self) -> None:
        with self.assertRaises(PolicyViolation):
            validate(_pack("nope"), _reg())


if __name__ == "__main__":
    unittest.main()
