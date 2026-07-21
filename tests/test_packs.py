"""v2 단계 B — 도메인 팩 단위 테스트."""
from __future__ import annotations

import unittest

from processing.pipeline.packs import GENERAL_PACK, MEDICAL_PACK, DomainPack, for_domain


class TestDomainPacks(unittest.TestCase):
    def test_general_and_medical_fields(self) -> None:
        self.assertEqual(GENERAL_PACK.policy, "general_default")
        self.assertEqual(MEDICAL_PACK.policy, "medical_strict")
        for pack in (GENERAL_PACK, MEDICAL_PACK):
            self.assertEqual(set(pack.per_asset), {"classify", "extract", "embed", "persist"})

    def test_for_domain_mapping_and_fallback(self) -> None:
        self.assertIs(for_domain("general"), GENERAL_PACK)
        self.assertIs(for_domain("medical"), MEDICAL_PACK)
        self.assertIs(for_domain("review"), GENERAL_PACK)   # 미지정/유보는 일반으로 보수적 폴백
        self.assertIs(for_domain("unknown"), GENERAL_PACK)


class TestDomainPackFrozen(unittest.TestCase):
    """US-E1(2026-07-20) — per_asset/cross_asset 매핑 내용까지 읽기전용(제자리 수정 차단)."""

    def test_mapping_content_is_readonly(self) -> None:
        with self.assertRaises(TypeError):
            GENERAL_PACK.per_asset["extract"] = "hacked"     # type: ignore[index]
        with self.assertRaises(TypeError):
            GENERAL_PACK.cross_asset["score"] = "hacked"     # type: ignore[index]

    def test_equality_semantics_preserved(self) -> None:
        # MappingProxyType 은 == 를 원본 매핑에 위임 — run_relations 라우팅 비교 의미 보존.
        same = DomainPack(
            name="x", per_asset={"a": "1"},
            cross_asset=dict(GENERAL_PACK.cross_asset), policy="general_default",
        )
        self.assertEqual(same.cross_asset, GENERAL_PACK.cross_asset)  # proxy == proxy
        self.assertEqual(same.cross_asset, dict(GENERAL_PACK.cross_asset))  # proxy == dict
        self.assertEqual(dict(GENERAL_PACK.per_asset),
                         {"classify": "cascade_v1", "extract": "by_modality",
                          "embed": "by_modality", "persist": "asset_upsert"})

    def test_read_access_unchanged(self) -> None:
        # 소비 관용(.get / [] / items) 전부 정상(동작 불변).
        self.assertEqual(GENERAL_PACK.per_asset["extract"], "by_modality")
        self.assertEqual(GENERAL_PACK.cross_asset.get("missing"), None)
        self.assertIn("candidates", dict(GENERAL_PACK.cross_asset).keys())


if __name__ == "__main__":
    unittest.main()
