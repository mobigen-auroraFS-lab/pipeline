"""닫힌 어휘 StrEnum ↔ DB CHECK 동기 검증 (spec 042)."""
from __future__ import annotations

import unittest

from processing.ingest.status import AssetStatus
from src.domain.status_vocab import (
    AccessTier,
    GraphEdgeStatus,
    RegistryFieldStatus,
    RelationResolutionStatus,
)


class StatusVocabSyncTest(unittest.TestCase):
    """마이그레이션 CHECK IN 목록과 StrEnum 값 집합이 일치해야 한다."""

    def test_access_tier_matches_v290_check(self):
        self.assertEqual(
            frozenset(AccessTier),
            frozenset(("public", "authenticated", "authorized", "regulated")),
        )

    def test_graph_edge_status_matches_v230_check(self):
        self.assertEqual(
            frozenset(GraphEdgeStatus),
            frozenset(("proposed", "active", "rejected")),
        )

    def test_relation_resolution_status_matches_v260_check(self):
        self.assertEqual(
            frozenset(RelationResolutionStatus),
            frozenset(("pending", "resolved", "isolated", "failed")),
        )

    def test_registry_field_status_matches_core_check(self):
        self.assertEqual(
            frozenset(RegistryFieldStatus),
            frozenset(("active", "inactive")),
        )

    def test_asset_status_matches_v160_check(self):
        self.assertEqual(
            frozenset(AssetStatus),
            frozenset(
                (
                    "received",
                    "routing",
                    "classifying",
                    "extracting",
                    "registered",
                    "failed",
                    "deferred",
                )
            ),
        )

    def test_access_tier_ordinal_order_in_access_tier_module(self):
        from src.registry.access_tier import TIER_ORDER

        self.assertEqual(TIER_ORDER, tuple(AccessTier))
