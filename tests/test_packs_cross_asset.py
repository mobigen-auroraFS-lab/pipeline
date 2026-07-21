import unittest


class TestCrossAssetPack(unittest.TestCase):
    def test_general_pack_has_cross_asset_slots(self):
        from processing.pipeline.packs import for_domain
        pack = for_domain("general")
        self.assertEqual(pack.cross_asset["candidates"], "embedding_topk")
        self.assertEqual(pack.cross_asset["score"], "llm_propose")
        self.assertEqual(pack.cross_asset["persist_edges"], "graph_upsert")

    def test_medical_falls_back_to_general_cross_asset(self):
        from processing.pipeline.packs import for_domain
        self.assertIn("candidates", for_domain("medical").cross_asset)
