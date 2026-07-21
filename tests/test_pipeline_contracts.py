"""v2 단계 A — 스테이지 계약/타입/scratch 단위 테스트."""
from __future__ import annotations

import unittest

from processing.dispatch.types import ExtractContext
from processing.pipeline import contracts as C
from processing.pipeline.cross_types import Candidate, Decision, Evidence, ScoredPair


class TestExtractContextScratch(unittest.TestCase):
    def test_scratch_defaults_empty_and_independent(self) -> None:
        a = ExtractContext(file_path="/x.txt", modality="txt")
        b = ExtractContext(file_path="/y.txt", modality="txt")
        self.assertEqual(a.scratch, {})
        a.scratch["clip_vec"] = [0.1]
        self.assertEqual(b.scratch, {})  # 인스턴스별 독립(가변 기본값 공유 금지)


class TestStageContracts(unittest.TestCase):
    def test_protocols_exist(self) -> None:
        for name in ("ClassifyStage", "ExtractStage", "EmbedStage", "PersistStage",
                     "CandidateStage", "ScoreStage", "DecideStage", "EdgePersistStage"):
            self.assertTrue(hasattr(C, name), name)

    def test_existing_extract_fn_is_callable_extract_stage(self) -> None:
        from processing.skills.text_skill import _extract_text_meta
        self.assertTrue(callable(_extract_text_meta))  # ExtractStage 구조(=__call__) 충족


class TestCrossTypes(unittest.TestCase):
    def test_construct_and_frozen(self) -> None:
        c = Candidate(source_id="a", target_id="b", block_key="dob_sex", method="blocking")
        ev = Evidence(field="dob", comparator="exact", similarity=1.0, weight=2.3)
        sp = ScoredPair(candidate=c, score=10.0, evidence=[ev])
        d = Decision(candidate=c, verdict="match", score=10.0)
        self.assertEqual(sp.candidate.target_id, "b")
        self.assertEqual(d.verdict, "match")
        with self.assertRaises(AttributeError):  # frozen dataclass → FrozenInstanceError(⊂AttributeError)
            c.source_id = "z"  # frozen


if __name__ == "__main__":
    unittest.main()
