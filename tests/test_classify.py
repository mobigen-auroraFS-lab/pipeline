"""분류 cascade 단위 테스트(도메인-불가지). 외부 모델/네트워크/DB 불필요.

- 프로파일/provider: test_classify_profiles.py
- 시그니처 능력/의료 프로파일: test_domain_medical.py
- 본 파일: count_hits · stage3 동적 라벨 · cascade 엔진(기본 RegistryProvider 포함) 행동.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from processing.classify import cascade, stage3_gemma
from processing.classify.medical_terms import MEDICAL_TERMS
from processing.classify.profiles import DomainProfile, SigHit
from processing.classify.stage2_zeroshot import count_hits
from processing.classify.types import DOMAIN_GENERAL, DOMAIN_MEDICAL, DOMAIN_REVIEW


class TmpBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, name: str, data: bytes) -> str:
        p = self.dir / name
        p.write_bytes(data)
        return str(p)


class TestCountHits(unittest.TestCase):
    def test_korean_substring(self) -> None:
        n, terms = count_hits("환자 진단 처방", MEDICAL_TERMS)
        self.assertGreaterEqual(n, 2)
        self.assertIn("환자", terms)

    def test_latin_word_boundary(self) -> None:
        _, terms = count_hits("CT scan and MRI ordered", MEDICAL_TERMS)
        self.assertIn("ct", terms)
        self.assertIn("mri", terms)

    def test_latin_substring_not_matched(self) -> None:
        n, _ = count_hits("Please contact the factory team", MEDICAL_TERMS)
        self.assertEqual(n, 0)

    def test_zero_hits(self) -> None:
        n, terms = count_hits("주식 시장 동향", MEDICAL_TERMS)
        self.assertEqual(n, 0)
        self.assertEqual(terms, [])

    def test_arbitrary_lexicon(self) -> None:
        n, _ = count_hits("이번 계약 소송 건", frozenset({"계약", "소송"}))
        self.assertEqual(n, 2)

    def test_empty_lexicon(self) -> None:
        n, _ = count_hits("anything", frozenset())
        self.assertEqual(n, 0)


class TestStage3(unittest.TestCase):
    def test_label_in_allowed(self) -> None:
        label, sig = stage3_gemma.classify(
            "환자 진단", ["medical", "general"], complete=lambda p: '{"label": "medical"}'
        )
        self.assertEqual(label, DOMAIN_MEDICAL)
        self.assertEqual(sig["stage3"], "llm")

    def test_dynamic_label_legal(self) -> None:
        label, _ = stage3_gemma.classify(
            "계약", ["medical", "legal", "general"], complete=lambda p: '{"label": "legal"}'
        )
        self.assertEqual(label, "legal")

    def test_not_allowed_returns_review(self) -> None:
        label, sig = stage3_gemma.classify(
            "x", ["medical", "general"], complete=lambda p: '{"label": "blah"}'
        )
        self.assertEqual(label, DOMAIN_REVIEW)
        self.assertEqual(sig["stage3"], "llm_unclear")

    def test_complete_raises_review(self) -> None:
        def boom(_p: str) -> str:
            raise RuntimeError("no endpoint")
        label, sig = stage3_gemma.classify("x", ["medical", "general"], complete=boom)
        self.assertEqual(label, DOMAIN_REVIEW)
        self.assertEqual(sig["stage3"], "error")

    def test_prompt_lists_labels(self) -> None:
        seen: dict[str, str] = {}
        def cap(p: str) -> str:
            seen["p"] = p
            return '{"label": "general"}'
        stage3_gemma.classify("본문", ["medical", "legal", "general"], complete=cap)
        self.assertIn("medical", seen["p"])
        self.assertIn("legal", seen["p"])


# --- cascade 엔진: 주입 provider 로 다도메인 ---

def _dicom(head: bytes) -> SigHit | None:
    return SigHit("dicom") if len(head) >= 132 and head[128:132] == b"DICM" else None


class _Provider:
    def __init__(self, profiles: list[DomainProfile]) -> None:
        self._p = profiles

    def all_profiles(self) -> list[DomainProfile]:
        return list(self._p)


_MED = DomainProfile("medical", frozenset({"환자", "진단", "처방"}), "medical", (_dicom,))
_LEGAL = DomainProfile("legal", frozenset({"계약", "소송", "판결"}), "legal", ())


class TestCascadeEngine(TmpBase):
    def setUp(self) -> None:
        super().setUp()
        self.provider = _Provider([_MED, _LEGAL])

    def test_stage1_unique_signature(self) -> None:
        path = self._write("a.dcm", b"\x00" * 128 + b"DICM")
        r = cascade.classify(path, "unknown", provider=self.provider)
        self.assertEqual(r.final_label, DOMAIN_MEDICAL)
        self.assertEqual(r.decided_stage, 1)
        self.assertEqual(r.confidence, 1.0)

    def test_stage2_top_confirmed(self) -> None:
        path = self._write("n.txt", "환자 진단 처방".encode())
        r = cascade.classify(path, "txt", provider=self.provider)
        self.assertEqual(r.final_label, DOMAIN_MEDICAL)
        self.assertEqual(r.decided_stage, 2)
        self.assertEqual(r.stage2_scores["medical"]["hits"], 3)

    def test_stage2_other_domain(self) -> None:
        path = self._write("n.txt", "계약 소송 판결".encode())
        r = cascade.classify(path, "txt", provider=self.provider)
        self.assertEqual(r.final_label, "legal")
        self.assertEqual(r.decided_stage, 2)

    def test_stage2_general(self) -> None:
        path = self._write("n.txt", "여행 후기와 맛집".encode())
        r = cascade.classify(path, "txt", provider=self.provider)
        self.assertEqual(r.final_label, DOMAIN_GENERAL)
        self.assertEqual(r.decided_stage, 2)

    def test_ambiguous_to_stage3(self) -> None:
        # 1 hit < _HIT_THRESHOLD(2) → 모호 → Stage 3.
        path = self._write("n.txt", "환자 대기실 안내".encode())
        calls: dict[str, list[str]] = {}
        def fake_s3(text: str, labels: list[str], **kw: object):
            calls["labels"] = labels
            return DOMAIN_GENERAL, {"stage3": "llm"}
        r = cascade.classify(path, "txt", provider=self.provider, _llm_classify=fake_s3)
        self.assertEqual(r.decided_stage, 3)
        self.assertEqual(r.final_label, DOMAIN_GENERAL)
        self.assertIn("medical", calls["labels"])
        self.assertIn("legal", calls["labels"])
        self.assertIn(DOMAIN_GENERAL, calls["labels"])

    def test_stage1_conflict_falls_through(self) -> None:
        also = DomainProfile("imaging", frozenset(), "imaging", (_dicom,))
        prov = _Provider([_MED, also])
        path = self._write("a.dcm", b"\x00" * 128 + b"DICM")
        r = cascade.classify(path, "unknown", provider=prov)
        self.assertNotEqual(r.decided_stage, 1)


class TestCascadeDefaultProvider(TmpBase):
    """provider 미지정 → RegistryProvider(등록된 의료 프로파일) 기본 경로."""

    def test_default_dicom_medical(self) -> None:
        path = self._write("a.dcm", b"\x00" * 128 + b"DICM")
        r = cascade.classify(path, "unknown")
        self.assertEqual(r.final_label, DOMAIN_MEDICAL)
        self.assertEqual(r.decided_stage, 1)

    def test_default_medical_text(self) -> None:
        path = self._write("note.txt", "환자 진단 처방 내역".encode())
        r = cascade.classify(path, "txt")
        self.assertEqual(r.final_label, DOMAIN_MEDICAL)
        self.assertEqual(r.decided_stage, 2)

    def test_default_general_text(self) -> None:
        path = self._write("note.txt", "여행 후기와 맛집 추천".encode())
        r = cascade.classify(path, "txt")
        self.assertEqual(r.final_label, DOMAIN_GENERAL)
        self.assertEqual(r.decided_stage, 2)


if __name__ == "__main__":
    unittest.main()
