"""의료 프로파일 시그니처 규칙 + 등록 테스트."""
from __future__ import annotations

import unittest

from processing.classify.domains.medical import MEDICAL_PROFILE, _dicom, _fhir, _hl7v2
from processing.classify.types import DOMAIN_MEDICAL


class TestMedicalSignatures(unittest.TestCase):
    def test_dicom(self) -> None:
        hit = _dicom(b"\x00" * 128 + b"DICM" + b"\x00" * 10)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.signature, "dicom")

    def test_dicom_too_short_none(self) -> None:
        self.assertIsNone(_dicom(b"\x00" * 10))

    def test_hl7v2(self) -> None:
        hit = _hl7v2(b"MSH|^~\\&|HIS|HOSP|...\r")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.signature, "hl7v2")

    def test_fhir_known_resource(self) -> None:
        hit = _fhir(b'{"resourceType": "Patient", "id": "1"}')
        self.assertIsNotNone(hit)
        self.assertEqual(hit.signature, "fhir")
        self.assertEqual(hit.detail["resourceType"], "Patient")

    def test_fhir_unknown_resource_none(self) -> None:
        self.assertIsNone(_fhir(b'{"resourceType": "Widget"}'))

    def test_non_medical_none(self) -> None:
        data = "오늘 날씨가 좋네요 hello".encode()
        self.assertIsNone(_dicom(data))
        self.assertIsNone(_hl7v2(data))
        self.assertIsNone(_fhir(data))

    def test_profile_shape(self) -> None:
        self.assertEqual(MEDICAL_PROFILE.domain, DOMAIN_MEDICAL)
        self.assertEqual(MEDICAL_PROFILE.llm_label, DOMAIN_MEDICAL)
        self.assertIn("환자", MEDICAL_PROFILE.lexicon)
        self.assertEqual(len(MEDICAL_PROFILE.signatures), 3)


if __name__ == "__main__":
    unittest.main()
