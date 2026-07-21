"""DomainProfile / RegistryProvider 단위 테스트."""
from __future__ import annotations

import unittest

from processing.classify.profiles import (
    DOMAIN_PROFILES,
    DomainProfile,
    RegistryProvider,
    SigHit,
    register_profile,
)


class TestProfiles(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = dict(DOMAIN_PROFILES)
        DOMAIN_PROFILES.clear()

    def tearDown(self) -> None:
        DOMAIN_PROFILES.clear()
        DOMAIN_PROFILES.update(self._saved)

    def test_register_and_provider_returns_profile(self) -> None:
        p = DomainProfile(domain="legal", lexicon=frozenset({"계약", "소송"}), llm_label="legal")
        register_profile(p)
        profiles = RegistryProvider().all_profiles()
        self.assertEqual([x.domain for x in profiles], ["legal"])
        self.assertEqual(profiles[0].lexicon, frozenset({"계약", "소송"}))

    def test_sighit_holds_signature_and_detail(self) -> None:
        h = SigHit("fhir", {"resourceType": "Patient"})
        self.assertEqual(h.signature, "fhir")
        self.assertEqual(h.detail["resourceType"], "Patient")

    def test_register_overwrites_same_domain(self) -> None:
        register_profile(DomainProfile(domain="x", llm_label="x"))
        register_profile(DomainProfile(domain="x", lexicon=frozenset({"a"}), llm_label="x"))
        self.assertEqual(len(RegistryProvider().all_profiles()), 1)
        self.assertEqual(RegistryProvider().all_profiles()[0].lexicon, frozenset({"a"}))


if __name__ == "__main__":
    unittest.main()
