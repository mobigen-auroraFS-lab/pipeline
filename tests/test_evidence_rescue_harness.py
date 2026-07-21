"""045 G2 — evidence rescue·P29 골든 라이브 하니스."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

_FX = Path(__file__).resolve().parent / "fixtures" / "search"
_RUN = os.getenv("RUN_OS_E2E") == "1"


class P29GoldenFixtureTest(unittest.TestCase):
    """순수: P29 스키마·라벨 계약."""

    @classmethod
    def setUpClass(cls) -> None:
        golden = json.loads((_FX / "golden_os.json").read_text(encoding="utf-8"))
        cls.p29 = next(q for q in golden["queries"] if q["id"] == "P29")

    def test_p29_present_fishing_prep(self) -> None:
        self.assertEqual(self.p29["query"], "낚시 준비물")
        self.assertEqual(self.p29["category"], "present")
        self.assertIn("낚시", self.p29["topics"])
        self.assertIn("018f0000-0000-7000-8000-000000000249", self.p29["relevant"])
        self.assertIn("018f0000-0000-7000-8000-000000000220", self.p29["partial"])

    def test_manifest_documents_noise_assets(self) -> None:
        manifest = (_FX / "golden_p29_manifest.md").read_text(encoding="utf-8")
        self.assertIn("캠핑", manifest)
        self.assertIn("019e91a8", manifest)


def _run_search(query: str, *, rescue: str | None = None) -> dict:
    env = os.environ.copy()
    if rescue is not None:
        env["SEARCH_EVIDENCE_RESCUE_ENABLED"] = rescue
    raw = subprocess.check_output(
        [sys.executable, "-m", "processing.app.run_search", "--env", "dev", "--query", query, "--limit", "20"],
        env=env,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    return json.loads(raw)


def _ranked_ids(result: dict) -> list[str]:
    rows = sorted(
        (r for bucket in (result.get("results") or {}).values() for r in (bucket or [])),
        key=lambda r: (-float(r.get("similarity") or 0.0), str(r.get("id") or r.get("asset_id") or "")),
    )
    seen: list[str] = []
    for r in rows:
        rid = str(r.get("id") or r.get("asset_id") or "")
        if rid and rid not in seen:
            seen.append(rid)
    return seen


@unittest.skipUnless(_RUN, "RUN_OS_E2E=1 + 실 OS 필요")
class EvidenceRescueLiveHarnessTest(unittest.TestCase):
    """라이브: q=테스트 RESCUE 0/1 분기."""

    def test_rescue_off_keeps_weak_lexical(self) -> None:
        off = _run_search("테스트", rescue="0")
        on = _run_search("테스트", rescue="1")
        off_ids = _ranked_ids(off)
        on_ids = _ranked_ids(on)
        self.assertGreater(len(off_ids), len(on_ids))
        plan = (off.get("meta") or {}).get("search_plan") or {}
        self.assertEqual(plan.get("lexical_rescue"), "restricted")


@unittest.skipUnless(_RUN, "RUN_OS_E2E=1 + 실 OS 필요")
class P29GoldenLiveTest(unittest.TestCase):
    """라이브: P29 recall@20·gate_passed 경로."""

    @classmethod
    def setUpClass(cls) -> None:
        golden = json.loads((_FX / "golden_os.json").read_text(encoding="utf-8"))
        cls.p29 = next(q for q in golden["queries"] if q["id"] == "P29")

    def test_p29_recall_at_20(self) -> None:
        result = _run_search(self.p29["query"])
        seen = set(_ranked_ids(result)[:20])
        rel = set(self.p29["relevant"])
        hit = len(rel & seen)
        self.assertGreater(hit, 0, f"relevant 0/{len(rel)} @20 — {seen}")
        recall = hit / len(rel)
        self.assertGreaterEqual(recall, 1 / len(rel))

    def test_p29_gate_passed_text_bucket(self) -> None:
        result = _run_search(self.p29["query"])
        gate = ((result.get("meta") or {}).get("os_gate") or {}).get("text") or {}
        self.assertTrue(gate.get("gate_passed"), gate)


if __name__ == "__main__":
    unittest.main()
