"""코어 계약 ↔ 테스트 더블 드리프트 감지.

**왜 필요한가**: 이 레포의 관계 경로 테스트는 코어 함수(`propose_relations_for_asset` 등)를
가짜 설정·가짜 seam 으로 감싸 돌린다. 코어가 설정을 하나 더 읽거나 인자를 하나 더 넘기기 시작하면
그 더블이 조용히 낡고, **테스트가 `AttributeError`/`TypeError` 로 죽는다.**

실제로 그렇게 됐다(2026-08-05 발견): 코어에 `persist_min_conf_similarity`·
`auto_approve_exclude_kinds`·`stats`·`source_keywords`·`source_filename`·`keywords` 열이 추가된 뒤
이 레포 테스트 4건이 실패한 상태로 방치돼 있었다. 게다가 **한 개를 고치면 다음 것이 드러나는**
식이어서 원인을 한 번에 못 보게 만든다.

그래서 이 파일은 **더블을 고치는 대신 드리프트 자체를 실패로 만든다**. 실패 메시지가 "무엇을
어디에 추가해야 하는지"를 알려주므로, 다음 사람이 원인 추적에 시간을 쓰지 않는다.

⚠️ 이 테스트는 **코어 소스를 읽는다**(설치된 `src` 패키지). 코어가 없으면 skip 된다.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

try:
    from src.relations import asset_entry as _ae

    _CORE = True
except Exception:  # pragma: no cover - 코어 미설치 환경
    _CORE = False

from tests.test_asset_relations import _FAKE_CFG

# 이 레포의 더블들이 **현재 받아들이는** 키워드 인자. 코어가 인자를 늘리면 아래 테스트가 실패하고,
# 그때 `tests/test_asset_relations.py` 의 `_fake_prompt`·`_fake_sync` 에 같은 인자를 추가해야 한다.
# (일부러 `**kwargs` 로 삼키지 않는다 — 삼키면 계약 변화가 묻힌다.)
_DOUBLE_PROMPT_KWARGS = {
    "source_summary", "source_media_type", "candidates", "relation_kinds_catalog",
    "source_topic", "source_keywords", "source_filename",
}
_DOUBLE_SYNC_KWARGS = {
    "source_asset_id", "edges", "allowed_target_ids", "auto_approve_min",
    "target_emb_scores", "auto_approve_emb_min", "collect",
    "persist_min_conf_similarity", "auto_approve_exclude_kinds", "stats",
}

_CFG_READ = re.compile(r"cfg\.relations\.([a-z_]+)")


def _call_kwargs(source: str, func_name: str) -> set[str]:
    """소스에서 특정 함수 **호출부가 넘기는** 키워드 인자 이름을 모은다.

    시그니처(정의)가 아니라 호출부를 보는 이유: 더블은 코어가 **실제로 넘기는** 인자만 받으면 된다.
    정의 전체와 비교하면 코어가 안 쓰는 선택 인자까지 더블에 요구해 잡음이 된다
    (실측: ``build_relation_proposal_prompt`` 는 호출부가 안 넘기는 override 인자를 3개 더 갖고 있다).

    Args:
        source: 검사할 파이썬 소스 텍스트(코어 모듈 원문).
        func_name: 호출부를 찾을 함수 이름(예: ``sync_graph_edges``).

    Returns:
        그 함수 호출에 쓰인 키워드 인자 이름 집합. ``**kwargs`` 전개는 이름이 없어 제외된다.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        called = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if called != func_name:
            continue
        names.update(kw.arg for kw in node.keywords if kw.arg)
    return names


@unittest.skipUnless(_CORE, "코어(src) 미설치 — 계약 검사 불가")
class TestCoreSettingsDoubleCoversReads(unittest.TestCase):
    """코어가 읽는 `cfg.relations.*` 가 가짜 설정에 전부 있는지."""

    def test_fake_cfg_covers_every_setting_core_reads(self) -> None:
        src_text = Path(_ae.__file__).read_text(encoding="utf-8")
        needed = set(_CFG_READ.findall(src_text))
        self.assertTrue(needed, "cfg.relations.* 참조를 하나도 못 찾았다 — 정규식이 낡았을 수 있다")
        have = set(vars(_FAKE_CFG.relations))
        missing = sorted(needed - have)
        self.assertEqual(
            missing, [],
            "코어가 읽는데 _FAKE_CFG.relations 에 없는 설정이다 — tests/test_asset_relations.py 의 "
            "_FAKE_CFG(및 TestLineageRecordsEdgePairs 안의 cfg)에 추가할 것:\n  " + "\n  ".join(missing))


@unittest.skipUnless(_CORE, "코어(src) 미설치 — 계약 검사 불가")
class TestCoreSeamCallSitesMatchDoubles(unittest.TestCase):
    """코어 호출부가 넘기는 키워드 인자가 더블이 받는 집합 안에 있는지."""

    def setUp(self) -> None:
        self.src = Path(_ae.__file__).read_text(encoding="utf-8")

    def test_prompt_builder_call_kwargs_are_accepted_by_double(self) -> None:
        used = _call_kwargs(self.src, "build_relation_proposal_prompt")
        self.assertTrue(used, "프롬프트 빌더 호출부를 못 찾았다 — 코어 구조가 바뀌었을 수 있다")
        extra = sorted(used - _DOUBLE_PROMPT_KWARGS)
        self.assertEqual(
            extra, [],
            "코어가 build_relation_proposal_prompt 에 새 인자를 넘긴다 — tests/test_asset_relations.py 의 "
            "_fake_prompt 시그니처와 위 _DOUBLE_PROMPT_KWARGS 에 추가할 것:\n  " + "\n  ".join(extra))

    def test_sync_graph_edges_call_kwargs_are_accepted_by_double(self) -> None:
        used = _call_kwargs(self.src, "sync_graph_edges")
        self.assertTrue(used, "sync_graph_edges 호출부를 못 찾았다 — 코어 구조가 바뀌었을 수 있다")
        extra = sorted(used - _DOUBLE_SYNC_KWARGS)
        self.assertEqual(
            extra, [],
            "코어가 sync_graph_edges 에 새 인자를 넘긴다 — tests/test_asset_relations.py 의 _fake_sync "
            "시그니처와 위 _DOUBLE_SYNC_KWARGS 에 추가할 것:\n  " + "\n  ".join(extra))


if __name__ == "__main__":
    unittest.main()
