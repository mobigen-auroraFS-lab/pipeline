"""073 — aboutness 개체 추출·저장(순수 단위·client 주입·네트워크 0)."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from processing.classify.aboutness import extract_about, extract_and_persist_about, persist_about


def _client_returning(content: str) -> MagicMock:
    """``complete_json`` 이 호출하는 ``chat.completions.create`` 응답 모킹(query_preprocess 동형)."""
    c = MagicMock()
    c.chat.completions.create.return_value.choices = [MagicMock()]
    c.chat.completions.create.return_value.choices[0].message.content = content
    return c


class TestExtractAbout(unittest.TestCase):
    def test_parses_about_list(self) -> None:
        c = _client_returning('{"about": ["씨름"]}')
        self.assertEqual(extract_about("씨름의 정의와 역사. 고구려 고분벽화에도 등장.", client=c), ["씨름"])

    def test_caps_at_three_and_strips(self) -> None:
        c = _client_returning('{"about": [" 김치 ", "발효식품", "한식", "넷째", "다섯째"]}')
        self.assertEqual(extract_about("요약", client=c), ["김치", "발효식품", "한식"])

    def test_prompt_excludes_background_mentions(self) -> None:
        # 073 핵심 지시("배경·유래 언급 제외" — 언급≠주제 차단)가 프롬프트에 실제로 담긴다.
        c = _client_returning('{"about": ["씨름"]}')
        extract_about("요약", client=c)
        sent = c.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("배경", sent)
        self.assertIn("제외", sent)

    def test_temperature_zero(self) -> None:
        # 헌법 §3: 단일 seam 기본 temperature=0 유지(비0 리터럴 미전달).
        c = _client_returning('{"about": ["x"]}')
        extract_about("요약", client=c)
        self.assertEqual(c.chat.completions.create.call_args.kwargs["temperature"], 0.0)

    def test_blank_summary_no_llm_call(self) -> None:
        c = _client_returning('{"about": ["x"]}')
        self.assertEqual(extract_about("", client=c), [])
        self.assertEqual(extract_about("   ", client=c), [])
        self.assertEqual(extract_about(None, client=c), [])
        c.chat.completions.create.assert_not_called()

    def test_schema_violation_falls_back_empty(self) -> None:
        # fail-safe(FR-001): 비-dict·비-list·빈 응답 전부 [] — 추출 실패가 적재를 깨지 않는다.
        for bad in ("[1,2]", '{"about": "명사"}', '{"other": 1}', ""):
            self.assertEqual(extract_about("요약", client=_client_returning(bad)), [])


class TestPersistAbout(unittest.TestCase):
    def test_jsonb_merge_upsert_shape(self) -> None:
        conn = MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        persist_about(conn, "aid-1", ["씨름", "민속놀이"])
        sql, params = cur.execute.call_args.args
        self.assertIn("ext_meta", sql)
        self.assertIn("||", sql)  # jsonb 병합(기존 키 보존·마이그레이션 0)
        self.assertEqual(json.loads(params[0]), {"about": ["씨름", "민속놀이"]})
        self.assertEqual(params[1], "aid-1")

    def test_extract_and_persist_stores_empty_too(self) -> None:
        # 빈 [] 도 저장 — "시도함" 기록으로 백필 --only-missing 이 무한 재시도하지 않게(멱등 근거).
        conn = MagicMock()
        cur = conn.cursor.return_value.__enter__.return_value
        out = extract_and_persist_about(conn, "aid-2", summary="", client=_client_returning("x"))
        self.assertEqual(out, [])
        _, params = cur.execute.call_args.args
        self.assertEqual(json.loads(params[0]), {"about": []})


if __name__ == "__main__":
    unittest.main()
