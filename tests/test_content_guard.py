"""065 G6 T601 — 무내용 가드(FR-701·SC-08) 단위 테스트(mock·LLM/DB 불필요).

배경(실측): 기악 오디오 등 무내용 자산의 STT 전사가 비었거나 얇으면, 요약기 LLM 이 자유 출력으로
placeholder 문장("제시된 텍스트 내용 없음")을 만들어 summary 에 들어가고 → 자기주제가 '미분류'로
분류되며 OS·관계·검색까지 오염됐다. 이 테스트는 무내용 신호(빈/얇은 STT)에서 **요약기 LLM 을 아예
호출하지 않고 summary 를 빈값**으로 두는 상류 가드를 봉인한다(placeholder 원천 차단).

검증 대상:
    - ``summarize_and_extract_keywords_from_audio`` 가 빈/얇은 STT 에서 ``complete_json`` 미호출 +
      ``{"summary": "", "keywords": []}`` (+stt 보존) 반환.
    - 임계 이상 STT 는 기존대로 요약기 LLM 호출(동작 보존).
    - 임계는 env ``TOPIC_MIN_SELF_TEXT`` 로 조정 가능(기본 15자).
    - 분류 하류(``classify_asset_topic``)는 빈 summary → 빈 self_text → None(미부여) — 상류 가드가
      빈값을 보장하므로 placeholder 문자열 매칭은 하지 않는다.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import src.llm.text_summarizer as ts


class TestAudioContentGuard(unittest.TestCase):
    """T601 — 오디오 STT 무내용 가드: 빈/얇은 전사 → 요약기 LLM 미호출·summary ''."""

    def test_empty_stt_skips_llm_returns_empty_summary(self) -> None:
        with patch.object(ts, "complete_json") as m_llm:
            out = ts.summarize_and_extract_keywords_from_audio(text="")
        m_llm.assert_not_called()  # 무내용 → 요약기 LLM 미호출(placeholder 원천 차단)
        self.assertEqual(out["summary"], "")
        self.assertEqual(out["keywords"], [])
        self.assertEqual(out["stt"], "")  # stt 원문 보존(임베딩 슬롯 재사용)

    def test_whitespace_only_stt_skips_llm(self) -> None:
        with patch.object(ts, "complete_json") as m_llm:
            out = ts.summarize_and_extract_keywords_from_audio(text="   \n \t  ")
        m_llm.assert_not_called()
        self.assertEqual(out["summary"], "")
        self.assertEqual(out["keywords"], [])

    def test_thin_stt_below_threshold_skips_llm(self) -> None:
        # 기본 임계 15자 미만(공백 trim 후) → 미호출.
        with patch.object(ts, "complete_json") as m_llm:
            out = ts.summarize_and_extract_keywords_from_audio(text="음 아 어")
        m_llm.assert_not_called()
        self.assertEqual(out["summary"], "")
        self.assertEqual(out["stt"], "음 아 어")  # stt 원문은 얇아도 보존

    def test_contentful_stt_calls_llm_as_before(self) -> None:
        # 임계 이상 → 기존대로 요약기 LLM 호출(동작 보존). settings·파싱은 기존 경로.
        long_text = "오늘 경기에서 선수들이 뛰어난 활약을 펼쳤고 관중들이 열광했다"
        with patch.object(
            ts, "complete_json",
            return_value={"summary": "경기 요약", "keywords": ["경기", "선수"]},
        ) as m_llm, patch.object(ts, "get_current_settings") as m_cfg:
            m_cfg.return_value.summary_max_chars = 500
            m_cfg.return_value.top_k_keywords = 10
            out = ts.summarize_and_extract_keywords_from_audio(text=long_text)
        m_llm.assert_called_once()
        self.assertEqual(out["summary"], "경기 요약")
        self.assertEqual(out["keywords"], ["경기", "선수"])
        self.assertEqual(out["stt"], long_text)

    def test_threshold_env_override(self) -> None:
        # env 로 임계를 크게 잡으면 중간 길이도 무내용 취급(미호출).
        mid_text = "짧은 발화 한 줄"  # 기본 15 미만이면 어차피 컷 — 큰 임계로 확실히 컷
        with patch.dict("os.environ", {"TOPIC_MIN_SELF_TEXT": "100"}), \
                patch.object(ts, "complete_json") as m_llm:
            out = ts.summarize_and_extract_keywords_from_audio(text=mid_text)
        m_llm.assert_not_called()
        self.assertEqual(out["summary"], "")


class TestClassifyEmptySummaryUnassigned(unittest.TestCase):
    """T601(분류) — 빈/무내용 summary → 빈 self_text → 미부여(None). placeholder 매칭 없음."""

    def test_build_self_text_empty_summary_no_keywords(self) -> None:
        from processing.classify.asset_topic import build_self_text

        # 상류 가드가 summary 를 ''로 준 무내용 자산: 자기 텍스트도 ''.
        self.assertEqual(build_self_text("", None, None), "")
        self.assertEqual(build_self_text("", [], []), "")

    def test_classify_empty_summary_returns_none_no_llm(self) -> None:
        from processing.classify import asset_topic

        client = unittest_client_never()
        with patch.object(asset_topic, "topic_candidates_for_self_text") as m_knn:
            out = asset_topic.classify_asset_topic(
                object(), "A", self_text="", client=client
            )
        self.assertIsNone(out)  # 빈 self_text → 미부여
        m_knn.assert_not_called()  # kNN·LLM 자체를 건너뜀


def unittest_client_never():
    """호출되면 실패시키는 client 대역 — LLM 미호출을 강제 검증."""
    from unittest.mock import MagicMock

    client = MagicMock()
    client.chat.completions.create.side_effect = AssertionError("LLM 호출되면 안 됨")
    return client


if __name__ == "__main__":
    unittest.main()
