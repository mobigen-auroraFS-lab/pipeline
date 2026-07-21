"""count_tokens 는 임베딩 모델의 **토크나이저만**(경량 AutoTokenizer) 쓴다 — GB SentenceTransformer 미로드.

배경: 종전 count_tokens 는 토큰 수를 세려고 ``get_embedding_model``(SentenceTransformer 전체·GB)을 로드했고,
넘기는 모델도 활성 임베딩(st_api=bge-m3)이 아니라 로컬 KoSimCSE 하드코딩이라 **임베딩과 불일치**했다.
개선: (1) ``AutoTokenizer`` 로 토크나이저만 로드(가중치 X·컨테이너 메모리·콜드로드↓), (2) 호출부가
**활성 임베딩 모델**(``active_embed_model``)의 토크나이저를 쓰게 정합. num_tokens 는 메타데이터 전용이라
저위험(값이 활성 모델 기준으로 바뀌지만 오히려 정합). 실모델·네트워크 0(모두 mock).
"""

from __future__ import annotations

import unittest
from unittest import mock


class TestCountTokensTokenizerOnly(unittest.TestCase):
    def test_uses_autotokenizer_not_sentence_transformer(self) -> None:
        # 핵심: count_tokens 가 _get_tokenizer(AutoTokenizer) 만 쓰고 SentenceTransformer 전체는 로드 안 함.
        import processing.extractors.text_meta_extractor as tme

        fake_tok = mock.MagicMock()
        fake_tok.encode.return_value = [1, 2, 3, 4]
        with mock.patch.object(tme, "_get_tokenizer", return_value=fake_tok) as mk_tok:
            n = tme.count_tokens("안녕하세요", model_name="BAAI/bge-m3")
        self.assertEqual(n, 4)
        mk_tok.assert_called_once_with("BAAI/bge-m3")
        fake_tok.encode.assert_called_once()

    def test_empty_text_is_zero_without_loading(self) -> None:
        # 빈 텍스트는 토크나이저 로드 없이 0.
        import processing.extractors.text_meta_extractor as tme

        with mock.patch.object(tme, "_get_tokenizer") as mk_tok:
            self.assertEqual(tme.count_tokens("", model_name="m"), 0)
        mk_tok.assert_not_called()

    def test_get_tokenizer_loads_autotokenizer(self) -> None:
        # _get_tokenizer 는 transformers.AutoTokenizer.from_pretrained(model_name) 경유(가중치 로드 X).
        import processing.extractors.text_meta_extractor as tme

        tme._get_tokenizer.cache_clear()
        fake_tok = mock.MagicMock()
        with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=fake_tok) as mk:
            got = tme._get_tokenizer("BAAI/bge-m3")
        self.assertIs(got, fake_tok)
        mk.assert_called_once_with("BAAI/bge-m3")
        tme._get_tokenizer.cache_clear()


class TestExtractTextMetaAlignsModel(unittest.TestCase):
    def test_counts_with_active_embed_model(self) -> None:
        # 정합 봉인: _extract_text_meta 가 count 대상 모델로 **활성 임베딩 모델**(active_embed_model)을
        # 넘긴다 — 로컬 KoSimCSE 하드코딩(임베딩 불일치) 대신.
        import processing.skills.text_skill as ts

        captured: dict = {}

        def _fake_extract(**kw):
            captured.update(kw)
            return {}

        cfg = mock.MagicMock()
        ctx = mock.MagicMock(settings=cfg, modality="txt", file_path="/tmp/x.txt")
        with (
            mock.patch("processing.extractors.text_meta_extractor.extract_text_meta", side_effect=_fake_extract),
            mock.patch("src.llm.text_summarizer.summarize_and_extract_keywords", return_value={}),
            mock.patch.object(ts, "active_embed_model", return_value="BAAI/bge-m3") as mk_active,
            mock.patch.object(ts, "split_core_ext", return_value=({}, {})),
        ):
            ts._extract_text_meta(ctx)
        mk_active.assert_called()  # 활성 모델 해소 경유
        self.assertEqual(captured.get("embedding_model_name"), "BAAI/bge-m3")


if __name__ == "__main__":
    unittest.main()
