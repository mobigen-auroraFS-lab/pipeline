"""텍스트/문서 본문 통계 메타(언어·인코딩·문장수·토큰수·길이) 추출 — text_skill 이 호출.

본문을 청크로 순회하며 집계만 한다. 임베딩 벡터는 ``src/embedders/text_embedder.py`` 가
별도로 만든다(추출/임베딩 분리 설계). 토큰 수는 **토크나이저만** 필요하므로, GB 단위 임베딩 모델
전체(SentenceTransformer)를 로드하지 않고 경량 ``AutoTokenizer`` 만 로드한다(``count_tokens`` 참조).
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from src.file.data_loader import (
    MAX_INPUT_CHARS,
    choose_encoding,
    iter_document_chunks,
    normalize_file_kind,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = ["EmbeddingTextMeta", "count_tokens", "extract_text_meta"]


class EmbeddingTextMeta(TypedDict):
    language: str
    encoding: str
    num_sentences: int
    num_tokens: int
    length: int


@lru_cache(maxsize=4)
def _get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """토큰 수 세기용 **토크나이저만** 로드(가중치 X). 프로세스 캐시.

    종전엔 ``get_embedding_model``(SentenceTransformer 전체·GB)을 로드해 그 ``.tokenizer`` 만 썼다.
    임베딩이 원격(st_api=bge-m3)이면 로컬 임베딩 모델은 애초에 없고, 로컬(st)이어도 토큰 수 하나
    세자고 GB 가중치를 올릴 이유가 없다 → ``AutoTokenizer`` 로 토크나이저만 로드해 컨테이너 메모리·
    콜드로드를 줄인다. 무거운 import 는 함수 내부(모듈 import 시 transformers 미로딩)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text: str, *, model_name: str) -> int:
    # model_name = 활성 임베딩 모델(호출부가 active_embed_model 로 넘김) — 토큰 수를 임베딩과 정합.
    if not text:
        return 0
    tokenizer = _get_tokenizer(model_name)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


def _detect_language_from_counts(*, hangul_count: int, latin_count: int) -> str:
    # 글자 종류 비율 기반 휴리스틱(사전·LLM 없이 결정적). 한글은 라틴보다 낮은 임계(0.3)를
    # 쓴다 — 한글 문서도 영문 용어가 흔히 섞여 한글 비중이 낮게 나오기 때문.
    # 라틴 0.5 미만이고 한글도 0.3 미만이면 판별 불가로 unknown.
    total_letters = hangul_count + latin_count
    if total_letters == 0:
        return "unknown"
    if (hangul_count / total_letters) >= 0.3:
        return "ko"
    if (latin_count / total_letters) >= 0.5:
        return "en"
    return "unknown"


def _count_sentences(text: str) -> int:
    if not text.strip():
        return 0
    parts = re.split(r"[.!?]+|\n+", text)
    return len([p for p in parts if p.strip()])


def extract_text_meta(
    file_path: str | Path,
    *,
    file_kind: str,
    encoding: str = "utf-8",
    chunk_size: int = 512,
    # 기본은 토큰 수 세기용 폴백 — 운영 호출부(text_skill)는 active_embed_model 을 명시 주입한다.
    embedding_model_name: str = "BM-K/KoSimCSE-roberta-multitask",
) -> EmbeddingTextMeta:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    kind = normalize_file_kind(file_kind)
    if kind is None:
        raise ValueError("file_kind는 필수입니다.")

    effective_encoding = choose_encoding(path, encoding).upper()

    num_sentences = 0
    num_tokens = 0
    length = 0
    hangul_count = 0
    latin_count = 0

    for chunk in iter_document_chunks(
        path,
        file_kind=kind,
        encoding=encoding,
        chunk_size=chunk_size,
        overlap_size=0,
        max_input_chars=MAX_INPUT_CHARS,
    ):
        if not chunk:
            continue
        length += len(chunk)
        num_sentences += _count_sentences(chunk)
        num_tokens += count_tokens(chunk, model_name=embedding_model_name)
        hangul_count += len(re.findall(r"[가-힣]", chunk))
        latin_count += len(re.findall(r"[A-Za-z]", chunk))

    language = _detect_language_from_counts(
        hangul_count=hangul_count,
        latin_count=latin_count,
    )

    return {
        "language": language,
        "encoding": effective_encoding,
        "num_sentences": num_sentences,
        "num_tokens": num_tokens,
        "length": length,
    }
