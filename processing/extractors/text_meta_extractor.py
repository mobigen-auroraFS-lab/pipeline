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

    ⚠️ 토크나이저만 필요할 때 임베딩 모델 전체를 올리면 안 된다 — 수 GB 를 잡아먹는다.
    임베딩이 원격(st_api=bge-m3)이면 로컬 임베딩 모델은 애초에 없고, 로컬(st)이어도 토큰 수 하나
    세자고 GB 가중치를 올릴 이유가 없다 → ``AutoTokenizer`` 로 토크나이저만 로드해 컨테이너 메모리·
    콜드로드를 줄인다. 무거운 import 는 함수 내부(모듈 import 시 transformers 미로딩)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text: str, *, model_name: str) -> int:
    """모델 기준 토큰 수를 센다 — 청크 크기를 정할 때 글자 수 대신 이 값을 본다.

    Args:
        text: 대상 텍스트.
        model_name: 토크나이저를 고를 기준 모델.

    Returns:
        토큰 개수.
    """
    # model_name = 활성 임베딩 모델(호출부가 active_embed_model 로 넘김) — 토큰 수를 임베딩과 정합.
    if not text:
        return 0
    tokenizer = _get_tokenizer(model_name)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(token_ids)


def _detect_language_from_counts(*, hangul_count: int, latin_count: int) -> str:
    """글자 종류 비율로 언어를 판별한다(사전·모델 없이 결정적).

    한글 쪽 기준을 더 낮게 잡는다 — 한글 문서에도 영문 용어가 흔히 섞여 한글 비중이
    낮게 나오기 때문이다. 양쪽 다 기준에 못 미치면 판별 불가로 둔다.

    Args:
        hangul_count: 한글 글자 수.
        latin_count: 라틴 문자 수.

    Returns:
        언어 코드 또는 판별 불가 표시. **억지로 하나를 고르지 않는다** — 잘못된 언어 라벨은
        형태소 분석·검색 경로를 엉뚱한 쪽으로 보낸다.
    """
    total_letters = hangul_count + latin_count
    if total_letters == 0:
        return "unknown"
    # 한글은 3할만 넘어도 한국어로 본다 — 한글 문서에 영문 용어·고유명사가 섞이는 것이 흔해,
    # 절반을 요구하면 실제 한국어 문서가 판별 불가로 떨어진다.
    if (hangul_count / total_letters) >= 0.3:
        return "ko"
    # 영문은 절반을 요구한다 — 한글이 이미 3할 미만이라 여기 오면 대체로 영문 문서다.
    if (latin_count / total_letters) >= 0.5:
        return "en"
    return "unknown"


def _count_sentences(text: str) -> int:
    """문장 수를 대략 센다(마침표 계열 구두점 기준 — 정확한 문장 분리가 목적이 아니다)."""
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
    """텍스트 파일에서 임베딩·검색에 쓸 메타를 뽑는다(길이·언어·문장 수·청크 등).

    Args:
        file_path: 대상 파일.
        file_kind: 파일 종류(읽기 방식을 정한다).
        encoding: 텍스트 인코딩.
        chunk_size: 청크 하나의 최대 토큰 수.
        embedding_model_name: **토큰 수를 셀 기준** 모델. 임베딩을 만들지는 않는다 —
            운영 호출부는 실제 사용 모델을 명시로 넘겨 청크 크기를 맞춘다.

    Returns:
        메타 dict.

    Raises:
        FileNotFoundError: 파일이 없을 때.
    """
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
