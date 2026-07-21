"""F-3.4 디스패처 입출력 데이터클래스.

``AssetRecord`` 는 모든 추출 함수(extract_text/image/video/audio)의 **통일 출력 계약**이다.
영속화(F-1.3 ``persist_asset``)가 이 한 가지 형태만 받아 ``asset``/``asset_metadata``/
``asset_embedding`` 에 단일 경로로 적재한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EmbeddingItem:
    """``asset_embedding`` 한 행에 대응. 채널·청크별 1536D 벡터."""

    channel: str  # 'st' | 'clip' | (옵션 B) 'medclip' 등
    vector: list[float]
    model_name: str
    model_version: str | None = None
    chunk_index: int = 0


@dataclass
class ExtractContext:
    """추출 함수에 넘기는 입력 컨텍스트.

    **불변식**: ``dispatch_extract_meta`` 와 ``dispatch_embed`` 는 동일한 ctx 인스턴스를 공유해야 한다.
    extract 단계가 ``scratch`` 에 기록한 중간 산출물(CLIP 벡터, STT 결과, 키프레임 경로 등)을
    embed 단계가 읽어 재계산을 피한다.
    """

    file_path: str
    modality: str  # MediaKind/OfficeKind 값 (txt/pdf/json/word/excel/powerpoint/image/video/audio)
    domain: str = "general"  # 'general' | 'medical' | 'review'
    settings: Any = None  # PipelineSettings (선택)
    db: Any = None  # PostgresUtil (선택)
    # extract → embed 핸드오프 버퍼. skill 이 고비용 중간 산출물을 키 이름으로 저장한다.
    scratch: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssetRecord:
    """추출 결과의 통일 표현. asset_metadata(core/ext/tags) + asset_embedding(embeddings).

    037(OS 전용): 종전 PG FTS 평문 입력 필드를 제거했다. 풀텍스트 색인은 OpenSearch
    동기화가 ``ext_meta`` 에서 생성하므로 적재 계약에 별도 FTS 입력이 필요 없다.
    """

    core_meta: dict[str, Any] = field(default_factory=dict)  # 파일/시스템 메타
    ext_meta: dict[str, Any] = field(default_factory=dict)  # 도메인 신호(summary/keywords/labels 등)
    tags: list[str] = field(default_factory=list)
    embeddings: list[EmbeddingItem] = field(default_factory=list)
