"""F-1.3 자산 등록·적재 (모델 A).

``run_extract_meta.py`` 의 ``media_items``/``media_chunks`` 직접 INSERT 를 신규
``asset``/``asset_metadata``/``asset_embedding`` 로 재배선한 통일 영속화 계층.

모델 A 분리
    - ``create_asset``: 파일 픽업 직후 ``asset`` 행을 ``received`` 로 조기 INSERT(asset_id 확보).
    - ``finalize_asset``: 추출 결과(``AssetRecord``)의 메타·임베딩을 적재하고 상태를 ``registered`` 로.
      (호출 전 상태가 ``extracting`` 이어야 함 — 상태 머신 검증)

두 함수 모두 psycopg ``Connection`` 을 받아 오케스트레이터가 트랜잭션 경계를 제어한다
(단계별 짧은 트랜잭션 + 실패 시 fresh 트랜잭션으로 mark_failed — T1-6 참고).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from psycopg import Connection
from psycopg.rows import dict_row

from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.database.ids import uuid7
from processing.dispatch.types import AssetRecord
from src.file.file_type_defs import modality_of  # 저장 경계 canonical 매핑(053)
from processing.ingest.status import AssetStatus, InvalidTransitionError, fetch_status, set_status


def find_registered_asset_by_hash(conn: Connection[Any], file_hash: str) -> uuid.UUID | None:
    """동일 내용(file_hash)으로 이미 적재된 자산의 asset_id. 없으면 None(중복 적재 방지용).

    run_ingest 가 파일 픽업 직후 이 함수로 중복을 검사해, 기존 자산이 있으면 파이프라인을 건너뛴다.

    009(#4) dedup 범위 확장: 중복 식별 대상은 ``status IN ('registered','deferred')`` 다.
    함수명은 'registered' 지만 의미는 "이미 적재되어 보존 중인 자산"으로 넓어졌다.
      - ``deferred`` 포함 이유: DICOM 등 의료 표준 포맷이 추출기 부재로 보류(deferred)된 자산은
        실패가 아니라 계획적 대기 상태다. 같은 파일을 재수집할 때마다 중복 ``deferred`` 행이
        쌓이지 않도록(SC-003) 중복으로 본다. 단계 D에서 deferred→registered 정상 전이가
        일어나도 그땐 같은 asset_id 를 재사용하므로 이 dedup 이 정상 전이를 막지 않는다.
      - ``failed`` 미포함(재처리 허용 유지): 실패한 같은 해시는 중복으로 보지 않아 재수집 시
        새로 처리한다(실패 복구 경로 — FR-004, SC-004). 그래서 IN 목록에 'failed' 를 넣지 않는다.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT asset_id FROM asset "
            "WHERE file_hash = %s AND status IN ('registered', 'deferred') LIMIT 1",
            (file_hash,),
        )
        row = cur.fetchone()
    return row["asset_id"] if row else None


def create_asset(
    conn: Connection[Any],
    *,
    fs_path: str,
    modality: str,
    domain: str = "general",
    file_hash: str | None = None,
    file_size: int | None = None,
) -> uuid.UUID:
    """``asset`` 행을 ``received`` 상태로 INSERT 하고 asset_id(UUIDv7) 반환(모델 A 조기 INSERT).

    식별자는 앱에서 UUIDv7 로 생성해 명시적으로 INSERT 한다(PG17 네이티브 uuidv7() 부재).

    인자 ``modality`` 는 file_kind(``route.modality``) 이지만 저장은 canonical(``modality_of``, 053 A안)
    로 좁힌다 — 저장 chokepoint 단일화(우회 경로 없음). 추출은 이 컬럼을 안 읽고 ``route_file`` 로
    file_kind 를 재판정하므로 저장값 정규화는 추출과 독립이다(내부 추출 무변경).
    """
    asset_id = uuid7()
    canonical = modality_of(modality)   # file_kind → text/image/video/audio/unknown
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO asset (asset_id, modality, fs_path, file_hash, file_size, domain_label, status)
            VALUES (%s, %s, %s, %s, %s, %s, 'received')
            """,
            (asset_id, canonical, fs_path, file_hash, file_size, domain),
        )
    return asset_id


def finalize_asset(conn: Connection[Any], asset_id: uuid.UUID, record: AssetRecord) -> None:
    """``AssetRecord`` 의 메타·임베딩을 적재하고 상태를 ``registered`` 로 전이한다.

    ``asset_metadata`` 1행(core/ext jsonb, tags) + ``asset_embedding`` N행을 삽입.
    마지막에 ``set_status(..., REGISTERED)`` — 현재 상태가 ``extracting`` 이어야 한다(상태 머신 검증).

    임베딩이 없을 때(record.embeddings 빈 리스트) executemany 를 건너뛴다.
    벡터는 항상 ``FIX_EMBEDDING_DIMENSION``(1536D) ::vector 캐스트로 저장 — DB CHECK 제약과 일치해야 한다.

    037(OS 전용): 종전 PG FTS 컬럼(v270 에서 드롭)을 제거했다. 풀텍스트 색인은 OpenSearch 동기화가
    ``ext_meta`` 기반으로 생성하므로 적재 시 FTS 평문을 더 이상 채우지 않는다(INSERT 4컬럼만).

    069 B7(P2-7): 서두에서 현재 상태가 ``extracting`` 인지 먼저 확인해, 아니면 INSERT 이전에
    ``InvalidTransitionError`` 로 막는다(SELECT 1회 추가). 말미 ``set_status`` 도 같은 전이를
    검증하지만, autocommit 새 호출자에서는 그전에 metadata/embedding INSERT 가 이미 커밋돼
    중복·고아 행이 영속될 수 있다. 정상 경로(extracting)는 결과가 완전히 동일하다(회귀 0).
    """
    current = fetch_status(conn, asset_id)
    if current != AssetStatus.EXTRACTING:
        raise InvalidTransitionError(
            f"finalize_asset 는 extracting 상태에서만 호출 가능: 현재 {current.value}"
        )
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO asset_metadata (asset_id, core_meta, ext_meta, tags)
            VALUES (%s, %s::jsonb, %s::jsonb, %s)
            """,
            (
                asset_id,
                json.dumps(record.core_meta, ensure_ascii=False),
                json.dumps(record.ext_meta, ensure_ascii=False),
                list(record.tags),
            ),
        )
        if record.embeddings:
            # 채널(st/clip)·청크별 1행씩 bulk INSERT — 같은 트랜잭션 안에서 메타와 함께 커밋
            cur.executemany(
                f"""
                INSERT INTO asset_embedding (asset_id, channel, chunk_index, embedding, model_name, model_version)
                VALUES (%s, %s, %s, %s::vector({FIX_EMBEDDING_DIMENSION}), %s, %s)
                """,
                [
                    (asset_id, e.channel, e.chunk_index, e.vector, e.model_name, e.model_version)
                    for e in record.embeddings
                ],
            )
    set_status(conn, asset_id, AssetStatus.REGISTERED)
