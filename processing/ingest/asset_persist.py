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

from processing.dispatch.types import AssetRecord
from processing.ingest.status import AssetStatus, InvalidTransitionError, fetch_status, set_status
from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.database.ids import uuid7
from src.file.file_type_defs import modality_of  # 저장할 때 큰 갈래로 좁히는 매핑


def find_registered_asset_by_hash(conn: Connection[Any], file_hash: str) -> uuid.UUID | None:
    """동일 내용(file_hash)으로 이미 적재된 자산의 asset_id. 없으면 None(중복 적재 방지용).

    run_ingest 가 파일 픽업 직후 이 함수로 중복을 검사해, 기존 자산이 있으면 파이프라인을 건너뛴다.

    중복 식별 대상은 ``status IN ('registered','deferred')`` 다.
    함수명은 'registered' 지만 의미는 "이미 적재되어 보존 중인 자산"으로 넓어졌다.
      - ``deferred`` 포함 이유: DICOM 등 의료 표준 포맷이 추출기 부재로 보류(deferred)된 자산은
        실패가 아니라 계획적 대기 상태다. 같은 파일을 재수집할 때마다 중복 ``deferred`` 행이
        쌓이지 않도록 중복으로 본다. 나중에 보류→등록 정상 전이가
        일어나도 그땐 같은 asset_id 를 재사용하므로 이 dedup 이 정상 전이를 막지 않는다.
      - ``failed`` 미포함(재처리 허용 유지): 실패한 같은 해시는 중복으로 보지 않아 재수집 시
        새로 처리한다(실패 복구 경로). 그래서 IN 목록에 'failed' 를 넣지 않는다.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT asset_id FROM asset "
            "WHERE file_hash = %s AND status IN ('registered', 'deferred') LIMIT 1",
            (file_hash,),
        )
        row = cur.fetchone()
    return row["asset_id"] if row else None


def find_duplicate_terminal_asset(conn: Connection[Any], asset_id: uuid.UUID) -> uuid.UUID | None:
    """이 자산과 내용이 같으면서 **이미 끝난** 다른 자산의 id. 없으면 ``None``.

    Args:
        conn: DB 커넥션(읽기 전용 — 이 함수는 쓰지 않는다).
        asset_id: 기준 자산. 이 자산의 ``file_hash`` 로 찾고 자기 자신은 제외한다.

    Returns:
        내용이 같은 종료 자산 하나의 id. 해시가 없거나 그런 자산이 없으면 ``None``.
        여럿이면 먼저 만들어진 것을 준다(같은 입력에 같은 답 — 사유 문자열이 흔들리지 않게).

    설계 배경: `docs/설계_변경이력.md` 2026-09-11 (2)
    """
    # `find_registered_asset_by_hash` 와 무엇이 다른가: 저쪽은 **파일을 집어 들 때**(경로만 아는
    #   시점) 해시로 묻고, 이쪽은 이미 만들어진 자산 행을 기준으로 자기 자신을 빼고 묻는다.
    # 왜 또 묻나(2026-09-10 실측): 수집 시점 검사는 종료 상태(registered·deferred)만 본다. 같은
    #   배치로 함께 들어온 사본은 그때 둘 다 `received` 라 서로를 보지 못한다. 국보 사진 6건이 그
    #   창으로 빠져나가 추출까지 마친 뒤 등록에서 유일 색인에 걸렸고, 파이프는 그것을 추출 실패로
    #   보아 3번 재시도한 끝에 `failed` 로 굳혔다. 처리 시점에는 먼저 들어온 사본이 이미 끝나 있다.
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT other.asset_id FROM asset me "
            "JOIN asset other ON other.file_hash = me.file_hash "
            "WHERE me.asset_id = %s AND me.file_hash IS NOT NULL "
            "  AND other.asset_id <> me.asset_id "
            "  AND other.status IN ('registered', 'deferred') "
            "ORDER BY other.created_at, other.asset_id LIMIT 1",
            (asset_id,),
        )
        row = cur.fetchone()
    return row["asset_id"] if row else None


def clear_file_hash(conn: Connection[Any], asset_id: uuid.UUID) -> None:
    """이 자산의 ``file_hash`` 를 비운다(DB 쓰기) — 중복 보류 직전에만 쓴다.

    Args:
        conn: DB 커넥션(호출자 트랜잭션 안에서 돈다).
        asset_id: 해시를 비울 자산. 🔴 ``set_status(DEFERRED)`` **보다 먼저** 같은 트랜잭션에서
            부른다 — 뒤집으면 유일 색인이 먼저 터진다(지연 검사가 아니다).

    설계 배경: `docs/설계_변경이력.md` 2026-09-11 (2)
    """
    # 왜 비우나(2026-09-11 실측): 유일 색인 `uq_asset_file_hash_dedup` 의 조건이
    #   `status IN ('registered','deferred') AND file_hash IS NOT NULL` 이다. 내용이 같은 자산을
    #   해시를 둔 채 `deferred` 로 바꾸면 그 전이 자체가 색인 위반으로 터진다(롤백 시험으로 확인).
    # 잃는 것과 남는 것: 해시는 잃지만 무엇의 중복인지는 남는다 — 사유 문자열
    #   `duplicate_content:<원본 asset_id>` 와 계보 `ingest.deferred.v1` 의 `duplicate_of` 에 적힌다.
    #   해시는 파일에서 다시 구할 수 있고, 내용의 대표는 살아남은 원본 자산이 갖는다.
    with conn.cursor() as cur:
        cur.execute("UPDATE asset SET file_hash = NULL WHERE asset_id = %s", (asset_id,))


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

    **DB에 쓴다.** 식별자는 DB 가 아니라 앱에서 만들어 넣는다 — 값이 시간순으로 정렬되는
    형식이라 목록 조회의 보조 정렬 키로 그대로 쓸 수 있다.

    Args:
        conn: DB 연결.
        fs_path: 파일 경로.
        modality: 파일 종류. ⚠️ **저장할 때 큰 갈래로 좁힌다** — 세부 종류를 그대로 넣으면
            집계 축이 무한정 늘어난다. 좁히는 지점을 여기 하나로 두어 우회 경로를 없앴다.
            추출 단계는 이 컬럼을 읽지 않고 경로로 다시 판정하므로 영향이 없다.
        domain: 도메인 라벨.
        file_hash: 파일 해시. 중복 판정에 쓴다.
        file_size: 파일 크기(바이트).

    Returns:
        새로 만든 자산 id.
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

    **DB에 쓴다** — 메타 1행 + 임베딩 N행을 넣고 마지막에 상태를 완료로 전이한다.

    ⚠️ **쓰기 전에 상태를 먼저 확인한다.** 마지막 상태 전이에서도 같은 검증을 하지만,
    자동 커밋으로 부르는 호출자에게는 그때 이미 메타·임베딩이 커밋돼 **고아 행이 남는다**.
    그래서 조회 한 번을 더 들여 시작 전에 막는다.

    Args:
        conn: DB 연결(트랜잭션 경계는 호출자).
        asset_id: 대상 자산.
        record: 저장할 메타·태그·임베딩. **임베딩이 비어 있어도 정상**이다 — 그때는 벡터
            INSERT 를 통째로 건너뛴다(요약이 없거나 임베딩 대상이 아닌 자산).

    Raises:
        InvalidTransitionError: 추출 단계가 아닌 자산에 대해 불렀을 때.
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
