"""PG(`asset_*`) 전체 → OpenSearch 재색인 **복구 도구** CLI (검색 엔진 도입 — spec 020 G4).

정상 경로(새 자산 실시간 반영)는 run_ingest 증분 훅(G3)이 담당한다. 본 CLI 는 그 정상 경로가
아니라 **복구/관리** 도구다 — OpenSearch 인덱스가 **손실**되거나 PG 와 **드리프트**(어긋남)
났거나 스키마가 바뀌었을 때, PG 의 `registered` 자산 전체를 OpenSearch 로 **재색인**해 깨끗이
맞춘다(US2). `_id=asset_id` upsert 라 **재실행 멱등**(중복 0, FR-003).

CQRS — PG 는 **읽기 전용**(SELECT 만, FR-004·헌법 6조), 쓰기는 OpenSearch 에만.
`--recreate` 는 인덱스 삭제 후 재생성(스키마 변경 시·**파괴적 옵트인**); 기본은 비파괴 upsert.

설계 — 순수 조립부 / 실행(IO) 경계 (docs/테스트_가이드.md §0 하이브리드, 017/019 measure 러너 동형)
    - **순수 조립부**(단위 검증, OS·DB 무관): `run_resync(client, conn, *, channel, index, recreate)`
      는 동기화 코어 `sync_all` 을 **주입 seam**(`sync_fn`)으로 호출하고 결과(status·ok·errors)를
      보고만 한다. `tests/test_run_opensearch_resync.py` 가 가짜 client/conn/sync_fn 을 주입해
      OS·DB 없이 조립(인자 전달·결과 보고)을 단위로 덮는다.
    - **실행(IO) 부트스트랩**(G5·사람): `main()` 만 load_dotenv→init_settings→get_client→
      PostgresUtil 읽기전용 트랜잭션에서 실제 `sync_all` 을 돌린다 — 실OS·실DB 재색인은 G5 사람 단계.

사용법
    conda activate AuroraFS
    python -m processing.app.run_opensearch_resync --env dev               # 활성 채널·설정 인덱스, 비파괴 upsert
    python -m processing.app.run_opensearch_resync --env dev --recreate    # 인덱스 재생성(스키마 변경 시)
    python -m processing.app.run_opensearch_resync --env dev --channel st_bge --index assets_bge
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any

# sync_all 은 opensearch_sync 의 순수/지연 import 설계상 모듈 상단에서 안전하게 가져올 수 있다
# (opensearch-py 는 sync_all 내부에서 실제 호출 시에만 지연 import). 따라서 본 모듈 import 만으로는
# opensearch-py 미설치 환경에서도 깨지지 않는다 — 단위 테스트가 OS 없이 run_resync 를 덮을 수 있는 이유.
from src.search.opensearch_sync import sync_all


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PG → OpenSearch 전체 재색인 복구 도구 (PG 읽기 전용·재실행 멱등)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    p.add_argument("--channel", default=None, help="임베딩 채널(미지정=활성 프로파일·018)")
    p.add_argument("--index", default=None, help="OpenSearch 인덱스(미지정=OPENSEARCH_INDEX)")
    p.add_argument(
        "--recreate",
        action="store_true",
        help="인덱스를 삭제 후 재생성(파괴적·스키마 변경 시만). 기본은 비파괴 upsert.",
    )
    # 027: 검색 정규화 융합이 서버 파이프라인에서 클라이언트 융합으로 이동해 --ensure-pipeline 옵션은
    # 제거됐다(등록할 서버 파이프라인이 없음). 재색인 도구는 인덱스 동기화에만 집중한다.
    return p


def run_resync(
    client: Any,
    conn: Any,
    *,
    channel: str,
    index: str,
    recreate: bool,
    nori_user_words: Any = None,
    noise_patterns: Any = (),
    sync_fn: Callable[..., tuple[str, int, list[Any]]] = sync_all,
) -> dict[str, Any]:
    """복구 러너의 **순수 조립부** — 전체 재동기화 코어(`sync_all`)를 호출하고 결과를 보고한다.

    `sync_fn` 은 동기화 seam(기본 `opensearch_sync.sync_all`)으로, 단위 테스트가 가짜를 주입해
    OS·DB 없이 인자 전달·결과 보고를 검증한다. PG 읽기 전용→OS 쓰기·멱등(`_id=asset_id`)은 코어의
    책임이고, 여기서는 (이미 해소된) channel·index·recreate 를 그대로 흘려보내고 코어가 돌려준
    ``(상태, 색인수, 오류목록)`` 을 보고용 dict 로 모은다(상태·색인 맥락을 함께 담아 출력·검수 용이).
    """
    # 026: 인덱스 analyzer 사전·파일명 정제 패턴은 settings 단일 출처를 IO 층이 주입한다(미지정=기본).
    status, ok, errors = sync_fn(
        client, conn, index=index, channel=channel, recreate=recreate,
        nori_user_words=nori_user_words, noise_patterns=noise_patterns,
    )
    return {
        "status": status,
        "ok": ok,
        "errors": list(errors),
        "channel": channel,
        "index": index,
        "recreate": recreate,
    }


def format_report(report: dict[str, Any], *, doc_count: int | None = None) -> str:
    """복구 결과를 사람이 읽는 한 줄 요약으로(순수). 오류가 있으면 상위 2건 샘플을 덧붙인다."""
    line = (
        f"  인덱스 상태: {report['status']} | 색인 성공: {report['ok']} | "
        f"오류: {len(report['errors'])} | channel='{report['channel']}' index='{report['index']}'"
    )
    if doc_count is not None:
        line += f" | 인덱스 총문서: {doc_count}"
    if report["errors"]:
        line += f"\n  ⚠️ 오류 샘플: {report['errors'][:2]}"
    return line


# ── 실행(IO) 부트스트랩 — 실OS·실DB 재색인은 G5(사람) ─────────────────────────────
# 1) load_dotenv(.env.{env}) → 2) init_settings → 3) channel·index 해소(미지정=활성·설정) →
# 4) get_client → 5) PostgresUtil 읽기전용 트랜잭션에서 run_resync(=sync_all) → 6) 결과 출력.
# PG 는 SELECT 만(FR-004·헌법 6조). 위 run_resync 는 OS·DB 없이 단위 검증되는 순수 조립부다.
# 무거운 의존(dotenv·settings·PostgresUtil·get_client→opensearch-py)은 실행 시에만 지연 import 한다.
def main() -> int:
    args = _build_parser().parse_args()


    from src.config.bootstrap import bootstrap_env
    from src.config.settings import get_current_settings
    from src.database.postgres_util import PostgresUtil
    from src.search.opensearch_sync import check_pgvector_version, get_client, resolve_channel

    bootstrap_env(args.env)

    cfg = get_current_settings()
    channel = resolve_channel(args.channel)  # 미지정=활성 프로파일(018)
    index = args.index or cfg.opensearch.index  # 미지정=OPENSEARCH_INDEX

    client = get_client()
    info = client.info()
    print(
        f"[OpenSearch 복구 재색인] {cfg.opensearch.url} (v{info['version']['number']}) "
        f"→ index='{index}' channel='{channel}' recreate={args.recreate}"
    )

    db = PostgresUtil()

    def _resync_txn(conn: Any) -> dict[str, Any]:
        # 선검사: 동기화 SELECT 의 avg(embedding) 집계는 pgvector>=0.5 의존 → 미달이면 재색인 전에
        # 원인 분명한 오류로 중단(모호한 SQL 오류 회피). 그 뒤 읽기전용 전체 재동기화를 조립·실행.
        check_pgvector_version(conn)
        return run_resync(
            client, conn, channel=channel, index=index, recreate=args.recreate,
            nori_user_words=cfg.opensearch.nori_user_words,
            noise_patterns=cfg.opensearch.filename_noise_patterns,
        )

    with db:
        # 읽기 전용 조회 트랜잭션(원본 PG 무수정, FR-004). 멱등(_id=asset_id upsert)이라 재시도 안전.
        report = db.execute_in_transaction(_resync_txn, idempotent=True)

    doc_count = client.count(index=index).get("count")
    print(format_report(report, doc_count=doc_count))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
