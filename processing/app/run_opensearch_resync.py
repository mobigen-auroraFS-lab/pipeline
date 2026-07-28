"""DB 의 자산 전체를 검색 엔진으로 다시 색인하는 **복구 도구**.

**흐름에서의 위치**: 정상 경로가 아니다. 새 자산은 적재가 끝날 때마다 그 자리에서 색인되고,
이 도구는 색인이 **유실·손상됐거나 DB 와 어긋났거나 매핑을 바꿨을 때**만 쓴다.

DB 는 **읽기만** 한다(헌법 6조) — 쓰기는 검색 엔진 쪽에만. 같은 자산을 다시 넣으면 덮어쓰므로
여러 번 돌려도 안전하다.

⚠️ ``--recreate`` 는 **인덱스를 지우고 다시 만든다.** 재색인이 끝날 때까지 검색이 비어 보이므로
매핑을 바꿀 때만 쓴다. 기본은 지우지 않는 덮어쓰기다.

IO 경계를 나눠 뒀다: 조립부는 동기화 함수를 **주입받아** 부르고 결과만 보고하므로 검색 엔진·DB
없이 단위 검증되고, 실제 연결·트랜잭션은 ``main`` 만 만든다.

사용법
    python -m processing.app.run_opensearch_resync --env dev               # 덮어쓰기(기본)
    python -m processing.app.run_opensearch_resync --env dev --recreate    # 인덱스 재생성
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
    """명령행 옵션을 정의한다(환경·채널·인덱스·재생성 여부)."""
    p = argparse.ArgumentParser(
        description="PG → OpenSearch 전체 재색인 복구 도구 (PG 읽기 전용·재실행 멱등)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    p.add_argument("--channel", default=None, help="임베딩 채널(미지정=활성 프로파일)")
    p.add_argument("--index", default=None, help="OpenSearch 인덱스(미지정=OPENSEARCH_INDEX)")
    p.add_argument(
        "--recreate",
        action="store_true",
        help="인덱스를 삭제 후 재생성(파괴적·스키마 변경 시만). 기본은 비파괴 upsert.",
    )
    # 점수 융합을 서버가 아니라 클라이언트에서 하게 바뀌어 ``--ensure-pipeline`` 옵션은
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
    """동기화 코어를 부르고 결과를 보고용으로 모은다 — 조립만 하는 층이다.

    값 해소(어느 채널·어느 인덱스인지)는 호출부가 이미 끝냈고, 여기서는 그대로 흘려보낸다.

    Args:
        client: 검색 엔진 클라이언트.
        conn: DB 연결. **읽기만 한다**(헌법 6조).
        channel: 임베딩 채널.
        index: 대상 인덱스.
        recreate: ⚠️ **참이면 인덱스를 지우고 다시 만든다** — 되돌릴 수 없다.
        nori_user_words: 형태소 분석 사용자 사전. 설정에서 주입한다.
        noise_patterns: 파일명 정제 패턴. 설정에서 주입한다.
        sync_fn: 동기화 함수. **바꿔 끼울 수 있게 열어 뒀다** — 검색 엔진·DB 없이 인자
            전달과 결과 보고를 단위 검증한다.

    Returns:
        상태·성공 수·오류 목록에 어느 채널·인덱스였는지를 함께 담은 dict(검수용).
    """
    # 형태소 사전·파일명 정제 패턴은 설정 한 곳에서 IO 층이 주입한다(미지정=기본값).
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
    """복구 결과를 사람이 읽는 한 줄로 만든다(순수 함수).

    Args:
        report: 조립부가 돌려준 결과 dict.
        doc_count: 색인의 전체 문서 수. ``None`` 이면 그 항목을 빼고 찍는다.

    Returns:
        요약 문자열. **오류는 앞 2건만** 덧붙인다 — 전부 찍으면 콘솔이 넘쳐 정작 상태를
        못 본다(자세한 내용은 로그에 있다).
    """
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
# PG 는 SELECT 만(헌법 6조). 위 run_resync 는 검색 엔진·DB 없이 단위 검증되는 순수 조립부다.
# 무거운 의존(dotenv·settings·PostgresUtil·get_client→opensearch-py)은 실행 시에만 지연 import 한다.
def main() -> int:
    """PostgreSQL 의 자산을 OpenSearch 로 **전부 다시 색인**한다(복구 도구).

    색인이 유실·손상됐거나 매핑을 바꿨을 때 쓴다. PG 는 읽기만 하고, 같은 자산을 다시
    넣어도 덮어쓰므로 여러 번 돌려도 안전하다.

    ⚠️ ``--recreate`` 를 주면 **인덱스를 지우고 다시 만든다** — 재색인이 끝날 때까지
    검색 결과가 비어 보인다.

    Returns:
        0=성공.
    """
    args = _build_parser().parse_args()


    from src.config.bootstrap import bootstrap_env
    from src.config.settings import get_current_settings
    from src.database.postgres_util import PostgresUtil
    from src.search.opensearch_sync import check_pgvector_version, get_client, resolve_channel

    bootstrap_env(args.env)

    cfg = get_current_settings()
    channel = resolve_channel(args.channel)  # 미지정이면 활성 프로파일을 따른다
    index = args.index or cfg.opensearch.index  # 미지정=OPENSEARCH_INDEX

    client = get_client()
    info = client.info()
    print(
        f"[OpenSearch 복구 재색인] {cfg.opensearch.url} (v{info['version']['number']}) "
        f"→ index='{index}' channel='{channel}' recreate={args.recreate}"
    )

    db = PostgresUtil()

    def _resync_txn(conn: Any) -> dict[str, Any]:
        """한 커넥션 안에서 사전 점검 → 전체 재색인을 수행한다.

        확장 버전을 **먼저** 확인하는 이유: 조회 SQL 이 특정 버전 이상에서만 되는 집계를
        쓰는데, 그냥 시작하면 한참 뒤 모호한 오류로 멈춘다.
        """
        check_pgvector_version(conn)
        return run_resync(
            client, conn, channel=channel, index=index, recreate=args.recreate,
            nori_user_words=cfg.opensearch.nori_user_words,
            noise_patterns=cfg.opensearch.filename_noise_patterns,
        )

    with db:
        # 읽기 전용 트랜잭션 — 원본을 고치지 않는다. 같은 자산을 다시 넣으면 덮어쓰므로 재시도도 안전하다.
        report = db.execute_in_transaction(_resync_txn, idempotent=True)

    doc_count = client.count(index=index).get("count")
    print(format_report(report, doc_count=doc_count))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
