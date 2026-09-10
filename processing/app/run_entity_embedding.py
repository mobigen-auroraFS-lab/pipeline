"""개체 임베딩 배치 — 개체마다 검색용 벡터를 채운다 (spec 090 G2).

무엇을 하는 배치인가: 노출 임계를 넘은 개체(멀티모달 메타)의 **검색 재료**를 조립해 임베딩하고
``entity_embedding`` 에 저장한다. 이것이 채워져야 개체를 **글자가 겹치지 않아도** 찾을 수 있다
(`발효`→김치 · `여자 가수`→아이유).

🔴 **재료가 그대로면 임베딩을 부르지 않는다.** 개체는 배치가 다시 만들 때마다 바뀌는데
(087 재판정에서 925→1,065) 매번 전량 재임베딩하면 비용이 그만큼 든다. 코어
``upsert_entity_embedding`` 이 ``material_hash`` 로 판단하고, 이 배치는 그 결과를
``created``/``updated``/``reused`` 로 세어 보고한다(합격선 C6: 재실행 시 재사용 ≥95%).

🔴 **고아를 먼저 지운다.** ``entity_embedding`` 에는 FK 가 없다(참조 대상의 유니크가 부분
인덱스라 PostgreSQL 이 받지 않는다). 개체가 사라져도 벡터가 남으므로 배치가 시작할 때
정리한다 — 087 재판정에서 고아 113건이 났던 전례가 있다.

사용 예::

    # 무엇이 만들어질지만 본다(임베딩 호출 0 · 쓰기 0)
    python -m processing.app.run_entity_embedding --env dev --dry-run

    # 전량 채운다
    python -m processing.app.run_entity_embedding --env dev

    # 재료가 같아도 다시 만든다(모델 교체·품질 재확인 시)
    python -m processing.app.run_entity_embedding --env dev --force
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from src.config import search_constants
from src.mm_meta.entity_embedding import (
    build_search_material,
    count_entity_embeddings,
    fetch_embedding_targets,
    fetch_entity_vectors,
    purge_orphan_embeddings,
    upsert_entity_embedding,
)
from src.mm_meta.persist import (
    MM_META_VISIBLE_STATUSES,
    fetch_member_keywords_all,
    fetch_meta_members,
)
from src.mm_meta.rules import MIN_BUNDLE_SIZE
from src.search.entity_index import (
    bulk_index_entities,
    delete_entity_doc,
    ensure_entity_index,
    entity_to_doc,
)

# 한 번에 임베딩을 요청할 개수. 개체 재료는 짧아(중위 68자) 배치를 크게 잡아도 요청이 커지지
# 않지만, 실패 시 다시 보낼 양이 그만큼 늘어난다.
DEFAULT_BATCH_SIZE = 32

# 색인 문서에 실을 구성 자산 요약 수(092). 🔴 **상한이 필요한 이유**: 개체 하나에 자산이 1만 건이면
# 전량을 읽어도 색인에 쓸 것은 앞 몇 건뿐이다. ``ORDER BY asset_id`` 뒤에 상한이 걸리므로 같은
# 개체는 매번 같은 요약이 실린다(결정성).
# ⚠️ 요약은 **샘플**이라 자산이 많을수록 대표성이 떨어진다 — 그래서 자산 수와 무관하게 길이가
#   일정한 **키워드 집계**를 함께 싣는다(아래 상수).
DEFAULT_MEMBER_SUMMARIES = 3
# 색인 문서에 실을 구성 자산 키워드 수(빈도 내림차순). 규모가 커져도 문서 길이가 일정하다.
DEFAULT_MEMBER_KEYWORDS = 10


def _bump(counter: dict[str, int], key: str) -> None:
    """카운터를 하나 올린다.

    Args:
        counter: 셈 사전.
        key: 올릴 열쇠.
    """
    counter[key] = counter.get(key, 0) + 1


def run_entity_embedding(
    conn: Any,
    *,
    targets: Sequence[Mapping[str, Any]],
    embed_fn: Any,
    model_name: str,
    model_version: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    member_keywords: Mapping[tuple[str, str], Sequence[str]] | None = None,
) -> dict[str, Any]:
    """대상 개체의 임베딩을 채운다.

    Args:
        conn: DB 커넥션(트랜잭션은 호출한 쪽이 관리한다).
        targets: ``fetch_embedding_targets`` 결과.
        embed_fn: 재료 문자열 → 벡터. 주입받는다 — 테스트가 임베딩 서버 없이 돌게.
        model_name: 쓴 모델 이름.
        model_version: 모델 버전(선택).
        force: True 면 재료 해시가 같아도 다시 만든다. 🔴 재료 뒤에 표식을 붙여 해시를
            바꾸는 식의 우회를 쓰지 않는다 — 저장되는 재료가 실제와 달라지면 다음 배치가
            영원히 재생성을 반복한다. 대신 기존 행을 지우고 새로 넣는다.
        dry_run: True 면 임베딩·쓰기 없이 무엇이 대상인지만 센다.
        member_keywords: ``{(타입, 표기): [구성 자산 키워드]}``(092). 재료에 함께 실어 개체
            설명문에 없는 낱말로도 찾히게 한다 — 실측 다어절 재현율 73.3% → 86.7%.
            ``None`` 이면 090 재료 그대로다(되돌림).

    Returns:
        ``{total, created, updated, reused, skipped, failed, chars, elapsed_s, errors}``.
    """
    counter: dict[str, int] = {}
    errors: list[str] = []
    chars: list[int] = []
    started = time.time()

    for target in targets:
        try:
            material = build_search_material(
                name=target["name"],
                entity_type=target["entity_type"],
                description=target.get("description"),
                keywords=target.get("keywords") or (),
                member_keywords=(member_keywords or {}).get(
                    (str(target["entity_type"]), str(target["entity_uid"])), ()),
            )
        except Exception as exc:                       # noqa: BLE001 — 한 건 실패가 배치를 멈추지 않는다
            _bump(counter, "failed")
            errors.append(f"{target.get('entity_type')}/{target.get('entity_uid')}: 재료 조립 실패 {exc}")
            continue
        chars.append(len(material))

        if dry_run:
            _bump(counter, "skipped")
            continue
        try:
            if force:
                # 해시 비교를 건너뛰려면 기존 행을 비운다(재료를 조작하지 않는다).
                conn.execute(
                    "DELETE FROM entity_embedding WHERE entity_type=%s AND entity_uid=%s",
                    (target["entity_type"], target["entity_uid"]),
                )
            outcome = upsert_entity_embedding(
                conn,
                entity_type=target["entity_type"],
                entity_uid=target["entity_uid"],
                material=material,
                embed_fn=embed_fn,
                model_name=model_name,
                model_version=model_version,
            )
            _bump(counter, outcome)
        except Exception as exc:                       # noqa: BLE001
            _bump(counter, "failed")
            errors.append(f"{target.get('entity_type')}/{target.get('entity_uid')}: {exc}")

    return {
        "total": len(targets),
        "created": counter.get("created", 0),
        "updated": counter.get("updated", 0),
        "reused": counter.get("reused", 0),
        "skipped": counter.get("skipped", 0),
        "failed": counter.get("failed", 0),
        "chars": chars,
        "elapsed_s": round(time.time() - started, 2),
        "errors": errors[:20],
    }


def sync_entity_index(
    conn: Any,
    *,
    client: Any,
    index: str,
    targets: Sequence[Mapping[str, Any]],
    model_name: str,
    member_summaries: int = DEFAULT_MEMBER_SUMMARIES,
    member_keywords: int = DEFAULT_MEMBER_KEYWORDS,
) -> dict[str, Any]:
    """개체를 검색 엔진에 색인한다(092 · 임베딩 뒤에 이어서 돈다).

    왜 임베딩 배치가 함께 하나: 038 이 자산에서 "적재=색인" 을 정한 것과 같은 이유다. 배치를 둘로
    나누면 "임베딩은 됐는데 색인이 안 된" 상태가 생기고, 그때 검색은 조용히 옛 결과를 준다.

    벡터는 **방금 저장한 것을 읽어** 쓴다(``fetch_entity_vectors``) — 같은 재료로 두 번 임베딩하면
    비용도 두 배이고 두 값이 미세하게 달라질 여지도 생긴다.

    Args:
        conn: DB 커넥션(읽기).
        client: OpenSearch 클라이언트.
        index: 개체 인덱스 이름. 🔴 자산 인덱스와 달라야 한다(매핑 충돌 방지).
        targets: 색인할 개체들(``fetch_embedding_targets`` 결과).
        model_name: 이 모델로 만든 벡터만 색인한다(모델이 섞이면 유사도가 뜻을 잃는다).
        member_summaries: 문서에 실을 구성 자산 요약 수(개체당 조회 상한이기도 하다).
        member_keywords: 문서에 실을 구성 자산 키워드 수(빈도순).

    Returns:
        ``{indexed, skipped_no_vector, elapsed_s}``. 벡터가 없는 개체는 색인하지 않는다 —
        BM25 만으로도 검색되지만 kNN 이 빠져 융합 점수가 한쪽으로 치우친다.
    """
    started = time.time()
    vectors = fetch_entity_vectors(conn, model_name=model_name)
    keywords = fetch_member_keywords_all(conn, top_n=member_keywords)
    docs: list[dict[str, Any]] = []
    skipped = 0
    for target in targets:
        key = (str(target["entity_type"]), str(target["entity_uid"]))
        vector = vectors.get(key)
        if vector is None:
            skipped += 1
            continue
        summaries = [
            summary
            for _, summary in fetch_meta_members(
                conn, target["entity_type"], target["entity_uid"], limit=member_summaries)
        ]
        docs.append(entity_to_doc(target, vector=vector, member_summaries=summaries,
                                  member_keywords=keywords.get(key, [])))
    ensure_entity_index(client, index)
    indexed = bulk_index_entities(client, index, docs)
    return {"indexed": indexed, "skipped_no_vector": skipped,
            "elapsed_s": round(time.time() - started, 2)}


def format_report(report: Mapping[str, Any]) -> str:
    """배치 결과를 사람이 읽을 여러 줄로 만든다.

    Args:
        report: ``run_entity_embedding`` 결과.

    Returns:
        출력할 문자열.

    🔴 **재사용률을 첫 줄에 놓는다** — 이것이 낮으면 ``material_hash`` 가 안 도는 것이고,
    그러면 배치마다 임베딩 비용이 그대로 나간다(합격선 C6).
    """
    done = report["created"] + report["updated"] + report["reused"]
    lines = ["═" * 58]
    if done:
        rate = report["reused"] / done * 100
        lines.append(f"  재사용      {report['reused']}/{done}건 ({rate:.0f}%)"
                     f"{'  ✅' if rate >= 95 or report['created'] else ''}")
    lines += [
        f"  대상        {report['total']}개체",
        f"  신규        {report['created']}건",
        f"  갱신        {report['updated']}건",
    ]
    if report["skipped"]:
        lines.append(f"  건너뜀      {report['skipped']}건 (dry-run)")
    if report["failed"]:
        lines.append(f"  🔴 실패     {report['failed']}건")
    chars = sorted(report["chars"])
    if chars:
        lines.append(f"  재료 길이   중위 {chars[len(chars) // 2]}자 · "
                     f"최소 {chars[0]} · 최대 {chars[-1]}")
    lines.append(f"  소요        {report['elapsed_s']}초")
    if report["errors"]:
        lines.append("  오류:")
        lines += [f"    {e[:100]}" for e in report["errors"][:5]]
    lines.append("═" * 58)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """CLI 파서를 만든다.

    Returns:
        인자 파서.
    """
    p = argparse.ArgumentParser(description="개체 임베딩 배치 (개체마다 검색용 벡터를 채운다)")
    p.add_argument("--env", choices=("dev", "prod"), default="dev")
    p.add_argument("--dry-run", action="store_true",
                   help="임베딩·쓰기 0 — 무엇이 대상인지와 재료 길이만 본다")
    p.add_argument("--no-index", action="store_true",
                   help="검색 엔진 색인을 건너뛴다(임베딩만). 기본은 색인까지 한다 — "
                        "038 '적재=색인' 관례와 같은 이유로, 나뉘면 '임베딩은 됐는데 "
                        "색인이 안 된' 상태가 생긴다")
    p.add_argument("--index", default=None,
                   help=f"개체 인덱스 이름(기본 {search_constants.ENTITY_INDEX_DEFAULT}). "
                        "🔴 자산 인덱스 이름을 주면 자산 검색이 깨진다")
    p.add_argument("--force", action="store_true",
                   help="재료가 같아도 다시 만든다(모델 교체·품질 재확인 시)")
    p.add_argument("--min-members", type=int, default=MIN_BUNDLE_SIZE,
                   help=f"최소 구성 자산 수(기본 {MIN_BUNDLE_SIZE} = 화면 노출 임계)")
    p.add_argument("--limit", type=int, default=None, help="이번 실행에서 처리할 개체 수 상한")
    p.add_argument("--no-purge", action="store_true",
                   help="시작 시 고아 정리를 건너뛴다(진단용 · 기본은 정리한다)")
    return p


def purge_orphan_index_docs(client: Any, index: str, conn: Any) -> int:
    """DB(``node``)에 없는 개체의 **색인 문서**를 지운다 — PG 고아 정리(``purge_orphan_embeddings``)의 색인 판.

    🔴 왜 따로 필요한가(2026-09-10 실측). 데이터 테이블을 비운 뒤에도 개체 색인에는 옛 개체 82건이 남아, 개체 화면
    검색이 「DB 에 없는 개체」를 물어 빈 결과를 냈다. PG 쪽 고아 정리만 있고 색인 쪽이 없었던 것이다.
    색인 문서 수는 개체 수(수백~수천)라 한 번에 읽어 대조한다(1만 개를 넘으면 스크롤로 바꿔야 한다).

    Args:
        client: OpenSearch 클라이언트.
        index: 개체 인덱스 이름.
        conn: DB 커넥션(읽기).

    Returns:
        지운 문서 수. 인덱스가 없으면 0.
    """
    if not client.indices.exists(index=index):
        return 0
    rows = conn.execute(
        "SELECT entity_type, entity_uid FROM node WHERE node_kind='entity'"
    ).fetchall()
    keep = {(str(r[0]), str(r[1])) for r in rows}
    res = client.search(index=index, body={"size": 10000, "_source": ["entity_type", "entity_uid"],
                                           "query": {"match_all": {}}})
    removed = 0
    for hit in res.get("hits", {}).get("hits", []):
        src = hit.get("_source") or {}
        key = (str(src.get("entity_type", "")), str(src.get("entity_uid", "")))
        if key not in keep and delete_entity_doc(client, index, key[0], key[1]):
            removed += 1
    return removed


def run_embedding_pass(db: Any, cfg: Any, *, min_members: int = MIN_BUNDLE_SIZE, limit: int | None = None,
                       force: bool = False, dry_run: bool = False, purge: bool = True,
                       index_name: str | None = search_constants.ENTITY_INDEX_DEFAULT,
                       embed_fn: Callable[[str], list[float]] | None = None,
                       client: Any = None) -> dict[str, Any]:
    """개체 임베딩 한 바퀴 — 고아 정리 → 대상 읽기 → 임베딩 저장 → 검색 엔진 색인(+색인 고아 정리).

    CLI(``main``)와 mm_meta DAG 의 ``embed_entities`` 태스크가 **같은 함수**를 부른다(2026-09-10 · 090 의
    「CLI 전용」을 「DAG 뒤에 자동」으로 바꾼 것). 자산의 「적재=색인」(038)과 같은 이유 — 묶음만 갱신되고
    개체 색인이 따라오지 않으면 개체 화면 검색이 조용히 옛 결과를 준다.

    Args:
        db: ``PostgresUtil``(트랜잭션 seam).
        cfg: ``init_settings`` 결과(임베딩 API 설정을 읽는다).
        min_members: 최소 구성 자산 수(화면 노출 임계).
        limit: 이번에 처리할 개체 상한. ``None`` 이면 전량.
        force: 재료가 같아도 다시 만든다.
        dry_run: 임베딩·쓰기·색인 0 — 대상만 본다.
        purge: 시작 시 PG 고아 정리(색인 고아 정리는 색인 단계에서 함께).
        index_name: 개체 인덱스. ``None`` 이면 색인을 건너뛴다.
        embed_fn: 재료 → 벡터. ``None`` 이면 설정의 임베딩 API 를 쓴다(테스트가 갈아끼운다).
        client: OpenSearch 클라이언트. ``None`` 이면 설정으로 만든다(색인 단계에서만 필요).

    Returns:
        리포트 dict — ``targets``·``stored_total``·``purged``·``index`` (``indexed``·``index_purged``·``error``) 등.
    """
    model_name = cfg.embed.api_model

    def _embed_default(text: str) -> list[float]:
        """재료 하나를 설정의 임베딩 API 로 임베딩한다(코어 seam 경유).

        Args:
            text: 검색 재료.

        Returns:
            모델 원본 차원의 벡터.
        """
        from src.embedders.text_embedder_api import embed_texts_api
        return embed_texts_api([text], base_url=cfg.embed.api_base_url, model=model_name,
                               api_key=(cfg.embed.api_key or None), timeout_s=cfg.embed.api_timeout_s,
                               batch_size=1)[0]

    embed = embed_fn or _embed_default

    def _load(conn: Any) -> list[dict[str, Any]]:
        """대상 개체를 읽기 트랜잭션 한 번에 읽는다.

        Args:
            conn: DB 커넥션.

        Returns:
            대상 개체 목록.
        """
        return fetch_embedding_targets(conn, min_members=min_members,
                                       statuses=MM_META_VISIBLE_STATUSES, limit=limit)

    targets = db.execute_in_transaction(_load, idempotent=True)

    def _work(conn: Any) -> dict[str, Any]:
        """고아를 정리하고 임베딩을 채운다.

        Args:
            conn: DB 커넥션.

        Returns:
            배치 리포트(고아 정리 수 포함).
        """
        purged = purge_orphan_embeddings(conn) if (purge and not dry_run) else 0
        member_kw = fetch_member_keywords_all(conn, top_n=DEFAULT_MEMBER_KEYWORDS)
        rep = run_entity_embedding(conn, targets=targets, embed_fn=embed, model_name=model_name,
                                   force=force, dry_run=dry_run, member_keywords=member_kw)
        rep["purged"] = purged
        rep["stored_total"] = count_entity_embeddings(conn)
        return rep

    report = db.execute_in_transaction(_work, idempotent=False)
    report["targets"] = len(targets)
    report["index"] = None
    if dry_run or not index_name:
        return report
    # 색인 실패는 임베딩 결과를 무르지 않는다(임베딩은 커밋됨 · 색인은 멱등이라 다시 돌리면 된다).
    try:
        if client is None:
            from src.search.opensearch_sync import get_client
            client = get_client()

        def _index(conn: Any) -> dict[str, Any]:
            """색인 재료를 읽어 검색 엔진에 싣고, DB 에 없는 개체 문서를 지운다.

            Args:
                conn: DB 커넥션(읽기).

            Returns:
                ``sync_entity_index`` 리포트 + ``index_purged``.
            """
            idx = sync_entity_index(conn, client=client, index=index_name, targets=targets,
                                    model_name=model_name)
            idx["index_purged"] = purge_orphan_index_docs(client, index_name, conn) if purge else 0
            return idx

        report["index"] = db.execute_in_transaction(_index, idempotent=True)
    except Exception as exc:                   # noqa: BLE001 — 색인 실패는 배치를 무르지 않는다
        report["index"] = {"error": str(exc)[:200]}
    return report


def main(argv: list[str] | None = None) -> int:
    """대상을 읽어 임베딩을 채우고 리포트를 출력한다.

    Args:
        argv: 명령행 인자. ``None`` 이면 실제 명령행.

    Returns:
        종료 코드 — 실패가 하나라도 있으면 1(배치 모니터가 실패를 놓치지 않게).
    """
    args = _build_parser().parse_args(argv)
    from pathlib import Path

    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    env_path = Path(__file__).resolve().parents[2] / f".env.{args.env}"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)
    cfg = init_settings(args.env)
    db = PostgresUtil()
    report = run_embedding_pass(
        db, cfg, min_members=args.min_members, limit=args.limit, force=args.force, dry_run=args.dry_run,
        purge=not args.no_purge,
        index_name=None if args.no_index else (args.index or search_constants.ENTITY_INDEX_DEFAULT),
    )
    print(f"대상 {report['targets']}개체 (구성 자산 {args.min_members}건 이상)"
          f"{' · dry-run' if args.dry_run else ''}{' · force' if args.force else ''}")
    if report.get("purged"):
        print(f"고아 정리   {report['purged']}건")
    print(format_report(report))
    print(f"저장 총계    {report['stored_total']}건")
    idx = report.get("index")
    if idx and "error" in idx:
        print(f"⚠️ 색인 실패(임베딩은 저장됨 · 다시 돌리면 된다): {idx['error']}", file=sys.stderr)
    elif idx:
        print(f"색인        {idx['indexed']}건 → {args.index or search_constants.ENTITY_INDEX_DEFAULT} "
              f"({idx['elapsed_s']}초)"
              + (f" · 벡터 없어 건너뜀 {idx['skipped_no_vector']}건" if idx.get("skipped_no_vector") else "")
              + (f" · 색인 고아 정리 {idx['index_purged']}건" if idx.get("index_purged") else ""))
    db.close()
    return 1 if report["failed"] else 0

if __name__ == "__main__":
    sys.exit(main())
