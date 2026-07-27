"""F-3.5 관계 제안 배치 CLI — registered 자산에 대해 ``graph_edge`` 엣지를 생성한다(관계 카탈로그는 relation_kind).

수집/추출(run_ingest)과 분리된 후속 배치다. 임베딩이 있는 ``status='registered'`` 자산만 대상으로
``propose_relations_for_asset`` 을 호출한다(온프레미스 LLM). 자산 단위 격리(한 건 실패가 배치를 멈추지 않음).

예)
    python -m processing.app.run_relations --env dev --all
    python -m processing.app.run_relations --env dev <asset_uuid> <asset_uuid> ...
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from psycopg import Connection

from src.database.postgres_util import PostgresUtil

# builtins import 부수효과로 DEFAULT_REGISTRY 에 cross_asset 전략이 등록된다(register_defaults).
# 슬롯 resolve(_resolve_cross_asset_slots)가 빈 레지스트리를 만나지 않도록 진입부에서 강제 로드.
# run_ingest 와 동일 관용(별칭 _builtins) — 부수효과 import 라 직접 참조하지 않는다.
from processing.pipeline import builtins as _builtins  # noqa: F401 — DEFAULT_REGISTRY 등록 부수효과
from processing.pipeline.cross_runner import run_cross_asset
from processing.pipeline.packs import GENERAL_PACK, DomainPack, for_domain
from processing.pipeline.registry import DEFAULT_REGISTRY, StrategyRegistry
from src.relations.asset_candidates import EmbeddingKindFilter
from src.relations.asset_entry import propose_relations_for_asset
from src.relations.resolution_persist import (
    decide_resolution_status,
    fetch_unresolved_asset_ids,
    upsert_resolution,
)

_LOG = logging.getLogger("meta_extract.run_relations")

# cross_asset 슬롯 중 레지스트리에서 Callable 로 resolve 되는 슬롯(008, 일반 경로 가드용).
# 'decide'(confidence)는 propose_relations_for_asset 내부 auto_approve 임계로 처리되어
# 별도 Callable 이 등록돼 있지 않으므로 resolve 대상에서 제외한다(packs.py·builtins.py 주석 참조).
# **무변경**: 일반 팩 가드(_resolve_cross_asset_slots 기본 인자)에서만 쓴다.
_RESOLVED_CROSS_SLOTS: tuple[str, ...] = ("candidates", "score", "persist_edges")

# 제네릭 cross_asset 러너(016)가 실행하는 contracts.py 계약 4슬롯 — 'decide' 를 포함한다.
# 비일반 팩(예 샘플)은 decide 슬롯에 별도 Callable(sample_decide)을 등록하므로 4슬롯 전부를
# resolve 해 run_cross_asset 에 넘긴다. 일반 경로의 _RESOLVED_CROSS_SLOTS(3슬롯)와는 별개다.
_GENERIC_CROSS_SLOTS: tuple[str, ...] = ("candidates", "score", "decide", "persist_edges")

# 큐 last_reason 비식별 표식(009, 헌법 10조 — PHI/풀경로 금지).
# 고립(엣지0)은 표식 1개, 예외는 예외 **타입명**만 기록한다(메시지·경로 미포함).
_REASON_ISOLATED = "isolated:no_edges"


def _fetch_domain_label(db: PostgresUtil, asset_id: str) -> str:
    """자산의 domain_label(분류 결과). 없으면 'general'.

    NULL 이나 자산 미존재는 모두 'general' 로 폴백한다. for_domain 의 보수적 폴백 정책과 일치.
    """

    def _run(conn: Connection[Any]) -> str:
        """트랜잭션 안에서 도메인 라벨 한 건을 읽는다."""
        with conn.cursor() as cur:
            cur.execute("SELECT domain_label FROM asset WHERE asset_id = %s LIMIT 1", (asset_id,))
            row = cur.fetchone()
        return str(row[0]) if row and row[0] else "general"

    return db.execute_in_transaction(_run, idempotent=True)


def _fetch_registered_asset_ids(db: PostgresUtil) -> list[str]:
    """임베딩을 가진 ``registered`` 자산 id 전부(생성순).

    **필터 조건**: status='registered' + asset_embedding 존재.
    임베딩이 없으면 candidates 단계에서 후보 자체가 나오지 않아 관계 제안이 무의미하므로 제외한다.
    deferred/failed 상태 자산은 임베딩 미적재이므로 이 조건으로 자연히 걸러진다.
    """

    def _run(conn: Connection[Any]) -> list[str]:
        """트랜잭션 안에서 대상 자산 id 를 모아 온다."""
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.asset_id
                FROM asset a
                WHERE a.status = 'registered'
                  AND EXISTS (SELECT 1 FROM asset_embedding e WHERE e.asset_id = a.asset_id)
                ORDER BY a.created_at
                """
            )
            return [str(r[0]) for r in cur.fetchall()]

    return db.execute_in_transaction(_run, idempotent=True)


def _resolve_cross_asset_slots(
    pack: DomainPack,
    *,
    registry: StrategyRegistry = DEFAULT_REGISTRY,
    slots: tuple[str, ...] = _RESOLVED_CROSS_SLOTS,
) -> dict[str, Callable[..., Any]]:
    """팩의 cross_asset 슬롯명을 레지스트리에서 Callable 로 resolve(미배선 가드, FR-002).

    **목적(헌법 4조)**: 도메인 차이를 코드 if/else 가 아니라 "팩이 고른 전략 이름"으로 표현한다.
    여기서 슬롯명을 registry.resolve 로 검증함으로써, 의료 ER(단계 D)이 전용 cross_asset
    전략(예: blocking_5keys)을 **레지스트리에 등록만 하면** core 파이프라인 수정 없이 갈리는
    자리를 만든다. 등록 전 상태(미배선)면 KeyError → NotImplementedError 로 승격해 자산 단위
    격리(run_relations 의 except)로 흘려보낸다 — 배치는 중단되지 않는다.

    **slots 인자(016)**: 어떤 슬롯 집합을 resolve 할지 선택한다. 기본은 일반 경로 가드용
    _RESOLVED_CROSS_SLOTS(3슬롯, 'decide' 제외 — 일반은 propose 내부 auto_approve 처리)이며,
    제네릭 러너 경로(비일반 팩)는 _GENERIC_CROSS_SLOTS(4슬롯, 'decide' 포함)를 넘긴다.
    기본값이 _RESOLVED_CROSS_SLOTS 라 기존 호출·테스트는 무영향이다.

    Returns:
        슬롯 이름 → resolve 된 Callable. (검증 통과 시에만 반환)
    Raises:
        NotImplementedError: 슬롯이 가리키는 전략이 레지스트리에 미등록(단계 D 전 의료 등).
    """
    resolved: dict[str, Callable[..., Any]] = {}
    for slot in slots:
        name = pack.cross_asset.get(slot)
        if name is None:
            raise NotImplementedError(
                f"cross_asset 슬롯 '{slot}' 미정의(도메인 {pack.name})"
            )
        try:
            resolved[slot] = registry.resolve(slot, name)
        except KeyError as e:
            # 미등록 전략 — 의료 cross_asset 전략이 단계 D 전이라 배선되지 않은 경우 등.
            raise NotImplementedError(
                f"cross_asset 전략 미구현(도메인 {pack.name}): {slot}={name}"
            ) from e
    return resolved


def _fetch_unresolved_asset_ids(db: PostgresUtil) -> list[str]:
    """``--retry`` 대상 자산 id — pending(미해소) + 미시도만, 결정적 정렬.

    베이스(status='registered' + 임베딩 존재)는 ``_fetch_registered_asset_ids`` 와 공유하고,
    ``relation_resolution`` LEFT JOIN 으로 (rr.status='pending' OR rr.asset_id IS NULL) 만 고른다.
    resolved/failed(DLQ) 자산은 제외된다. 정렬은 attempts ASC, created_at ASC(결정적, 헌법 3조).
    실제 쿼리는 resolution_persist.fetch_unresolved_asset_ids(conn-우선)에 위임한다.
    """

    def _run(conn: Connection[Any]) -> list[str]:
        """트랜잭션 안에서 재시도 대상 조회를 위임한다."""
        return fetch_unresolved_asset_ids(conn)

    return db.execute_in_transaction(_run, idempotent=True)


def _record_resolution(
    db: PostgresUtil, aid: str, edges_upserted: int, *, error: Exception | None, max_attempts: int
) -> None:
    """한 자산 처리 결과를 큐에 반영 — **별도 fresh 트랜잭션**으로 격리(run_ingest 패턴 차용).

    핵심(SC-008): 한 자산의 큐 upsert 실패가 다른 자산 처리나 이미 적재된 관계를 롤백하면 안 된다.
    그래서 큐 갱신은 propose_relations_for_asset 트랜잭션 **밖**, 자산별 독립 fresh 트랜잭션에서 수행하고,
    여기서 또 예외가 나면 로그만 남기고 흡수한다(배치·다른 자산에 전파 금지).

    attempts 는 별도 조회(``_fetch_attempts``) 후 이 fresh 트랜잭션에서 upsert 한다 — read-then-write 가
    서로 다른 트랜잭션이라, 배치가 자산을 **순차 처리하는 단일 워커** 전제에서만 안전하다. 병렬화 시
    두 워커가 같은 자산의 attempts 를 겹쳐 읽어 카운트가 어긋날 수 있다(TOCTOU — 병렬화 시 재설계 필요).

    last_reason 은 비식별만(헌법 10조): 고립은 _REASON_ISOLATED 표식, 예외는 예외 **타입명**만.
    예외 메시지·파일 경로는 PHI/풀경로 누출 위험이 있어 큐에 담지 않는다.
    """
    cur_attempts = _fetch_attempts(db, aid)
    status, next_attempts = decide_resolution_status(
        edges_upserted, cur_attempts, error=error, max_attempts=max_attempts
    )
    # 035 #2: last_reason 3분기 — 예외=타입명(비식별), 고립(엣지0)=표식, 성공(엣지≥1)=None(reason 불요).
    if error is not None:
        reason: str | None = type(error).__name__
    elif edges_upserted == 0:
        reason = _REASON_ISOLATED
    else:
        reason = None
    try:
        def _run(conn: Connection[Any]) -> None:
            """큐 한 행만 갱신한다 — 관계 저장과 **다른 트랜잭션**이라 서로 롤백시키지 않는다."""
            upsert_resolution(conn, aid, status=status, attempts=next_attempts, reason=reason)

        db.execute_in_transaction(_run, idempotent=False)
    except Exception as exc:  # noqa: BLE001 — 큐 갱신 실패 격리(다른 자산 미롤백)
        _LOG.warning("resolution queue upsert failed %s: %s", aid, exc)


def _fetch_attempts(db: PostgresUtil, aid: str) -> int:
    """자산의 현재 큐 attempts(없으면 0). decide_resolution_status 입력으로 쓴다."""

    def _run(conn: Connection[Any]) -> int:
        """트랜잭션 안에서 누적 시도 횟수를 읽는다."""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT attempts FROM relation_resolution WHERE asset_id = %s LIMIT 1", (aid,)
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    return db.execute_in_transaction(_run, idempotent=True)


def run_relations(
    asset_ids: list[str],
    *,
    db: PostgresUtil,
    top_k: int | None = None,
    # 036: 관계 후보 임베딩 채널 기본은 "st"(BGE 텍스트 임베더) — 4모달리티가 캡션으로 공유하는 단일
    # 텍스트 공간을 쓴다. 'both'(st+clip)는 텍스트·시각 코사인을 MAX 로 섞어 척도가 달라 emb_score 가 흐려진다.
    embedding_kind: EmbeddingKindFilter = "st",
    max_attempts: int | None = None,
    _domain_fn: Callable[[PostgresUtil, str], str] | None = None,
) -> dict[str, list[Any]]:
    """자산 리스트 순회하며 관계 제안. 자산 단위 예외 흡수 + 미해소 큐 갱신.

    반환값 구조: {"done": [(aid, edges_upserted, edges_kept), ...], "failed": [(aid, msg), ...]}.
    배치 전체 성공 여부는 ``failed`` 리스트가 비어있는지로 판별한다(main 의 종료코드 참조).

    **cross_asset 슬롯 resolve(008, FR-001~003)**: 자산 domain_label → for_domain 으로 팩을 고르고,
    팩의 cross_asset 슬롯을 _resolve_cross_asset_slots 로 레지스트리에서 resolve 한다(미배선 가드).
    일반 팩(슬롯이 GENERAL_PACK.cross_asset 과 동일)은 결과 동치(회귀 0, FR-001/SC-001)를
    구조적으로 보장하기 위해 기존 propose_relations_for_asset 로 위임한다 — 슬롯 전환의 목적은
    "의료 전용 전략이 끼워질 자리 만들기"이지 일반 동작을 바꾸는 것이 아니다.

    **비일반 팩 → 제네릭 러너(016)**: 일반 묶음이 아닌 팩(예 샘플)은 4슬롯(decide 포함)을
    _GENERIC_CROSS_SLOTS 로 resolve 해 cross_runner.run_cross_asset 으로 contracts.py 계약
    순서대로 실행한다(NotImplementedError 였던 자리 교체). 미등록 전략 팩은 여전히
    _resolve_cross_asset_slots 가 NotImplementedError 를 내고 바깥 except 가 failed 로 격리한다
    (Acceptance 2 유지). 헌법 4조: 도메인명 코드 분기 없이 for_domain + cross_asset 비교로만 갈린다.

    **_domain_fn seam**: domain_label 조회를 주입 가능하게 한 seam. 기본(None)은 모듈 수준
    _fetch_domain_label 를 **호출 시점에 이름으로** 조회한다 — 이래야 기존
    ``mock.patch.object(rr, "_fetch_domain_label", ...)`` 가 그대로 먹어 회귀 0 이 보장된다
    (정의 시점 바인딩이면 모듈 패치가 무시됨). 실 DB e2e 픽스처는 이 seam 으로 'sample' 라벨을
    직접 주입해 비일반 러너 라우팅을 검증한다(G4).

    **미해소 큐 갱신(009, SC-008)**: 각 자산 처리 직후 결과(edges_upserted/예외)로
    ``decide_resolution_status`` 를 호출해 relation_resolution 큐를 **별도 fresh 트랜잭션**으로
    갱신한다(자산별 격리). ``max_attempts`` 미지정 시 설정 ``relation_retry_max_attempts`` 를 쓴다.
    """
    # seam 기본값 해소: None 이면 모듈 수준 _fetch_domain_label 를 **호출 시점에** 잡는다.
    # (def 기본값으로 직접 바인딩하면 mock.patch.object(rr, "_fetch_domain_label", ...) 가 안 먹어
    #  기존 테스트가 깨진다 — 회귀 0 을 위해 런타임 이름 조회로 둔다.)
    domain_fn = _domain_fn if _domain_fn is not None else _fetch_domain_label
    if max_attempts is None:
        from src.config.settings import get_current_settings

        max_attempts = get_current_settings().relations.retry_max_attempts
    result: dict[str, list[Any]] = {"done": [], "failed": []}
    for aid in asset_ids:
        edges_u = 0
        err: Exception | None = None
        try:
            domain = domain_fn(db, aid)  # seam(기본 _fetch_domain_label) — G4 e2e 가 'sample' 주입
            pack = for_domain(domain)  # 미지정/review → GENERAL_PACK 폴백(FR-003)
            if pack.cross_asset == GENERAL_PACK.cross_asset:
                # 일반 cross_asset 묶음 — 기존 경로에 위임해 결과를 100% 동치로 유지(FR-001, 헌법 8조).
                # 슬롯별 전략을 잘게 호출하는 대신 검증된 propose_relations_for_asset 를 재사용한다.
                # 일반 가드용 슬롯 resolve(미배선 가드, FR-002): 일반 팩은 전부 등록돼 있어 통과한다.
                _resolve_cross_asset_slots(pack)
                cat_s, cat_k, edges_u, edges_k = propose_relations_for_asset(
                    db, aid, top_k=top_k, embedding_kind=embedding_kind
                )
            else:
                # 비일반 팩(예 샘플): contracts.py 계약 4슬롯(decide 포함)을 resolve 해 제네릭
                # 러너로 실행한다(016). 미등록 전략 팩이면 _resolve 가 NotImplementedError →
                # 바깥 except 가 failed 격리(Acceptance 2 유지). 헌법 4조: 러너는 도메인 무관.
                resolved = _resolve_cross_asset_slots(pack, slots=_GENERIC_CROSS_SLOTS)
                # 루프 변수(resolved/aid)는 기본인자로 바인딩 — 늦은 바인딩 footgun 차단(ruff B023).
                edges_u = db.execute_in_transaction(
                    lambda conn, _resolved=resolved, _aid=aid: run_cross_asset(_resolved, conn, _aid),
                    idempotent=False,
                )
                # 러너 경로는 카탈로그 카운트가 N/A — 로깅 정합용 0 으로 둔다(엣지 수만 의미).
                cat_s = cat_k = edges_k = 0
            result["done"].append((aid, edges_u, edges_k))
            _LOG.info(
                "relations %s: kinds=%s/%s edges=%s/%s", aid, cat_s, cat_k, edges_u, edges_k
            )
        except Exception as exc:  # noqa: BLE001 — 자산 단위 격리
            err = exc
            _LOG.warning("relations failed %s: %s", aid, exc)
            result["failed"].append((aid, str(exc)))
        # 자산 처리 결과를 큐에 반영(엣지0/예외/성공 모두). fresh 트랜잭션 격리.
        _record_resolution(db, aid, edges_u, error=err, max_attempts=max_attempts)
    # 065 FR-404: 관계 배치 후 OS topics 재색인 훅(056 FR-301) 제거 — 065 는 주제를 자산 자기주제
    # 정본(asset_topic)에서 부여하므로 관계 생성/변경이 자산 주제를 바꾸지 않는다(재색인 무의미).
    # 관계 파이프라인(후보→LLM→confidence→active·큐)은 불변, 주제 결정권만 회수됐다.
    return result


# ── 초기 설정(부트스트랩) 절차 ────────────────────────────────────────────────
# [import 시점] 이 진입점은 processing.pipeline.builtins 를 import 해 DEFAULT_REGISTRY 에 cross_asset
#   전략(candidates/score/persist_edges)을 등록한다(register_defaults 부수효과). run_relations 가
#   팩의 cross_asset 슬롯을 registry.resolve 로 검증(미배선 가드)하기 때문이다. 일반 팩은 검증 통과
#   후 propose_relations_for_asset 로 위임해 결과를 기존과 동치로 유지한다(FR-001).
# [런타임·main() 안·순서 중요]:
#   1) load_dotenv(.env.{env}, override=False)  2) init_settings(env): 필수 환경변수 검증+frozen 설정
#   3) PostgresUtil() + `with db:`: 연결 풀+PG17 검증.  온프레미스 LLM 클라이언트는 propose 내부 첫 호출 시 지연 생성.
def main() -> int:
    """자산들에 대해 관계 제안 배치를 돌린다(명령행 진입점).

    대상은 두 가지 — 기본은 **등록된 자산 전체**, ``--retry`` 를 주면 아직 관계를 못 만든
    자산만 골라 다시 시도한다.

    Returns:
        0=성공.
    """
    import argparse
    import json

    from src.config.bootstrap import bootstrap_env

    parser = argparse.ArgumentParser(description="관계 제안 배치 (graph_edge 적재)")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--all", action="store_true", help="registered 자산 전체 대상")
    parser.add_argument(
        "--retry",
        action="store_true",
        help="미해소(pending) + 미시도 자산만 골라 재시도(relation_resolution 큐 기반). --all 동시 지정 시 --retry 우선",
    )
    parser.add_argument("--top-k", dest="top_k", type=int, default=None)
    parser.add_argument(
        # 036: 기본 bge-only(st_bge). 필요 시 --embedding-kind both|clip 로 시각 채널 포함 가능.
        "--embedding-kind", dest="embedding_kind", choices=["st", "clip", "both"], default="st"
    )
    parser.add_argument("asset_ids", nargs="*", metavar="ASSET_ID")
    args = parser.parse_args()

    bootstrap_env(args.env)

    db = PostgresUtil()
    with db:
        # 선택 분기: --retry(미해소+미시도) > --all(registered 전체) > 명시 asset_ids.
        # --all·명시 asset_ids 경로는 무변경(회귀 0) — --retry 만 큐 기반 LEFT JOIN 선택을 쓴다.
        if args.retry:
            asset_ids = _fetch_unresolved_asset_ids(db)
        elif args.all:
            asset_ids = _fetch_registered_asset_ids(db)
        else:
            asset_ids = list(args.asset_ids)
        if not asset_ids:
            print(json.dumps({"done": 0, "failed": 0}, ensure_ascii=False))
            return 0
        result = run_relations(
            asset_ids, db=db, top_k=args.top_k, embedding_kind=args.embedding_kind
        )
    print(json.dumps({k: len(v) for k, v in result.items()}, ensure_ascii=False))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
