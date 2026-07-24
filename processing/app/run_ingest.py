"""F-1.x 수집→라우팅→(분류)→추출→등록 오케스트레이터 (모델 A) — CLI 진입점.

로컬 파일 리스트를 순회하며 각 파일을:
    1. route_file 로 modality 판정(없는 파일은 skip).
    2. create_asset 로 ``received`` 조기 INSERT(별도 트랜잭션, 커밋).
    3. set_status 로 routing→classifying→extracting (단계별 짧은 트랜잭션).
    4. dispatch_extract 로 추출(트랜잭션 밖에서 실행 — LLM/IO).
    5. finalize_asset 로 메타·임베딩 적재 + ``registered`` (한 트랜잭션).
    6. 어디서든 예외 → **fresh 트랜잭션**으로 mark_failed (이전 트랜잭션이 abort 돼도 안전). 다음 파일 계속.

069 US-E FR-E3: 재사용 스텝(``collect_file``·``process_asset``·``_make_opensearch_indexer``·
``CollectResult``·타입)은 **ingest 계층 ``processing/ingest/pipeline_steps.py`` 로 이관**됐다(배치·DAG 가
app 진입점을 거꾸로 import 하던 레이어링 해소·레포 분리 대비). 이 모듈은 그것들을 재import 해 CLI
end-to-end(``run_ingest``)로 엮고, 하위호환으로 재export 한다(테스트의 ``run_ingest.collect_file`` patch·
dag 참조 유지). ``run_ingest``/``main``/``_configure_logging`` 만 여기 CLI 조립으로 남는다.

``classify_fn``/``extract_fn`` 은 **테스트·e2e 전용 override** 주입점이다(운영 호출부 없음).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from src.config.settings import get_current_settings
from src.database.lineage_persist import record_lineage
from src.database.postgres_util import PostgresUtil

# FR-E3: 파이프라인 스텝은 pipeline_steps(ingest 계층)로 이관됨 — 여기서 재import 해 CLI 오케스트레이션에
# 쓰고, 하위호환으로 재export 한다(run_ingest.<name> import·patch 경로 보존·정본은 pipeline_steps).
from processing.ingest.pipeline_steps import (
    REASON_DUPLICATE,  # noqa: F401 — 하위호환 재export
    REASON_MISSING,  # noqa: F401 — 하위호환 재export(router 정의를 pipeline_steps 경유로 재노출)
    ClassifyFn,
    CollectResult,  # noqa: F401 — 하위호환 재export
    ExtractFn,
    OsIndexFn,  # noqa: F401 — 하위호환 재export
    _make_opensearch_indexer,
    collect_file,
    process_asset,
)
from processing.ingest.status import InvalidTransitionError, mark_failed
from processing.pipeline import (
    builtins as _builtins,  # noqa: F401 — DEFAULT_REGISTRY 등록 부수효과(import 계약)
)
from processing.pipeline.registry import DEFAULT_REGISTRY

_LOG = logging.getLogger("meta_extract.run_ingest")


def _configure_logging() -> None:
    if _LOG.handlers:
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    _LOG.addHandler(h)
    _LOG.setLevel(logging.INFO)
    _LOG.propagate = False


def run_ingest(
    files: list[str],
    *,
    db: PostgresUtil,
    extract_fn: ExtractFn | None = None,   # 테스트·e2e 전용 override(미주입=팩 기본 extract/embed)
    classify_fn: ClassifyFn | None = None,  # 테스트·e2e 전용 override(미주입=cascade_v1)
    registry=DEFAULT_REGISTRY,
    settings: Any = None,
) -> dict[str, list[Any]]:
    """파일 리스트를 적재. 반환: {'registered': [asset_id], 'failed': [(asset_id,err)], 'skipped': [(path,reason)]}.

    파일마다 ``collect_file``(→received) → ``process_asset``(→registered/deferred)을 이어 호출하는
    standalone end-to-end 오케스트레이터다(헌법 8조 — 분할 전 동작과 자산/상태/lineage/dedup/os_index
    바이트 동일). 파일 단위 격리: 한 파일의 어떤 실패도 배치를 멈추지 않고 즉시 ``failed`` 로 마킹한다.
    """
    _configure_logging()
    cfg = settings or get_current_settings()
    result: dict[str, list[Any]] = {"registered": [], "failed": [], "skipped": [], "deferred": []}
    # OpenSearch 증분 색인기 — 배치당 1회 생성(opt-in off 면 no-op·미import). 클라이언트는 첫 색인에서
    # 만들어 배치 전체 재사용(자산마다 새 연결 X). process_asset 의 finalize 직후 os_index(asset_id) 로 호출.
    os_index = _make_opensearch_indexer(db=db, settings=cfg)

    for path in files:
        # 파일 단위 격리: 한 파일의 어떤 실패도 배치를 멈추지 않는다.
        asset_id: uuid.UUID | None = None
        try:
            # 1) 수집 — received 자산화(또는 누락/중복 스킵). dedup·create·received lineage 가 여기서 끝난다.
            with db.transaction() as conn:
                collected = collect_file(conn, path)
            if collected.skip_reason is not None:
                result["skipped"].append((path, collected.skip_reason))
                continue
            asset_id = collected.asset_id

            # 2) 처리 — received→registered(또는 deferred). 추출/적재/인라인 색인 + 단계별 lineage.
            outcome = process_asset(
                asset_id, db=db, fs_path=path,
                modality=collected.route.modality, domain=collected.route.domain,
                extract_fn=extract_fn, classify_fn=classify_fn,
                registry=registry, settings=cfg, os_index=os_index,
            )
            if outcome == "deferred":
                result["deferred"].append(asset_id)
            else:
                result["registered"].append(asset_id)
        except Exception as exc:  # noqa: BLE001 — route/create/추출/적재 모든 실패 흡수
            reason = f"{type(exc).__name__}: {exc}"
            _LOG.warning("failed: asset_id=%s %s (%s)", asset_id, path, reason)
            if asset_id is None:
                # asset 생성 전(route_file/create_asset) 실패 → 경로 기준 기록
                result["failed"].append((None, f"{path}: {reason}"))
            else:
                # fresh 트랜잭션으로 실패 기록(이전 트랜잭션 abort 가능성 격리)
                with db.transaction() as conn:
                    try:
                        mark_failed(conn, asset_id, reason)
                        record_lineage(conn, asset_id, activity="ingest.failed.v1", agent="run_ingest",
                                       payload={"reason": reason})
                    except InvalidTransitionError:
                        # 이미 종료 상태면 무시. 009: ConcurrentTransitionError(동시 전이 충돌 —
                        # 다른 워커가 먼저 종료시킨 경우)도 이 계열의 하위라 같은 경로로 흡수돼
                        # 배치가 멈추지 않는다(호출부 시그니처 무변경).
                        pass
                result["failed"].append((asset_id, reason))

    _LOG.info(
        "ingest done: registered=%s failed=%s skipped=%s deferred=%s",
        len(result["registered"]),
        len(result["failed"]),
        len(result["skipped"]),
        len(result["deferred"]),
    )
    return result


# ── 초기 설정(부트스트랩) 절차 ────────────────────────────────────────────────
# [import 시점·자동·호출순서 무관] 이 모듈(및 pipeline_steps)을 import 하는 것만으로 완료되는 등록(부수효과):
#   · 전략 레지스트리 DEFAULT_REGISTRY ← `from processing.pipeline import builtins`(여기 + pipeline_steps)가
#     register_defaults 를 실행 → classify/cascade_v1·extract·embed/by_modality·persist + cross-asset 슬롯.
#   · 도메인 프로파일 DOMAIN_PROFILES ← builtins→cascade→`processing.classify.domains`→medical 체인 →
#     {"medical"}(DICOM/HL7/FHIR 시그니처+어휘). general 은 미등록 — cascade 엔진 내장 폴백.
# [런타임·main() 안·순서 중요] 부트스트랩은 src.config.bootstrap.bootstrap_env(env) 로 일원화(FR-E2):
#   .env.{env} 로드(override=False) → init_settings(env) 검증. 이후 PostgresUtil() + `with db:` (PG17 검증).
def main() -> int:
    """CLI: 로컬 파일 수집 후 asset_* 적재. 예) python -m processing.app.run_ingest --env dev --input-dir DIR"""
    import argparse
    import json

    from src.config.bootstrap import bootstrap_env
    from processing.ingest.collector import collect_files

    parser = argparse.ArgumentParser(description="수집→라우팅→추출→등록 (asset_* 적재)")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--input-dir", dest="input_dir", default=None)
    parser.add_argument("--file-list", dest="file_list", default=None)
    parser.add_argument("paths", nargs="*", metavar="FILE")
    args = parser.parse_args()

    bootstrap_env(args.env)

    files = collect_files(
        input_dir=args.input_dir,
        file_list=args.file_list,
        paths=args.paths or None,
    )

    db = PostgresUtil()
    with db:
        result = run_ingest(files, db=db)
    print(json.dumps({k: len(v) for k, v in result.items()}, ensure_ascii=False))
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
