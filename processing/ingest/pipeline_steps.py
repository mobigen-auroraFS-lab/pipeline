"""수집·처리 파이프라인 재사용 스텝 (069 US-E FR-E3) — collect_file · process_asset · 인라인 색인기.

종전 ``src/app/run_ingest.py`` 에 있던 오케스트레이터-불가지 순수 스텝을 **ingest 계층으로 이관**한다.
이유(레이어링·레포 분리): 이 스텝들의 소비처는 CLI ``run_ingest``(app) 뿐 아니라 배치 ``batch_runner``
(processing.ingest)·Airflow DAG(dag_collect·dag_process)다. 종전엔 ``batch_runner``(하위 ingest 계층)가
``processing.app.run_ingest``(상위 app 진입점)를 **거꾸로 import** 해 레이어링이 뒤집혀 있었다(순환 지뢰).
스텝을 여기(ingest 계층)로 내리면 배치·DAG 가 같은 계층을 참조하고, 파이프라인 코드가 app 진입점 없이
자족한다(향후 처리/백엔드 레포 분리 대비).

``run_ingest.py`` 는 이 심볼들을 재import 해 CLI 오케스트레이션에 쓰고 하위호환으로 재export 한다
(테스트의 ``run_ingest.collect_file`` patch 는 run_ingest 가 CLI 에서 그 이름을 호출하는 정본 위치라 유효).
동작은 분할 전과 **바이트 동일**(상태전이·lineage·dedup·os_index 시퀀스 무변경, 헌법 8조).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from psycopg import Connection

from processing.classify.aboutness import extract_and_persist_about
from processing.classify.asset_topic import build_self_text, classify_asset_topic
from processing.classify.types import ClassificationResult
from src.database.lineage_persist import record_lineage
from src.database.postgres_util import PostgresUtil
from processing.dispatch.types import AssetRecord, ExtractContext
from src.file.file_type_defs import modality_of  # 저장·감사 canonical 매핑(053)
from src.file.hashing import file_hash_and_size
from processing.ingest.asset_persist import create_asset, finalize_asset, find_registered_asset_by_hash
from processing.ingest.classification_persist import record_classification
from processing.ingest.router import REASON_MISSING, RouteResult, route_file
from processing.ingest.status import AssetStatus, set_status
from processing.pipeline import builtins as _builtins  # noqa: F401 — DEFAULT_REGISTRY 등록 부수효과
from processing.pipeline.packs import for_domain
from processing.pipeline.policy import validate as policy_validate
from processing.pipeline.registry import DEFAULT_REGISTRY
from src.registry.ext_meta_field_registry import validate_ext_meta

REASON_DUPLICATE = "duplicate"

# CLI(run_ingest)의 _configure_logging 이 설정하는 것과 **같은 로거**("meta_extract.run_ingest")를 쓴다
# — getLogger 는 같은 이름에 같은 객체를 돌려주므로, 스텝을 이 모듈로 옮겨도 로그 대상·핸들러가 불변이다.
_LOG = logging.getLogger("meta_extract.run_ingest")

ExtractFn = Callable[[ExtractContext], AssetRecord]
ClassifyFn = Callable[[str, str], ClassificationResult]  # (file_path, modality) -> 분류 결과
OsIndexFn = Callable[[Any], None]  # finalize 직후 인라인 색인기 — asset_id 만 받는 best-effort 콜러블


@dataclass(frozen=True)
class CollectResult:
    """collect_file 결과(수집 단계 산출).

    - 성공: ``asset_id`` 에 received 자산 UUID, ``skip_reason`` 은 None, ``route`` 로 탐지된 모달리티/도메인.
    - 스킵: ``asset_id`` 는 None, ``skip_reason`` 에 run_ingest 결과 ``skipped`` 에 넣을 사유 문자열
      (파일 누락=``REASON_MISSING`` · 동일 해시 중복=``f"{REASON_DUPLICATE}:{기존id}"``). 자산 미생성.

    ``route`` 는 스킵/성공 모두 채워 process_asset 이 modality·domain 으로 ctx 를 만들 수 있게 한다.
    """

    asset_id: uuid.UUID | None
    route: RouteResult
    skip_reason: str | None


def _make_opensearch_indexer(*, db: PostgresUtil, settings: Any) -> Callable[[Any], None]:
    """ingest 배치용 best-effort 증분 색인기를 만든다(spec 020, FR-002 · 클라이언트 1회 재사용).

    반환한 콜러블 ``index(asset_id)`` 를 자산 finalize(PG 커밋) **직후**마다 부른다.

    안전 게이트(프로덕션 적재 경로 — 회귀 0·격리):
      · **opt-in off(기본)이면 콜러블이 즉시 반환** — ``opensearch_sync_enabled`` 미설정/False 면 아무
        것도 하지 않고, ``src.search.opensearch_sync``·opensearch-py 를 **import 조차 하지 않는다**(아래
        지연 import). 미도입 환경의 run_ingest 동작이 완전 불변(SC-001). PR4b: ``settings`` 는 완전한
        ``PipelineSettings``(``.opensearch`` 보유)를 요구한다 — malformed 는 즉시 AttributeError(fail-fast).
      · **클라이언트 재사용**: OpenSearch 클라이언트·활성 채널은 **첫 색인 성공에서 1회** 만들어
        ``cache`` 에 담아 배치 전체에서 재사용한다 — 디렉터리/파일리스트 수집에서 자산마다 새 연결을
        열던 낭비를 없앤다. "배치당 1회"는 **성공 경로 보증**이다: OS 가 **지속 다운**이면 ``get_client``
        실패로 ``cache['client']`` 가 미설정이라 자산마다 셋업을 재시도한다(상한 = 배치 크기, 무한 아님).
        이는 OS 가 배치 도중 복구되면 그때부터 재사용을 재개하는 **일과성 복구**가 의도다. 부분 캐시
        손상은 없다(게이트는 ``'client'`` 단일 키 기준, ``channel``/``index_asset`` 은 멱등 재대입).
      · **격리**: 각 색인을 ``try/except Exception`` 으로 감싸 OS 미도달·색인 오류를 ``_LOG.warning``
        만 남기고 삼킨다 — ingest 를 중단·롤백하지 않는다(SC-003). finalize 트랜잭션 커밋 뒤라 OS
        색인은 본질적으로 PG 와 분리된 best-effort 작업이다.
    """
    enabled = settings.opensearch.sync_enabled
    cache: dict[str, Any] = {}  # 첫 성공 셋업 후 client·index_asset·channel 을 담아 배치 내 재사용

    def index(asset_id: Any) -> None:
        if not enabled:
            return  # off(기본) — OpenSearch 코드 미접촉, 기존 동작 불변
        try:
            if "client" not in cache:
                # 지연 import — 플래그 off 환경(opensearch-py 미설치 가능)의 순수성을 보존한다.
                from src.config.settings import active_embed_channel
                from src.search.opensearch_sync import get_client, index_asset

                cache["channel"] = active_embed_channel(settings)  # 적재·검색과 같은 채널(018)
                cache["client"] = get_client(settings.opensearch.url)  # 배치당 1회 생성·재사용
                cache["index_asset"] = index_asset
            # PG 는 읽기전용(SELECT만, FR-004) → OpenSearch 에만 쓰기(CQRS). finalize 와 별도 트랜잭션.
            with db.transaction() as conn:
                cache["index_asset"](
                    cache["client"], conn, str(asset_id),
                    index=settings.opensearch.index, channel=cache["channel"],
                    noise_patterns=settings.opensearch.filename_noise_patterns,
                )
        except Exception as exc:  # noqa: BLE001 — OS 색인 실패가 적재를 막지 않는다(best-effort 격리)
            _LOG.warning("opensearch 증분 색인 실패(무시): asset_id=%s (%s)", asset_id, exc)

    return index


def collect_file(conn: Connection[Any], fs_path: str) -> CollectResult:
    """파일 1건을 ``received`` 자산으로 만든다(수집 단계 — 모델 0·저비용).

    오케스트레이터-불가지 순수 함수(CLI run_ingest·Airflow dag_collect·테스트가 공유, FR-011):
      1. ``route_file`` 로 모달리티 탐지(헤더/확장자 기반·모델 미적재).
      2. 파일 누락(``REASON_MISSING``)이면 자산 미생성 스킵(해시조차 계산하지 않음 — 기존 동작 보존).
      3. 내용 해시 기준 중복 방지(009#4): 동일 ``file_hash`` 가 이미 적재(registered/deferred)면 스킵
         (``f"{REASON_DUPLICATE}:{기존id}"``). DICOM 등 보류 자산 재수집이 중복 행을 만들지 않는다.
         ``failed`` 는 dup=None 으로 떨어져 재처리된다(``find_registered_asset_by_hash`` IN 범위 제어).
      4. 그 외엔 ``create_asset(status='received')`` 조기 INSERT + ``ingest.received.v1`` lineage.

    트랜잭션 경계는 호출자가 제어한다(``conn`` 우선 인자 — registry/*_persist 패턴). dedup SELECT 와
    create INSERT 가 같은 트랜잭션에서 일어나도 적재 행·lineage·반환은 분할 전과 동일하다(원자 검사-생성).

    ``unknown_modality``(routable=False·reason≠missing)는 스킵하지 않고 received 를 만든다 — 이후
    process_asset 의 추출에서 ``UnsupportedModalityError`` 로 failed 처리되는 기존 흐름을 보존한다.
    """
    route = route_file(fs_path)
    if not route.routable and route.reason == REASON_MISSING:
        _LOG.info("skip(missing): %s", fs_path)
        return CollectResult(asset_id=None, route=route, skip_reason=route.reason)

    file_hash, file_size = file_hash_and_size(fs_path)
    dup = find_registered_asset_by_hash(conn, file_hash)
    if dup is not None:
        _LOG.info("skip(duplicate of %s): %s", dup, fs_path)
        return CollectResult(asset_id=None, route=route, skip_reason=f"{REASON_DUPLICATE}:{dup}")

    asset_id = create_asset(
        conn,
        fs_path=fs_path,
        modality=route.modality,
        domain=route.domain,
        file_hash=file_hash,
        file_size=file_size,
    )
    # 053(FR-203): 감사도 canonical 일관. 원 file_kind 는 payload.fs_path 확장자로 재도출 가능.
    record_lineage(conn, asset_id, activity="ingest.received.v1", agent="run_ingest",
                   generated={"modality": modality_of(route.modality)}, payload={"fs_path": fs_path})
    return CollectResult(asset_id=asset_id, route=route, skip_reason=None)


def process_asset(
    asset_id: uuid.UUID,
    *,
    db: PostgresUtil,
    fs_path: str,
    modality: str,
    domain: str = "general",
    extract_fn: ExtractFn | None = None,   # 테스트·e2e 전용 override(미주입=팩 기본 extract/embed)
    classify_fn: ClassifyFn | None = None,  # 테스트·e2e 전용 override(미주입=cascade_v1)
    registry=DEFAULT_REGISTRY,
    settings: Any,
    os_index: OsIndexFn | None = None,
) -> str:
    """``received`` 자산 1건을 ``routing→classifying→extracting→registered``(또는 ``deferred``)로 전이한다.

    오케스트레이터-불가지 순수 함수(CLI run_ingest·Airflow dag_process·테스트가 공유, FR-011). 기존
    run_ingest 의 'received 이후' 단계 로직(분류·deferred 판별·도메인 팩·추출/임베딩·적재)을 **무변경**으로
    옮긴 것이다 — 상태전이·lineage 시퀀스·인라인 색인 호출이 분할 전과 동일하다(헌법 8조).

    반환: ``'registered'`` 또는 ``'deferred'``. 처리 중 예외는 **호출자에게 전파**한다 — 실패 정책은
    호출자 책임이다(CLI run_ingest=즉시 mark_failed 종료, 배치 dag_process=비종료 유지·재시도/cap).

    ``os_index`` 는 finalize(PG 커밋) **직후** ``os_index(asset_id)`` 로 1회 호출하는 인라인 색인 seam이다
    (배치당 1회 생성·재사용을 호출자가 주입). 미주입이면 no-op(배치 밖 단독 처리 안전).
    """
    if os_index is None:
        os_index = lambda _asset_id: None  # noqa: E731 — 미주입 시 색인 no-op(주입 seam 안전 기본값)

    # 1) routing → classifying (한 트랜잭션 묶음 커밋)
    # status 는 '단계 진입 전'에 찍는 진행형 마커다(완료는 다음 전이로 암시). 그래서:
    #  - routing: 실작업(route_file, collect_file)이 row 생성 전 끝나 사후 마커이고, classifying 과 같은
    #    트랜잭션이라 단독 관측되지 않는다 — received→classifying 사이를 FSM 순차성상 거쳐갈 뿐.
    #  - classifying: 바로 아래 실분류(cascade) 진입 직전을 정확히 반영.
    with db.transaction() as conn:
        set_status(conn, asset_id, AssetStatus.ROUTING)
        record_lineage(conn, asset_id, activity="ingest.routing.v1", agent="run_ingest")
        set_status(conn, asset_id, AssetStatus.CLASSIFYING)
        record_lineage(conn, asset_id, activity="ingest.classifying.v1", agent="run_ingest")

    # 2) 도메인 분류: override 우선, 없으면 레지스트리 기본 분류기(ctx 기반)
    ctx = ExtractContext(file_path=fs_path, modality=modality, domain=domain, settings=settings, db=db)
    if classify_fn is not None:
        classification = classify_fn(fs_path, modality)
    else:
        classification = registry.resolve("classify", "cascade_v1")(ctx)
    with db.transaction() as conn:
        record_classification(conn, asset_id, classification)
        record_lineage(conn, asset_id, activity="ingest.classified.v1",
                       agent=("classify_fn" if classify_fn is not None else "cascade_v1"),
                       generated={"final_label": classification.final_label,
                                  "decided_stage": classification.decided_stage,
                                  "confidence": classification.confidence})
    domain = classification.final_label
    ctx.domain = domain

    # 시그니처로 확정된 포맷(stage1)이지만 해당 어댑터가 없으면 보류(deferred) — 실패 아닌 계획적 대기.
    # 현재 시그니처를 등록한 프로파일이 medical(DICOM/HL7/FHIR)뿐이라 사실상 의료 포맷만 보류되지만,
    # 코드 자체는 도메인-불가지(메커니즘은 범용; 어느 도메인이든 추출기 없는 시그니처 포맷이면 보류).
    # cascade v2 이후 stage1_scores 는 {domain: {signature, ...}} 중첩 구조.
    signature = None
    if classification.decided_stage == 1:
        signature = classification.stage1_scores.get(domain, {}).get("signature")
    if signature:
        with db.transaction() as conn:
            set_status(conn, asset_id, AssetStatus.DEFERRED, reason=f"{domain}_format:{signature}")
            record_lineage(conn, asset_id, activity="ingest.deferred.v1", agent="run_ingest",
                           payload={"domain": domain, "signature": signature})
        _LOG.info("deferred(%s/%s): asset_id=%s %s", domain, signature, asset_id, fs_path)
        return "deferred"

    # 도메인 팩 선택 + 정책 검증(컴포지션 시점)
    pack = for_domain(domain)
    policy_validate(pack, registry)

    with db.transaction() as conn:
        set_status(conn, asset_id, AssetStatus.EXTRACTING)
        record_lineage(conn, asset_id, activity="ingest.extracting.v1", agent="run_ingest")

    # 3) 추출/임베딩 — override(full record) 또는 팩 경로(extract_meta + embed)
    if extract_fn is not None:
        record = extract_fn(ctx)
    else:
        extract = registry.resolve("extract", pack.per_asset["extract"])
        embed = registry.resolve("embed", pack.per_asset["embed"])
        record = extract(ctx)
        record.embeddings = embed(ctx, record)

    # 4) 검증 + 적재 + registered — 한 트랜잭션
    with db.transaction() as conn:
        # 039 ext_meta 키·값 검증 — tier(040)는 write 에서 검사하지 않음(042 read 전용).
        validate_ext_meta(conn, domain, record.ext_meta)
        finalize_asset(conn, asset_id, record)
        record_lineage(conn, asset_id, activity="ingest.registered.v1",
                       agent=("extract_fn" if extract_fn is not None else pack.per_asset["extract"]),
                       generated={"channels": sorted({e.channel for e in record.embeddings}),
                                  "n_embeddings": len(record.embeddings),
                                  "models": sorted({e.model_name for e in record.embeddings})})

    _LOG.info("registered: asset_id=%s %s", asset_id, fs_path)

    # 자기주제 분류(065·FR-301/302) — registered 커밋 **직후·OS 색인 전**에 (topic, subtopic)
    # 정본을 부여한다. 색인 전이라 신규 자산은 첫 OS doc 부터 topics 를 포함한다(관계 배치 대기 불요).
    # 자기 텍스트는 in-memory ``record.ext_meta`` 에서 구성해 주입한다(방금 적재한 값 재조회 회피).
    # **완전 격리(FR-204)**: 분류 실패는 warn 로그만 남기고 registered 를 유지하며 색인을 진행한다
    # (자산은 등록됨·주제만 미부여·백필 재시도 대상). 별도 짧은 트랜잭션(finalize 와 분리).
    try:
        self_text = build_self_text(
            record.ext_meta.get("summary"),
            record.ext_meta.get("keywords"),
            record.ext_meta.get("labels"),
        )
        with db.transaction() as conn:
            classify_asset_topic(
                conn, asset_id, self_text=self_text, settings=settings, client=None
            )
    except Exception as exc:  # noqa: BLE001 — 분류 실패가 적재(registered)를 막지 않는다(FR-204 격리)
        _LOG.warning("자기주제 분류 실패(무시): asset_id=%s (%s)", asset_id, exc)

    # aboutness 개체 확정(073 FR-001) — 검색 OR-증거 필터의 적재시점 층. 065 분류와 동일 격리
    # 패턴(실패=warn·registered 유지·백필 재시도 대상). os_index **전**에 저장해 첫 OS doc 부터
    # about 을 포함한다(asset_to_doc 이 ext_meta['about'] 을 읽음).
    # 의료 제외(리뷰 지적 — 백필 CLI 의 medical 제외와 대칭): 의료는 현재 시그니처 기반 deferred 라
    # 여기 도달하지 않지만, 미인식 의료 자산이 registered 로 진행돼도 summary 를 LLM 에 보내지
    # 않도록 이중 방어한다(검색 자체가 의료 배제(FR-011)라 about 도 불요).
    if domain != "medical":
        try:
            with db.transaction() as conn:
                extract_and_persist_about(
                    conn, asset_id, summary=record.ext_meta.get("summary"), client=None
                )
        except Exception as exc:  # noqa: BLE001 — 추출 실패가 적재(registered)를 막지 않는다(073 격리)
            _LOG.warning("aboutness 추출 실패(무시): asset_id=%s (%s)", asset_id, exc)

    # 증분 색인(opt-in·격리) — 위 finalize 트랜잭션이 **PG 커밋된 직후** 호출한다(FR-002).
    # off(기본)면 즉시 반환해 OpenSearch 코드를 전혀 건드리지 않으므로 기존 동작 불변(SC-001).
    # 색인기 내부 try/except 로 OS 실패를 삼켜 적재를 중단·롤백하지 않는다(SC-003).
    os_index(asset_id)
    return "registered"
