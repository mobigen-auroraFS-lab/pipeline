"""여러 오케스트레이터가 공유하는 수집·처리 스텝 — 수집 · 처리 · 인라인 색인기.

**흐름에서의 위치**: CLI 도, 배치도, 스케줄러 태스크도 **모두 이 두 함수를 부른다**. 그래서
이 모듈은 자기를 부르는 쪽을 알지 못하고, 실패를 어떻게 다룰지도 정하지 않는다.

⚠️ **이 스텝들이 CLI 쪽에 있으면 안 된다.** 배치가 CLI 진입점을 import 하면 계층이 거꾸로
뒤집혀 순환 import 지뢰가 된다. 여기(수집 계층)에 두면 배치·스케줄러가 같은 계층만 보고,
파이프라인 코드가 진입점 없이 자족한다.
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
from src.file.file_type_defs import modality_of  # 저장·기록에 쓸 큰 갈래 매핑
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
      (파일 누락 · 같은 내용이 이미 적재됨). 그때는 자산 행을 만들지 않는다.

    경로 판정 결과는 **스킵이든 아니든 항상 채운다** — 호출부가 그것으로 처리 문맥을 만든다.
    """

    asset_id: uuid.UUID | None
    route: RouteResult
    skip_reason: str | None


def _make_opensearch_indexer(*, db: PostgresUtil, settings: Any) -> Callable[[Any], None]:
    """적재 직후 자산 하나를 색인할 콜러블을 만든다(배치당 하나 만들어 재사용).

    돌려받은 함수를 자산 저장이 **커밋된 직후**마다 부른다.

    안전장치 셋
      · **꺼져 있으면 즉시 반환**한다 — 검색 엔진 라이브러리를 **import 조차 하지 않는다**
        (지연 import). 검색 엔진이 없는 환경에서도 적재가 그대로 돈다.
      · **연결을 배치 전체에서 재사용**한다. 자산마다 새로 열면 연결 수가 자산 수만큼 늘어난다.
        단, 첫 연결이 실패하면 캐시에 담기지 않아 자산마다 다시 시도한다 — 배치 도중 검색
        엔진이 복구되면 그때부터 재사용을 재개하려는 의도다(재시도 상한 = 배치 크기).
      · **색인 실패를 삼킨다.** 저장은 이미 커밋됐고 색인은 그와 분리된 부수 작업이므로,
        색인이 실패했다고 적재를 되돌리면 안 된다. 경고 로그만 남긴다.

    Args:
        db: 트랜잭션을 열 수 있는 DB 핸들.
        settings: 설정. ⚠️ **완전한 설정 객체여야 한다** — 색인 설정을 못 읽으면 그 자리에서
            터뜨린다(조용히 꺼지는 것보다 낫다).

    Returns:
        ``index(asset_id)`` 콜러블.
    """
    enabled = settings.opensearch.sync_enabled
    cache: dict[str, Any] = {}  # 첫 성공 셋업 후 client·index_asset·channel 을 담아 배치 내 재사용

    def index(asset_id: Any) -> None:
        """자산 하나를 검색 색인에 반영한다(적재 직후 훅).

        토글이 꺼져 있으면 아무 것도 하지 않고, 색인 실패도 **삼킨다** — 검색 색인 문제가
        적재 자체를 실패시키면 안 되기 때문이다(복구 도구로 나중에 다시 넣을 수 있다).
        클라이언트는 캐시에 담아 배치 안에서 재사용한다.
        """
        if not enabled:
            return  # 꺼져 있음 — 검색 엔진 코드를 전혀 건드리지 않는다
        try:
            if "client" not in cache:
                # 지연 import — 플래그 off 환경(opensearch-py 미설치 가능)의 순수성을 보존한다.
                from src.config.settings import active_embed_channel
                from src.search.opensearch_sync import get_client, index_asset

                cache["channel"] = active_embed_channel(settings)  # 적재·검색이 같은 채널을 봐야 한다
                cache["client"] = get_client(settings.opensearch.url)  # 배치당 1회 생성·재사용
                cache["index_asset"] = index_asset
            # DB 는 읽기만, 쓰기는 검색 엔진에만(헌법 6조). 저장 트랜잭션과 분리된 별도 트랜잭션이다.
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

    **DB에 쓴다.** 흐름: 경로로 모달리티 판정(모델 미적재) → 파일이 없으면 스킵(해시조차 계산
    하지 않는다) → 내용 해시가 같은 자산이 이미 있으면 스킵 → 그 밖이면 대기 상태로 행을 만든다.

    **중복 판정은 내용 해시로 한다** — 경로가 달라도 같은 파일이면 한 번만 적재된다. 실패한
    자산은 중복 판정에서 빠져 다시 처리된다.

    ⚠️ **모달리티를 몰라도 스킵하지 않는다** — 대기 상태로 만들어 두면 다음 단계 추출이 그것을
    실패로 확정한다. 여기서 버리면 "들어왔는데 아무 기록도 없는 파일"이 생긴다.

    Args:
        conn: 호출자가 연 연결. 중복 검사와 행 생성이 **같은 트랜잭션**에서 일어나야 둘 사이에
            끼어들어 중복 행이 생기지 않는다.
        fs_path: 수집할 파일 경로.

    Returns:
        만든 자산 id(스킵이면 ``None``)·경로 판정 결과·스킵 사유. **경로 판정 결과는 스킵이든
        아니든 항상 채운다** — 호출부가 그것으로 처리 문맥을 만든다.
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
    # 기록에도 큰 갈래로 좁힌 모달리티를 남긴다(집계와 어긋나지 않게). 세부 종류가 필요하면
    # 함께 남긴 경로의 확장자에서 다시 구할 수 있다.
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

    **DB에 쓴다.** CLI·스케줄러·테스트가 함께 쓰는 함수라 **누가 부르는지 알지 못한다**.

    ⚠️ **예외를 삼키지 않고 호출자에게 올린다** — 실패를 어떻게 다룰지는 부르는 쪽마다
    다르기 때문이다(CLI 는 즉시 실패 처리, 배치는 남겨 뒀다 재시도).

    Args:
        asset_id: 처리할 자산. **대기 상태여야 한다** — 첫 상태 전이가 그 조건을 걸고,
            0행이면 남이 이미 집어 간 것이라 예외가 난다.
        db: 트랜잭션을 열 수 있는 DB 핸들.
        fs_path: 파일 경로.
        modality: 파일 종류.
        domain: 도메인 라벨. 어느 팩을 쓸지 여기서 갈린다.
        extract_fn: 추출을 갈아끼울 때만. 미주입이면 팩이 고른 전략.
        classify_fn: 분류를 갈아끼울 때만. 미주입이면 팩이 고른 전략.
        registry: 전략을 찾을 레지스트리.
        settings: 설정(필수 키워드).
        os_index: 저장 커밋 **직후** 한 번 부를 색인 함수. 미주입이면 아무 일도 하지 않는다
            (배치 밖 단독 처리에서도 안전하도록).

    Returns:
        ``'registered'``(정상 완료) 또는 ``'deferred'``(계획적 보류).
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
        # 확장 메타 키·값 검증 — 열람 등급은 여기서 검사하지 않는다(읽기 경로 소관).
        validate_ext_meta(conn, domain, record.ext_meta)
        finalize_asset(conn, asset_id, record)
        record_lineage(conn, asset_id, activity="ingest.registered.v1",
                       agent=("extract_fn" if extract_fn is not None else pack.per_asset["extract"]),
                       generated={"channels": sorted({e.channel for e in record.embeddings}),
                                  "n_embeddings": len(record.embeddings),
                                  "models": sorted({e.model_name for e in record.embeddings})})

    _LOG.info("registered: asset_id=%s %s", asset_id, fs_path)

    # 주제 확정 — 저장 커밋 **직후·색인 전**에 한다. 색인 전이라야 새 자산이 **첫 색인 문서부터**
    # 주제를 갖는다(관계 배치가 돌 때까지 기다리지 않는다).
    # 자기 텍스트는 방금 만든 메모리 값에서 구성한다 — 막 적재한 값을 다시 조회할 이유가 없다.
    # ⚠️ **실패를 완전히 격리한다**: 경고만 남기고 자산은 등록된 채 둔다. 주제 미부여는 나중에
    #    백필로 메울 수 있지만, 여기서 예외를 올리면 멀쩡히 적재된 자산이 실패로 뒤집힌다.
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
    except Exception as exc:  # noqa: BLE001 — 분류 실패가 적재를 뒤집지 않게 전부 흡수
        _LOG.warning("자기주제 분류 실패(무시): asset_id=%s (%s)", asset_id, exc)

    # "무엇에 관한 자산인가" 개체 확정 — 검색이 글자 비교만으로 걸러 낼 수 있게 **적재 시점에**
    # 한 번 굳혀 둔다(검색 때마다 모델을 돌리지 않는다). 주제 분류와 같은 격리 패턴이고,
    # 색인 **전**에 저장해야 첫 색인 문서부터 포함된다.
    # 도메인에 따라 건너뛰지 않는다 — 전 도메인을 균일하게 처리한다. LLM 은 온프레미스 단일
    # 통로만 쓰므로 외부로 나가지 않는다(헌법 2조).
    try:
        with db.transaction() as conn:
            extract_and_persist_about(
                conn, asset_id, summary=record.ext_meta.get("summary"), client=None
            )
    except Exception as exc:  # noqa: BLE001 — 개체 추출 실패가 적재를 뒤집지 않게 전부 흡수
        _LOG.warning("aboutness 추출 실패(무시): asset_id=%s (%s)", asset_id, exc)

    # 색인 — 저장이 **커밋된 뒤에** 부른다. 커밋 전에 색인하면 롤백된 자산이 검색에 남는다.
    # 꺼져 있으면 즉시 반환하고, 실패는 색인기 안에서 삼켜 적재를 되돌리지 않는다.
    os_index(asset_id)
    return "registered"
