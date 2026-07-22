"""processing.app — 플랫폼의 모든 실행 진입점(CLI · HTTP API)을 모은 얇은 오케스트레이션 레이어.

각 모듈은 도메인 로직을 직접 들고 있지 않다 — argparse 로 인자를 받아 부트스트랩
(load_dotenv(.env.{env}) → init_settings → PostgresUtil)한 뒤, 처리 조합 레이어(``processing.pipeline``)와
코어(``src.registry``·``src.search``·``src.relations``)로 위임만 한다. v2 "고정 뼈대"의 바깥 껍데기다.

진입점 지도(정기 운영 CLI):
  · run_ingest            — per-asset 수집·적재 (route→classify→extract→embed→persist)
  · run_relations         — cross-asset 관계 제안 배치 (candidates→propose→graph_edge)
  · run_search            — 하이브리드 검색 CLI(OpenSearch BM25+kNN 클라이언트 융합·037)
  · run_opensearch_resync — PG→OpenSearch 전체 재색인 복구 도구(spec 020)

레포 분리(077·078): HTTP 백엔드(구 ``portal_api`` FastAPI·spec 010)는 백엔드 레포
``dataplatform-service``(``service.api``)로 이관됐다. 이 레포는 **처리 파이프라인(``processing.*``)만** 담당하고,
공유 코어(``src.*``)는 별도 레포 ``DataFlatform``(pip ``meta-extract``)을 설치해 참조한다.
수동 백필(구 ``run_about_backfill``·``run_topic_backfill``)은 다른 백필들과 관례를 맞춰
``scripts/backfill_aboutness``·``scripts/backfill_asset_topic`` 로 옮겼다(정기 파이프라인 아님·수동 유지보수).

부트스트랩 관용(공통): main() 이 .env.{env} 를 override=False(OS 기존 환경변수 우선)로 로드한 뒤
init_settings 로 frozen 설정을 만든다. **per-asset/cross-asset 진입점(run_ingest·run_relations)만**
`processing.pipeline.builtins` 를 import 해 DEFAULT_REGISTRY 에 슬롯 전략을 등록하는 부수효과를 일으킨다 —
검색·복구 진입점은 레지스트리·도메인 팩이 필요 없어 import 하지 않는다.
"""
