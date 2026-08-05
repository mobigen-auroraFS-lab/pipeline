# dataplatform-pipeline

멀티모달 데이터 통합 플랫폼의 **처리 파이프라인**입니다. 파일 수집 → 분류 → 추출 → 임베딩 → 적재 →
색인 → 관계 생성을 Airflow 로 오케스트레이션합니다.

> 국책과제 **RS-2025-02215256** 산출물.

## 세 레포의 관계

| 레포 | 파이썬 패키지 | 역할 |
|---|---|---|
| dataplatform-core | `src.*` | 규약·계약·순수 로직 + DB 스키마 정본 |
| **dataplatform-pipeline**(이 레포) | `processing.*` | 실행 오케스트레이션(Airflow DAG·CLI) |
| dataplatform-service | `service.*` | HTTP API |

**이 레포는 코어를 필요로 합니다.** 코어가 없으면 `processing.*` 이 import 되지 않습니다.

## 아키텍처 — 브로커 없이 "PostgreSQL 상태 = 작업 큐"

별도 메시지 브로커(RabbitMQ 등)를 두지 않습니다. 자산의 처리 상태가 PostgreSQL 에 있고,
Airflow DAG 가 그 상태를 조회해 다음 할 일을 집습니다. 상태가 곧 큐이므로 큐와 DB 가 어긋날 수 없습니다.

DAG 3종:

| DAG | 하는 일 |
|---|---|
| `dag_collect` | inbox 감시 → 자산 등록(수집) |
| `dag_process` | per-asset 처리 — 분류·추출·요약·임베딩·적재·색인 |
| `dag_relations` | cross-asset 관계 후보 생성 → LLM 제안 → 엣지 저장 |

## 요구사항

| 항목 | 버전 |
|---|---|
| Python | **3.13 이상** |
| Apache Airflow | **3.x** |
| PostgreSQL | 17 + `pgvector` (코어 스키마) · Airflow 메타DB 별도 |
| OpenSearch | `analysis-nori` 플러그인 |
| 시스템 도구 | `ffmpeg`(영상·오디오) · `tesseract`(이미지 OCR) |

## 설치

**코어를 먼저 설치합니다.**

```bash
# ① 코어 — 나란히 clone 해서 참조형으로 설치(개발) 또는 태그로 설치형
git clone https://github.com/OWNER/dataplatform-core.git
pip install -e ./dataplatform-core

# ② 이 레포
pip install -e .
```

코어를 설치하면 코어 런타임 의존(psycopg·numpy·torch·transformers·pillow·opensearch-py 등)이
**전이로** 따라옵니다. 이 레포의 `pyproject.toml` 에는 코어가 제공하지 않는 파이프라인 전용
의존(opencv·faster-whisper·scenedetect·soundfile·pytesseract)만 있습니다.

> `apache-airflow` 는 어느 목록에도 없습니다 — 실행 환경이 제공하는 버전과 핀이 충돌하는 것을 막기 위함입니다.

## 스키마·시드 (최초 1회)

파이프라인을 돌리기 전에 **코어 레포에서** 다음을 마칩니다.

```bash
alembic -c alembic.ini upgrade head                        # DB 스키마
python -m scripts.seed_topic_registry --env dev --apply    # ★ 닫힌 taxonomy 시드
```

> ⚠️ **시드를 생략하면 관계 생성 결과가 0건이 됩니다.**

## 환경변수

레포 루트에 `.env.dev` 를 두면 `META_ENV=dev` 로 코어 설정 로더가 읽습니다. 필요한 변수 **이름만** 적습니다.

### 🔴 필수 — 없으면 기동 시점에 실패합니다

코어 설정 로더가 다음 11개를 **필수로 요구**합니다(미설정 시 `ValueError: 필수 환경변수 누락: <이름>`
으로 즉시 중단 — 잘못된 설정으로 조용히 도는 것을 막는 fail-fast).

```dotenv
META_MODEL=              # 온프레미스 LLM 모델 이름
ENCODING=utf-8
CHUNK_SIZE=1000
OVERLAP_SIZE=100
SUMMARY_MAX_CHARS=500
TOP_K_KEYWORDS=10
TEXT_EMBED_MODEL=
TEXT_EMBED_CHUNK_SIZE=512
TEXT_EMBED_NORMALIZE=true
OPENAI_BASE_URL=         # OpenAI 호환 엔드포인트(= 온프레미스 LLM 서버)
OPENAI_API_KEY=
```

### 그 외

| 구분 | 변수 |
|---|---|
| DB | `POSTGRES_HOST` · `POSTGRES_PORT` · `POSTGRES_DB` · `POSTGRES_USER` · `POSTGRES_PASSWORD` |
| 검색 | `OPENSEARCH_HOST` · `OPENSEARCH_PORT` |
| LLM | `LLM_BASE_URL` · `LLM_MODEL` |
| 데이터 경로 | `WATCHER_INBOX_DIR`(수집 대기) · `WATCHER_ARCHIVE_DIR`(보관) |
| DAG 튜닝(선택) | `META_ENV` · `DAG_COLLECT_SCHEDULE` · `DAG_PROCESS_SCHEDULE` · `DAG_RELATIONS_SCHEDULE` · `DAG_PROCESS_LIMIT` · `DAG_PROCESS_MAX_FAILURES` · `DAG_PROCESS_POOL` · `DAG_RELATIONS_LIMIT` |

> ⚠️ 보관 디렉터리는 **HTTP API 레포와 공유**합니다(다운로드·썸네일이 같은 파일을 읽습니다).
> 두 레포에 같은 경로를 지정하십시오.

## 실행

Airflow 없이 로컬에서 바로 확인하려면:

```bash
python -m processing.app.run_ingest    --env dev <파일>            # per-asset 수집·처리
python -m processing.app.run_relations --env dev --all             # cross-asset 관계 생성
python -m processing.app.run_search    --env dev --query "<질의>"   # 검색(코어 위임)
python -m processing.app.run_opensearch_resync --env dev           # 색인 재생성
```

Airflow 로 상시 운영하려면:

```bash
cd deploy/airflow
./run.sh start      # scheduler · dag-processor · api-server 일괄 기동
./run.sh status     # 상태 확인 (중지: stop · 재기동: restart)
```

## 테스트

```bash
python -m unittest discover -s tests    # 순수 단위(실 DB·모델 불필요분은 자동 skip)
```

## 구조

```
processing/
  app/          실행 진입점(run_ingest · run_relations · run_search · run_opensearch_resync)
  classify/     도메인·모달리티 분류
  dispatch/     라우팅
  extractors/   모달리티별 메타데이터 추출
  ingest/       수집·적재·상태 전이·배치 러너
  pipeline/     v2 모듈 조합 레이어(계약·레지스트리·도메인 팩·정책)
  preprocess/   전처리(STT · 키프레임 · 장면 분할)
  skills/       요약·캡션 등 보조 기능
deploy/airflow/ DAG 3종 + 네이티브 실행 스크립트
tests/          단위 테스트
```

## 설계 제약

- **학습 기반 방식을 쓰지 않습니다** — 사전학습 모델은 추론 전용입니다.
- 도메인을 코드로 분기하지 않습니다 — 고정 뼈대 + **도메인 팩**이 스테이지 전략을 고릅니다.
- 코드·주석·로그는 한국어로 작성합니다.
