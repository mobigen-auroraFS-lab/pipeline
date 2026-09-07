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
git clone <이 레포와 같은 계정>/dataplatform-core.git   # 예: gh repo clone <owner>/dataplatform-core
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

템플릿이 있습니다 — 복사해서 값만 채우면 됩니다:

```bash
cp .env.example .env.dev      # .env.dev 는 커밋되지 않습니다(.gitignore)
```

### 설정을 주는 두 가지 방법

| 방법 | 어디에 | 우선순위 |
|---|---|---|
| **A. `.env.<환경>` 파일** | **실행하는 디렉터리** → 없으면 레포 루트 순으로 찾습니다 | 낮음 |
| **B. 환경변수 직접 주입** | 배포·컨테이너·CI(`export` · `env_file:` · `env:`) | **높음**(A 를 덮어씁니다) |

방법 B 로 파일 값을 그대로 올리려면:

```bash
set -a; . ./.env.dev; set +a
```

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

Airflow 로 상시 운영하려면 — DAG 폴더를 지정해 네이티브로 띄웁니다.

```bash
export AIRFLOW_HOME=~/airflow-home                    # 메타DB·설정 위치(임의)
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/deploy/airflow/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export META_ENV=dev                                   # 코어 설정 프로파일

airflow db migrate                                    # 최초 1회
airflow scheduler &                                   # 스케줄러
airflow dag-processor &                               # DAG 파싱(3.x 는 별 프로세스)
airflow api-server &                                  # UI/API
```

> ⚠️ **DAG 태스크는 앱 환경변수를 프로세스 환경에서 물려받습니다.** `META_MODEL` 등이 빠지면
> 태스크가 `init_settings` 에서 즉시 실패합니다 → Airflow 를 띄우기 **전에** `.env.dev` 값을
> 환경으로 올리십시오: `set -a; . ./.env.dev; set +a`
>
> ⚠️ Airflow 메타DB는 앱 DB와 **분리**하십시오(같은 이름을 쓰면 충돌합니다).
> GPU·모델을 쓰는 태스크는 동시 실행을 1로 제한하는 pool 을 두는 것이 안전합니다.

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

## 트러블슈팅

### `ValueError: 필수 환경변수 누락: META_MODEL`

설정이 **하나도** 로드되지 않았다는 뜻입니다. 값이 틀린 게 아니라 대개 `.env` 파일을 못 찾은 것입니다.

1. `.env.dev` 가 **실행하는 디렉터리** 또는 레포 루트에 있는지 확인하십시오(`cp .env.example .env.dev`).
2. `--env dev` 로 실행했는지 확인하십시오 — `--env prod` 는 `.env.prod` 를 찾습니다.
3. 그래도 안 되면 환경변수를 직접 주입하십시오: `set -a; . ./.env.dev; set +a`
   (§환경변수 › 방법 B — 설치 방식과 무관하게 항상 동작합니다).

### 코어를 못 찾습니다 (`ModuleNotFoundError: No module named 'src'`)

이 레포는 코어(`dataplatform-core`)를 필요로 합니다. §설치 순서대로 **코어를 먼저** 설치하십시오.

### 관계 생성 결과가 0건입니다

**코어 레포에서** 닫힌 주제 분류체계 시드를 적재하지 않았을 때 나타납니다:
`python -m scripts.seed_topic_registry --env dev --apply`

### DAG 가 Airflow UI 에 보이지 않습니다

DAG 파일이 import 단계에서 실패하면 목록에 나타나지 않습니다. 먼저 로컬에서 확인하십시오:
`python -c "from airflow.dag_processing.dagbag import DagBag; b=DagBag(dag_folder='deploy/airflow/dags'); print(sorted(b.dag_ids), b.import_errors)"`
(Airflow 3.1 이전은 `airflow.models.dagbag` 경로입니다.)

## 이 레포에 대해

이 레포는 이 프로젝트의 **공개 개발 레포**입니다 — 소스는 여기서 직접 개발합니다(2026-08-06 이후). 코드·테스트·
Airflow DAG 와 "어떻게 돌리나"(이 README)만 담고, **왜 이렇게 설계했나**(기획·설계 문서·설계 변경 이력·결정 기록)는
별도 비공개 문서 레포에 있습니다. 그래서 커밋 메시지는 짧고, 근거는 `근거: 설계이력 YYYY-MM-DD` 한 줄로 그 문서를 가리킵니다.

- 코어는 git 태그(`vMAJOR.MINOR.PATCH`)를 기준으로 설치합니다. 코어 공개 API 변경은 코어 `CHANGELOG.md` 에 있습니다.
- 문의는 과제 담당자에게 해주십시오.
