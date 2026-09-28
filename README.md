# pipeline

멀티모달 데이터 통합 플랫폼에서 파일을 받아 검색 가능한 상태로 만드는 처리 레포입니다.

## 이 레포지토리는 무엇인가

지정한 폴더에 파일이 들어오면 자산으로 등록하고, 종류를 가려 요약과 키워드를 뽑고,
벡터로 바꿔 데이터베이스와 검색 색인에 넣습니다. 그다음 자산들 사이의 관계를 찾고, 여러
자산에 공통으로 등장하는 개체를 묶습니다. 이 과정을 Airflow 로 자동화합니다.

| 저장소 | 하는 일 |
|---|---|
| [core](https://github.com/mobigen-auroraFS-lab/core) | 공통 코드 · 데이터베이스 스키마 |
| **pipeline** (이 레포) | 파일 수집 · 분류 · 메타데이터 추출 · 색인 · 자산 간 관계 생성 |
| [service](https://github.com/mobigen-auroraFS-lab/service) | 웹 화면이 사용하는 HTTP API |

**작업 대기열로 메시지 브로커를 쓰지 않습니다.** 대신 PostgreSQL 에 적힌 자산의 상태가
대기열 역할을 합니다(`received` → `processing` → `registered` / `deferred` / `failed`).
상태를 바꿀 때 조건을 걸어 갱신하므로, 작업자가 여럿이어도 같은 자산을 두 번 집지 않습니다.

## 디렉터리 구조

```
processing/
  app/             # 실행 진입점. 명령줄에서 직접 돌리는 스크립트들
  ingest/          # 파일 수집, 자산 등록, 상태 관리, 배치 실행
  classify/        # 자산의 주제 분류
  extractors/      # 파일에서 기본 정보 추출 (재생 시간, 해상도 등)
  preprocess/      # 영상 키프레임 추출, 음성 인식 등 무거운 전처리
  skills/          # 종류별(텍스트·이미지·영상·오디오) 처리 묶음
  pipeline/        # 처리 단계를 조합하는 틀과 도메인별 설정
  dispatch/        # 자산 종류에 따라 어느 처리를 돌릴지 결정

deploy/airflow/dags/   # Airflow DAG 5개
tests/                 # 단위 테스트
```

새 파일 종류를 지원하려면 `skills/` 에 추가하고 `dispatch/` 에 연결합니다. 처리 순서
자체를 바꾸려면 `pipeline/` 을 봅니다.

## 사용 환경

### 하드웨어

**세 레포 중 이 레포가 가장 무겁습니다.** 메타데이터 추출과 임베딩이 전부 여기서 돌기
때문입니다.

| 구분 | 최소 | 권장 | 넉넉히 |
|---|---|---|---|
| CPU | 4 코어 | 8 코어 | 16 코어 |
| 메모리 | 8 GB | 16 GB | 32 GB |
| GPU | 불필요 | 불필요 | 선택 |

권장 사양의 근거는 실제 처리 경험입니다. 10코어·메모리 16GB 장비에서 자산 20,505건을
처음부터 끝까지 처리했습니다. 운영 클러스터에서는 아래처럼 할당했습니다.

| 구성요소 | 평소 | 최대 | 실제 사용량 |
|---|---|---|---|
| Airflow 스케줄러 (실제 처리가 도는 곳) | 2 코어 · 6 GB | 4 코어 · 12 GB | 4.1 GB |
| PostgreSQL | 0.5 코어 · 1 GB | 2 코어 · 4 GB | 0.1 GB |
| OpenSearch | 0.5 코어 · 3 GB | 2 코어 · 5 GB | 2.6 GB |
| 그 밖 (API · DAG 처리기 · 도구) | 1.4 코어 · 3 GB | 6 코어 · 13 GB | 0.6 GB |
| 합계 | 4.4 코어 · 13 GB | 14 코어 · 34 GB | 7.8 GB |

메모리는 OpenSearch 와 Airflow 스케줄러가 대부분을 씁니다. 검색 엔진은 켜 두기만 해도
2~3GB 를 잡고 있고, 스케줄러는 영상을 처리할 때 순간적으로 크게 씁니다. 메모리가 모자라면
영상 처리 도중에 프로세스가 죽습니다. CPU 보다 메모리를 먼저 늘리십시오.

GPU 는 기본 구성에서 필요 없습니다. 임베딩과 LLM 을 별도 서버에 맡기기 때문입니다. 이
서버에서 직접 모델을 돌리려면(`EMBED_ACTIVE_CHANNEL` 을 로컬로 바꾸거나
`EMBED_ENABLE_CLIP=true`) 그때 GPU 를 검토하십시오. 없어도 CPU 로 동작하지만 느립니다.

디스크는 원본 파일 보관 폴더가 대부분을 차지합니다. 이 폴더는 service 레포와 같은 경로를
공유해야 합니다. 파일 내려받기와 미리보기가 데이터베이스에 적힌 경로를 그대로 읽기
때문입니다.

| 항목 | 자산 1건당 | 1만 건 | 10만 건 |
|---|---|---|---|
| 원본 파일 | 1.6 MB | 16 GB | 160 GB |
| PostgreSQL | 66 KB | 0.7 GB | 6.6 GB |
| OpenSearch 색인 | 6.8 KB | 70 MB | 0.7 GB |
| 모델 캐시 | — | 2~10 GB | 2~10 GB |

모델 캐시는 자산 수와 무관한 고정값이지만 구성에 따라 차이가 큽니다. 임베딩을 별도 서버에
맡기면 음성 인식 모델만 내려받아 2GB 정도, 임베딩과 이미지 모델까지 이 서버에서 돌리면
10GB 를 넘습니다.

### 소프트웨어

| 항목 | 요구 버전 | 개발 확인 |
|---|---|---|
| Python | 3.13 이상 | 3.13.13 |
| Apache Airflow | 3.x | 3.2.1 |
| PostgreSQL | 17 + pgvector 확장 | 17.9 · pgvector 0.8.2 |
| OpenSearch | 3.x | 3.6.0 |
| ffmpeg | — | 8.1.1 |
| tesseract (한국어 데이터 포함) | — | 5.5.2 |
| faster-whisper | — | 1.2.1 |
| core 라이브러리 | v0.7.0 이상 | — |

Airflow 는 `pyproject.toml` 에 넣지 않았습니다. 실행 환경(도커 이미지나 conda)이 제공하는
버전과 충돌하지 않게 하기 위해서입니다.

Airflow 는 자체 데이터베이스를 씁니다. 플랫폼 데이터베이스와 **분리**해야 합니다.

## 설치 방법

```bash
# 1. 시스템 도구
brew install ffmpeg tesseract tesseract-lang              # macOS
sudo apt install ffmpeg tesseract-ocr tesseract-ocr-kor   # Linux

# 2. core 라이브러리
pip install "meta-extract @ git+https://github.com/mobigen-auroraFS-lab/core.git@v0.7.0"

# 3. 이 레포
pip install -e .
```

### 데이터베이스 준비 (최초 1회)

스키마는 core 레포가 관리합니다. 이 레포에는 마이그레이션 도구가 없으므로 core 를 내려받아
실행합니다.

```bash
git clone --branch v0.7.0 https://github.com/mobigen-auroraFS-lab/core.git
cd core
pip install -e ".[migrate]"
alembic -c alembic.ini upgrade head
python -m scripts.seed_topic_registry --env dev --apply
```

마지막 줄을 빠뜨리면 자산 간 관계가 하나도 만들어지지 않습니다. 오류는 나지 않습니다.

## 실행 및 운영 방법

### 설정

core 가 요구하는 값(core README 참고)에 더해 아래를 설정합니다.

| 변수 | 용도 |
|---|---|
| `WATCHER_INBOX_DIR` | 처리할 파일을 넣어 두는 폴더 |
| `WATCHER_ARCHIVE_DIR` | 처리가 끝난 원본을 보관하는 폴더. service 레포와 공유합니다 |
| `META_ENV` | `dev` 또는 `prod` |
| `DAG_PROCESS_LIMIT` | 한 번에 처리할 자산 수 |
| `DAG_PROCESS_MAX_FAILURES` | 연속 실패 허용 횟수 |
| `MM_META_DISCOVERY_MODE` | `propose`(후보만 보고, 기본값) 또는 `auto`(자동 등록) |

설정 파일의 값을 **프로세스 환경변수로 올려야** 합니다. Airflow 작업이 그 환경을
물려받기 때문입니다.

```bash
set -a; . ./.env.dev; set +a
```

### 명령줄에서 직접 돌리기

Airflow 없이 한 단계씩 확인할 때 씁니다.

```bash
# 수집과 처리
python -m processing.app.run_ingest    --env dev <파일 또는 폴더>
python -m processing.app.run_relations --env dev --all
python -m processing.app.run_search    --env dev --query "<검색어>"

# 개체 관련
python -m processing.app.run_mm_classify      --env dev   # 분류 기준으로 자산 판정
python -m processing.app.run_mm_meta_binding  --env dev   # 개체 묶기
python -m processing.app.run_entity_embedding --env dev   # 개체 색인
python -m processing.app.run_entity_label     --env dev   # 개체 라벨 판정

# 색인 복구
python -m processing.app.run_opensearch_resync --env dev              # 빠진 것만 채움
python -m processing.app.run_opensearch_resync --env dev --recreate   # 지우고 다시 만듦
```

재색인 도구는 검색 결과를 합치는 데 쓰는 설정(`assets-hybrid`)도 함께 등록합니다. 이것이
없으면 파일 검색 API 가 500 오류를 냅니다. 이미 직접 관리하고 있다면
`--no-ensure-pipeline` 으로 끌 수 있습니다.

### Airflow 로 돌리기

```bash
set -a; . ./.env.dev; set +a          # 먼저 실행해야 작업이 환경을 물려받습니다
export AIRFLOW_HOME=~/airflow-home
export AIRFLOW__CORE__DAGS_FOLDER=$PWD/deploy/airflow/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export META_ENV=dev

airflow db migrate                    # 최초 1회
airflow scheduler &
airflow dag-processor &
airflow api-server &
```

DAG 는 다섯 개이고 시간을 엇갈려 배치했습니다. 앞 단계가 끝난 뒤 다음이 돌도록 한 것입니다.
각 DAG 는 처리할 것이 남아 있으면 끝에서 자기를 다시 호출하므로, 한 주기에 다 끝내지
못해도 이어서 진행합니다.

| DAG | 하는 일 | 변수 | 기본 주기 |
|---|---|---|---|
| `dag_collect` | 폴더를 살펴 새 파일을 자산으로 등록 | `DAG_COLLECT_SCHEDULE` | 5분마다 |
| `dag_process` | 등록된 자산을 처리하고 색인 | `DAG_PROCESS_SCHEDULE` | 매시 정각 |
| `dag_relations` | 자산 간 관계 생성 | `DAG_RELATIONS_SCHEDULE` | 매시 30분 |
| `dag_mm_meta` | 개체 묶기 | `DAG_MM_META_SCHEDULE` | 매시 45분 |
| `dag_mm_classify` | 분류 기준으로 자산 판정 | `DAG_MM_CLASSIFY_SCHEDULE` | 매시 50분 |

### 운영 시 확인할 것

| 상황 | 할 일 |
|---|---|
| core 를 새 버전으로 올렸을 때 | core 재설치 → 테스트 → 재색인 도구 실행 |
| 파일을 많이 넣기 전 | 개체 제외 목록(`EXCLUDED_ENTITIES`·`STOP_PATTERNS`) 확인. 자동화되지 않는 유일한 단계입니다 |
| `MM_META_DISCOVERY_MODE=auto` 로 바꿨을 때 | 대량 수집이 끝나면 `propose` 로 되돌리기 |
| GPU 나 모델을 쓰는 작업 | Airflow 풀로 동시 실행 수를 1 로 제한 |

재색인과 수집은 여러 번 돌려도 안전합니다. 같은 파일은 해시로 걸러냅니다.

## 실행 예제

```bash
$ python -m processing.app.run_ingest --env dev ./inbox/sample.mp4
[run_ingest] collected=1 registered=1 skipped=0 failed=0

$ python -m processing.app.run_ingest --env dev ./inbox/manifest.json
[run_ingest] collected=0 registered=0 skipped=1 (ledger_file)

$ python -m processing.app.run_opensearch_resync --env dev --recreate
[OpenSearch 복구 재색인] http://<host>:9200 (v3.6.0) → index='assets' channel='st_api' recreate=True
  인덱스 상태: recreated | 색인 성공: 1526 | 오류: 0 | 인덱스 총문서: 1526 | 파이프라인: created

$ python -m processing.app.run_relations --env dev --all
[run_relations] 후보 N쌍 · 제안 M건 · 저장 K건
```

두 번째 예의 `ledger_file` 은 목록 파일이라 자산으로 등록하지 않았다는 뜻입니다.
건수는 데이터에 따라 달라집니다.

## 기타

- **테스트** — `python -m unittest discover -s tests` (662건 · 62건 건너뜀) 와
  `ruff check processing tests`. `.env` 를 환경변수로 올린 셸에서 돌리면 이미지 모델 테스트
  4건이 매번 실패하므로, 테스트는 설정을 올리지 않은 셸에서 돌립니다.

- **core 와의 버전 관계** — core 태그가 먼저 올라간 뒤 이 레포를 맞춥니다.

## 자주 겪는 문제

| 증상 | 원인 |
|---|---|
| `필수 환경변수 누락` | 설정 파일을 환경변수로 올리지 않았습니다 |
| `No module named 'src'` | core 라이브러리가 설치되지 않았습니다 |
| 관계가 하나도 안 생김 | 주제 분류 기초 데이터를 넣지 않았습니다 |
| Airflow 화면에 DAG 가 안 보임 | `DAGS_FOLDER` 경로나 `dag-processor` 실행 여부를 확인하십시오 |
| 색인 설정을 고쳤는데 그대로임 | 재색인 결과가 `analysis-stale` 이면 `--recreate` 로 다시 만드십시오 |

## 제3자 오픈소스

전체 목록과 라이선스 전문은 `NOTICE` 파일에 있습니다.

| 구성요소 | 라이선스 |
|---|---|
| PyTorch · SceneDetect · soundfile | BSD 3-Clause |
| Apache Airflow · OpenCV · pytesseract · sentence-transformers | Apache License 2.0 |
| faster-whisper | MIT |
| psycopg | LGPL 3.0 |

psycopg 는 LGPL 입니다. 파이썬에서 불러 쓰는 것은 이 소프트웨어의 라이선스에 영향을 주지
않지만, 사용 사실을 `NOTICE` 에 밝혀야 합니다.

ffmpeg 와 tesseract 는 이 소프트웨어에 포함되지 않고 실행 환경에 설치해 사용합니다.
각각의 라이선스는 해당 프로젝트를 따릅니다.
