#!/usr/bin/env bash
# =============================================================================
# 파이프라인(네이티브 Airflow) 실행 제어 — 이 레포 단독으로 올리고 내린다.
#
#   ./run.sh start      # scheduler·dag-processor·api-server 기동
#   ./run.sh stop       # 3종 중지(역순)
#   ./run.sh restart
#   ./run.sh status
#
# 흐름에서의 위치: 이 스크립트가 **태스크 실행 환경의 출발점**이다. LocalExecutor 태스크는
#   scheduler 의 자식이라 여기서 export 한 것을 그대로 물려받는다 — 앱 설정(POSTGRES_*·
#   META_MODEL 등)도, 아래 macOS 회피책도 여기 없으면 태스크까지 닿지 않는다.
#
# 전제(최초 1회): 3레포 editable 설치 + (네이티브) psycopg2-binary·asyncpg 설치
#   + 메타DB 생성(createdb "$AIRFLOW_META_DB").  상세 = 같은 폴더 README §네이티브 실행.
#
# 비밀번호를 파일·로그에 남기지 않는다: PG 접속값은 환경변수가 있으면 그것을 쓰고, 없으면
#   코어 .env.dev 를 **서브셸 안에서만** 읽어 조립한다(값이 이 셸로 새지 않는다).
#
# 깨지면 안 되는 것
#   · conda.sh 소싱은 `set -u` 앞에 — 뒤로 옮기면 소싱 자체가 깨진다.
#   · 기동은 프로세스그룹 리더로, 종료는 그룹째 — 안 그러면 태스크가 고아로 남는다.
#   · start/stop 은 락으로 직렬화 — 동시에 돌면 pid 파일이 엇갈린다.
# =============================================================================
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"       # .../deploy/airflow
REPO_ROOT="$(cd "$HERE/../.." && pwd)"                      # 파이프 레포 루트
WS_ROOT="$(cd "$REPO_ROOT/.." && pwd)"                      # 레포들이 모인 상위 폴더

# ── 설정(환경변수로 override 가능) ───────────────────────────────────────────
CONDA_BASE="${CONDA_BASE:-/opt/anaconda3}"
CONDA_ENV="${CONDA_ENV:-AuroraFS}"
CORE_DIR="${CORE_DIR:-$WS_ROOT/dataplatform-core}"      # PG 접속값 원본(.env.dev)
DATA_ROOT="${DATA_ROOT:-$WS_ROOT/dataplatform-data}"         # 공유 데이터 루트(inbox/archive)
AIRFLOW_HOME="${AIRFLOW_HOME:-$HOME/airflow-meta}"
AIRFLOW_META_DB="${AIRFLOW_META_DB:-airflow_native}"        # 기존 실사용 'airflow' 와 분리(충돌 방지)
RUN_DIR="${RUN_DIR:-$HOME/.dataflatform/pipeline}"          # pid·log
# ────────────────────────────────────────────────────────────────────────────

# conda 활성화(비대화형 셸). ※ conda.sh 소싱은 set -u 앞에 둔다 — conda.sh 내부가 미정의 변수를
#   참조해 -u 를 먼저 켜면 소싱 자체가 깨진다(순서 중요).
if [[ -r "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  # ⚠️ activate 실패를 조용히 넘기지 않는다 — 넘기면 시스템 python 으로 진행해
  #    "왜 import 가 안 되지" 로 한참 헤매게 된다.
  conda activate "$CONDA_ENV" || { echo "오류: conda 환경 활성 실패($CONDA_ENV) — 환경 존재/이름 확인" >&2; exit 1; }
else
  echo "경고: conda.sh 없음($CONDA_BASE) — CONDA_BASE 확인(사전 활성화된 env 로 진행)" >&2
fi
set -uo pipefail

# ── 앱 레벨 env 주입(META_MODEL·POSTGRES_*·EMBED_*·OPENSEARCH_* 등) ──────────────
# DAG 태스크는 init_settings/PostgresUtil 이 os.environ 에서 읽으므로 앱 설정을 Airflow 프로세스
# 환경에 실어야 한다 — 아래 SQL 조립 $(...) 서브셸의 source 는 그 안에만 갇혀 프로세스로 새지 않아
# META_MODEL 등이 빠지면 collect/process 태스크가 init_settings 에서 즉사한다. 코어 .env.dev 를
# set -a 로 export(없으면 운영자 사전 export 로 보고 경고만). 이 아래 명시 export(META_ENV=dev·
# WATCHER_* 등)가 뒤에 와서 프로파일·경로는 run.sh 가 authoritative.
if [[ -f "$CORE_DIR/.env.dev" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$CORE_DIR/.env.dev" >/dev/null 2>&1 || true
  set +a
else
  echo "경고: $CORE_DIR/.env.dev 없음 — DAG 의 init_settings 가 META_MODEL 부재로 실패할 수 있음" >&2
fi

# Airflow(네이티브) 환경변수
export AIRFLOW_HOME
export AIRFLOW__CORE__EXECUTOR=LocalExecutor
export AIRFLOW__CORE__DAGS_FOLDER="$HERE/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_ALL_ADMINS=True   # dev 로그인 없음(운영 노출 시 끈다)
export META_ENV=dev
export DATA_ROOT
export WATCHER_INBOX_DIR="$DATA_ROOT/inbox"
export WATCHER_ARCHIVE_DIR="$DATA_ROOT/archive"

# ── macOS 전용 회피책(태스크 무한 정지) ──────────────────────────────────────
# ⚠️ **macOS 에서만 건다.** 셋 다 Apple 프레임워크가 원인인 증상의 회피책이라 Linux 에는 의미가
#    없고, 특히 no_proxy 는 **무해하지 않다** — 외부 프록시를 거쳐야 하는 배포에서 프록시를
#    꺼 버린다. Linux 의 getproxies() 는 애초에 시스템 조회(_scproxy)를 하지 않는다.
#
# ⚠️ **여기서 export 해야 한다.** LocalExecutor 태스크는 scheduler 의 자식이라 부모 환경을
#    물려받는다. DAG 파일 안에서 os.environ 에 넣으면 늦다 — BLAS 스레드 수는 numpy 가
#    로드되는 순간 읽힌다.
#
# 증상(2026-07-28 실측·macOS): dag_process·dag_relations 태스크가 CPU 100% 로 60분 넘게 무진행.
#   SIGTERM 도 안 먹었다(네이티브 호출 안에 갇혀 파이썬 신호 핸들러가 못 돈다 → SIGKILL 필요).
#   같은 파일을 CLI 단독으로 돌리면 158초에 정상 완료 — 데이터가 아니라 실행 환경 문제다.
#   스택 최심부는 둘 다 fork-safe 하지 않은 Apple 프레임워크였다:
#     · dag_process   : numpy matmul → cblas_sgemm → dispatch_apply  (faster-whisper 멜 스펙트로그램)
#     · dag_relations : _scproxy.get_proxies → SystemConfiguration   (HTTP 요청 전 프록시 조회)
if [[ "$(uname -s)" == "Darwin" ]]; then
  # Accelerate(BLAS)의 dispatch_apply 병렬 분배를 끈다 — 멈추던 경로 자체를 안 타게 한다.
  # ※ OMP_NUM_THREADS 는 **일부러 건드리지 않는다**. 멈춘 곳은 Accelerate 이지 OpenMP 가 아니고,
  #   1 로 묶으면 정상 동작 중인 whisper 추론(ctranslate2)까지 직렬화돼 STT 가 크게 느려진다.
  export VECLIB_MAXIMUM_THREADS=1
  # HTTP 요청마다 macOS 시스템 프록시를 조회(_scproxy)하지 않게 한다. getproxies() 가 환경변수를
  # 먼저 보고 결과가 비지 않으면 시스템 조회로 내려가지 않는다. 호출 대상이 전부 localhost
  # (OpenSearch·온프레미스 LLM)라 프록시를 건너뛰어 잃는 것이 없다.
  export no_proxy='*'
  export NO_PROXY='*'
  # 자식 프로세스에서 ObjC 런타임 초기화 검사를 끈다(위 두 증상과 같은 뿌리).
  export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
  echo "  · macOS 회피책 적용: VECLIB_MAXIMUM_THREADS=1 · no_proxy=* · OBJC_DISABLE_INITIALIZE_FORK_SAFETY"
fi

# 메타DB 접속문자열(이미 있으면 그대로) — 없으면 POSTGRES_*/코어 .env.dev 에서 조립
if [[ -z "${AIRFLOW__DATABASE__SQL_ALCHEMY_CONN:-}" ]]; then
  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="$(
    set -a
    if [[ -z "${POSTGRES_USER:-}" && -f "$CORE_DIR/.env.dev" ]]; then
      # shellcheck disable=SC1091
      source "$CORE_DIR/.env.dev" >/dev/null 2>&1 || true
    fi
    # ⚠️ user·password 를 URL-인코딩한다 — 특수문자(@ : / # 등) 비밀번호가 접속 URL 을 깨뜨리지
    #   않게(예 p@ss → p%40ss). host/port/db 는 자격증명이 아니라 그대로. conda python 으로 조립(shell 값 명시 전달).
    POSTGRES_USER="${POSTGRES_USER:-}" POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-}" \
    POSTGRES_HOST="${POSTGRES_HOST:-localhost}" POSTGRES_PORT="${POSTGRES_PORT:-5432}" \
    AIRFLOW_META_DB="$AIRFLOW_META_DB" python - <<'PY'
import os, urllib.parse as up
enc = lambda s: up.quote(s, safe="")
print("postgresql+psycopg2://%s:%s@%s:%s/%s" % (
    enc(os.environ["POSTGRES_USER"]), enc(os.environ["POSTGRES_PASSWORD"]),
    os.environ["POSTGRES_HOST"], os.environ["POSTGRES_PORT"], os.environ["AIRFLOW_META_DB"]))
PY
  )"
fi
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN

mkdir -p "$RUN_DIR" "$WATCHER_INBOX_DIR" "$WATCHER_ARCHIVE_DIR"

SVCS=(scheduler dag-processor api-server)

# ── 유틸 ─────────────────────────────────────────────────────────────────────
# 동시 start/stop 직렬화용 락. mkdir 을 쓰는 이유는 '있으면 실패'가 원자적이기 때문이다 —
# flock 은 macOS 에 없다.
_LOCK_DIR="$RUN_DIR/.lock"
acquire_lock() {
  if ! mkdir "$_LOCK_DIR" 2>/dev/null; then
    echo "다른 run.sh 가 실행 중입니다(락: $_LOCK_DIR). 끝나길 기다리거나, 비정상 종료로 남았으면 그 폴더를 지우세요." >&2
    exit 1
  fi
  trap 'rmdir "$_LOCK_DIR" 2>/dev/null || true' EXIT   # 스크립트 종료 시 자동 해제
}

# ⚠️ pid 가 살아있는 것만으로는 부족하다 — 서비스가 죽은 뒤 OS 가 그 pid 를 무관 프로세스에
#   재할당했을 때 '실행 중' 오인이나 stop 의 남의 프로세스 kill 을 막는다.
_is_our_airflow() {
  local p="$1"
  kill -0 "$p" 2>/dev/null || return 1
  ps -p "$p" -o command= 2>/dev/null | grep -q "airflow" || return 1
}

is_running() {
  local p
  [[ -f "$RUN_DIR/$1.pid" ]] || return 1
  p="$(cat "$RUN_DIR/$1.pid" 2>/dev/null)" || return 1
  _is_our_airflow "$p"
}

start_one() {
  local name="$1"; local -a cmd
  case "$name" in
    scheduler)     cmd=(airflow scheduler) ;;
    dag-processor) cmd=(airflow dag-processor) ;;
    api-server)    cmd=(airflow api-server) ;;
    *) echo "  알 수 없는 서비스: $name"; return 1 ;;
  esac
  if is_running "$name"; then printf '  = %-13s 이미 실행 중(pid %s)\n' "$name" "$(cat "$RUN_DIR/$name.pid")"; return 0; fi
  # nohup(SIGHUP 무시) + </dev/null(터미널 stdin 분리) + set -m(자기 프로세스그룹 리더로) 로 띄운다 —
  #   터미널·SSH 종료(SIGHUP)에도 살아남고, 기록한 pid 가 곧 프로세스그룹 리더라 stop 이 그룹째 종료(C4)할 수 있다.
  #   set -m 은 job control 이므로 tty 가 필요(대화형 `./run.sh start`). tty 없는 비대화형(cron 등)에선 조용히
  #   생략되고 일반 백그라운드로 뜬다 — 그 경우 stop 이 단일 pid 로 폴백해 여전히 종료된다(자식 회수만 못함).
  set -m 2>/dev/null || true
  ( cd "$REPO_ROOT" && exec nohup "${cmd[@]}" ) >"$RUN_DIR/$name.log" 2>&1 </dev/null &
  local pid=$!
  set +m 2>/dev/null || true
  echo "$pid" >"$RUN_DIR/$name.pid"
  printf '  ▶ %-13s pid %-7s → %s\n' "$name" "$pid" "$RUN_DIR/$name.log"
}
stop_one() {
  local name="$1"                       # name 먼저 선언(set -u: 같은 local 문서 $name 자기참조 금지)
  local pidf="$RUN_DIR/$name.pid" pid
  if [[ ! -f "$pidf" ]]; then printf '  - %-13s pid 없음(미기동?)\n' "$name"; return; fi
  pid="$(cat "$pidf")"
  # 살아있는 '우리 airflow' 가 아니면(종료됨 또는 재할당된 무관 pid) 파일만 정리하고 끝낸다.
  # ⚠️ 이 확인을 빼면 **남의 프로세스를 죽인다**.
  if ! _is_our_airflow "$pid"; then printf '  - %-13s 이미 종료(또는 무관 pid)\n' "$name"; rm -f "$pidf"; return; fi
  printf '  ▪ %-13s 종료(pid %s)…' "$name" "$pid"
  # ⚠️ 단일 pid 가 아니라 **프로세스그룹째** 종료한다 — LocalExecutor 태스크·gunicorn 워커 등
  #    자식까지 회수해야 한다. 부모만 죽이면 태스크가 고아로 남아 계속 돈다.
  #   그룹 종료가 안 되는 구버전 pid(비-리더)면 단일 pid 로 폴백.
  kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then
    printf ' 강제(SIGKILL)'; kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  printf ' 완료\n'; rm -f "$pidf"
}

do_start() {
  echo "═══ 파이프라인 기동 ═══"
  if ! python - <<'PY'
import importlib.util as u, sys
miss=[m for m in ("airflow","asyncpg","psycopg2","src","processing") if u.find_spec(m) is None]
if miss: print("  누락 모듈:", ", ".join(miss)); print("  → 네이티브 설치(asyncpg 포함)+3레포 editable 설치 필요"); sys.exit(1)
PY
  then return 1; fi
  [[ "$AIRFLOW_META_DB" == "airflow" ]] && echo "  ⚠️ 메타DB 가 공유 'airflow' — 기존 Airflow 와 충돌 위험(AIRFLOW_META_DB=airflow_native 권장)"
  echo "  airflow db migrate + pool(gpu=1)  (메타DB=$AIRFLOW_META_DB)…"
  if ! airflow db migrate >"$RUN_DIR/db_migrate.log" 2>&1; then
    echo "  db migrate 실패(로그 $RUN_DIR/db_migrate.log) — 최초 1회: createdb \"$AIRFLOW_META_DB\""; return 1
  fi
  airflow pools set gpu 1 "단일 추론 직렬화(dag_process 동시 1)" >/dev/null 2>&1 || true
  for s in "${SVCS[@]}"; do start_one "$s"; done
  sleep 3
  for s in "${SVCS[@]}"; do
    if is_running "$s"; then printf '  ✔ %-13s (pid %s)\n' "$s" "$(cat "$RUN_DIR/$s.pid")"
    else printf '  ✗ %-13s 즉시 종료 — 로그:\n' "$s"; tail -n 3 "$RUN_DIR/$s.log" 2>/dev/null | sed 's/^/      /'; fi
  done
  echo "  · Airflow UI : http://localhost:8080  (로그인 없음 — dev)"
  echo "  · DAG 활성화(수동): (cd \"$REPO_ROOT\" && airflow dags unpause dag_collect dag_process dag_relations)"
}
do_stop() {
  echo "═══ 파이프라인 중지 ═══"
  for (( i=${#SVCS[@]}-1; i>=0; i-- )); do stop_one "${SVCS[$i]}"; done
}
do_status() {
  echo "═══ 파이프라인 상태 ═══"
  for s in "${SVCS[@]}"; do
    if is_running "$s"; then printf '  ✔ %-13s (pid %s)\n' "$s" "$(cat "$RUN_DIR/$s.pid")"
    else printf '  ✗ %-13s (중지)\n' "$s"; fi
  done
}

case "${1:-}" in
  start)   acquire_lock; do_start ;;
  stop)    acquire_lock; do_stop ;;
  restart) acquire_lock; do_stop; sleep 1; do_start ;;
  status)  do_status ;;                 # 읽기 전용 — 락 불요
  *) echo "사용법: $(basename "$0") <start|stop|restart|status>"; exit 2 ;;
esac
