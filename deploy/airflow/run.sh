#!/usr/bin/env bash
# =============================================================================
# 파이프라인(네이티브 Airflow) 실행 제어 — 이 레포 단독으로 올리고 내린다.
#
#   ./run.sh start      # scheduler·dag-processor·api-server 기동
#   ./run.sh stop       # 3종 중지(역순)
#   ./run.sh restart
#   ./run.sh status
#
# 전제(최초 1회): 3레포 editable 설치 + (네이티브) psycopg2-binary·asyncpg 설치
#   + 메타DB 생성(createdb "$AIRFLOW_META_DB").  상세 = 같은 폴더 README §네이티브 실행.
#
# PG 접속값: 환경변수 SQL_ALCHEMY_CONN / POSTGRES_* 가 있으면 그것을 쓰고,
#   없으면 코어 .env.dev($CORE_DIR)에서 '서브셸로만' 읽어 조립(비밀번호 비하드코딩).
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

# conda 활성화(비대화형 셸)
if [[ -r "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"; conda activate "$CONDA_ENV"
else
  echo "경고: conda.sh 없음($CONDA_BASE) — CONDA_BASE 확인" >&2
fi
set -uo pipefail

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

# 메타DB 접속문자열(이미 있으면 그대로) — 없으면 POSTGRES_*/코어 .env.dev 에서 조립
if [[ -z "${AIRFLOW__DATABASE__SQL_ALCHEMY_CONN:-}" ]]; then
  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="$(
    set -a
    if [[ -z "${POSTGRES_USER:-}" && -f "$CORE_DIR/.env.dev" ]]; then
      # shellcheck disable=SC1091
      source "$CORE_DIR/.env.dev" >/dev/null 2>&1 || true
    fi
    printf 'postgresql+psycopg2://%s:%s@%s:%s/%s' \
      "${POSTGRES_USER:-}" "${POSTGRES_PASSWORD:-}" "${POSTGRES_HOST:-localhost}" "${POSTGRES_PORT:-5432}" "$AIRFLOW_META_DB"
  )"
fi
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN

mkdir -p "$RUN_DIR" "$WATCHER_INBOX_DIR" "$WATCHER_ARCHIVE_DIR"

SVCS=(scheduler dag-processor api-server)

# ── 유틸 ─────────────────────────────────────────────────────────────────────
is_running() { local p; [[ -f "$RUN_DIR/$1.pid" ]] && p="$(cat "$RUN_DIR/$1.pid")" 2>/dev/null && kill -0 "$p" 2>/dev/null; }

start_one() {
  local name="$1"; local -a cmd
  case "$name" in
    scheduler)     cmd=(airflow scheduler) ;;
    dag-processor) cmd=(airflow dag-processor) ;;
    api-server)    cmd=(airflow api-server) ;;
    *) echo "  알 수 없는 서비스: $name"; return 1 ;;
  esac
  if is_running "$name"; then printf '  = %-13s 이미 실행 중(pid %s)\n' "$name" "$(cat "$RUN_DIR/$name.pid")"; return 0; fi
  ( cd "$REPO_ROOT" && exec "${cmd[@]}" ) >"$RUN_DIR/$name.log" 2>&1 &
  local pid=$!; echo "$pid" >"$RUN_DIR/$name.pid"
  printf '  ▶ %-13s pid %-7s → %s\n' "$name" "$pid" "$RUN_DIR/$name.log"
}
stop_one() {
  local name="$1"                       # name 먼저 선언(set -u: 같은 local 문서 $name 자기참조 금지)
  local pidf="$RUN_DIR/$name.pid" pid
  if [[ ! -f "$pidf" ]]; then printf '  - %-13s pid 없음(미기동?)\n' "$name"; return; fi
  pid="$(cat "$pidf")"
  if ! kill -0 "$pid" 2>/dev/null; then printf '  - %-13s 이미 종료\n' "$name"; rm -f "$pidf"; return; fi
  printf '  ▪ %-13s 종료(pid %s)…' "$name" "$pid"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  if kill -0 "$pid" 2>/dev/null; then printf ' 강제(SIGKILL)'; kill -KILL "$pid" 2>/dev/null || true; fi
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
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; sleep 1; do_start ;;
  status)  do_status ;;
  *) echo "사용법: $(basename "$0") <start|stop|restart|status>"; exit 2 ;;
esac
