"""030 G3(T008) — Airflow DAG 무결성·얇은 래퍼 단위 테스트 [US5·FR-011·SC-008].

목적
    `deploy/airflow/dags/` 의 세 DAG(dag_collect·dag_process·dag_relations)이
      1. **Airflow 미기동(메타DB·스케줄러 없이)에서 DagBag 파싱만으로** 무오류로 로드되고
         (import_errors 0·cycle 0·3개 DAG 존재·max_active_runs==1·catchup==False),
      2. **각 태스크 callable 이 G1/G2 순수 함수의 얇은 래퍼**임(collect_file·process_received_batch·
         scan_unresolved_assets+run_relations 를 호출하고 DAG 에 비즈니스 로직 0 — FR-011)
    을 단언한다.

DagBag 파싱은 임시 ``AIRFLOW_HOME`` + ``LOAD_EXAMPLES=False`` 로 메타DB·스케줄러 없이 수행한다
    (SC-008: Airflow 미기동 단위 테스트). 실 DB·LLM·파일·모델 불필요. 래퍼 검증은 callable 을
    DagBag 에서 꺼내 ``init_settings``/``PostgresUtil``/G1·G2 함수를 모킹한 뒤 호출해, 위임 호출이
    실제로 일어남을 확인한다(소스 문자열 검사보다 강함).

Airflow 가 설치되지 않은 환경(거버넌스상 apache-airflow 미설치)에서는 전 클래스를 skip 한다 —
    헌법 8조 회귀 0(SC-010): 단위 집합은 의존 부재에도 green 을 유지한다.
"""

from __future__ import annotations

import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

# Airflow 가용성 게이트 — DagBag 파싱 전에 임시 AIRFLOW_HOME 을 세팅해야 config 가 그 경로로 초기화된다.
_AIRFLOW_HOME = tempfile.mkdtemp(prefix="airflow_dagbag_test_")
os.environ.setdefault("AIRFLOW_HOME", _AIRFLOW_HOME)
os.environ.setdefault("AIRFLOW__CORE__LOAD_EXAMPLES", "False")

try:
    from airflow.dag_processing.dagbag import (
        DagBag,  # Airflow 3.x 신경로(models.dagbag 는 deprecated)
    )

    _AIRFLOW_AVAILABLE = True
except Exception:  # noqa: BLE001 — airflow 미설치/임포트 실패 시 전 테스트 skip(회귀 0)
    _AIRFLOW_AVAILABLE = False

# 프로젝트 루트 기준 DAG 폴더(테스트가 어디서 실행돼도 절대경로). dags 는 repo 정본 —
#   네이티브 Airflow(run.sh)가 AIRFLOW__CORE__DAGS_FOLDER 로 이 dags 를 직접 가리킨다(도커 제거·소스=repo).
_DAG_FOLDER = str(Path(__file__).resolve().parents[1] / "deploy" / "airflow" / "dags")

_EXPECTED_DAGS = {"dag_collect", "dag_process", "dag_relations"}
_COLLECT_TASK = "collect_inbox"
_PROCESS_TASK = "process_batch"
_RELATIONS_TASK = "propose_relations"

_AID = uuid.UUID("018f0000-0000-7000-8000-000000000010")

_dagbag_cache: dict[str, object] = {}


def _dagbag():
    """임시 AIRFLOW_HOME·LOAD_EXAMPLES=False 로 DagBag 1회 파싱(메타DB 불요·캐시 재사용).

    Airflow 3.0 은 ``DagBag.__init__`` 에서 ``include_examples``/``safe_mode`` 인자를 제거했다
    (예제 로드는 ``AIRFLOW__CORE__LOAD_EXAMPLES`` env 로 제어 — 위 모듈 상단에서 False 설정).
    ``requirements.txt`` 가 ``apache-airflow>=2.9.0`` 로 핀 없이 최신(3.x)을 당기므로, 2.x/3.x 양쪽
    호환을 위해 **시그니처가 실제로 받는 kwarg 만** 전달한다(3.x 에서 사라진 인자는 자동 생략).
    """
    if "bag" not in _dagbag_cache:
        import inspect

        params = inspect.signature(DagBag.__init__).parameters
        kwargs: dict[str, object] = {"dag_folder": _DAG_FOLDER}
        if "include_examples" in params:
            kwargs["include_examples"] = False
        if "safe_mode" in params:
            kwargs["safe_mode"] = False
        _dagbag_cache["bag"] = DagBag(**kwargs)
    return _dagbag_cache["bag"]


def _callable(dag_id: str, task_id: str):
    return _dagbag().dags[dag_id].get_task(task_id).python_callable


def _fake_db(*, fetchone=None):
    """가짜 PostgresUtil — ``with db:`` · ``with db.transaction() as conn:`` · ``with conn.cursor() as cur``
    컨텍스트를 모두 지원하고 cur.fetchone/fetchall 을 제어한다(실 DB 불요)."""
    db = mock.MagicMock(name="db")
    db.__enter__.return_value = db
    db.__exit__.return_value = False
    txn = db.transaction.return_value
    conn = txn.__enter__.return_value
    txn.__exit__.return_value = False
    curcm = conn.cursor.return_value
    cur = curcm.__enter__.return_value
    curcm.__exit__.return_value = False
    cur.fetchone.return_value = fetchone
    cur.fetchall.return_value = []
    return db, conn, cur


def _fake_context(xcom_value):
    """게이트 callable 용 가짜 태스크 컨텍스트 — ``context['ti'].xcom_pull(...)`` 가 주어진 값을 돌려준다."""
    ti = mock.MagicMock(name="ti")
    ti.xcom_pull.return_value = xcom_value
    return {"ti": ti}


@unittest.skipUnless(_AIRFLOW_AVAILABLE, "apache-airflow 미설치 — DagBag 파싱 불가")
class TestDagBagIntegrity(unittest.TestCase):
    """SC-008: Airflow 미기동에서 DagBag 파싱만으로 세 DAG 가 무오류·안전 가드 유지."""

    def test_no_import_errors(self) -> None:
        # import_errors 에는 파싱 실패·DAG cycle(AirflowDagCycleException)이 모두 기록된다 →
        # 빈 dict 이면 import 오류 0 + cycle 0 을 함께 보증한다.
        self.assertEqual(_dagbag().import_errors, {})

    def test_three_expected_dags_present(self) -> None:
        self.assertTrue(_EXPECTED_DAGS.issubset(set(_dagbag().dags)),
                        msg=f"누락: {_EXPECTED_DAGS - set(_dagbag().dags)}")

    def test_singleton_run_and_no_catchup(self) -> None:
        bag = _dagbag()
        for dag_id in _EXPECTED_DAGS:
            dag = bag.dags[dag_id]
            # 단일 GPU·중복 처리 차단 — 동시 1 run·과거 미실행 보충 금지(FR-010·US4·US3).
            self.assertEqual(dag.max_active_runs, 1, msg=f"{dag_id} max_active_runs")
            self.assertFalse(dag.catchup, msg=f"{dag_id} catchup")
            self.assertIsNotNone(dag.schedule, msg=f"{dag_id} schedule")

    def test_dag_task_structure(self) -> None:
        # (a) push 체이닝 — collect/process/relations 는 [래퍼 → 게이트(ShortCircuit) → 다음/자기 DAG 트리거].
        #     067: 연속 드레인 self-retrigger — process/relations 는 처리분이 남으면 자기 자신을 재트리거
        #     (gate_more_*→trigger_more). 게이트는 신규 산출이 있을 때만 트리거를 통과시킨다(빈 처리 회피).
        bag = _dagbag()
        expected_tasks = {
            # 061: dag_collect +gate_inbox_nonempty(인입 게이트), dag_process +archive_processed(꼬리 아카이브).
            # 067: dag_process +gate_more_received/trigger_more · dag_relations +gate_more_unresolved/trigger_more(self-retrigger).
            "dag_collect": {"gate_inbox_nonempty", _COLLECT_TASK, "gate_new_received", "trigger_process"},
            "dag_process": {_PROCESS_TASK, "gate_new_registered", "trigger_relations", "archive_processed",
                            "gate_more_received", "trigger_more"},
            "dag_relations": {_RELATIONS_TASK, "gate_more_unresolved", "trigger_more"},
        }
        for dag_id, tasks in expected_tasks.items():
            self.assertEqual({t.task_id for t in bag.dags[dag_id].tasks}, tasks,
                             msg=f"{dag_id} 태스크 구성")
        # 기본 래퍼는 여전히 python_callable 을 가진 얇은 래퍼(FR-011).
        for dag_id, task_id in (("dag_collect", _COLLECT_TASK), ("dag_process", _PROCESS_TASK),
                                ("dag_relations", _RELATIONS_TASK)):
            task = bag.dags[dag_id].get_task(task_id)
            self.assertTrue(callable(getattr(task, "python_callable", None)),
                            msg=f"{dag_id}.{task_id} python_callable")

    def test_push_chaining_to_next_dag(self) -> None:
        # (a) TriggerDagRunOperator push 체이닝 — collect→process→relations. 게이트가 빈 산출에 트리거를 막고,
        #     cron 은 안전망으로 유지(트리거 유실·크래시 시 다음 주기에 PG 상태로 복구 — self-healing).
        bag = _dagbag()
        coll = bag.dags["dag_collect"]
        self.assertEqual(coll.get_task("trigger_process").trigger_dag_id, "dag_process")
        # collect_inbox → gate_new_received → trigger_process (게이트가 신규 received 없으면 트리거 스킵).
        self.assertIn("gate_new_received", coll.get_task(_COLLECT_TASK).downstream_task_ids)
        self.assertIn("trigger_process", coll.get_task("gate_new_received").downstream_task_ids)
        proc = bag.dags["dag_process"]
        self.assertEqual(proc.get_task("trigger_relations").trigger_dag_id, "dag_relations")
        self.assertIn("gate_new_registered", proc.get_task(_PROCESS_TASK).downstream_task_ids)
        self.assertIn("trigger_relations", proc.get_task("gate_new_registered").downstream_task_ids)
        # 067 self-retrigger — process → gate_more_received → trigger_more(dag_process 자기 재트리거).
        self.assertEqual(proc.get_task("trigger_more").trigger_dag_id, "dag_process")
        self.assertIn("gate_more_received", proc.get_task(_PROCESS_TASK).downstream_task_ids)
        self.assertIn("trigger_more", proc.get_task("gate_more_received").downstream_task_ids)
        # 067 self-retrigger — relations 는 종단이 아니라 propose → gate_more_unresolved → trigger_more(dag_relations 자기 재트리거).
        rel = bag.dags["dag_relations"]
        self.assertEqual(rel.get_task("trigger_more").trigger_dag_id, "dag_relations")
        self.assertIn("gate_more_unresolved", rel.get_task(_RELATIONS_TASK).downstream_task_ids)
        self.assertIn("trigger_more", rel.get_task("gate_more_unresolved").downstream_task_ids)

    def test_archive_wiring(self) -> None:
        # 061: 인입 게이트 → collect_inbox(빈 인입서 collect 스킵), process_batch → archive_processed(꼬리·병렬).
        bag = _dagbag()
        coll = bag.dags["dag_collect"]
        self.assertIn(_COLLECT_TASK, coll.get_task("gate_inbox_nonempty").downstream_task_ids)
        proc = bag.dags["dag_process"]
        self.assertIn("archive_processed", proc.get_task(_PROCESS_TASK).downstream_task_ids)
        # archive_processed 는 GPU pool 미부착(파일 이동·UPDATE 뿐).
        self.assertNotEqual(proc.get_task("archive_processed").pool, "gpu")

    def test_process_dag_pins_gpu_pool(self) -> None:
        # 단일 GPU OOM 차단 — dag_process 태스크는 크기 1 Pool('gpu')에 묶인다(FR-010).
        task = _dagbag().dags["dag_process"].get_task(_PROCESS_TASK)
        self.assertEqual(task.pool, "gpu")

    def test_dag_modules_keep_top_level_light(self) -> None:
        # 모듈 최상위는 정의만(스케줄러가 자주 파싱) — 무거운 src 심볼(설정·DB·배치)은 callable 안에서
        # 런타임 import 해야 한다. 모듈 전역에 노출돼 있으면 top-level 에서 import 된 것이라 위반.
        import inspect
        import sys

        heavy = ("PostgresUtil", "init_settings", "process_received_batch",
                 "run_relations", "collect_file", "scan_unresolved_assets")
        for dag_id, task_id in (
            ("dag_collect", _COLLECT_TASK),
            ("dag_process", _PROCESS_TASK),
            ("dag_relations", _RELATIONS_TASK),
        ):
            cb = _callable(dag_id, task_id)
            module = inspect.getmodule(cb) or sys.modules.get(cb.__module__)
            self.assertIsNotNone(module, msg=f"{dag_id} 모듈 해소")
            for name in heavy:
                self.assertNotIn(name, vars(module),
                                 msg=f"{dag_id}: '{name}' 이 모듈 최상위에 노출됨(런타임 import 여야)")


@unittest.skipUnless(_AIRFLOW_AVAILABLE, "apache-airflow 미설치 — DagBag 파싱 불가")
class TestThinWrappers(unittest.TestCase):
    """FR-011: 각 DAG 태스크 callable 이 G1/G2 순수 함수를 호출하는 얇은 래퍼임을 호출로 증명."""

    def test_collect_callable_calls_collect_file_per_inbox_file(self) -> None:
        cb = _callable("dag_collect", _COLLECT_TASK)
        db, _conn, _cur = _fake_db(fetchone=None)  # fs_path 미존재 → 두 파일 모두 수집 진행
        with mock.patch.dict(os.environ, {"WATCHER_INBOX_DIR": "/inbox", "WATCHER_ARCHIVE_DIR": "/archive", "META_ENV": "dev"}), \
                mock.patch("src.config.settings.init_settings") as m_init, \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch("processing.ingest.collector.collect_files",
                           return_value=["/inbox/a.txt", "/inbox/b.txt"]) as m_files, \
                mock.patch("processing.ingest.pipeline_steps.collect_file") as m_collect:
            cb()
        m_init.assert_called_once()
        m_files.assert_called_once()
        # 인입 각 파일마다 G1 collect_file(conn, path) 호출 — 래퍼는 루프·트랜잭션만, 수집 로직 0.
        self.assertEqual(m_collect.call_count, 2)
        self.assertEqual([c.args[1] for c in m_collect.call_args_list],
                         ["/inbox/a.txt", "/inbox/b.txt"])

    def test_collect_callable_skips_already_collected(self) -> None:
        cb = _callable("dag_collect", _COLLECT_TASK)
        db, _conn, _cur = _fake_db(fetchone=(1,))  # fs_path 이미 존재 → 멱등 스킵(FR-008)
        with mock.patch.dict(os.environ, {"WATCHER_INBOX_DIR": "/inbox", "WATCHER_ARCHIVE_DIR": "/archive", "META_ENV": "dev"}), \
                mock.patch("src.config.settings.init_settings"), \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch("processing.ingest.collector.collect_files",
                           return_value=["/inbox/a.txt"]), \
                mock.patch("processing.ingest.pipeline_steps.collect_file") as m_collect:
            cb()
        m_collect.assert_not_called()

    def test_collect_counts_only_created_received(self) -> None:
        # 해시 dup·누락(collect_file → asset_id=None)은 collected 에 세지 않는다 → 게이트가 헛 트리거 안 함.
        cb = _callable("dag_collect", _COLLECT_TASK)
        db, _conn, _cur = _fake_db(fetchone=None)  # fs_path 미존재 → collect_file 까지 진행
        with mock.patch.dict(os.environ, {"WATCHER_INBOX_DIR": "/inbox", "WATCHER_ARCHIVE_DIR": "/archive", "META_ENV": "dev"}), \
                mock.patch("src.config.settings.init_settings"), \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch("processing.ingest.collector.collect_files",
                           return_value=["/inbox/a.txt", "/inbox/b.txt"]), \
                mock.patch("processing.ingest.pipeline_steps.collect_file",
                           return_value=mock.Mock(asset_id=None, skip_reason=None)) as m_collect:
            collected = cb()
        self.assertEqual(m_collect.call_count, 2)   # 두 파일 다 시도는 함
        self.assertEqual(collected, 0)              # 생성 0(전부 dup/누락) → 트리거 게이트 False

    def test_collect_archives_duplicate_to_dup_dir(self) -> None:
        # 061(C2): 중복(skip_reason=duplicate:...)은 archive/dup 로 즉시 이동(재해싱 churn 제거).
        cb = _callable("dag_collect", _COLLECT_TASK)
        db, _c, _cur = _fake_db(fetchone=None)
        with mock.patch.dict(os.environ, {"WATCHER_INBOX_DIR": "/inbox", "WATCHER_ARCHIVE_DIR": "/archive", "META_ENV": "dev"}), \
                mock.patch("src.config.settings.init_settings"), \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch("processing.ingest.collector.collect_files", return_value=["/inbox/dup.txt"]), \
                mock.patch("processing.ingest.pipeline_steps.collect_file",
                           return_value=mock.Mock(asset_id=None, skip_reason="duplicate:abc")), \
                mock.patch("processing.ingest.archiver.execute_move") as m_move:
            cb()
        m_move.assert_called_once()
        self.assertEqual(m_move.call_args.args[0], "/inbox/dup.txt")     # src=인입 파일
        self.assertIn("/archive/dup/", m_move.call_args.args[1])         # dest=archive/dup/
        self.assertTrue(m_move.call_args.args[1].endswith("dup.txt"))

    def test_inbox_gate_true_with_files_false_when_empty(self) -> None:
        # 061(C5): gate_inbox_nonempty 는 인입에 파일이 있을 때만 True(빈 인입서 collect 스킵).
        gate = _callable("dag_collect", "gate_inbox_nonempty")
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ, {"WATCHER_INBOX_DIR": d}):
                self.assertFalse(gate())                                 # 빈 인입 → False
            with open(os.path.join(d, "x.txt"), "w") as f:
                f.write("x")
            with mock.patch.dict(os.environ, {"WATCHER_INBOX_DIR": d}):
                self.assertTrue(gate())                                  # 파일 존재 → True

    def test_process_callable_calls_process_received_batch(self) -> None:
        from processing.ingest.batch_runner import BatchReport

        cb = _callable("dag_process", _PROCESS_TASK)
        db, _c, _cur = _fake_db()
        with mock.patch.dict(os.environ, {"META_ENV": "dev"}), \
                mock.patch("src.config.settings.init_settings") as m_init, \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch("processing.ingest.batch_runner.process_received_batch",
                           return_value=BatchReport()) as m_batch:
            cb()
        m_init.assert_called_once()
        m_batch.assert_called_once()
        # db 를 첫 인자로, 배치 한도·cap·고착 임계를 키워드로 전달(얇은 래퍼·배치 로직 0).
        self.assertIs(m_batch.call_args.args[0], db)
        for kw in ("limit", "max_failures", "older_than_s"):
            self.assertIn(kw, m_batch.call_args.kwargs)

    def test_process_callable_passes_settings_for_inline_opensearch(self) -> None:
        # 코드리뷰 #1(FR-002·US1§2): dag_process 자동 적재분도 OpenSearch 인라인 색인을 하려면
        # init_settings 가 돌려준 활성 설정이 배치로 전달돼야 한다(배치가 _make_opensearch_indexer 에 사용).
        from processing.ingest.batch_runner import BatchReport

        cb = _callable("dag_process", _PROCESS_TASK)
        db, _c, _cur = _fake_db()
        with mock.patch.dict(os.environ, {"META_ENV": "dev"}), \
                mock.patch("src.config.settings.init_settings") as m_init, \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch("processing.ingest.batch_runner.process_received_batch",
                           return_value=BatchReport()) as m_batch:
            cb()
        self.assertIs(m_batch.call_args.kwargs.get("settings"), m_init.return_value)

    def test_relations_callable_scans_then_delegates_run_relations(self) -> None:
        cb = _callable("dag_relations", _RELATIONS_TASK)
        db, _c, _cur = _fake_db()
        with mock.patch.dict(os.environ, {"META_ENV": "dev"}), \
                mock.patch("src.config.settings.init_settings") as m_init, \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch("processing.ingest.batch_runner.scan_unresolved_assets",
                           return_value=[_AID]) as m_scan, \
                mock.patch("processing.app.run_relations.run_relations",
                           return_value={"done": [(str(_AID), 1, 0)], "failed": []}) as m_rr:
            cb()
        m_init.assert_called_once()
        m_scan.assert_called_once()
        # G2 스캔 → run_relations(내부에서 propose_relations_for_asset 호출 + relation_resolution 전이).
        m_rr.assert_called_once()
        self.assertEqual(m_rr.call_args.args[0], [str(_AID)])

    def test_relations_callable_noop_when_nothing_unresolved(self) -> None:
        cb = _callable("dag_relations", _RELATIONS_TASK)
        db, _c, _cur = _fake_db()
        with mock.patch.dict(os.environ, {"META_ENV": "dev"}), \
                mock.patch("src.config.settings.init_settings"), \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch("processing.ingest.batch_runner.scan_unresolved_assets",
                           return_value=[]), \
                mock.patch("processing.app.run_relations.run_relations") as m_rr:
            cb()
        # 미해소 0건이면 run_relations 를 부르지 않는다(빈 배치 호출 회피).
        m_rr.assert_not_called()

    def test_collect_gate_passes_only_when_new_received(self) -> None:
        # (a) 게이트 — 신규 수집이 있을 때만 trigger_process 통과(빈 인입에 GPU 배치 헛 기동 차단).
        gate = _callable("dag_collect", "gate_new_received")
        self.assertTrue(gate(**_fake_context(2)))       # 신규 2건 → 통과
        self.assertFalse(gate(**_fake_context(0)))      # 0건 → ShortCircuit 스킵
        self.assertFalse(gate(**_fake_context(None)))   # XCom 없음 → 스킵

    def test_process_gate_passes_only_when_new_registered(self) -> None:
        # (a) 게이트 — 신규 registered 가 있을 때만 trigger_relations 통과.
        gate = _callable("dag_process", "gate_new_registered")
        self.assertTrue(gate(**_fake_context({"registered": 3, "deferred": 1})))   # 신규 3건 → 통과
        self.assertFalse(gate(**_fake_context({"registered": 0, "deferred": 2})))  # 0건 → 스킵
        self.assertFalse(gate(**_fake_context(None)))                              # XCom 없음 → 스킵


@unittest.skipUnless(_AIRFLOW_AVAILABLE, "apache-airflow 미설치 — dag_process 모듈 로드 불가")
class TestIntEnvValidation(unittest.TestCase):
    """A1 — ``_int_env`` 는 음수·0(minimum 미만)을 기본값으로 되돌린다.

    방치 시 DAG_PROCESS_LIMIT 음수는 PG 'LIMIT must not be negative' 배치 크래시, 0 은 매 run 0건
    (무음 스톨), MAX_FAILURES 0 은 첫 실패 즉시 격리를 유발한다. dag_process 모듈에서 함수를 꺼내 검증.
    """

    @staticmethod
    def _fn():
        import inspect

        mod = inspect.getmodule(_callable("dag_process", _PROCESS_TASK))
        return mod._int_env

    def test_valid_positive_returned(self) -> None:
        with mock.patch.dict(os.environ, {"X_LIM": "50"}):
            self.assertEqual(self._fn()("X_LIM", 10), 50)

    def test_negative_falls_back_to_default(self) -> None:
        with mock.patch.dict(os.environ, {"X_LIM": "-1"}):
            self.assertEqual(self._fn()("X_LIM", 10), 10)

    def test_zero_falls_back_to_default(self) -> None:
        with mock.patch.dict(os.environ, {"X_LIM": "0"}):
            self.assertEqual(self._fn()("X_LIM", 10), 10)

    def test_nonint_falls_back_to_default(self) -> None:
        with mock.patch.dict(os.environ, {"X_LIM": "abc"}):
            self.assertEqual(self._fn()("X_LIM", 10), 10)

    def test_unset_uses_default(self) -> None:
        os.environ.pop("X_LIM", None)
        self.assertEqual(self._fn()("X_LIM", 10), 10)

    def test_custom_minimum_allows_zero(self) -> None:
        # 하한을 낮추면 0 허용 — 방어는 기본 minimum=1 일 때만(세 소비처가 그렇게 호출).
        with mock.patch.dict(os.environ, {"X_LIM": "0"}):
            self.assertEqual(self._fn()("X_LIM", 10, minimum=0), 0)


if __name__ == "__main__":
    unittest.main()
