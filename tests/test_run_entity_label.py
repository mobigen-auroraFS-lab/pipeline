"""개체 갈래 배치 — 증분 재판정 단위 테스트 (LLM·DB 불필요 · spec 104 T008·T010).

무엇을 덮는가: ``processing/app/run_entity_label.py`` 의 ``run_label_pass``·``run_entity_label``·CLI 배선.
판정(LLM)은 가짜 판정 함수(``judge_fn``)로, DB 는 메모리 상태를 든 가짜 ``db``·가짜 커넥션으로 갈아
끼운다. 코어의 **순수 선별 함수**(``select_label_work``·``label_material_hash``·``build_entity_material``)
는 그대로 돌려, 배선이 코어 계약과 실제로 맞물리는지를 본다.

지키려는 것(spec 104 합격선):
  - SC-01 재료가 그대로면 판정하지 않는다 — 두 번째 실행 판정 0회(LLM 호출 0).
  - SC-02 굶주림 해소 — 상한으로 반복 실행하면 결국 노출 개체 전부가 판정된다. 상한을 SQL 에 걸면
    구성원 많은 상위만 매번 다시 판정되고 나머지는 영영 판정되지 않는다(2026-10-07 클러스터 실측).
  - SC-03·SC-04 재료·판이 바뀐 (개체, 스킬) 짝만 다시 판정한다.
  - 🔴 판정에 쓴 재료와 저장한 지문이 같은 재료에서 나온다 — 다른 재료로 판정하면 지문이 거짓이 된다.
  - 판정 실패는 행을 남기지 않아 다음 실행에 다시 고른다(ADR D4 · 087 계약).
  - ``--plan`` 은 판정·쓰기 0 으로 집계와 대상 목록만 돌려준다.
  - (리뷰 반영) 읽기는 짧은 트랜잭션 둘로 나뉘고 구성 요약은 **일괄 SQL 1회**다 — 옛 개체당 SQL 과
    같은 결과여야 재료·지문이 바뀌지 않는다. 문안 판은 코어 공유 문안 판을 합성한다. 재료 조립에
    실패한 개체 하나가 배치를 멈추지 않는다. 활성 스킬 0 은 경고와 함께 판정 0.

가짜 함수는 코어 실함수의 **시그니처로 먼저 바인딩**한 뒤 동작한다 — 파이프가 코어에 없는 인자를
넘기면(코어 계약 드리프트) 가짜도 같이 실패해, 단위 테스트가 통과하는데 운영에서 깨지는 일을 막는다.
"""

from __future__ import annotations

import inspect
import io
import os
import unittest
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from typing import Any
from unittest import mock

from processing.app import run_entity_label as rel
from src.mm_classify import JudgeFailure, SkillJudgement
from src.mm_classify import judge as core_judge
from src.mm_classify import persist as core_persist
from src.mm_classify import prompt as core_prompt
from src.mm_meta import entity_label as core_el
from src.mm_meta.entity_label import LabelState, build_entity_material, label_material_hash

# 러너 로거 이름 — 경고 1줄을 이 이름으로 잡는다.
_LOGGER = "meta_extract.run_entity_label"

# 갈래 표 ``prompt_version`` 칸 폭(v304 · VARCHAR(40)) — 넘으면 저장이 전부 실패한다.
_PROMPT_VERSION_DB_WIDTH = 40

# 한 판의 읽기 트랜잭션(멱등) — ① 스킬·노출 개체·구성 요약(일괄) ② 저장 상태. 그 뒤는 쓰기뿐이다.
_READS = [True, True]

# 옛 개체당 구성 요약 SQL(리뷰 전 · 개체마다 1회 · 위치 인자 4개). 일괄 SQL 이 **같은 결과**를 내야
# 하는 기준이다 — 다르면 재료가 바뀌어 첫 실행 뒤 전량이 hash_changed 로 다시 판정된다.
_PER_ENTITY_SUMMARY_SQL = """
SELECT m.ext_meta->>'summary'
  FROM node n
  JOIN graph_edge ge    ON ge.dst_node = n.node_id
  JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id
  JOIN node sn          ON sn.node_id = ge.src_node
  JOIN asset_metadata m ON m.asset_id = sn.asset_id
 WHERE n.node_kind = 'entity' AND n.entity_type = %s AND n.entity_uid = %s
   AND rk.kind_code = 'mm_member' AND ge.status = ANY(%s)
   AND m.ext_meta->>'summary' IS NOT NULL
 ORDER BY sn.asset_id
 LIMIT %s
"""


def _norm(sql: str) -> str:
    """SQL 을 비교용으로 정규화한다(공백 한 칸 · 소문자)."""
    return " ".join(sql.lower().split())

_FOOD = {
    "skill_code": "food",
    "name": "음식 갈래",
    "version": 1,
    "policy": {"selection": "multi", "unassigned": "해당없음", "max_labels": 5},
    "labels": [
        {"code": "korean", "name": "한식", "definition": "한국 음식이 중심", "not": "외국 음식"},
        {"code": "snack", "name": "분식", "definition": "간단한 길거리 음식", "not": "정찬"},
    ],
}
_MUSIC = {
    "skill_code": "music",
    "name": "음악 갈래",
    "version": 1,
    "policy": {"selection": "multi", "unassigned": "해당없음", "max_labels": 5},
    "labels": [
        {"code": "song", "name": "노래", "definition": "가창이 있는 곡", "not": "연주곡"},
        {"code": "lyrics", "name": "노랫말", "definition": "가사 텍스트", "not": "악보"},
    ],
}

_MIN = rel.DEFAULT_MIN_MEMBERS

# 노출 개체 5 + 노출 임계 미만 1. 같은 표기 「김밥」이 음식·작품 두 타입으로 있다 — 개체 키는
# (타입, 표기) 짝이어야 한다(실례: 더 자두의 곡 「김밥」).
_ENTITIES = (
    {"entity_type": "인물", "entity_uid": "아이유", "name": "아이유",
     "description": "가수이자 배우.", "members": _MIN + 6},
    {"entity_type": "음식", "entity_uid": "김밥", "name": "김밥",
     "description": "밥을 김에 만 음식.", "members": _MIN + 4},
    {"entity_type": "장소", "entity_uid": "경주시", "name": "경주시",
     "description": "신라의 옛 수도.", "members": _MIN + 2},
    {"entity_type": "음식", "entity_uid": "떡볶이", "name": "떡볶이",
     "description": None, "members": _MIN + 1},
    {"entity_type": "작품", "entity_uid": "김밥", "name": "김밥",
     "description": "더 자두의 노래.", "members": _MIN},
    {"entity_type": "인물", "entity_uid": "단역", "name": "단역",
     "description": "한 번 등장.", "members": _MIN - 1},
)
_VISIBLE = [(e["entity_type"], e["entity_uid"]) for e in _ENTITIES if e["members"] >= _MIN]

# 구성 자산 요약 — 아이유는 4건이지만 재료에는 앞 MEMBER_SUMMARIES 건만 실린다.
_SUMMARIES = {
    ("인물", "아이유"): ["콘서트 영상", "노래 가사", "인터뷰 기사", "광고 사진"],
    ("음식", "김밥"): ["김밥 만드는 법"],
    ("작품", "김밥"): ["김밥 노래 음원"],
}


class _Conn:
    """가짜 커넥션이자 메모리 DB — 노출 개체·구성 요약·스킬 행·갈래 행(``entity_mm_skill_label``).

    파이프가 직접 SQL 을 치는 곳은 구성 요약 일괄 조회 하나라서, 커서는 그 질의만 흉내 낸다. 나머지
    (스킬·대상·상태 읽기와 갈래 쓰기)는 코어 함수 자리에 꽂는 가짜가 이 상태를 읽고 쓴다.
    무엇을 했는지는 ``log`` 에 순서대로 남는다 — 가짜 db 가 트랜잭션마다 잘라 ``ops`` 로 묶는다.
    """

    def __init__(
        self,
        entities: tuple[dict[str, Any], ...] = _ENTITIES,
        skills: tuple[dict[str, Any], ...] = (_FOOD, _MUSIC),
    ) -> None:
        self.entities = [dict(e) for e in entities]
        self.skill_rows = [dict(s) for s in skills]
        self.summaries: dict[tuple[str, str], list[str]] = {
            k: list(v) for k, v in _SUMMARIES.items()
        }
        self.rows: list[dict[str, Any]] = []
        self.target_limits: list[int | None] = []
        self.summary_queries: list[dict[str, Any]] = []
        self.replace_calls: list[dict[str, Any]] = []
        self.log: list[str] = []  # skills · targets · summaries · state · replace

    # --- 구성 요약 SQL 흉내 -------------------------------------------------
    def cursor(self, *_a: Any, **_k: Any) -> _Cursor:
        """구성 요약 일괄 질의만 받는 가짜 커서."""
        return _Cursor(self)

    # --- 조회 도우미 ---------------------------------------------------------
    def entity(self, etype: str, euid: str) -> dict[str, Any]:
        """(타입, 표기) 로 개체 행을 찾는다."""
        return next(
            e for e in self.entities if (e["entity_type"], e["entity_uid"]) == (etype, euid)
        )

    def rows_of(self, etype: str, euid: str, code: str) -> list[dict[str, Any]]:
        """(개체, 스킬) 짝의 갈래 행."""
        return [
            r for r in self.rows
            if (r["entity_type"], r["entity_uid"], r["skill_code"]) == (etype, euid, code)
        ]

    def skill_row(self, code: str) -> dict[str, Any]:
        """스킬 코드로 스킬 행을 찾는다."""
        return next(s for s in self.skill_rows if s["skill_code"] == code)


class _Cursor:
    """구성 요약 **일괄** SQL 하나만 흉내 내는 커서(그 밖의 SQL 이 오면 실패시킨다).

    일괄 SQL 의 약속을 그대로 낸다 — 요청한 (타입, 표기) 짝마다 구성 요약을 자산 순 앞 N건, 행은
    ``(entity_type, entity_uid, summary)`` 이고 ``ORDER BY entity_type, entity_uid, rn`` 순서다.
    """

    def __init__(self, conn: _Conn) -> None:
        self._conn = conn
        self._out: list[tuple[str, str, str]] = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def execute(self, sql: str, params: dict[str, Any]) -> None:
        flat = _norm(sql)
        if "mm_member" not in flat or "summary" not in flat or "row_number()" not in flat:
            raise AssertionError(f"구성 요약 일괄 질의만 흉내 낸다 — 예상 밖 SQL: {sql[:80]!r}")
        self._conn.log.append("summaries")
        self._conn.summary_queries.append(dict(params))
        keys = sorted(set(zip(params["types"], params["uids"], strict=True)))
        n = params["per_entity"]
        self._out = [
            (etype, euid, s)
            for etype, euid in keys
            for s in self._conn.summaries.get((etype, euid), [])[:n]
        ]

    def fetchall(self) -> list[tuple[str, str, str]]:
        return list(self._out)


class _FakeDb:
    """``PostgresUtil.execute_in_transaction`` 흉내 — 트랜잭션마다 멱등 여부와 한 일을 기록한다."""

    def __init__(self, conn: _Conn) -> None:
        self.conn = conn
        self.transactions: list[bool] = []
        self.ops: list[tuple[str, ...]] = []  # 트랜잭션마다 그 안에서 한 일(conn.log 조각)

    def execute_in_transaction(self, fn: Callable[[Any], Any], *, idempotent: bool = False) -> Any:
        self.transactions.append(bool(idempotent))
        start = len(self.conn.log)
        try:
            return fn(self.conn)
        finally:
            self.ops.append(tuple(self.conn.log[start:]))


class _Judge:
    """가짜 판정 — 호출을 기록하고, 지정한 (개체 이름, 스킬) 짝은 실패(ok=False)로 돌려준다."""

    def __init__(self, *, fail: set[tuple[str, str]] | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []  # (skill_code, 이름 키워드, 재료)
        self.fail = set(fail or ())

    def __call__(self, skill: Any, material: str, keywords: list[str], *, client: Any = None):
        self.calls.append((skill.skill_code, keywords[0], material))
        if (keywords[0], skill.skill_code) in self.fail:
            return SkillJudgement(ok=False, failure=JudgeFailure.RESPONSE_SHAPE, detail="가짜 실패")
        first = skill.labels[0]
        return SkillJudgement(ok=True, label_names=(first.name,), label_codes=(first.code,))

    def materials_of(self, name: str) -> set[str]:
        """그 이름 개체를 판정할 때 넘어간 재료 모음."""
        return {m for _c, n, m in self.calls if n == name}


def _signed(real: Callable[..., Any], fake: Callable[..., Any]) -> Callable[..., Any]:
    """가짜 함수를 코어 실함수의 시그니처로 먼저 바인딩한다(인자 드리프트를 테스트에서 잡는다)."""
    sig = inspect.signature(real)

    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        sig.bind(*args, **kwargs)
        return fake(*args, **kwargs)

    return _wrapper


def _fake_active_skills(conn: _Conn) -> list[dict[str, Any]]:
    conn.log.append("skills")
    return [dict(r) for r in sorted(conn.skill_rows, key=lambda r: r["skill_code"])]


def _fake_targets(conn: _Conn, *, min_members: int, statuses: Any, limit: int | None = None):
    conn.log.append("targets")
    conn.target_limits.append(limit)
    rows = [dict(e) for e in conn.entities if e["members"] >= min_members]
    rows.sort(key=lambda e: (-e["members"], e["entity_uid"]))
    return rows if limit is None else rows[:limit]


def _fake_state(conn: _Conn) -> dict[tuple[str, str, str], LabelState]:
    conn.log.append("state")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in conn.rows:
        groups.setdefault((r["entity_type"], r["entity_uid"], r["skill_code"]), []).append(r)
    out: dict[tuple[str, str, str], LabelState] = {}
    for key, rs in groups.items():
        hashes = {r["material_hash"] for r in rs}
        svs = {r["skill_version"] for r in rs}
        pvs = {r["prompt_version"] for r in rs}
        out[key] = LabelState(
            material_hash=None if None in hashes else max(hashes),
            skill_version=max(svs),
            prompt_version=max(pvs),
            mixed=len(hashes) > 1 or len(svs) > 1 or len(pvs) > 1,
        )
    return out


def _fake_replace(conn: _Conn, *, entity_type: str, entity_uid: str, skill: Any,
                  skill_version: int, judgement: Any, prompt_version: str,
                  decided_by: str = "llm", material_hash: str | None = None) -> int:
    conn.log.append("replace")
    conn.replace_calls.append({
        "entity_type": entity_type, "entity_uid": entity_uid, "skill_code": skill.skill_code,
        "skill_version": skill_version, "prompt_version": prompt_version,
        "material_hash": material_hash,
    })
    if not judgement.ok:  # 코어 계약 — 실패 판정은 쓰지 않는다
        return 0
    conn.rows = [
        r for r in conn.rows
        if (r["entity_type"], r["entity_uid"], r["skill_code"])
        != (entity_type, entity_uid, skill.skill_code)
    ]
    for code in judgement.label_codes:
        conn.rows.append({
            "entity_type": entity_type, "entity_uid": entity_uid,
            "skill_code": skill.skill_code, "label_code": code,
            "skill_version": skill_version, "prompt_version": prompt_version,
            "decided_by": decided_by, "material_hash": material_hash,
        })
    return len(judgement.label_codes)


def _expected_material(conn: _Conn, etype: str, euid: str) -> str:
    """운영과 같은 규칙으로 만든 그 개체의 판정 재료(테스트 쪽 기대값)."""
    e = conn.entity(etype, euid)
    return build_entity_material(
        name=e["name"],
        entity_type=etype,
        description=e["description"],
        member_summaries=conn.summaries.get((etype, euid), []),
        max_member_summaries=rel.MEMBER_SUMMARIES,
    )


def _judged_entities(judge: _Judge) -> set[str]:
    return {name for _c, name, _m in judge.calls}


class _PassMixin(unittest.TestCase):
    """가짜 db·판정으로 ``run_label_pass`` 한 판을 돌리는 도우미."""

    def run_pass(self, conn: _Conn, judge: _Judge, **kw: Any) -> tuple[dict[str, Any], _FakeDb]:
        db = _FakeDb(conn)
        with mock.patch.multiple(
            rel,
            fetch_active_skills=_signed(core_persist.fetch_active_skills, _fake_active_skills),
            fetch_label_targets=_signed(core_el.fetch_label_targets, _fake_targets),
            fetch_label_state=_signed(core_el.fetch_label_state, _fake_state),
            replace_entity_labels=_signed(core_el.replace_entity_labels, _fake_replace),
        ):
            report = rel.run_label_pass(db, judge_fn=judge, **kw)
        return report, db


class TestFirstAndSecondRun(_PassMixin):
    def test_first_run_judges_every_visible_pair_and_stores_hash(self) -> None:
        # 상태가 없으면 노출 개체 × 활성 스킬 전부가 미판정(new)이다.
        conn, judge = _Conn(), _Judge()
        report, _db = self.run_pass(conn, judge)

        self.assertEqual(len(judge.calls), len(_VISIBLE) * 2)
        self.assertNotIn("단역", _judged_entities(judge))  # 노출 임계 미만은 대상이 아니다
        self.assertEqual(report["visible"], len(_VISIBLE))
        self.assertEqual(report["need"], len(_VISIBLE))
        self.assertEqual(report["selected"], len(_VISIBLE))
        self.assertEqual(report["targets"], len(_VISIBLE))  # targets = 이번에 판정한 개체 수
        self.assertEqual(report["skipped_unchanged"], 0)
        self.assertEqual(report["pairs_new"], len(_VISIBLE) * 2)
        self.assertEqual((report["judged"], report["failed"], report["rows"]),
                         (len(_VISIBLE) * 2, 0, len(_VISIBLE) * 2))

        for etype, euid in _VISIBLE:
            want_hash = label_material_hash(_expected_material(conn, etype, euid))
            for code in ("food", "music"):
                rows = conn.rows_of(etype, euid, code)
                self.assertEqual(len(rows), 1, (etype, euid, code))
                self.assertEqual(rows[0]["material_hash"], want_hash)
                self.assertEqual(rows[0]["prompt_version"], rel.PROMPT_VERSION)
                self.assertEqual(rows[0]["skill_version"], 1)

    def test_judged_material_is_the_hashed_material(self) -> None:
        # 🔴 판정에 넘긴 재료와 저장한 지문이 같은 재료여야 한다 — 다르면 「재료 그대로」 판단이 거짓이다.
        conn, judge = _Conn(), _Judge()
        self.run_pass(conn, judge)
        want = _expected_material(conn, "인물", "아이유")
        self.assertEqual(judge.materials_of("아이유"), {want})
        self.assertIn("노래 가사", want)
        self.assertNotIn("광고 사진", want)  # MEMBER_SUMMARIES(3) 넘는 요약은 실리지 않는다
        stored = {r["material_hash"] for r in conn.rows if r["entity_uid"] == "아이유"}
        self.assertEqual(stored, {label_material_hash(want)})

    def test_second_run_with_same_material_judges_nothing(self) -> None:
        # SC-01 — 재료·판이 그대로면 LLM 을 한 번도 부르지 않는다.
        conn = _Conn()
        self.run_pass(conn, _Judge())
        before = [dict(r) for r in conn.rows]

        judge2 = _Judge()
        report, db = self.run_pass(conn, judge2)

        self.assertEqual(judge2.calls, [])
        self.assertEqual(conn.rows, before)  # 행이 흔들리지 않는다(SC-05)
        self.assertEqual((report["targets"], report["judged"], report["rows"]), (0, 0, 0))
        self.assertEqual(report["skipped_unchanged"], len(_VISIBLE))
        self.assertEqual((report["need"], report["selected"]), (0, 0))
        self.assertEqual(db.transactions, _READS)  # 읽기(짧은 둘)뿐 · 쓰기 트랜잭션 0

    def test_reads_are_short_transactions_and_summaries_are_one_batch(self) -> None:
        # 🔴 긴 읽기 트랜잭션 금지(리뷰 M2) — 옛 배선은 노출 개체(수천 개)마다 요약 SQL 을 한 번씩
        #    한 트랜잭션 안에서 쳐서, 그동안 표 잠금을 쥔 채 몇 초를 끌었다(v307 ALTER 가 그 뒤에서
        #    기다리고, 그 뒤로 화면 조회가 줄을 선다). 이제 ① 스킬·노출 개체·구성 요약(일괄 1회)
        #    ② 저장 상태 — 짧은 읽기 둘로 끝나고, 그 뒤는 개체×스킬 단위 쓰기뿐이다.
        conn = _Conn()
        _report, db = self.run_pass(conn, _Judge())
        self.assertEqual(db.ops[:2], [("skills", "targets", "summaries"), ("state",)])
        self.assertEqual(db.transactions[:2], _READS)
        writes = len(_VISIBLE) * 2
        self.assertEqual(db.ops[2:], [("replace",)] * writes)
        self.assertEqual(db.transactions[2:], [False] * writes)

        # 요약은 개체 수와 무관하게 **한 번** — 노출 개체 전부의 (타입, 표기) 짝을 한꺼번에 넘긴다.
        self.assertEqual(len(conn.summary_queries), 1)
        q = conn.summary_queries[0]
        self.assertEqual(sorted(zip(q["types"], q["uids"], strict=True)), sorted(_VISIBLE))
        self.assertEqual(q["per_entity"], rel.MEMBER_SUMMARIES)
        self.assertEqual(list(q["statuses"]), list(rel.VISIBLE_STATUSES))

    def test_no_summary_query_without_targets_or_when_summaries_off(self) -> None:
        # 노출 개체가 없으면 빈 배열로 SQL 을 치지 않는다 · 요약을 끄면(0) 요약 SQL 자체가 없다.
        empty = _Conn(entities=())
        report, _db = self.run_pass(empty, _Judge())
        self.assertEqual(empty.summary_queries, [])
        self.assertEqual(report["visible"], 0)

        conn, judge = _Conn(), _Judge()
        with mock.patch.object(rel, "MEMBER_SUMMARIES", 0):
            self.run_pass(conn, judge)
        self.assertEqual(conn.summary_queries, [])
        self.assertNotIn("구성 자료", next(iter(judge.materials_of("아이유"))))

    def test_legacy_rows_without_hash_are_rejudged_once(self) -> None:
        # 지문 칸이 생기기 전의 행(NULL)은 첫 실행에서 한 번 다시 판정하고, 그 뒤엔 건너뛴다.
        conn = _Conn()
        for etype, euid in _VISIBLE:
            for code in ("food", "music"):
                conn.rows.append({
                    "entity_type": etype, "entity_uid": euid, "skill_code": code,
                    "label_code": "x", "skill_version": 1, "prompt_version": rel.PROMPT_VERSION,
                    "decided_by": "llm", "material_hash": None,
                })
        judge = _Judge()
        report, _db = self.run_pass(conn, judge)
        self.assertEqual(len(judge.calls), len(_VISIBLE) * 2)
        self.assertEqual(report["pairs_hash_null"], len(_VISIBLE) * 2)
        self.assertNotIn(None, {r["material_hash"] for r in conn.rows})

        judge2 = _Judge()
        self.run_pass(conn, judge2)
        self.assertEqual(judge2.calls, [])


class TestLimitStarvation(_PassMixin):
    def test_repeated_runs_with_limit_cover_every_visible_entity(self) -> None:
        # SC-02 — 상한 2 로 반복하면 이미 판정된 상위가 자리를 차지하지 않고 다음 개체로 넘어간다.
        conn = _Conn()
        seen: list[set[str]] = []
        judged_keys: list[tuple[str, str]] = []
        for _ in range(4):
            judge = _Judge()
            report, _db = self.run_pass(conn, judge, limit=2)
            seen.append({c[1] for c in judge.calls})
            for etype, euid in _VISIBLE:
                if conn.rows_of(etype, euid, "food") and (etype, euid) not in judged_keys:
                    judged_keys.append((etype, euid))
            self.assertLessEqual(report["selected"], 2)

        # 구성원 많은 순으로 2개씩 · 4번째 실행은 할 일이 없다.
        self.assertEqual(seen[0], {"아이유", "김밥"})
        self.assertEqual(seen[1], {"경주시", "떡볶이"})
        self.assertEqual(seen[2], {"김밥"})
        self.assertEqual(seen[3], set())
        self.assertEqual(sorted(judged_keys), sorted(_VISIBLE))  # 노출 개체 전부 판정됨(미판정 0)

    def test_limit_is_applied_after_filtering_not_in_sql(self) -> None:
        # 🔴 상한을 대상 SQL 에 걸면 굶주림이 재발한다 — 대상은 항상 상한 없이 전부 읽는다.
        conn = _Conn()
        self.run_pass(conn, _Judge(), limit=2)
        self.run_pass(conn, _Judge(), limit=2, plan_only=True)
        self.assertEqual(conn.target_limits, [None, None])

    def test_limit_counts_entities_not_pairs(self) -> None:
        # 상한은 개체 수 — 한 개체의 두 스킬은 함께 판정한다(재료 조립 1회).
        conn, judge = _Conn(), _Judge()
        report, _db = self.run_pass(conn, judge, limit=1)
        self.assertEqual(report["targets"], 1)
        self.assertEqual(sorted(c[0] for c in judge.calls), ["food", "music"])
        self.assertEqual(report["need"], len(_VISIBLE))  # 상한 전 밀린 일은 그대로 보고한다


class TestRejudgeTriggers(_PassMixin):
    def test_changed_description_rejudges_only_that_entity(self) -> None:
        # SC-03 — 설명문이 다시 만들어진 개체만 다시 판정한다.
        conn = _Conn()
        self.run_pass(conn, _Judge())
        conn.entity("장소", "경주시")["description"] = "신라 천 년의 수도. 불국사가 있다."

        judge = _Judge()
        report, _db = self.run_pass(conn, judge)

        self.assertEqual(_judged_entities(judge), {"경주시"})
        self.assertEqual(sorted(c[0] for c in judge.calls), ["food", "music"])
        self.assertEqual(report["pairs_hash_changed"], 2)
        self.assertEqual(report["skipped_unchanged"], len(_VISIBLE) - 1)
        want = label_material_hash(_expected_material(conn, "장소", "경주시"))
        self.assertEqual({r["material_hash"] for r in conn.rows if r["entity_uid"] == "경주시"},
                         {want})

    def test_changed_member_summary_rejudges_only_that_entity(self) -> None:
        # 구성 요약(재료의 일부)이 바뀌어도 재료가 바뀐 것이다 — 같은 표기 다른 타입은 건드리지 않는다.
        conn = _Conn()
        self.run_pass(conn, _Judge())
        conn.summaries[("작품", "김밥")] = ["김밥 노래 뮤직비디오"]

        judge = _Judge()
        self.run_pass(conn, judge)
        self.assertEqual(len(judge.calls), 2)
        self.assertEqual(judge.materials_of("김밥"), {_expected_material(conn, "작품", "김밥")})

    def test_skill_version_bump_rejudges_only_that_skill(self) -> None:
        # SC-04 — 스킬 정의가 개정되면(DB 판 카운터) 그 스킬 짝만 다시 판정한다.
        conn = _Conn()
        self.run_pass(conn, _Judge())
        conn.skill_row("music")["version"] = 2
        food_before = [dict(r) for r in conn.rows if r["skill_code"] == "food"]

        judge = _Judge()
        report, _db = self.run_pass(conn, judge)

        self.assertEqual({c[0] for c in judge.calls}, {"music"})
        self.assertEqual(len(judge.calls), len(_VISIBLE))
        self.assertEqual(report["pairs_skill_version"], len(_VISIBLE))
        self.assertEqual({r["skill_version"] for r in conn.rows if r["skill_code"] == "music"},
                         {2})
        self.assertEqual([r for r in conn.rows if r["skill_code"] == "food"], food_before)

    def test_prompt_version_change_rejudges_every_pair(self) -> None:
        # SC-04 — 판정 문안 판이 바뀌면 모든 짝이 대상이다.
        conn = _Conn()
        self.run_pass(conn, _Judge())
        judge = _Judge()
        with mock.patch.object(rel, "PROMPT_VERSION", "entity_label.test-v2"):
            report, _db = self.run_pass(conn, judge)
        self.assertEqual(len(judge.calls), len(_VISIBLE) * 2)
        self.assertEqual(report["pairs_prompt_version"], len(_VISIBLE) * 2)
        self.assertEqual({r["prompt_version"] for r in conn.rows}, {"entity_label.test-v2"})

    def test_only_needed_skills_are_judged_per_entity(self) -> None:
        # 한 개체에서 일부 스킬만 판정 대상이면 그 스킬만 판정한다(나머지 스킬 호출 0).
        conn = _Conn(skills=(_FOOD,))
        self.run_pass(conn, _Judge())
        conn.skill_rows.append(dict(_MUSIC))  # 새 스킬 등록 → 모든 개체의 music 짝만 미판정

        judge = _Judge()
        report, _db = self.run_pass(conn, judge)

        self.assertEqual({c[0] for c in judge.calls}, {"music"})
        self.assertEqual(len(judge.calls), len(_VISIBLE))
        self.assertEqual(report["pairs_new"], len(_VISIBLE))
        self.assertEqual(report["by_skill"].get("food", {}).get("judged", 0), 0)
        self.assertEqual(report["judged_by_reason"], {"new": len(_VISIBLE)})


class TestFailureIsRetried(_PassMixin):
    def test_failed_pair_leaves_no_row_and_is_picked_next_run(self) -> None:
        # ADR D4 — 실패는 행을 남기지 않으므로 다음 실행에서 그 짝만 다시 고른다.
        conn = _Conn()
        report, _db = self.run_pass(conn, _Judge(fail={("떡볶이", "music")}))
        self.assertEqual(report["failed"], 1)
        self.assertEqual(conn.rows_of("음식", "떡볶이", "music"), [])
        self.assertEqual(len(conn.rows_of("음식", "떡볶이", "food")), 1)

        judge = _Judge()
        report2, _db = self.run_pass(conn, judge)
        self.assertEqual([(c[0], c[1]) for c in judge.calls], [("music", "떡볶이")])
        self.assertEqual(report2["pairs_new"], 1)
        self.assertEqual(len(conn.rows_of("음식", "떡볶이", "music")), 1)

        judge3 = _Judge()
        self.run_pass(conn, judge3)
        self.assertEqual(judge3.calls, [])

    def test_dry_run_judges_but_writes_nothing(self) -> None:
        # --dry-run 은 판정은 하되 쓰지 않는다 — 그래서 다음 실행도 같은 대상을 고른다.
        conn, judge = _Conn(), _Judge()
        report, db = self.run_pass(conn, judge, dry_run=True)
        self.assertEqual(len(judge.calls), len(_VISIBLE) * 2)
        self.assertEqual((conn.rows, conn.replace_calls), ([], []))
        self.assertTrue(report["dry_run"])
        self.assertEqual(db.transactions, _READS)


class TestPlan(_PassMixin):
    def test_plan_judges_and_writes_nothing_and_reports_targets(self) -> None:
        # --plan — 읽기·재료·선별까지만. LLM 0 · 쓰기 0 · 집계와 대상 목록을 돌려준다.
        conn, judge = _Conn(), _Judge()
        report, db = self.run_pass(conn, judge, plan_only=True, limit=2)

        self.assertEqual(judge.calls, [])
        self.assertEqual((conn.rows, conn.replace_calls), ([], []))
        self.assertEqual(db.transactions, _READS)
        self.assertTrue(report["plan"])
        self.assertEqual((report["judged"], report["failed"], report["rows"]), (0, 0, 0))
        self.assertEqual((report["visible"], report["need"], report["selected"]),
                         (len(_VISIBLE), len(_VISIBLE), 2))
        self.assertEqual(report["pairs_new"], len(_VISIBLE) * 2)

        items = report["plan_items"]
        self.assertEqual([(i["entity_type"], i["name"]) for i in items],
                         [("인물", "아이유"), ("음식", "김밥")])
        self.assertEqual(items[0]["members"], _MIN + 6)
        self.assertEqual(items[0]["skill_codes"], ("food", "music"))
        self.assertEqual(items[0]["reasons"], ("new", "new"))

    def test_plan_wins_over_dry_run(self) -> None:
        # 둘 다 주면 plan — 판정(LLM)조차 하지 않는다.
        conn, judge = _Conn(), _Judge()
        report, _db = self.run_pass(conn, judge, plan_only=True, dry_run=True)
        self.assertEqual(judge.calls, [])
        self.assertTrue(report["plan"])

    def test_plan_after_full_run_is_empty(self) -> None:
        conn = _Conn()
        self.run_pass(conn, _Judge())
        report, _db = self.run_pass(conn, _Judge(), plan_only=True)
        self.assertEqual((report["need"], report["selected"], report["plan_items"]), (0, 0, []))
        self.assertEqual(report["skipped_unchanged"], len(_VISIBLE))

    def test_format_plan_lists_at_most_twenty(self) -> None:
        many = tuple(
            {"entity_type": "음식", "entity_uid": f"음식{i:02d}", "name": f"음식{i:02d}",
             "description": None, "members": _MIN + i}
            for i in range(25)
        )
        conn = _Conn(entities=many)
        report, _db = self.run_pass(conn, _Judge(), plan_only=True)
        text = rel.format_plan(report)
        listed = [ln for ln in text.splitlines() if ln.lstrip().startswith("- ")]
        self.assertEqual(len(listed), 20)
        self.assertIn("음식/음식24", listed[0])  # 구성원 많은 순
        self.assertIn("외 5개", text)
        self.assertIn("25", text)  # 노출·필요 개체 수가 보인다


class TestPromptVersion(unittest.TestCase):
    """개체 문안 판 = 개체 자체 판 + 코어 공유 문안 판(리뷰 M1)."""

    def test_entity_prompt_version_carries_core_prompt_version(self) -> None:
        # 개체 판정은 자산 판정과 **같은 공유 프롬프트**(코어 mm_classify)를 쓴다. 공유 문안이 개정되면
        #  개체도 다시 판정해야 하므로 그 판을 개체 판에 싣는다.
        self.assertEqual(rel.PROMPT_VERSION, f"entity_label.v1+{core_prompt.PROMPT_VERSION}")
        self.assertIn(core_prompt.PROMPT_VERSION, rel.PROMPT_VERSION)
        self.assertTrue(rel.PROMPT_VERSION.startswith(rel.ENTITY_PROMPT_BASE + "+"))

    def test_prompt_version_fits_db_column(self) -> None:
        # 갈래 표 prompt_version 은 VARCHAR(40) — 넘으면 모든 저장이 실패한다(조용한 0건 배치).
        self.assertLessEqual(len(rel.PROMPT_VERSION), _PROMPT_VERSION_DB_WIDTH)

    def test_core_prompt_bump_changes_entity_prompt_version(self) -> None:
        # 코어 문안 판이 오르면 개체 판도 바뀐다 → 저장된 판과 달라져 전부 재선별된다.
        with mock.patch.object(core_prompt, "PROMPT_VERSION", "mm_classify.v9"):
            bumped = rel.entity_prompt_version()
        self.assertEqual(bumped, "entity_label.v1+mm_classify.v9")
        self.assertNotEqual(bumped, rel.PROMPT_VERSION)
        self.assertEqual(rel.entity_prompt_version(), rel.PROMPT_VERSION)  # 패치가 풀리면 원래 값

    def test_composed_version_tracks_the_prompt_the_judge_uses(self) -> None:
        # 합성하는 판이 「판정이 실제로 쓰는 프롬프트 조립기」의 판이어야 뜻이 있다 — 판정부가 다른
        #  모듈의 조립기를 쓰게 바뀌면 이 합성은 엉뚱한 판을 따라가게 된다.
        self.assertIs(core_judge.build_classification_prompt,
                      core_prompt.build_classification_prompt)


class TestMemberSummaryBatch(unittest.TestCase):
    """구성 요약 일괄 조회 — 옛 개체당 SQL 과 같은 결과(리뷰 M2·L5).

    실 SQL 을 돌리지 않고 볼 수 있는 범위: ① 일괄 SQL 이 기준(개체당) SQL 의 거르기·정렬을 빠짐없이
    갖고 상한을 개체마다 건다 ② 행을 개체별로 묶을 때 순서·개수를 잃지 않는다. 실 DB 에서 두 SQL 의
    결과가 같은지는 아래 ``RUN_DB_E2E`` 테스트가 본다(사람 실행).
    """

    def test_batch_sql_keeps_every_per_entity_filter_and_ranks_per_entity(self) -> None:
        ref, batch = _norm(_PER_ENTITY_SUMMARY_SQL), _norm(rel._MEMBER_SUMMARIES_SQL)
        shared = (
            "from node n",
            "join graph_edge ge on ge.dst_node = n.node_id",
            "join relation_kind rk on rk.relation_kind_id = ge.relation_kind_id",
            "join node sn on sn.node_id = ge.src_node",
            "join asset_metadata m on m.asset_id = sn.asset_id",
            "n.node_kind = 'entity'",
            "rk.kind_code = 'mm_member'",
            "m.ext_meta->>'summary' is not null",
            "order by sn.asset_id",
        )
        for clause in shared:
            with self.subTest(clause=clause):
                self.assertIn(clause, ref)    # 기준 SQL 에 실제로 있는 조각만 비교한다
                self.assertIn(clause, batch)
        self.assertIn("ge.status = any(%s)", ref)
        self.assertIn("ge.status = any(%(statuses)s)", batch)
        # 개체 조건 = (타입, 표기) **짝** — 표기만 맞추면 타입이 다른 동명 개체(김밥 음식/작품)가 섞인다.
        self.assertIn("unnest(%(types)s::text[], %(uids)s::text[])", batch)
        # 개체마다 자산 순으로 순위를 매겨 앞 N건 — 기준 SQL 의 ORDER BY sn.asset_id LIMIT N 과 같은 뜻.
        self.assertIn(
            "row_number() over (partition by n.entity_type, n.entity_uid order by sn.asset_id)",
            batch,
        )
        self.assertIn("rn <= %(per_entity)s", batch)
        # 전체 LIMIT 은 개체 사이를 가로질러 자른다(앞 개체가 몫을 다 가져간다) — 없어야 한다.
        self.assertNotRegex(batch, r"\blimit\b")
        # 요약 NULL 거르기는 순위를 매기기 **전**(안쪽 질의)이어야 한다 — 밖으로 빼면 NULL 요약이
        #  N 자리 중 일부를 차지해 실리는 요약이 줄어든다(기준 SQL 은 거른 뒤 LIMIT).
        self.assertLess(batch.index("m.ext_meta->>'summary' is not null"),
                        batch.index(") ranked"))

    def test_rows_are_grouped_per_entity_without_losing_order(self) -> None:
        conn = _Conn()
        keys = [("인물", "아이유"), ("음식", "김밥"), ("음식", "떡볶이"), ("작품", "김밥")]
        got = rel._fetch_member_summaries(conn, keys)
        n = rel.MEMBER_SUMMARIES
        self.assertEqual(got, {
            ("인물", "아이유"): _SUMMARIES[("인물", "아이유")][:n],   # 4건 중 앞 N건 · 자산 순 그대로
            ("음식", "김밥"): _SUMMARIES[("음식", "김밥")],
            ("작품", "김밥"): _SUMMARIES[("작품", "김밥")],           # 같은 표기 다른 타입은 따로
        })
        self.assertNotIn(("음식", "떡볶이"), got)                     # 요약 없는 개체는 키가 없다
        self.assertEqual(len(conn.summary_queries), 1)

    def test_empty_keys_issue_no_sql(self) -> None:
        conn = _Conn()
        self.assertEqual(rel._fetch_member_summaries(conn, []), {})
        self.assertEqual(conn.summary_queries, [])


@unittest.skipUnless(os.environ.get("RUN_DB_E2E") == "1", "실 DB 게이트(RUN_DB_E2E=1)")
class TestMemberSummaryBatchMatchesPerEntityDB(unittest.TestCase):
    """실 DB — 일괄 요약 SQL 결과가 옛 개체당 SQL 결과와 **노출 개체 전부에서** 같은지(조회 전용).

    다르면 재료가 바뀌어 지문이 달라지고, 배포 뒤 첫 실행이 노출 개체 전부를 hash_changed 로 다시
    판정한다. 쓰기 0 — 읽기 트랜잭션 하나에서 두 SQL 을 모두 친다(같은 시점 비교).
    """

    @classmethod
    def setUpClass(cls) -> None:
        from pathlib import Path

        from dotenv import load_dotenv

        from src.config.settings import init_settings
        from src.database.postgres_util import PostgresUtil

        load_dotenv(Path(__file__).resolve().parents[1] / ".env.dev", override=False)
        init_settings("dev")
        cls.db = PostgresUtil()
        cls.db.open_pool()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.db.close()

    def test_batch_equals_per_entity_for_every_visible_entity(self) -> None:
        def _read(conn: Any) -> tuple[list[tuple[str, str]], dict, dict]:
            targets = core_el.fetch_label_targets(
                conn, min_members=rel.DEFAULT_MIN_MEMBERS, statuses=rel.VISIBLE_STATUSES,
                limit=None,
            )
            keys = [(t["entity_type"], t["entity_uid"]) for t in targets]
            batch = rel._fetch_member_summaries(conn, keys)
            per: dict[tuple[str, str], list[str]] = {}
            with conn.cursor() as cur:
                for etype, euid in keys:
                    cur.execute(_PER_ENTITY_SUMMARY_SQL, (etype, euid, list(rel.VISIBLE_STATUSES),
                                                          rel.MEMBER_SUMMARIES))
                    rows = [str(r[0]) for r in cur.fetchall()]
                    if rows:
                        per[(etype, euid)] = rows
            return keys, batch, per

        keys, batch, per = self.db.execute_in_transaction(_read, idempotent=True)
        self.assertGreater(len(keys), 0, "노출 개체가 없다 — 비교할 것이 없다(적재 상태 확인)")
        self.assertEqual(batch, per)


class TestMaterialFailureIsolated(_PassMixin):
    """재료 조립 실패 격리 — 개체 하나가 배치·계획 전체를 막지 않는다(리뷰 M3)."""

    _BROKEN = (
        {"entity_type": "음식", "entity_uid": "이름없음1", "name": "",
         "description": "이름이 비었다.", "members": _MIN + 3},
        {"entity_type": "장소", "entity_uid": "이름없음2", "name": "   ",
         "description": None, "members": _MIN},
    )

    def test_broken_entities_are_skipped_counted_and_logged_once(self) -> None:
        conn, judge = _Conn(entities=_ENTITIES + self._BROKEN), _Judge()
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            report, _db = self.run_pass(conn, judge)

        self.assertEqual(report["material_failed"], 2)
        self.assertEqual(_judged_entities(judge), {e[1] for e in _VISIBLE})  # 나머지는 그대로
        self.assertEqual(len(judge.calls), len(_VISIBLE) * 2)
        self.assertEqual(report["visible"], len(_VISIBLE))  # 재료를 만든 노출 개체만 센다
        self.assertEqual([r for r in conn.rows if r["entity_uid"].startswith("이름없음")], [])
        self.assertEqual(len(logs.records), 1)                 # 경고는 판마다 1줄
        self.assertIn("2", logs.output[0])
        self.assertIn("음식/이름없음1", logs.output[0])
        self.assertIn("재료 실패 2", rel.format_report(report))

    def test_plan_survives_broken_entity(self) -> None:
        conn, judge = _Conn(entities=_ENTITIES + self._BROKEN[:1]), _Judge()
        with self.assertLogs(_LOGGER, level="WARNING"):
            report, _db = self.run_pass(conn, judge, plan_only=True)
        self.assertEqual(judge.calls, [])
        self.assertEqual(report["material_failed"], 1)
        self.assertEqual(len(report["plan_items"]), len(_VISIBLE))
        self.assertNotIn("이름없음1", {i["entity_uid"] for i in report["plan_items"]})
        self.assertIn("재료 실패 1", rel.format_plan(report))

    def test_no_failure_reports_zero(self) -> None:
        report, _db = self.run_pass(_Conn(), _Judge())
        self.assertEqual(report["material_failed"], 0)
        self.assertNotIn("재료 실패", rel.format_report(report))


class TestNoActiveSkills(_PassMixin):
    """활성 스킬 0 — 판정 0 이고 경고 1줄(리포트·로그). 조용한 0건 배치를 막는다(리뷰 L3)."""

    def test_no_skill_warns_and_judges_nothing(self) -> None:
        conn, judge = _Conn(skills=()), _Judge()
        with self.assertLogs(_LOGGER, level="WARNING") as logs:
            report, db = self.run_pass(conn, judge)
        self.assertEqual(judge.calls, [])
        self.assertEqual((report["judged"], report["rows"], report["selected"]), (0, 0, 0))
        self.assertEqual(db.transactions, _READS)
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn("활성 스킬", report["warnings"][0])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("활성 스킬", logs.output[0])
        self.assertIn(report["warnings"][0], rel.format_report(report))

    def test_no_skill_plan_also_warns(self) -> None:
        conn = _Conn(skills=())
        with self.assertLogs(_LOGGER, level="WARNING"):
            report, _db = self.run_pass(conn, _Judge(), plan_only=True)
        self.assertEqual(report["plan_items"], [])
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn(report["warnings"][0], rel.format_plan(report))

    def test_with_skills_there_is_no_warning(self) -> None:
        report, _db = self.run_pass(_Conn(), _Judge())
        self.assertEqual(report["warnings"], [])


class TestRunEntityLabelCompat(unittest.TestCase):
    """``run_entity_label`` 직접 호출 — 옛 호출(재료 함수) 호환과 개체별 스킬 부분집합."""

    def setUp(self) -> None:
        self.skills = [core_persist.skill_from_row(dict(_FOOD)),
                       core_persist.skill_from_row(dict(_MUSIC))]

    def test_old_call_with_material_fn_judges_every_skill(self) -> None:
        judge = _Judge()
        targets = [{"entity_type": "음식", "entity_uid": "김밥", "name": "김밥", "members": 5}]
        report = rel.run_entity_label(self.skills, targets, material_fn=lambda t: "재료-김밥",
                                      judge_fn=judge, dry_run=True)
        self.assertEqual([(c[0], c[2]) for c in judge.calls],
                         [("food", "재료-김밥"), ("music", "재료-김밥")])
        self.assertEqual(report["targets"], 1)

    def test_target_skill_codes_and_material_are_used(self) -> None:
        # 대상 행에 판정할 스킬·재료가 있으면 그것만 쓴다 — 재료를 다시 만들지 않는다.
        def _boom(_t: Any) -> str:
            raise AssertionError("대상 행에 재료가 있으면 재료 함수를 부르지 않는다")

        judge = _Judge()
        targets = [{"entity_type": "음식", "entity_uid": "김밥", "name": "김밥", "members": 5,
                    "material": "선별 때 만든 재료", "skill_codes": ("music",),
                    "reasons": ("hash_changed",)}]
        report = rel.run_entity_label(self.skills, targets, material_fn=_boom,
                                      judge_fn=judge, dry_run=True)
        self.assertEqual([(c[0], c[2]) for c in judge.calls], [("music", "선별 때 만든 재료")])
        self.assertEqual(report["judged_by_reason"], {"hash_changed": 1})

    def test_missing_material_source_raises_before_judging(self) -> None:
        judge = _Judge()
        targets = [{"entity_type": "음식", "entity_uid": "김밥", "name": "김밥", "members": 5}]
        with self.assertRaises(ValueError):
            rel.run_entity_label(self.skills, targets, judge_fn=judge, dry_run=True)
        self.assertEqual(judge.calls, [])


class TestCli(unittest.TestCase):
    def _main(self, argv: list[str], report: dict[str, Any]) -> tuple[int, str, mock.MagicMock]:
        out = io.StringIO()
        db = mock.MagicMock()
        # .env 를 실제로 읽으면 테스트 프로세스 환경이 오염된다(다른 테스트가 깨진다) — 막는다.
        with mock.patch("dotenv.load_dotenv"), \
                mock.patch("src.config.settings.init_settings"), \
                mock.patch("src.database.postgres_util.PostgresUtil", return_value=db), \
                mock.patch.object(rel, "run_label_pass", return_value=report) as m_pass, \
                redirect_stdout(out):
            code = rel.main(argv)
        return code, out.getvalue(), m_pass

    @staticmethod
    def _plan_report() -> dict[str, Any]:
        return {
            "plan": True, "dry_run": True, "limit": None, "targets": 1, "judged": 0,
            "failed": 0, "rows": 0, "visible": 3, "skipped_unchanged": 2, "need": 1,
            "selected": 1, "pairs_new": 2, "pairs_mixed": 0, "pairs_hash_null": 0,
            "pairs_hash_changed": 0, "pairs_skill_version": 0, "pairs_prompt_version": 0,
            "plan_items": [{"entity_type": "음식", "entity_uid": "김밥", "name": "김밥",
                            "members": 7, "skill_codes": ("food", "music"),
                            "reasons": ("new", "new")}],
        }

    def test_limit_accepts_zero_and_positive(self) -> None:
        parser = rel._build_parser()
        self.assertEqual(parser.parse_args(["--limit", "0"]).limit, 0)
        self.assertEqual(parser.parse_args(["--limit", "500"]).limit, 500)
        self.assertIsNone(parser.parse_args([]).limit)

    def test_negative_or_non_integer_limit_is_rejected(self) -> None:
        # 음수 상한은 「이번 판은 판정 0」으로 조용히 접혔다 — 오타(-1)를 명령행에서 바로 막는다.
        for bad in ("-1", "-500", "abc", "1.5"):
            with self.subTest(limit=bad):
                err = io.StringIO()
                with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                    rel._build_parser().parse_args(["--limit", bad])
                self.assertEqual(cm.exception.code, 2)   # argparse 사용법 오류
                self.assertIn("--limit", err.getvalue())

    def test_plan_flag_parses(self) -> None:
        self.assertTrue(rel._build_parser().parse_args(["--plan"]).plan)
        self.assertFalse(rel._build_parser().parse_args([]).plan)

    def test_plan_flag_wins_over_dry_run_and_prints_plan(self) -> None:
        code, text, m_pass = self._main(["--plan", "--dry-run", "--limit", "7"],
                                        self._plan_report())
        self.assertEqual(code, 0)
        self.assertTrue(m_pass.call_args.kwargs["plan_only"])
        self.assertEqual(m_pass.call_args.kwargs["limit"], 7)
        self.assertIn("음식/김밥", text)
        self.assertIn("new", text)


if __name__ == "__main__":
    unittest.main()
