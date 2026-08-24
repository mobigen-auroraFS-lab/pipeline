"""085 T110 — 분류 스킬 배치 러너 단위 테스트 (LLM·DB·OpenSearch 불필요).

무엇을 덮는가: ``processing/app/run_mm_classify.py`` 의 **조립부**다. 판정(LLM)·저장(DB)·색인(OS)은
전부 주입 seam 으로 갈아 끼우고, 러너가 스스로 책임지는 것만 검증한다 —

  1. **스킬별 별도 호출**(spec §1·§4): 스킬이 여럿이면 스킬마다 따로 판정한다. 한 프롬프트에 섞으면
     격리·버전 분리가 무너진다.
  2. **판정 → 저장 → OS 부분 갱신** 순서와, 🔴 **색인 값은 그 자산의 '전 스킬' 행에서 만든다**
     — 방금 판정한 스킬 라벨만 실어 보내면 다른 스킬 라벨이 색인에서 증발한다(코어 구현확정 G2).
  3. **실패는 행을 남기지 않는다**(spec §2): 판정 실패·전송 예외는 저장·색인 없이 다음 배치 재대상.
  4. ``--dry-run``(쓰기 0)·재실행 멱등·diff 리포트(라벨 분포·해당없음 비율·신규/갱신/실패).

설계 경계(docs/테스트_가이드.md §0): 조립부(``run_classify``)는 DB·LLM·OS 를 아예 모르고, 실제
커넥션·클라이언트는 ``main`` 만 잡는다(``run_opensearch_resync``·084 배치와 동형).
"""

from __future__ import annotations

import unittest
import uuid
from typing import Any
from unittest import mock

from processing.app import run_mm_classify as rc
from src.mm_classify import (
    UNASSIGNED_LABEL_CODE,
    JudgeFailure,
    SkillJudgement,
    load_skill,
)

# 테스트용 자산 id — 실 자산 id 를 쓰지 않는다(고정 더미 UUIDv7 형태).
_A1 = str(uuid.UUID("018f0000-0000-7000-8000-0000000000b1"))
_A2 = str(uuid.UUID("018f0000-0000-7000-8000-0000000000b2"))

_FOOD = {
    "skill": "음식 콘텐츠",
    "skill_code": "food_content",
    "version": 2,
    "policy": {"selection": "multi", "unassigned": "해당없음", "max_labels": 30},
    "labels": [
        {"code": "recipe", "name": "레시피",
         "definition": "조리 과정·재료 손질이 중심인 자산", "not": "음식의 역사·유래 설명"},
        {"code": "food_culture", "name": "식문화",
         "definition": "음식의 역사·유래·문화 배경", "not": "조리 과정 설명"},
    ],
}
_TRAVEL = {
    "skill": "여행 콘텐츠",
    "skill_code": "travel_content",
    "version": 1,
    "policy": {"selection": "single", "unassigned": "해당없음", "max_labels": 10},
    "labels": [
        {"code": "landscape", "name": "풍경", "definition": "자연 경관이 중심", "not": "인물 중심"},
    ],
}


def _skill(raw: dict[str, Any] = _FOOD):
    """검증된 스킬 객체(코어 ``load_skill`` 을 그대로 통과시킨다)."""
    return load_skill(raw)


def _material(asset_id: str, summary: str = "요약", keywords: tuple[str, ...] = ("김치",)):
    """판정 재료 한 건."""
    return {"asset_id": asset_id, "summary": summary, "keywords": list(keywords)}


def _ok(*codes: str) -> SkillJudgement:
    """성공 판정(라벨 코드만 지정 — 표시명은 리포트에 쓰지 않는다)."""
    return SkillJudgement(ok=True, label_names=tuple(codes), label_codes=tuple(codes))


class _FakePersist:
    """``replace_asset_labels`` 자리 — 호출 인자를 기록하고 삽입 행 수를 돌려준다."""

    def __init__(self, *, fail_on: set[str] | None = None):
        self.calls: list[tuple[str, str, int, SkillJudgement]] = []
        self._fail_on = set(fail_on or ())

    def __call__(self, skill: Any, asset_id: str, judgement: SkillJudgement) -> int:
        if asset_id in self._fail_on:
            raise RuntimeError("simulated persist failure")
        self.calls.append((skill.skill_code, asset_id, skill.version, judgement))
        return len(judgement.label_codes)


class TestSkillIsolation(unittest.TestCase):
    """spec §1 — 스킬 하나 = 호출 하나. 여러 스킬을 한 프롬프트에 섞지 않는다."""

    def test_each_skill_judged_separately(self) -> None:
        seen: list[tuple[str, str]] = []

        def _judge(skill, summary, keywords, **kwargs):  # noqa: ARG001
            seen.append((skill.skill_code, summary))
            return _ok("recipe") if skill.skill_code == "food_content" else _ok("landscape")

        persist = _FakePersist()
        report = rc.run_classify(
            [_skill(_FOOD), _skill(_TRAVEL)],
            targets_fn=lambda skill: [_material(_A1, summary=skill.skill_code)],
            judge_fn=_judge,
            persist_fn=persist,
        )
        # 자산 1건 × 스킬 2개 = 판정 2회(섞이지 않는다).
        self.assertEqual(seen, [("food_content", "food_content"), ("travel_content", "travel_content")])
        self.assertEqual([c[0] for c in persist.calls], ["food_content", "travel_content"])
        self.assertEqual([s["skill_code"] for s in report["skills"]],
                         ["food_content", "travel_content"])

    def test_persist_stamps_db_skill_version(self) -> None:
        # 재선별 기준이 DB 카운터라 판정 행에 **그 버전**이 찍혀야 한다(안 찍히면 무한 재선별).
        persist = _FakePersist()
        rc.run_classify(
            [_skill(_FOOD)],
            targets_fn=lambda skill: [_material(_A1)],
            judge_fn=lambda *a, **k: _ok("recipe"),
            persist_fn=persist,
        )
        self.assertEqual(persist.calls[0][2], 2)


class TestOpenSearchPartialUpdate(unittest.TestCase):
    """spec §5 — 판정 뒤 ``mm_skill_labels`` 부분 갱신. 색인 값은 **전 스킬 행**에서 만든다."""

    def test_index_keys_include_other_skills(self) -> None:
        # 🔴 이 자산에는 다른 스킬(travel_content) 판정도 있다. 방금 판정한 스킬 라벨만 실어 보내면
        #    그 라벨이 색인에서 증발한다(코어 fetch_asset_label_rows 가 존재하는 이유).
        rows = [
            {"skill_code": "food_content", "label_code": "recipe"},
            {"skill_code": "travel_content", "label_code": "landscape"},
        ]
        indexed: list[tuple[str, list[str]]] = []
        report = rc.run_classify(
            [_skill(_FOOD)],
            targets_fn=lambda skill: [_material(_A1)],
            judge_fn=lambda *a, **k: _ok("recipe"),
            persist_fn=_FakePersist(),
            labels_fn=lambda asset_id: rows,
            index_fn=lambda asset_id, keys: indexed.append((asset_id, keys)),
        )
        self.assertEqual(indexed, [(_A1, ["food_content/recipe", "travel_content/landscape"])])
        self.assertEqual(report["indexed"], 1)

    def test_unassigned_is_not_indexed(self) -> None:
        # 미부여는 DB 이력으로 남지만 패싯 축에는 올리지 않는다(노출 정책은 코어 한 곳).
        rows = [{"skill_code": "food_content", "label_code": UNASSIGNED_LABEL_CODE}]
        indexed: list[tuple[str, list[str]]] = []
        rc.run_classify(
            [_skill(_FOOD)],
            targets_fn=lambda skill: [_material(_A1)],
            judge_fn=lambda *a, **k: _ok(UNASSIGNED_LABEL_CODE),
            persist_fn=_FakePersist(),
            labels_fn=lambda asset_id: rows,
            index_fn=lambda asset_id, keys: indexed.append((asset_id, keys)),
        )
        # 빈 목록도 **그대로 실어 보낸다** — 생략하면 옛 라벨이 색인에 남는다(강등 잔재).
        self.assertEqual(indexed, [(_A1, [])])

    def test_no_index_fn_skips_opensearch(self) -> None:
        # OS 가 없는 환경(또는 --no-index)에서도 DB 판정은 정상 진행된다.
        persist = _FakePersist()
        report = rc.run_classify(
            [_skill(_FOOD)],
            targets_fn=lambda skill: [_material(_A1)],
            judge_fn=lambda *a, **k: _ok("recipe"),
            persist_fn=persist,
        )
        self.assertEqual(len(persist.calls), 1)
        self.assertEqual(report["indexed"], 0)


class TestFailureRules(unittest.TestCase):
    """실패는 행을 남기지 않는다 — 저장·색인 모두 건너뛴다(spec §2·§4)."""

    def test_judge_failure_is_not_persisted_or_indexed(self) -> None:
        persist = _FakePersist()
        indexed: list[Any] = []
        report = rc.run_classify(
            [_skill(_FOOD)],
            targets_fn=lambda skill: [_material(_A1)],
            judge_fn=lambda *a, **k: SkillJudgement(
                ok=False, failure=JudgeFailure.OUT_OF_VOCAB, detail="어휘 밖"),
            persist_fn=persist,
            labels_fn=lambda asset_id: [],
            index_fn=lambda asset_id, keys: indexed.append(asset_id),
        )
        self.assertEqual(persist.calls, [])
        self.assertEqual(indexed, [])
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["skills"][0]["failures"], {"out_of_vocab": 1})

    def test_exception_isolated_per_asset(self) -> None:
        def _judge(skill, summary, keywords, **kwargs):  # noqa: ARG001
            if summary == "폭발":
                raise RuntimeError("connection reset")
            return _ok("recipe")

        persist = _FakePersist()
        report = rc.run_classify(
            [_skill(_FOOD)],
            targets_fn=lambda skill: [_material(_A1, summary="폭발"), _material(_A2)],
            judge_fn=_judge,
            persist_fn=persist,
        )
        self.assertEqual([c[1] for c in persist.calls], [_A2])
        # 사유에는 예외 **타입명**만 담는다(헌법 10조 — 메시지·경로 미포함).
        self.assertEqual(report["skills"][0]["failures"], {"exception:RuntimeError": 1})
        self.assertNotIn("connection reset", str(report))

    def test_persist_exception_isolated(self) -> None:
        persist = _FakePersist(fail_on={_A1})
        report = rc.run_classify(
            [_skill(_FOOD)],
            targets_fn=lambda skill: [_material(_A1), _material(_A2)],
            judge_fn=lambda *a, **k: _ok("recipe"),
            persist_fn=persist,
        )
        self.assertEqual(report["judged"], 1)
        self.assertEqual(report["failed"], 1)


class TestReportAggregation(unittest.TestCase):
    """diff 리포트 — 라벨 분포·해당없음 비율(커버리지 갭)·신규/갱신."""

    def test_label_distribution_and_unassigned_ratio(self) -> None:
        def _judge(skill, summary, keywords, **kwargs):  # noqa: ARG001
            return {"1": _ok("recipe"), "2": _ok("recipe", "food_culture"),
                    "3": _ok(UNASSIGNED_LABEL_CODE), "4": _ok(UNASSIGNED_LABEL_CODE)}[summary]

        report = rc.run_classify(
            [_skill(_FOOD)],
            targets_fn=lambda skill: [_material(_A1, summary=s) for s in ("1", "2", "3", "4")],
            judge_fn=_judge,
            persist_fn=_FakePersist(),
        )
        skill_report = report["skills"][0]
        self.assertEqual(skill_report["labels"],
                         {"recipe": 2, "food_culture": 1, UNASSIGNED_LABEL_CODE: 2})
        self.assertEqual(skill_report["unassigned"], 2)
        self.assertEqual(skill_report["unassigned_ratio"], 0.5)
        self.assertEqual(skill_report["rows"], 5)  # multi — 자산당 1..N행

    def test_created_vs_updated(self) -> None:
        # 판정 **전** 행이 없으면 신규, 있으면 갱신(스킬 개정 백필).
        prior = {_A1: [], _A2: [{"skill_code": "food_content", "label_code": "food_culture"}]}
        report = rc.run_classify(
            [_skill(_FOOD)],
            targets_fn=lambda skill: [_material(_A1), _material(_A2)],
            judge_fn=lambda *a, **k: _ok("recipe"),
            persist_fn=_FakePersist(),
            labels_fn=lambda asset_id: prior[asset_id],
        )
        skill_report = report["skills"][0]
        self.assertEqual(skill_report["created"], 1)
        self.assertEqual(skill_report["updated"], 1)

    def test_format_report_shows_skill_labels_and_gap(self) -> None:
        report = rc.run_classify(
            [_skill(_FOOD)],
            targets_fn=lambda skill: [_material(_A1)],
            judge_fn=lambda *a, **k: _ok(UNASSIGNED_LABEL_CODE),
            persist_fn=_FakePersist(),
        )
        line = rc.format_report(report)
        self.assertIn("음식 콘텐츠", line)
        self.assertIn("해당없음", line)  # 커버리지 갭이 눈에 보여야 한다(파일럿 발견 2)

    def test_no_targets_makes_no_calls(self) -> None:
        judge = mock.MagicMock()
        report = rc.run_classify(
            [_skill(_FOOD)], targets_fn=lambda skill: [], judge_fn=judge,
            persist_fn=_FakePersist(),
        )
        judge.assert_not_called()
        self.assertEqual(report["targets"], 0)


class TestDryRunAndIdempotence(unittest.TestCase):
    """``--dry-run``(쓰기 0)·재실행 멱등."""

    def test_dry_run_writes_nothing_but_reports(self) -> None:
        persist = _FakePersist()
        indexed: list[Any] = []
        report = rc.run_classify(
            [_skill(_FOOD)],
            targets_fn=lambda skill: [_material(_A1)],
            judge_fn=lambda *a, **k: _ok("recipe"),
            persist_fn=persist,
            labels_fn=lambda asset_id: [],
            index_fn=lambda asset_id, keys: indexed.append(asset_id),
            dry_run=True,
        )
        self.assertEqual(persist.calls, [])
        self.assertEqual(indexed, [])
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["skills"][0]["labels"], {"recipe": 1})  # 무엇이 붙을지는 보고

    def test_missing_persist_fn_rejected_when_not_dry_run(self) -> None:
        with self.assertRaises(ValueError):
            rc.run_classify([_skill(_FOOD)], targets_fn=lambda skill: [], persist_fn=None)

    def test_same_input_same_report(self) -> None:
        kwargs = {
            "targets_fn": lambda skill: [_material(_A1), _material(_A2)],
            "judge_fn": lambda *a, **k: _ok("recipe"),
        }
        first = rc.run_classify([_skill(_FOOD)], persist_fn=_FakePersist(), **kwargs)
        second = rc.run_classify([_skill(_FOOD)], persist_fn=_FakePersist(), **kwargs)
        self.assertEqual(first, second)


class TestDbHelpers(unittest.TestCase):
    """DB 헬퍼는 코어 함수에 위임한다 — SQL 을 파이프에서 다시 쓰지 않는다."""

    def test_load_active_skills_delegates_and_restores_objects(self) -> None:
        row = {"skill_id": "s1", "skill_code": "food_content", "name": "음식 콘텐츠",
               "version": 3, "policy": _FOOD["policy"], "labels": _FOOD["labels"],
               "status": "active"}
        with mock.patch.object(rc, "fetch_active_skills", return_value=[row]) as m_fetch:
            skills = rc.load_active_skills(mock.MagicMock())
        m_fetch.assert_called_once()
        self.assertEqual([s.skill_code for s in skills], ["food_content"])
        self.assertEqual(skills[0].version, 3)  # 파일이 아니라 **DB 카운터**가 정본

    def test_fetch_skill_targets_uses_pending_then_materials(self) -> None:
        skill = _skill(_FOOD)
        with mock.patch.object(rc, "fetch_pending_asset_ids", return_value=[_A1]) as m_pending, \
                mock.patch.object(rc, "fetch_asset_materials",
                                  return_value=[_material(_A1)]) as m_mat:
            rows = rc.fetch_skill_targets(mock.MagicMock(), skill, limit=30)
        self.assertEqual(m_pending.call_args.kwargs["skill_code"], "food_content")
        self.assertEqual(m_pending.call_args.kwargs["skill_version"], 2)
        self.assertEqual(m_pending.call_args.kwargs["limit"], 30)
        self.assertEqual(m_mat.call_args.kwargs["asset_ids"], [_A1])
        self.assertEqual(rows, [_material(_A1)])

    def test_fetch_skill_targets_skips_materials_when_nothing_pending(self) -> None:
        with mock.patch.object(rc, "fetch_pending_asset_ids", return_value=[]), \
                mock.patch.object(rc, "fetch_asset_materials") as m_mat:
            rows = rc.fetch_skill_targets(mock.MagicMock(), _skill(_FOOD))
        m_mat.assert_not_called()  # 빈 목록으로 재료 질의를 돌리지 않는다
        self.assertEqual(rows, [])


class _FakeDB:
    """``execute_in_transaction(콜백)`` 을 가짜 커넥션으로 즉시 실행하는 최소 더블(실 DB 불요)."""

    def __init__(self) -> None:
        self.conn = mock.MagicMock(name="conn")

    def execute_in_transaction(self, fn, idempotent=True):  # noqa: ARG002 — 서명 호환만
        return fn(self.conn)


def _settings(*, enabled: bool = True, index: str = "assets"):
    """설정 더블 — 배치가 읽는 두 값만 갖춘다."""
    from types import SimpleNamespace

    return SimpleNamespace(
        mm_classify=SimpleNamespace(enabled=enabled),
        opensearch=SimpleNamespace(index=index),
    )


class TestRunBatchWiring(unittest.TestCase):
    """배선 — 명령행과 DAG 가 **같은 함수**를 부른다(배선을 두 벌 두면 한쪽만 고쳐진다)."""

    def _patches(self, *, skills=None, targets=None, settings=None):
        """배치가 부르는 DB·LLM·OS 경계를 전부 가짜로 바꾼다."""
        return {
            "skills": mock.patch.object(rc, "load_active_skills",
                                        return_value=list(skills or [_skill(_FOOD)])),
            "targets": mock.patch.object(rc, "fetch_skill_targets",
                                         return_value=list(targets or [])),
            "labels": mock.patch.object(rc, "fetch_asset_label_rows", return_value=[]),
            "judge": mock.patch.object(rc, "judge_asset_labels", return_value=_ok("recipe")),
            "persist": mock.patch.object(rc, "replace_asset_labels", return_value=1),
            "cfg": mock.patch("src.config.settings.get_current_settings",
                              return_value=settings or _settings()),
        }

    def test_delegates_to_run_classify_per_skill(self) -> None:
        patches = self._patches(targets=[_material(_A1), _material(_A2)])
        with patches["skills"] as m_skills, patches["targets"] as m_targets, patches["labels"], \
                patches["judge"], patches["persist"] as m_persist, patches["cfg"]:
            report = rc.run_batch(_FakeDB(), limit=30, no_index=True)
        m_skills.assert_called_once()                       # 활성 스킬 조회는 배치당 1회
        self.assertEqual(m_targets.call_args.kwargs["limit"], 30)
        self.assertEqual(m_persist.call_count, 2)           # 저장은 자산마다(트랜잭션 분리)
        self.assertEqual(report["judged"], 2)

    def test_disabled_toggle_does_nothing(self) -> None:
        patches = self._patches(targets=[_material(_A1)], settings=_settings(enabled=False))
        with patches["skills"] as m_skills, patches["targets"] as m_targets, patches["labels"], \
                patches["judge"] as m_judge, patches["persist"], patches["cfg"]:
            report = rc.run_batch(_FakeDB())
        for m in (m_skills, m_targets, m_judge):
            m.assert_not_called()
        self.assertEqual(report["skipped"], "disabled")

    def test_dry_run_writes_nothing(self) -> None:
        patches = self._patches(targets=[_material(_A1)])
        with patches["skills"], patches["targets"], patches["labels"], patches["judge"], \
                patches["persist"] as m_persist, patches["cfg"], \
                mock.patch("src.search.opensearch_sync.get_client") as m_client:
            report = rc.run_batch(_FakeDB(), dry_run=True)
        m_persist.assert_not_called()
        m_client.assert_not_called()   # dry-run 은 OS 클라이언트를 만들지도 않는다
        self.assertTrue(report["dry_run"])

    def test_index_wired_when_writing(self) -> None:
        patches = self._patches(targets=[_material(_A1)])
        with patches["skills"], patches["targets"], patches["labels"], patches["judge"], \
                patches["persist"], patches["cfg"], \
                mock.patch("src.search.opensearch_sync.get_client") as m_client, \
                mock.patch("src.search.opensearch_sync.update_asset_mm_skill_labels") as m_upd:
            report = rc.run_batch(_FakeDB())
        m_client.assert_called_once()
        m_upd.assert_called_once()
        # (client, index, asset_id, keys) — 인덱스는 설정값을 그대로 쓴다.
        self.assertEqual(m_upd.call_args.args[1], "assets")
        self.assertEqual(m_upd.call_args.args[2], _A1)
        self.assertEqual(report["indexed"], 1)

    def test_skill_filter_and_unknown_code(self) -> None:
        patches = self._patches(skills=[_skill(_FOOD), _skill(_TRAVEL)],
                                targets=[_material(_A1)])
        with patches["skills"], patches["targets"], patches["labels"], patches["judge"], \
                patches["persist"], patches["cfg"]:
            report = rc.run_batch(_FakeDB(), skill_codes=["travel_content"], no_index=True)
        self.assertEqual([s["skill_code"] for s in report["skills"]], ["travel_content"])

        with patches["skills"], patches["targets"], patches["labels"], patches["judge"], \
                patches["persist"], patches["cfg"], self.assertRaises(ValueError):
            # 오타를 조용히 "0건 처리"로 흡수하지 않는다(왜 안 도는지 추적하게 된다).
            rc.run_batch(_FakeDB(), skill_codes=["typo_code"], no_index=True)


class TestParser(unittest.TestCase):
    """명령행 — 기본값과 위험 옵션."""

    def test_defaults(self) -> None:
        args = rc._build_parser().parse_args([])
        self.assertEqual(args.env, "dev")
        self.assertFalse(args.dry_run)
        self.assertIsNone(args.limit)
        self.assertFalse(args.no_index)
        self.assertEqual(args.skills, [])

    def test_explicit(self) -> None:
        args = rc._build_parser().parse_args(
            ["--env", "prod", "--dry-run", "--limit", "100", "--no-index", "food_content"]
        )
        self.assertEqual(args.env, "prod")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.limit, 100)
        self.assertTrue(args.no_index)
        self.assertEqual(args.skills, ["food_content"])


if __name__ == "__main__":
    unittest.main()
