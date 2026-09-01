"""084 T006 — 멀티모달 메타 소속 배치 러너 단위 테스트 (LLM·DB 불필요).

무엇을 덮는가: ``processing/app/run_mm_meta_binding.py`` 의 **조립부**다. 판정(LLM)·저장(DB)은
전부 주입 seam 으로 갈아 끼우고, 러너가 스스로 책임지는 것만 검증한다 —

  1. **호출 순서**: 판정 → ``apply_rules`` → ``resolve_registered_aliases`` → (발굴 모드 분기) → 저장.
     한 단계라도 빠지면 등록 메타가 갈라지거나 광역 잡동사니가 살아남는다(spec §3·§6-1).
  2. **발굴 모드**(spec §1-1): ``propose`` 는 이미 존재하는 메타에 일치하는 판정만 소속시키고
     미등록 개체는 **후보 리포트**로만 낸다. ``auto`` 는 전량 생성한다.
  3. **병합 색인 발산 방지**(T017·T018 회귀 봉인): 색인은 배치 시작에 한 번 만들어 넘기고, 러너는
     자산 루프에서 **찾아보기만** 한다. 훑으면 터지는 가짜 색인으로 그것을 코드로 막는다.
  4. **실패 격리·이력 규율**: 판정 실패·전송 예외는 자산 하나만 건너뛰고, 실패는 **저장하지 않는다**
     (이력 미기록 = 다음 배치 재대상 · spec §2).
  5. ``--dry-run``(쓰기 0)·재실행 멱등·diff 리포트 집계.
  6. **타입 어휘 배선**(F05 · spec §10): 어휘는 배치 시작에 **한 번**, 색인과 **같은 트랜잭션**에서
     읽어 판정(``type_defs=``)에 싣고, 그 어휘로 정한 문안 판(``prompt_version_for``)을
     **재선별 술어와 저장 스탬프 양쪽**에 같은 값으로 쓴다. 🔴 둘이 갈리면 매 배치가 같은 자산을
     영원히 다시 판정한다(DAG 가 자기 자신을 다시 깨우는 연속 드레인이라 끝나지도 않는다).

설계 경계(docs/테스트_가이드.md §0): 조립부(``run_binding``·``run_describe``)는 DB·LLM 을 아예 모르고,
실제 커넥션·클라이언트는 ``main`` 만 잡는다(실행은 사람 · ``run_opensearch_resync`` 선례와 동형).
"""

from __future__ import annotations

import sys
import unittest
import uuid
from collections.abc import Iterator, Mapping
from typing import Any
from unittest import mock

from processing.app import run_mm_meta_binding as rb
from src.mm_meta import (
    ENTITY_TYPE_DEFS,
    LINEAGE_ACTIVITY,
    PROMPT_VERSION,
    PROMPT_VERSION_WITHOUT_TYPE_DEFS,
    RULE_VERSION,
    EntityJudgement,
    EntityTypeDef,
    ExtractedEntity,
    JudgeFailure,
)

# 테스트용 자산 id — 실 자산 id 를 쓰지 않는다(고정 더미 UUIDv7 형태).
_A1 = str(uuid.UUID("018f0000-0000-7000-8000-0000000000a1"))
_A2 = str(uuid.UUID("018f0000-0000-7000-8000-0000000000a2"))
_A3 = str(uuid.UUID("018f0000-0000-7000-8000-0000000000a3"))


def _material(asset_id: str, summary: str = "요약", keywords: tuple[str, ...] = ("키워드",)):
    """배치 재료 한 건(자산 id·요약·키워드)."""
    return {"asset_id": asset_id, "summary": summary, "keywords": list(keywords)}


def _ok(*entities: ExtractedEntity) -> EntityJudgement:
    """성공 판정(개체 0건도 성공이다)."""
    return EntityJudgement(ok=True, entities=tuple(entities))


class _LookupOnlyIndex(Mapping):
    """**찾아보기만** 허용하는 가짜 색인 — 훑으면 그 자리에서 실패한다.

    T017·T018 이 코어에 심어 둔 회귀 봉인과 같은 장치다. 러너가 자산 루프 안에서 색인을 복사하거나
    (``{**index}``) 훑으면 자산 수 × 개체 수로 발산한다(10만 × 10만 = 100억). 그 실수를 **테스트가
    코드로** 막는다.
    """

    def __init__(self, data: dict[Any, Any]):
        self._data = dict(data)
        self.lookups = 0

    def __getitem__(self, key: Any) -> Any:
        self.lookups += 1
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        self.lookups += 1
        return key in self._data

    def __iter__(self) -> Iterator[Any]:
        raise AssertionError("색인을 훑었다 — 자산 루프에서는 찾아보기만 해야 한다(T017 결함 ②)")

    def __len__(self) -> int:
        raise AssertionError("색인 길이를 쟀다 — 자산 루프에서는 찾아보기만 해야 한다")


class _FakePersist:
    """``upsert_entity_edges`` 자리에 끼우는 가짜 저장부 — 호출 인자를 기록하고 결과를 흉내 낸다."""

    def __init__(self, *, deleted: int = 0, fail_on: set[str] | None = None):
        self.calls: list[tuple[str, tuple[ExtractedEntity, ...]]] = []
        self._deleted = deleted
        self._fail_on = set(fail_on or ())

    def __call__(self, asset_id: str, entities) -> dict[str, Any]:
        items = tuple(entities)
        self.calls.append((asset_id, items))
        if asset_id in self._fail_on:
            raise RuntimeError("simulated persist failure")
        return {
            "asset_id": asset_id,
            "edges_deleted": self._deleted,
            "edges_inserted": len(items),
            "entities": [
                {"entity_type": e.entity_type, "entity_uid": e.uid, "name": e.name,
                 "keyword": e.keyword, "node_id": "n"}
                for e in items
            ],
            "prompt_version": PROMPT_VERSION,
            "rule_version": RULE_VERSION,
        }


class _FakeCursor:
    """``cursor(row_factory=dict_row)`` 흉내 — 실행한 SQL·파라미터를 남기고 준비된 행을 돌려준다."""

    def __init__(self, rows: list[dict[str, Any]], sink: list[tuple[str, tuple[Any, ...]]]):
        self._rows = rows
        self._sink = sink

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *a: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        self._sink.append((" ".join(sql.split()), tuple(params)))

    def fetchall(self) -> list[dict[str, Any]]:
        return list(self._rows)

    def fetchone(self) -> dict[str, Any] | None:
        """단일 행 조회(타입 어휘 ``fetch_meta_type_vocab`` 이 쓰는 모양).

        준비된 행이 없으면 ``None`` — 코어 계약상 "등록 행 없음"이라 코드 프리셋 폴백이 된다.
        """
        return dict(self._rows[0]) if self._rows else None


class _FakeConn:
    """조회 전용 가짜 커넥션(실 DB 불요)."""

    def __init__(self, rows: list[dict[str, Any]] | None = None):
        self.rows = rows or []
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def cursor(self, row_factory: Any = None) -> _FakeCursor:  # noqa: ARG002 — 서명 호환만
        return _FakeCursor(self.rows, self.executed)


class TestResolveDiscoveryMode(unittest.TestCase):
    """발굴 모드 해소(순수) — 기본 propose · 어휘 밖은 fail-fast."""

    def test_default_is_propose(self) -> None:
        # 미설정·빈 값은 기본값 propose — 잡동사니 메타를 만들지 않는 쪽이 기본이다(spec §1-1).
        self.assertEqual(rb.resolve_discovery_mode(None), rb.DISCOVERY_PROPOSE)
        self.assertEqual(rb.resolve_discovery_mode("   "), rb.DISCOVERY_PROPOSE)

    def test_auto_and_case_insensitive(self) -> None:
        self.assertEqual(rb.resolve_discovery_mode("auto"), rb.DISCOVERY_AUTO)
        self.assertEqual(rb.resolve_discovery_mode(" AUTO "), rb.DISCOVERY_AUTO)
        self.assertEqual(rb.resolve_discovery_mode("Propose"), rb.DISCOVERY_PROPOSE)

    def test_unknown_mode_fails_fast(self) -> None:
        # 오타를 조용히 기본값으로 흡수하면 "왜 메타가 안 생기나"(또는 그 반대)를 추적하게 된다.
        with self.assertRaises(ValueError):
            rb.resolve_discovery_mode("all")


class TestDiscoverySplit(unittest.TestCase):
    """등록 여부 분리(순수) — propose 모드의 판단 한 곳."""

    def test_matched_and_withheld(self) -> None:
        seoul = ExtractedEntity(keyword="서울", name="서울특별시", entity_type="장소")
        paris = ExtractedEntity(keyword="파리", name="파리", entity_type="장소")
        index = _LookupOnlyIndex({("장소", "서울특별시"): "서울특별시"})
        matched, withheld = rb.split_by_registration((seoul, paris), index)
        self.assertEqual([e.name for e in matched], ["서울특별시"])
        self.assertEqual([e.name for e in withheld], ["파리"])

    def test_empty_index_withholds_everything(self) -> None:
        e = ExtractedEntity(keyword="k", name="제주도", entity_type="장소")
        matched, withheld = rb.split_by_registration((e,), {})
        self.assertEqual(matched, ())
        self.assertEqual([x.name for x in withheld], ["제주도"])


class TestRunBindingPipelineOrder(unittest.TestCase):
    """판정 → 규칙 → 별칭 → 저장 — 한 단계도 건너뛰지 않는다."""

    def test_rules_applied_between_judge_and_persist(self) -> None:
        """판정 결과가 **규칙을 거쳐** 저장으로 간다 — 한 단계도 건너뛰지 않는다.

        🔴 제외 예시가 `대한민국` → `북극` 으로 바뀐 이유(087 T005 · 코어 커밋 `5fd8f4f`):
        광역 제외 20종 중 **19종이 코드 목록에서 '장소' 정의문(LLM 프롬프트)으로 이관**됐고,
        코드에는 `북극` 하나만 남았다("북극은 빼고 남극은 남긴다"는 **의미가 아니라 정책**이라
        LLM 이 맞추지 못한 유일한 항목이다).

        이 테스트는 ``judge_fn`` 을 가짜로 주입해 LLM 을 우회하므로 **정의문이 개입할 자리가 없다.**
        그래서 `대한민국` 을 넣으면 규칙이 걸러내지 못해 실패한다 — 코어는 갱신됐는데 이 파이프
        테스트가 함께 갱신되지 않아 남아 있던 것을 2026-09-01 에 맞췄다.
        검증 의도(규칙 단계가 실제로 돈다)는 그대로다.
        """
        # '북극'(코드 목록 유일 잔존)·'올림픽 정식 종목'(스톱패턴)은 규칙에서 떨어지고 '제주도'만 남는다.
        judged = _ok(
            ExtractedEntity(keyword="북극", name="북극", entity_type="장소"),
            ExtractedEntity(keyword="올림픽 정식 종목", name="올림픽", entity_type="사건"),
            ExtractedEntity(keyword="제주 장마", name="제주도", entity_type="장소"),
        )
        persist = _FakePersist()
        report = rb.run_binding(
            [_material(_A1)],
            mode=rb.DISCOVERY_AUTO,
            judge_fn=lambda *a, **k: judged,
            persist_fn=persist,
        )
        self.assertEqual(len(persist.calls), 1)
        self.assertEqual([e.name for e in persist.calls[0][1]], ["제주도"])
        self.assertEqual(report["edges_inserted"], 1)

    def test_registered_alias_substitution_before_persist(self) -> None:
        # 별칭 치환이 빠지면 '이지은' 이 별개 메타로 갈라진다(spec §6-1 매칭 합류).
        judged = _ok(ExtractedEntity(keyword="이지은", name="이지은", entity_type="인물"))
        alias = _LookupOnlyIndex({("인물", "이지은"): "아이유", ("인물", "아이유"): "아이유"})
        persist = _FakePersist()
        rb.run_binding(
            [_material(_A1)],
            mode=rb.DISCOVERY_AUTO,
            alias_index=alias,
            judge_fn=lambda *a, **k: judged,
            persist_fn=persist,
        )
        self.assertEqual([e.name for e in persist.calls[0][1]], ["아이유"])
        # 근거 키워드는 원문 그대로 보존된다(reason 스탬프의 kw=).
        self.assertEqual(persist.calls[0][1][0].keyword, "이지은")

    def test_official_index_is_lookup_only_across_assets(self) -> None:
        # T017 결함 ② 회귀 봉인 — 자산이 여러 건이어도 색인을 훑거나 복사하지 않는다.
        official = _LookupOnlyIndex({("장소", "서울"): "서울특별시"})
        alias = _LookupOnlyIndex({("장소", "서울특별시"): "서울특별시"})
        judged = _ok(ExtractedEntity(keyword="서울", name="서울", entity_type="장소"))
        persist = _FakePersist()
        report = rb.run_binding(
            [_material(_A1), _material(_A2), _material(_A3)],
            mode=rb.DISCOVERY_PROPOSE,
            official_index=official,
            alias_index=alias,
            judge_fn=lambda *a, **k: judged,
            persist_fn=persist,
        )
        # 색인이 실제로 먹었는지도 함께 본다(짧은 표기 → 공식형 → 등록 메타 일치 → 소속).
        self.assertEqual([e.name for e in persist.calls[0][1]], ["서울특별시"])
        self.assertEqual(report["assets_bound"], 3)

    def test_never_builds_indexes_inside_the_loop(self) -> None:
        # 색인 조립은 **배치 시작 1회**(main 책임)다 — 조립부는 DB 조회 함수를 아예 부르지 않는다.
        judged = _ok(ExtractedEntity(keyword="k", name="제주도", entity_type="장소"))
        with mock.patch.object(rb, "fetch_official_name_index") as m_off, \
                mock.patch.object(rb, "fetch_registered_alias_index") as m_alias:
            rb.run_binding(
                [_material(_A1), _material(_A2)],
                mode=rb.DISCOVERY_AUTO,
                judge_fn=lambda *a, **k: judged,
                persist_fn=_FakePersist(),
            )
        m_off.assert_not_called()
        m_alias.assert_not_called()

    def test_summary_limit_is_passed_to_judge(self) -> None:
        # 설정 MM_META_JUDGE_SUMMARY_CHARS 가 실제로 먹는 통로(T014).
        seen: dict[str, Any] = {}

        def _judge(summary, keywords, **kwargs):
            seen.update({"summary": summary, "keywords": list(keywords), **kwargs})
            return _ok()

        rb.run_binding(
            [_material(_A1, summary="요약본", keywords=("가", "나"))],
            mode=rb.DISCOVERY_AUTO,
            summary_max_chars=120,
            judge_fn=_judge,
            persist_fn=_FakePersist(),
        )
        self.assertEqual(seen["summary"], "요약본")
        self.assertEqual(seen["keywords"], ["가", "나"])
        self.assertEqual(seen["summary_max_chars"], 120)


class TestDiscoveryModes(unittest.TestCase):
    """spec §1-1 — propose(등록분만 소속·나머지는 후보) / auto(전량 생성)."""

    def _judged(self) -> EntityJudgement:
        return _ok(
            ExtractedEntity(keyword="한라산", name="제주도", entity_type="장소"),
            ExtractedEntity(keyword="루브르", name="루브르박물관", entity_type="조직"),
        )

    def test_propose_binds_only_registered_and_reports_candidates(self) -> None:
        alias = _LookupOnlyIndex({("장소", "제주도"): "제주도"})
        persist = _FakePersist()
        report = rb.run_binding(
            [_material(_A1), _material(_A2)],
            mode=rb.DISCOVERY_PROPOSE,
            alias_index=alias,
            judge_fn=lambda *a, **k: self._judged(),
            persist_fn=persist,
        )
        # 등록된 '제주도'만 저장되고, 미등록 '루브르박물관' 은 노드·엣지를 만들지 않는다.
        for _aid, entities in persist.calls:
            self.assertEqual([e.name for e in entities], ["제주도"])
        # 후보 리포트: 표기·타입·출현 자산 수.
        self.assertEqual(len(report["candidates"]), 1)
        cand = report["candidates"][0]
        self.assertEqual((cand["entity_type"], cand["name"]), ("조직", "루브르박물관"))
        self.assertEqual(cand["assets"], 2)
        self.assertEqual(report["withheld"], 2)
        # propose 는 정의상 새 메타를 만들지 않는다.
        self.assertEqual(report["new_metas"], [])

    def test_auto_binds_everything_and_reports_new_metas(self) -> None:
        alias = _LookupOnlyIndex({("장소", "제주도"): "제주도"})
        persist = _FakePersist()
        report = rb.run_binding(
            [_material(_A1)],
            mode=rb.DISCOVERY_AUTO,
            alias_index=alias,
            judge_fn=lambda *a, **k: self._judged(),
            persist_fn=persist,
        )
        self.assertEqual([e.name for e in persist.calls[0][1]], ["제주도", "루브르박물관"])
        self.assertEqual(report["candidates"], [])
        # 배치 시작 색인에 없던 것만 '신규 메타'다.
        self.assertEqual([m["name"] for m in report["new_metas"]], ["루브르박물관"])

    def test_propose_still_records_history_for_fully_withheld_asset(self) -> None:
        # 판정 자체는 성공했으므로 이력을 남긴다(개체 0) — 남기지 않으면 매 배치가 같은 자산에
        # LLM 을 다시 부른다(§9-1 "이력 있으면 재호출 0"). 후보 승인 후 재소속은 --rejudge 경로다.
        persist = _FakePersist()
        report = rb.run_binding(
            [_material(_A1)],
            mode=rb.DISCOVERY_PROPOSE,
            judge_fn=lambda *a, **k: self._judged(),
            persist_fn=persist,
        )
        self.assertEqual(len(persist.calls), 1)
        self.assertEqual(persist.calls[0][1], ())
        self.assertEqual(report["assets_empty"], 1)
        self.assertEqual(report["assets_bound"], 0)

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rb.run_binding([], mode="everything", persist_fn=_FakePersist())


class TestFailureIsolation(unittest.TestCase):
    """실패 규율 — 판정 실패는 저장하지 않고, 예외는 자산 하나만 건너뛴다."""

    def test_judge_failure_is_not_persisted(self) -> None:
        # 실패를 저장하면 "판정 완료"로 굳어 재시도가 영구히 멈춘다(spec §2).
        failed = EntityJudgement(ok=False, failure=JudgeFailure.RESPONSE_SHAPE, detail="빈 응답")
        persist = _FakePersist()
        report = rb.run_binding(
            [_material(_A1)],
            mode=rb.DISCOVERY_AUTO,
            judge_fn=lambda *a, **k: failed,
            persist_fn=persist,
        )
        self.assertEqual(persist.calls, [])
        self.assertEqual(report["judged_failed"], 1)
        self.assertEqual(report["failures"], {"response_shape": 1})

    def test_transport_exception_isolated_per_asset(self) -> None:
        def _judge(summary, keywords, **kwargs):  # noqa: ARG001
            if summary == "폭발":
                raise RuntimeError("connection reset")
            return _ok(ExtractedEntity(keyword="k", name="제주도", entity_type="장소"))

        persist = _FakePersist()
        report = rb.run_binding(
            [_material(_A1, summary="폭발"), _material(_A2)],
            mode=rb.DISCOVERY_AUTO,
            judge_fn=_judge,
            persist_fn=persist,
        )
        self.assertEqual([aid for aid, _ in persist.calls], [_A2])  # 두 번째는 계속 처리
        self.assertEqual(report["judged_failed"], 1)
        # 사유에는 예외 **타입명**만 담는다(헌법 10조 — 메시지·경로 미포함).
        self.assertEqual(report["failures"], {"exception:RuntimeError": 1})
        self.assertNotIn("connection reset", str(report))

    def test_persist_exception_isolated(self) -> None:
        persist = _FakePersist(fail_on={_A1})
        judged = _ok(ExtractedEntity(keyword="k", name="제주도", entity_type="장소"))
        report = rb.run_binding(
            [_material(_A1), _material(_A2)],
            mode=rb.DISCOVERY_AUTO,
            judge_fn=lambda *a, **k: judged,
            persist_fn=persist,
        )
        self.assertEqual(report["assets_bound"], 1)
        self.assertEqual(report["failures"], {"exception:RuntimeError": 1})


class TestDryRunAndIdempotence(unittest.TestCase):
    """``--dry-run``(쓰기 0)·재실행 멱등."""

    def test_dry_run_writes_nothing_but_reports(self) -> None:
        judged = _ok(ExtractedEntity(keyword="k", name="제주도", entity_type="장소"))
        persist = _FakePersist()
        report = rb.run_binding(
            [_material(_A1)],
            mode=rb.DISCOVERY_AUTO,
            dry_run=True,
            judge_fn=lambda *a, **k: judged,
            persist_fn=persist,
        )
        self.assertEqual(persist.calls, [])          # 쓰기 0
        self.assertTrue(report["dry_run"])
        self.assertEqual(report["assets_bound"], 1)  # 무엇이 바뀔지는 그대로 보고
        self.assertEqual(report["edges_inserted"], 1)

    def test_missing_persist_fn_is_rejected_when_not_dry_run(self) -> None:
        # 저장부 없이 실행하면 "조용한 0건"이 된다 — 쓰기 모드에서는 fail-fast.
        with self.assertRaises(ValueError):
            rb.run_binding([_material(_A1)], mode=rb.DISCOVERY_AUTO, persist_fn=None)

    def test_same_input_same_report(self) -> None:
        judged = _ok(ExtractedEntity(keyword="k", name="제주도", entity_type="장소"))
        kwargs = {"mode": rb.DISCOVERY_AUTO, "judge_fn": lambda *a, **k: judged}
        first = rb.run_binding([_material(_A1), _material(_A2)],
                               persist_fn=_FakePersist(), **kwargs)
        second = rb.run_binding([_material(_A1), _material(_A2)],
                                persist_fn=_FakePersist(), **kwargs)
        self.assertEqual(first, second)


class TestReportAggregation(unittest.TestCase):
    """diff 리포트 — 엣지 증감·상위 묶음·후보 정렬."""

    def test_counts_edges_and_shrunk_assets(self) -> None:
        judged = _ok(ExtractedEntity(keyword="k", name="제주도", entity_type="장소"))
        persist = _FakePersist(deleted=3)  # 이전 소속 3건 → 이번엔 1건(규칙 강화로 줄어듦)
        report = rb.run_binding(
            [_material(_A1)],
            mode=rb.DISCOVERY_AUTO,
            judge_fn=lambda *a, **k: judged,
            persist_fn=persist,
        )
        self.assertEqual(report["edges_deleted"], 3)
        self.assertEqual(report["edges_inserted"], 1)
        self.assertEqual(report["assets_shrunk"], 1)

    def test_top_bundles_sorted_by_size_then_name(self) -> None:
        def _judge(summary, keywords, **kwargs):  # noqa: ARG001
            names = {"1": ["제주도"], "2": ["제주도", "남극"], "3": ["제주도", "남극"]}[summary]
            return _ok(*[ExtractedEntity(keyword="k", name=n, entity_type="장소") for n in names])

        report = rb.run_binding(
            [_material(_A1, summary="1"), _material(_A2, summary="2"), _material(_A3, summary="3")],
            mode=rb.DISCOVERY_AUTO,
            judge_fn=_judge,
            persist_fn=_FakePersist(),
        )
        top = [(b["name"], b["assets"]) for b in report["top_bundles"]]
        self.assertEqual(top, [("제주도", 3), ("남극", 2)])

    def test_format_report_shows_mode_counts_and_candidates(self) -> None:
        report = rb.run_binding(
            [_material(_A1)],
            mode=rb.DISCOVERY_PROPOSE,
            judge_fn=lambda *a, **k: _ok(
                ExtractedEntity(keyword="루브르", name="루브르박물관", entity_type="조직")),
            persist_fn=_FakePersist(),
        )
        line = rb.format_report(report)
        self.assertIn("propose", line)
        self.assertIn("루브르박물관", line)  # 후보가 사람 눈에 보여야 등록으로 이어진다


class TestDescribeStage(unittest.TestCase):
    """T015 연계 — 설명은 **묶음이 확정된 뒤** 도는 별도 단계다."""

    def _target(self, name: str = "제주도", size: int = 13, reason: str = "missing"):
        return {"entity_type": "장소", "entity_uid": name, "name": name,
                "node_id": "n1", "bundle_size": size, "description": "", "reason": reason}

    def test_members_describe_save_in_order(self) -> None:
        from src.mm_meta import DESC_PROMPT_VERSION, MetaDescription

        saved: list[dict[str, Any]] = []
        report = rb.run_describe(
            [self._target()],
            members_fn=lambda t, u: [("text", "한라산 등반기")],
            describe_fn=lambda name, etype, members: MetaDescription(
                ok=True, description=f"{name}:{len(list(members))}"),
            save_fn=lambda **kw: saved.append(kw),
        )
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["description"], "제주도:1")
        # 저장하는 건수는 **묶음 크기**여야 낡음 판정이 성립한다(재료 줄 수가 아니다).
        self.assertEqual(saved[0]["member_count"], 13)
        self.assertEqual(saved[0]["prompt_version"], DESC_PROMPT_VERSION)
        self.assertEqual(report["described"], 1)
        self.assertEqual(report["reasons"], {"missing": 1})

    def test_failed_description_is_not_saved(self) -> None:
        from src.mm_meta import DescribeFailure, MetaDescription

        saved: list[dict[str, Any]] = []
        report = rb.run_describe(
            [self._target()],
            members_fn=lambda t, u: [("text", "x")],
            describe_fn=lambda *a: MetaDescription(
                ok=False, failure=DescribeFailure.RESPONSE_SHAPE, detail="빈 응답"),
            save_fn=lambda **kw: saved.append(kw),
        )
        self.assertEqual(saved, [])
        self.assertEqual(report["failed"], 1)
        self.assertEqual(report["failures"], {"response_shape": 1})

    def test_dry_run_and_exception_isolation(self) -> None:
        from src.mm_meta import MetaDescription

        saved: list[dict[str, Any]] = []

        def _describe(name, etype, members):  # noqa: ARG001
            if name == "폭발":
                raise RuntimeError("connection reset")
            return MetaDescription(ok=True, description="설명")

        report = rb.run_describe(
            [self._target(name="폭발"), self._target()],
            members_fn=lambda t, u: [("text", "x")],
            describe_fn=_describe,
            save_fn=lambda **kw: saved.append(kw),
            dry_run=True,
        )
        self.assertEqual(saved, [])                # dry-run 은 쓰지 않는다
        self.assertEqual(report["described"], 1)   # 두 번째는 계속 처리
        self.assertEqual(report["failures"], {"exception:RuntimeError": 1})


class TestFetchTargets(unittest.TestCase):
    """대상 선별 SQL — 조회 파라미터·정규화(실 DB 없이 가짜 커서로)."""

    def test_selection_binds_activity_and_versions(self) -> None:
        conn = _FakeConn([{"asset_id": uuid.UUID(_A1), "summary": None, "keywords": ["가", None]}])
        rows = rb.fetch_binding_targets(conn, limit=10)
        sql, params = conn.executed[0]
        self.assertIn("asset_lineage", sql)
        self.assertIn("NOT EXISTS", sql)
        self.assertIn(LINEAGE_ACTIVITY, params)
        self.assertIn(PROMPT_VERSION, params)
        self.assertIn(RULE_VERSION, params)
        self.assertIn(10, params)
        # 조회행 id → str · None 요약·키워드 정규화(프롬프트로 "None" 이 새지 않게).
        self.assertEqual(rows, [{"asset_id": _A1, "summary": "", "keywords": ["가"]}])

    def test_rejudge_drops_the_history_filter(self) -> None:
        # 후보 승인(수동 등록) 뒤 재소속 경로 — 이력이 있어도 다시 판정한다.
        conn = _FakeConn([])
        rb.fetch_binding_targets(conn, rejudge=True)
        sql, _params = conn.executed[0]
        self.assertNotIn("asset_lineage", sql)

    def test_explicit_asset_ids_scope_the_selection(self) -> None:
        conn = _FakeConn([])
        rb.fetch_binding_targets(conn, asset_ids=[_A1, _A2])
        sql, params = conn.executed[0]
        self.assertIn("a.asset_id = ANY", sql)
        self.assertIn([_A1, _A2], params)

    def test_keywords_guard_avoids_type_error(self) -> None:
        # jsonb_array_length 는 배열이 아닌 값에 터진다 — AND 단축평가는 보장되지 않으므로
        # CASE 로 감싸야 한 행 때문에 질의 전체가 죽지 않는다.
        conn = _FakeConn([])
        rb.fetch_binding_targets(conn)
        sql, _ = conn.executed[0]
        self.assertIn("CASE", sql)
        self.assertIn("jsonb_typeof", sql)


class _FakeDB:
    """``execute_in_transaction(콜백)`` 을 가짜 커넥션으로 즉시 실행하는 최소 더블(실 DB 불요)."""

    def __init__(self) -> None:
        self.conn = _FakeConn()
        self.calls = 0

    def execute_in_transaction(self, fn, idempotent=True):  # noqa: ARG002 — 서명 호환만
        self.calls += 1
        return fn(self.conn)


def _settings(*, binding: bool = True, describe: bool = True, chars: int = 250):
    """설정 더블 — 배치가 읽는 세 값만 갖춘다."""
    from types import SimpleNamespace

    return SimpleNamespace(
        mm_meta=SimpleNamespace(
            binding_enabled=binding, describe_enabled=describe, judge_summary_chars=chars
        )
    )


class TestRunBatchWiring(unittest.TestCase):
    """배선 — 🔴 **색인 조립은 배치당 1회**(자산 수에 비례하면 발산한다 · T017 결함 ②)."""

    def _patches(self, *, targets, settings=None):
        """배치가 부르는 DB·LLM 경계를 전부 가짜로 바꾼다."""
        judged = _ok(ExtractedEntity(keyword="k", name="제주도", entity_type="장소"))
        return {
            "official": mock.patch.object(rb, "fetch_official_name_index", return_value={}),
            "alias": mock.patch.object(rb, "fetch_registered_alias_index", return_value={}),
            "targets": mock.patch.object(rb, "fetch_binding_targets", return_value=targets),
            "kind": mock.patch.object(rb, "ensure_mm_member_kind", return_value="k1"),
            "judge": mock.patch.object(rb, "judge_asset_entities", return_value=judged),
            "persist": mock.patch.object(
                rb, "upsert_entity_edges",
                return_value={"edges_deleted": 0, "edges_inserted": 1, "entities": []}),
            "desc": mock.patch.object(rb, "fetch_meta_description_targets", return_value=[]),
            "orphan": mock.patch.object(rb, "fetch_orphan_metas", return_value=[]),
            "cfg": mock.patch("src.config.settings.get_current_settings",
                              return_value=settings or _settings()),
        }

    def test_indexes_built_once_regardless_of_asset_count(self) -> None:
        patches = self._patches(targets=[_material(_A1), _material(_A2), _material(_A3)])
        with patches["official"] as m_off, patches["alias"] as m_alias, patches["targets"], \
                patches["kind"], patches["judge"], patches["persist"] as m_persist, \
                patches["desc"], patches["orphan"], patches["cfg"]:
            result = rb.run_batch(_FakeDB(), mode=rb.DISCOVERY_AUTO)
        self.assertEqual(m_off.call_count, 1)     # 자산 3건이어도 색인은 1회
        self.assertEqual(m_alias.call_count, 1)
        self.assertEqual(m_persist.call_count, 3)  # 저장은 자산마다(트랜잭션 분리)
        self.assertEqual(result["binding"]["assets_bound"], 3)

    def test_disabled_toggle_does_nothing(self) -> None:
        patches = self._patches(targets=[_material(_A1)], settings=_settings(binding=False))
        with patches["official"] as m_off, patches["alias"], patches["targets"] as m_targets, \
                patches["kind"] as m_kind, patches["judge"] as m_judge, patches["persist"], \
                patches["desc"], patches["orphan"], patches["cfg"]:
            result = rb.run_batch(_FakeDB(), mode=rb.DISCOVERY_AUTO)
        # 끄기 ≠ 삭제 — 새 판정만 멈춘다(질의도 LLM 호출도 0).
        for m in (m_off, m_targets, m_kind, m_judge):
            m.assert_not_called()
        self.assertEqual(result["skipped"], "disabled")

    def test_dry_run_skips_kind_bootstrap_and_writes(self) -> None:
        patches = self._patches(targets=[_material(_A1)])
        with patches["official"], patches["alias"], patches["targets"], \
                patches["kind"] as m_kind, patches["judge"], patches["persist"] as m_persist, \
                patches["desc"], patches["orphan"], patches["cfg"]:
            result = rb.run_batch(_FakeDB(), mode=rb.DISCOVERY_AUTO, dry_run=True)
        m_kind.assert_not_called()     # 카탈로그 보장도 쓰기다 — dry-run 은 건드리지 않는다
        m_persist.assert_not_called()
        self.assertTrue(result["binding"]["dry_run"])

    def test_describe_targets_respect_limit(self) -> None:
        # 소속은 상한이 있는데 설명만 없으면, 처음 켜는 시점에 한 태스크가 전량을 LLM 에 태운다
        # (짧게 돌고 남으면 다시 도는 전제가 깨진다).
        many = [("place", f"uid{i}") for i in range(5)]
        patches = self._patches(targets=[])
        patches["desc"] = mock.patch.object(
            rb, "fetch_meta_description_targets", return_value=many)
        with patches["official"], patches["alias"], patches["targets"], patches["kind"], \
                patches["judge"], patches["persist"], patches["desc"], \
                patches["orphan"], patches["cfg"], \
                mock.patch.object(rb, "run_describe", return_value={"failed": 0}) as m_run:
            rb.run_batch(_FakeDB(), mode=rb.DISCOVERY_AUTO, limit=2)
        self.assertEqual(len(m_run.call_args.args[0]), 2)  # 잘라서 넘긴다

    def test_describe_targets_untouched_without_limit(self) -> None:
        # 상한 미지정(기본)은 기존 동작 그대로 — 전량을 넘긴다.
        many = [("place", f"uid{i}") for i in range(5)]
        patches = self._patches(targets=[])
        patches["desc"] = mock.patch.object(
            rb, "fetch_meta_description_targets", return_value=many)
        with patches["official"], patches["alias"], patches["targets"], patches["kind"], \
                patches["judge"], patches["persist"], patches["desc"], \
                patches["orphan"], patches["cfg"], \
                mock.patch.object(rb, "run_describe", return_value={"failed": 0}) as m_run:
            rb.run_batch(_FakeDB(), mode=rb.DISCOVERY_AUTO)
        self.assertEqual(len(m_run.call_args.args[0]), 5)

    def test_describe_stage_respects_setting_and_flag(self) -> None:
        patches = self._patches(targets=[], settings=_settings(describe=False))
        with patches["official"], patches["alias"], patches["targets"], patches["kind"], \
                patches["judge"], patches["persist"], patches["desc"] as m_desc, \
                patches["orphan"], patches["cfg"]:
            result = rb.run_batch(_FakeDB(), mode=rb.DISCOVERY_AUTO)
        m_desc.assert_not_called()     # MM_META_DESCRIBE_ENABLED=0
        self.assertIsNone(result["describe"])

        patches = self._patches(targets=[])
        with patches["official"], patches["alias"], patches["targets"], patches["kind"], \
                patches["judge"], patches["persist"], patches["desc"] as m_desc, \
                patches["orphan"], patches["cfg"]:
            result = rb.run_batch(_FakeDB(), mode=rb.DISCOVERY_AUTO, describe=False)
        m_desc.assert_not_called()     # --no-describe
        self.assertIsNone(result["describe"])

    def test_propose_with_empty_registry_warns(self) -> None:
        # 등록 메타 0 = "이번 배치는 후보 리포트만 낸다"는 정상 상태지만, 조용하면 "왜 묶음이
        # 안 생기나"를 코드에서 찾게 된다 — 배치가 먼저 말해 준다.
        # ⚠️ 이 판정은 **색인을 만든 자리**(run_batch)에서만 한다 — 조립부는 색인의 길이를 재는
        #    것조차 계약 밖이다(가짜 색인이 그것을 막는다).
        patches = self._patches(targets=[_material(_A1)])
        with patches["official"], patches["alias"], patches["targets"], patches["kind"], \
                patches["judge"], patches["persist"], patches["desc"], patches["orphan"], \
                patches["cfg"], \
                self.assertLogs("meta_extract.run_mm_meta_binding", level="WARNING") as logs:
            rb.run_batch(_FakeDB(), mode=rb.DISCOVERY_PROPOSE)
        self.assertTrue(any("후보" in line for line in logs.output), msg=logs.output)

    def test_selection_args_are_passed_through(self) -> None:
        patches = self._patches(targets=[])
        with patches["official"], patches["alias"], patches["targets"] as m_targets, \
                patches["kind"], patches["judge"], patches["persist"], patches["desc"], \
                patches["orphan"], patches["cfg"]:
            rb.run_batch(_FakeDB(), mode=rb.DISCOVERY_PROPOSE, limit=50, rejudge=True,
                         asset_ids=[_A1])
        kwargs = m_targets.call_args.kwargs
        self.assertEqual(kwargs["limit"], 50)
        self.assertTrue(kwargs["rejudge"])
        self.assertEqual(kwargs["asset_ids"], [_A1])


class TestParser(unittest.TestCase):
    """명령행 — 기본값과 위험 옵션."""

    def test_defaults(self) -> None:
        args = rb._build_parser().parse_args([])
        self.assertEqual(args.env, "dev")
        self.assertFalse(args.dry_run)
        self.assertIsNone(args.discovery_mode)  # 미지정=설정(env) 값
        self.assertFalse(args.rejudge)
        self.assertIsNone(args.limit)
        self.assertEqual(args.asset_ids, [])

    def test_explicit(self) -> None:
        args = rb._build_parser().parse_args(
            ["--env", "prod", "--dry-run", "--discovery-mode", "auto", "--limit", "50",
             "--rejudge", "--no-describe", _A1]
        )
        self.assertEqual(args.env, "prod")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.discovery_mode, "auto")
        self.assertEqual(args.limit, 50)
        self.assertTrue(args.rejudge)
        self.assertTrue(args.no_describe)
        self.assertEqual(args.asset_ids, [_A1])


# ── F05 타입 어휘 배선 ──────────────────────────────────────────────────────────
def _registered_defs() -> tuple[EntityTypeDef, ...]:
    """**등록 행에서 온 것처럼 보이는** 정의문 묶음(코드 프리셋과 다른 인스턴스).

    코어 ``fetch_meta_type_vocab`` 은 등록 행이면 새로 만든 튜플을, 행이 없으면 코드 프리셋
    ``ENTITY_TYPE_DEFS`` **그 객체**를 돌려준다 — 러너는 그 차이로 어휘 출처를 리포트에 적는다.

    Returns:
        닫힌 5종 정의문 튜플(내용은 프리셋과 달라도 배선 검증에는 상관없다).
    """
    return tuple(
        EntityTypeDef(code=d.code, name=d.name, definition="등록 정의", exclusion="등록 경계")
        for d in ENTITY_TYPE_DEFS
    )


class TestTypeVocabInRunBinding(unittest.TestCase):
    """어휘가 판정까지 흐르고, 스탬프가 **그 어휘로부터** 정해지는가(F05 잔여 ①)."""

    def test_type_defs_reach_the_judge(self) -> None:
        # 설정만 바꾸고 소비처가 없어 조용히 무동작이 되는 결함(MM_META_JUDGE_SUMMARY_CHARS 류)의
        # 재발 방지 — 어휘가 실제로 판정 함수까지 가야 정의문이 프롬프트에 실린다.
        defs = _registered_defs()
        seen: dict[str, Any] = {}

        def _judge(summary, keywords, **kwargs):  # noqa: ARG001
            seen.update(kwargs)
            return _ok()

        rb.run_binding(
            [_material(_A1)],
            mode=rb.DISCOVERY_AUTO,
            type_defs=defs,
            judge_fn=_judge,
            persist_fn=_FakePersist(),
        )
        self.assertIs(seen["type_defs"], defs)

    def test_no_type_defs_keeps_current_prompt(self) -> None:
        # 미주입이 곧 기존 동작이다(코어 하위호환 계약) — 판정에 None 이 그대로 간다.
        seen: dict[str, Any] = {}

        def _judge(summary, keywords, **kwargs):  # noqa: ARG001
            seen.update(kwargs)
            return _ok()

        rb.run_binding(
            [_material(_A1)], mode=rb.DISCOVERY_AUTO, judge_fn=_judge, persist_fn=_FakePersist()
        )
        self.assertIsNone(seen["type_defs"])

    def test_report_stamps_the_version_of_the_prompt_it_built(self) -> None:
        # 🔴 문안 v1 인데 스탬프만 v2 로 찍히면 정의문 효과 확인·재판정 범위 산정이 불가능하다.
        with_defs = rb.run_binding(
            [_material(_A1)],
            mode=rb.DISCOVERY_AUTO,
            type_defs=_registered_defs(),
            judge_fn=lambda *a, **k: _ok(),
            persist_fn=_FakePersist(),
        )
        without = rb.run_binding(
            [_material(_A1)],
            mode=rb.DISCOVERY_AUTO,
            judge_fn=lambda *a, **k: _ok(),
            persist_fn=_FakePersist(),
        )
        self.assertEqual(with_defs["prompt_version"], PROMPT_VERSION)
        self.assertEqual(without["prompt_version"], PROMPT_VERSION_WITHOUT_TYPE_DEFS)

    def test_vocab_source_labels_registered_preset_and_absent(self) -> None:
        # 운영자가 "등록한 정의문이 적용됐나"를 리포트만 보고 알아야 한다.
        self.assertEqual(rb.vocab_source_of(_registered_defs()), rb.VOCAB_SOURCE_REGISTERED)
        self.assertEqual(rb.vocab_source_of(ENTITY_TYPE_DEFS), rb.VOCAB_SOURCE_PRESET)
        self.assertEqual(rb.vocab_source_of(None), rb.VOCAB_SOURCE_NONE)
        self.assertEqual(rb.vocab_source_of(()), rb.VOCAB_SOURCE_NONE)

    def test_report_carries_vocab_source_and_count(self) -> None:
        report = rb.run_binding(
            [_material(_A1)],
            mode=rb.DISCOVERY_AUTO,
            type_defs=ENTITY_TYPE_DEFS,
            judge_fn=lambda *a, **k: _ok(),
            persist_fn=_FakePersist(),
        )
        self.assertEqual(report["vocab_source"], rb.VOCAB_SOURCE_PRESET)
        self.assertEqual(report["type_defs"], len(ENTITY_TYPE_DEFS))

    def test_format_report_shows_vocab_source_and_version(self) -> None:
        preset = rb.format_report({
            "mode": rb.DISCOVERY_AUTO, "vocab_source": rb.VOCAB_SOURCE_PRESET,
            "type_defs": 5, "prompt_version": PROMPT_VERSION,
        })
        self.assertIn("프리셋", preset)
        self.assertIn(PROMPT_VERSION, preset)
        registered = rb.format_report({
            "mode": rb.DISCOVERY_AUTO, "vocab_source": rb.VOCAB_SOURCE_REGISTERED,
            "type_defs": 5, "prompt_version": PROMPT_VERSION,
        })
        self.assertIn("등록", registered)


class _TxDB:
    """트랜잭션마다 **새 커넥션**을 주는 DB 더블 — "같은 트랜잭션인가"를 객체 동일성으로 검증한다.

    ``_FakeDB`` 는 커넥션 하나를 재사용해 그 질문에 답할 수 없다(무엇을 부르든 같은 객체다).
    """

    def __init__(self) -> None:
        self.conns: list[_FakeConn] = []

    def execute_in_transaction(self, fn, idempotent=True):  # noqa: ARG002 — 서명 호환만
        conn = _FakeConn()
        self.conns.append(conn)
        return fn(conn)


class TestTypeVocabWiringInRunBatch(unittest.TestCase):
    """🔴 배치 배선 — 어휘는 배치당 1회·색인과 한 트랜잭션, 스탬프는 술어와 **같은 값**."""

    def _patches(self, *, targets, vocab=None, settings=None):
        """배치가 부르는 DB·LLM 경계를 전부 가짜로 바꾼다(어휘 조회 포함)."""
        judged = _ok(ExtractedEntity(keyword="k", name="제주도", entity_type="장소"))
        return {
            "vocab": mock.patch.object(
                rb, "fetch_meta_type_vocab",
                return_value=_registered_defs() if vocab is None else vocab),
            "official": mock.patch.object(rb, "fetch_official_name_index", return_value={}),
            "alias": mock.patch.object(rb, "fetch_registered_alias_index", return_value={}),
            "targets": mock.patch.object(rb, "fetch_binding_targets", return_value=targets),
            "kind": mock.patch.object(rb, "ensure_mm_member_kind", return_value="k1"),
            "judge": mock.patch.object(rb, "judge_asset_entities", return_value=judged),
            "persist": mock.patch.object(
                rb, "upsert_entity_edges",
                return_value={"edges_deleted": 0, "edges_inserted": 1, "entities": []}),
            "desc": mock.patch.object(rb, "fetch_meta_description_targets", return_value=[]),
            "orphan": mock.patch.object(rb, "fetch_orphan_metas", return_value=[]),
            "cfg": mock.patch("src.config.settings.get_current_settings",
                              return_value=settings or _settings()),
        }

    def _run(self, patches, db=None, **kwargs):
        """패치를 모두 걸고 배치를 한 판 돌린다."""
        db = db if db is not None else _FakeDB()
        with patches["vocab"] as m_vocab, patches["official"] as m_off, patches["alias"], \
                patches["targets"] as m_targets, patches["kind"], patches["judge"] as m_judge, \
                patches["persist"] as m_persist, patches["desc"], patches["orphan"], \
                patches["cfg"]:
            result = rb.run_batch(db, mode=rb.DISCOVERY_AUTO, **kwargs)
        return result, {"vocab": m_vocab, "official": m_off, "targets": m_targets,
                        "judge": m_judge, "persist": m_persist}

    def test_vocab_read_once_in_the_index_transaction(self) -> None:
        # 자산 루프 안에서 읽으면 자산 수만큼 질의가 는다(색인 2벌과 같은 규율).
        db = _TxDB()
        patches = self._patches(targets=[_material(_A1), _material(_A2), _material(_A3)])
        _result, mocks = self._run(patches, db=db)
        self.assertEqual(mocks["vocab"].call_count, 1)
        # 색인과 **같은 커넥션** = 같은 읽기 트랜잭션(대상 목록과 어휘가 갈리지 않는다).
        self.assertIs(mocks["vocab"].call_args.args[0], mocks["official"].call_args.args[0])

    def test_type_defs_reach_the_judge_through_run_batch(self) -> None:
        defs = _registered_defs()
        patches = self._patches(targets=[_material(_A1)], vocab=defs)
        _result, mocks = self._run(patches)
        self.assertIs(mocks["judge"].call_args.kwargs["type_defs"], defs)

    def test_selection_predicate_and_persist_stamp_share_one_version(self) -> None:
        # 🔴 무한 재판정 봉인 — 대상 선별이 v2 로 비교하는데 저장 스탬프가 v1 이면(또는 그 반대)
        #    매 배치가 같은 자산을 영원히 다시 집는다.
        patches = self._patches(targets=[_material(_A1)])
        result, mocks = self._run(patches)
        selected = mocks["targets"].call_args.kwargs["prompt_version"]
        stamped = mocks["persist"].call_args.kwargs["prompt_version"]
        self.assertEqual(selected, stamped)
        self.assertEqual(selected, result["binding"]["prompt_version"])
        self.assertEqual(selected, PROMPT_VERSION)   # 정의문을 실었으니 v2

    def test_fallback_preset_also_keeps_predicate_and_stamp_together(self) -> None:
        # 폴백(등록 행 없음)도 정의문을 싣는다(코드 프리셋이 v2 문안이다) → 두 값이 함께 움직인다.
        patches = self._patches(targets=[_material(_A1)], vocab=ENTITY_TYPE_DEFS)
        result, mocks = self._run(patches)
        self.assertEqual(mocks["targets"].call_args.kwargs["prompt_version"],
                         mocks["persist"].call_args.kwargs["prompt_version"])
        self.assertEqual(result["binding"]["vocab_source"], rb.VOCAB_SOURCE_PRESET)
        self.assertEqual(result["binding"]["prompt_version"], PROMPT_VERSION)

    def test_registered_vocab_is_reported_as_registered(self) -> None:
        patches = self._patches(targets=[_material(_A1)])
        result, _mocks = self._run(patches)
        self.assertEqual(result["binding"]["vocab_source"], rb.VOCAB_SOURCE_REGISTERED)
        self.assertEqual(result["binding"]["type_defs"], len(ENTITY_TYPE_DEFS))

    def test_dry_run_still_loads_vocab_and_stamps_the_report(self) -> None:
        # 미리보기도 "무슨 문안으로 판정되나"를 보여 줘야 한다(어휘 조회는 읽기 전용이라 안전하다).
        patches = self._patches(targets=[_material(_A1)])
        result, mocks = self._run(patches, dry_run=True)
        mocks["persist"].assert_not_called()
        self.assertEqual(mocks["vocab"].call_count, 1)
        self.assertEqual(result["binding"]["prompt_version"], PROMPT_VERSION)


class TestFetchTargetsPromptVersion(unittest.TestCase):
    """재선별 술어의 ``pv`` 는 **주입**된다 — 판정 문안과 갈리지 않게 호출부가 정한다."""

    def test_injected_prompt_version_is_bound(self) -> None:
        conn = _FakeConn([])
        rb.fetch_binding_targets(conn, prompt_version=PROMPT_VERSION_WITHOUT_TYPE_DEFS)
        _sql, params = conn.executed[0]
        self.assertIn(PROMPT_VERSION_WITHOUT_TYPE_DEFS, params)
        self.assertNotIn(PROMPT_VERSION, params)


class TestMainExitCode(unittest.TestCase):
    """CLI 종료 코드 — 사람이 ``$?`` 로 성공을 판단하는 경로다(DAG 는 ``run_batch`` 를 직접 부른다)."""

    def _main(self, *, judged_failed: int, describe: dict | None) -> int:
        """``run_batch`` 결과만 갈아 끼우고 ``main`` 의 종료 코드를 본다."""
        result = {
            "binding": {"judged_failed": judged_failed},
            "describe": describe,
            "orphans": [],
        }
        # ``main`` 이 함수 안에서 import 하므로 원 모듈을 패치한다(모듈 속성이 아니다).
        with mock.patch("src.config.bootstrap.bootstrap_env"), \
                mock.patch("src.database.postgres_util.PostgresUtil"), \
                mock.patch.object(rb, "run_batch", return_value=result), \
                mock.patch.object(rb, "format_report", return_value=""), \
                mock.patch.object(sys, "argv", ["run_mm_meta_binding", "--env", "dev"]):
            return rb.main()

    def test_all_ok_is_zero(self) -> None:
        self.assertEqual(self._main(judged_failed=0, describe={"failed": 0}), 0)

    def test_judgement_failure_is_nonzero(self) -> None:
        self.assertEqual(self._main(judged_failed=2, describe={"failed": 0}), 1)

    def test_describe_failure_is_also_nonzero(self) -> None:
        # 이 경로가 예전에는 0 이었다 — 설명 생성이 전부 실패해도 성공으로 보였다.
        self.assertEqual(self._main(judged_failed=0, describe={"failed": 3}), 1)

    def test_describe_skipped_is_zero(self) -> None:
        # 설명 단계를 끈 경우(None)에 .get 을 부르면 터진다 — 그것도 막는다.
        self.assertEqual(self._main(judged_failed=0, describe=None), 0)


if __name__ == "__main__":
    unittest.main()
