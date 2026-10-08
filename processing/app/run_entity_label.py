"""개체 라벨 배치 — 묶음(멀티모달 메타)에 분류 스킬 갈래를 붙인다 (spec 087 T015).

무엇을 하나: 「아이유」라는 **대상**에 `음악` 을 붙인다. 지금 그 라벨은 **자산**에 붙어 있어서,
갈래로 좁히면 한 대상이 갈래마다 쪼개진다 — 실측 `경주시` 자료 6건 = 갈래없음 5 + 음료 1 +
디저트 1 → 목록 카드는 5건인데 상세로 들어가면 6건이었다(같은 결함 7개).

🔴 **자산 분류 배치(`run_mm_classify`)를 대체하지 않는다.** 둘은 답하는 질문이 다르다 —
자산 라벨 = "무엇을 받을까"(「한식 자료 90건 zip」) · 개체 라벨 = "무엇을 볼까"(탐색 계층).
두 배치는 서로를 읽지 않고 나란히 돈다(개체 추출 배치와 자산 분류 배치가 그런 것과 같다).

판정 엔진은 **085 를 그대로 쓴다** — 이 러너가 하는 일은 ①대상 선별 ②재료 조립 ③저장이다.
엔진을 복제하면 두 판정이 갈린다.

🔴 **시험 전제**(spec 087 §3) — 합격선 A4(정확도 ≥85%)·A5(쪼개짐 0) 미달이면 폐기한다.
폐기는 마이그레이션 v304 downgrade(`DROP TABLE`) + 이 파일 삭제로 끝난다.

**증분 재판정** — 「아직 판정 안 된 개체」와 「판정 재료가 바뀐 개체」만 판정한다. 판정할 때 재료의
지문(SHA-256)을 갈래 행에 함께 남기고, 다음 판은 지문·스킬 판·문안 판이 모두 그대로인 (개체, 스킬)
짝을 건너뛴다. 사서가 표지가 바뀐 책과 스티커 없는 책에만 분류 스티커를 다시 붙이는 것과 같다.
  - 🔴 상한(``limit``)은 **걸러 정렬한 뒤에** 개체 수로 건다. 대상 SQL 에 걸면 「구성원 많은 순 위에서
    N개」만 매시간 다시 판정되고, 그 밖의 개체는 영영 판정되지 않는다(굶주림). 그래서 노출 개체는
    상한 없이 전부 읽는다.
  - 🔴 판정에 넘기는 재료는 선별 때 지문을 뜬 **바로 그 문자열**이다 — 다시 조립하면 그 사이 설명문이
    바뀌었을 때 지문이 실제 판정 재료와 어긋난다.
  - 지문 칸이 생기기 전의 행은 지문이 비어 있어 **첫 실행에서 노출 개체 전부를 한 번** 다시 판정한다.
    그 뒤로는 재료가 바뀐 개체만 판정한다.
  - ``--plan`` 은 읽기·재료·선별까지만 해서 무엇을 판정할지 보여 준다(LLM 0 · 쓰기 0).
  - 읽기는 **짧은 트랜잭션 둘**(스킬·노출 개체·구성 요약 일괄 1회 / 저장 상태)이고, 재료 조립은 순수라
    트랜잭션 밖에서 한다 — 긴 읽기 트랜잭션은 표 잠금을 오래 쥐어 마이그레이션·화면 조회를 줄 세운다.
설계 배경: ``specs/104-entity-label-incremental`` ·
``docs/decisions/2026-10-07-entity-label-incremental.md``

실행:
    python -m processing.app.run_entity_label --env dev --plan      # 판정·쓰기 0 — 계획만
    python -m processing.app.run_entity_label --env dev --dry-run   # 쓰기 0(판정은 함)
    python -m processing.app.run_entity_label --env dev             # 실제 저장
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from src.domain.status_vocab import GraphEdgeStatus
from src.mm_classify import prompt as _core_prompt
from src.mm_classify.judge import judge_asset_labels
from src.mm_classify.model import ClassificationSkill
from src.mm_classify.persist import fetch_active_skills, skill_from_row
from src.mm_meta.entity_label import (
    EntityLabelError,
    LabelState,
    build_entity_material,
    fetch_label_state,
    fetch_label_targets,
    replace_entity_labels,
    select_label_work,
)
from src.mm_meta.rules import MIN_BUNDLE_SIZE

_LOG = logging.getLogger("meta_extract.run_entity_label")

# 노출 임계 — 화면에 뜨는 묶음만 대상이다. 1건짜리 개체에 갈래를 붙여도 쓰이지 않고, 개체
# 전체(1,065개)를 판정하는 것은 시험 단계에서 과하다. 묶음이 커지면 다음 배치가 집는다.
#   🔴 **코어에서 읽는다**(2026-08-31 · 087 후속). 전에는 이 파일과 백엔드 화면에 숫자 `3` 이
#      따로 박혀 있고 "같은 값이어야 한다"는 주석 한 줄이 유일한 방어선이었다 — 갈리면 "라벨은
#      있는데 화면에 없는" 개체가 생긴다(또는 그 반대). 이제 한 곳(코어)에서 읽어 결합을 코드로 만든다.
DEFAULT_MIN_MEMBERS = MIN_BUNDLE_SIZE

# 셈에 넣을 엣지 상태 — 화면과 같은 기준이어야 건수가 맞는다(2026-08-26 원칙: 적힌 숫자와 클릭
# 결과는 같은 것을 센다).
VISIBLE_STATUSES: tuple[str, ...] = (
    GraphEdgeStatus.ACTIVE.value,
    GraphEdgeStatus.PROPOSED.value,
)

# 재료에 실을 구성 자산 요약 개수.
#   🔴 3 으로 둔 근거(T016 파일럿 · 82개 전수 A/B): 설명문만 쓰면 해당없음 46/82, 구성 자산 요약
#   3건을 더하면 43/82 로 줄고 **더 잡은 17개가 대부분 옳았다** — `아이유`+노랫말(가사 자산 실재) ·
#   `김밥`+음악·노랫말(더 자두의 곡 「김밥」이 구성원) · `경주시`+맛집·외식.
#   처음엔 "한 파일의 특성이 대상 전체를 물들인다"고 보고 0 을 기본으로 뒀는데, 실측이 반대였다.
MEMBER_SUMMARIES = 3

# 개체 판정 자체의 문안 판 — 재료 조립·키워드 규칙을 바꾸거나 **LLM 모델을 바꾸면** 올린다(재판정
# 키에 모델 식별자가 없다 · plan 「운영 규칙」). 자산 판정과 같은 값으로 찍으면 두 판정을 구분할 수
# 없고 재선별 범위도 잡을 수 없다(084 pv 스탬프와 같은 규율).
ENTITY_PROMPT_BASE = "entity_label.v1"


def entity_prompt_version() -> str:
    """개체 판정 문안 판을 만든다 — ``<개체 자체 판>+<코어 공유 문안 판>``(예: ``entity_label.v1+mm_classify.v1``).

    개체 판정은 자산 판정과 **같은 공유 프롬프트**(코어 ``src.mm_classify.prompt``)로 LLM 에 묻는다.
    그래서 공유 문안이 개정되면(코어 ``PROMPT_VERSION`` 이 오르면) 개체도 다시 판정해야 한다 — 코어
    판을 개체 판에 실어 두면 저장된 판과 달라져 저절로 재선별된다. 코어 판은 **부르는 시점에** 모듈
    속성으로 읽는다(테스트가 코어 판을 바꿔 끼워 볼 수 있게).

    Returns:
        합성한 문안 판. 갈래 표 ``prompt_version`` 칸(VARCHAR(40)) 안이어야 한다(단위 테스트가 지킨다).
    """
    return f"{ENTITY_PROMPT_BASE}+{_core_prompt.PROMPT_VERSION}"


# 이번 프로세스의 개체 판정 문안 판(가져올 때 한 번 정한다 — 한 판 안에서 흔들리지 않게).
PROMPT_VERSION = entity_prompt_version()

# ``--plan`` 이 화면에 늘어놓을 개체 수. 첫 실행은 노출 개체 전부(수천 개)가 대상이라 다 찍으면
# 집계 줄이 스크롤에 묻힌다 — 전체 목록은 리포트 ``plan_items`` 에 그대로 있다.
PLAN_SHOW = 20

# 노출 개체 전부의 구성 자산 요약을 **한 번에** 읽는다 — 개체마다 「자산 id 순 앞 N건」.
#   🔴 옛 개체당 SQL(개체 하나 · ``ORDER BY sn.asset_id LIMIT N``)과 **결과가 정확히 같아야** 한다.
#      다르면 재료가 바뀌어 지문이 달라지고, 배포 뒤 첫 실행이 노출 개체 전부를 다시 판정한다.
#      같은 이유: ① 조인·거르기(mm_member · 가시 상태 · 요약 있음)가 같고, 요약 NULL 거르기가 순위보다
#      **먼저**(안쪽 질의 WHERE)라 LIMIT 앞에서 거르던 것과 같다. ② 순위를 개체(타입, 표기)마다
#      ``sn.asset_id`` 순으로 매겨 앞 N건만 남긴다 = 개체마다 ``ORDER BY … LIMIT N``. ③ 같은 자산 id 로
#      동점이 나도 요약 문자열이 같다(``asset_metadata`` 의 키가 ``asset_id``) — 동점 순서가 재료를 바꾸지 않는다.
#   🔴 전체 ``LIMIT`` 을 걸면 안 된다 — 개체 사이를 가로질러 잘라 앞 개체가 몫을 다 가져간다.
#   개체 조건은 (타입, 표기) **두 배열을 나란히** 풀어 짝으로 맞춘다(EXISTS 라 짝이 겹쳐도 행이 늘지
#   않는다). 표기만 맞추면 타입이 다른 동명 개체가 섞인다(`김밥` 음식/작품 · 코어 graph_query 와 같은 방식).
_MEMBER_SUMMARIES_SQL = """
SELECT entity_type, entity_uid, summary
  FROM (
        SELECT n.entity_type, n.entity_uid,
               m.ext_meta->>'summary' AS summary,
               ROW_NUMBER() OVER (PARTITION BY n.entity_type, n.entity_uid
                                  ORDER BY sn.asset_id) AS rn
          FROM node n
          JOIN graph_edge ge    ON ge.dst_node = n.node_id
          JOIN relation_kind rk ON rk.relation_kind_id = ge.relation_kind_id
          JOIN node sn          ON sn.node_id = ge.src_node
          JOIN asset_metadata m ON m.asset_id = sn.asset_id
         WHERE n.node_kind = 'entity'
           AND EXISTS (SELECT 1
                         FROM unnest(%(types)s::text[], %(uids)s::text[]) AS want(t, u)
                        WHERE want.t = n.entity_type AND want.u = n.entity_uid)
           AND rk.kind_code = 'mm_member' AND ge.status = ANY(%(statuses)s)
           AND m.ext_meta->>'summary' IS NOT NULL
       ) ranked
 WHERE rn <= %(per_entity)s
 ORDER BY entity_type, entity_uid, rn
"""


def _bump(counter: dict[str, int], key: str) -> None:
    """카운터 한 칸을 올린다.

    Args:
        counter: 대상 카운터 dict.
        key: 올릴 키.
    """
    counter[key] = counter.get(key, 0) + 1


def run_entity_label(
    skills: Sequence[ClassificationSkill],
    targets: Sequence[Mapping[str, Any]],
    *,
    material_fn: Callable[[Mapping[str, Any]], str] | None = None,
    judge_fn: Callable[..., Any] | None = None,
    persist_fn: Callable[[ClassificationSkill, Mapping[str, Any], Any], int] | None = None,
    dry_run: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    """개체마다 (필요한) 스킬을 판정·저장하고 diff 리포트를 만든다.

    **이 함수는 DB·LLM 을 모른다** — 재료 조립도 판정도 저장도 주입된 함수가 한다. 그래서 실 DB·
    실 LLM 없이 순서·정책·집계를 통째로 단위 검증할 수 있다(`run_mm_classify` 와 같은 관례).
    개체·스킬 하나의 실패가 배치를 멈추지 않는다(``try/except`` 격리). 실패는 행을 남기지 않으므로
    다음 배치가 다시 집는다.

    Args:
        skills: 활성 스킬 목록. 순서대로 처리한다(결정적 리포트).
        targets: 대상 개체 목록 — ``{entity_type, entity_uid, name, members}`` 에 선택 칸 셋:
            ``material``(있으면 그 재료로 판정 · 증분 선별이 지문을 뜬 재료) · ``skill_codes``(있으면
            그 스킬만 판정 · 없으면 ``skills`` 전부 — 예: ``("music",)`` 이면 음식 스킬은 건너뛴다) ·
            ``reasons``(``skill_codes`` 와 같은 순서의 재판정 사유 · 리포트 집계용).
        material_fn: ``(target) -> 판정 재료``. 대상 행에 ``material`` 이 없을 때만 부른다(옛 호출
            호환). ``None`` 인데 재료 없는 대상이 있으면 판정 전에 ``ValueError``.
        judge_fn: ``(skill, summary, keywords, *, client)`` 판정부. ``None``(기본)이면 코어
            ``judge_asset_labels`` 를 **호출 시점에 이름으로** 찾는다 — def 기본값으로 묶으면
            ``mock.patch.object`` 가 안 먹어 배선 테스트가 네트워크를 탄다.
        persist_fn: ``(skill, target, judgement) -> 기록 행수`` 저장부. 미주입이면 ``dry_run`` 일
            때만 허용되고 쓰기 모드에서는 ``ValueError`` — 저장부 없는 쓰기 배치는 조용한 0건이 된다.
        dry_run: 참이면 **아무 것도 쓰지 않고** 무엇이 붙을지만 보고한다. 판정 LLM 호출은 한다.
        client: 판정에 쓸 LLM 클라이언트. ``None`` 이면 코어 seam 의 운영 클라이언트.

    Returns:
        diff 리포트 dict — ``targets``(이번에 판정한 개체 수 = ``targets`` 길이)·``judged``/
        ``failed``(스킬 판정 단위)·``rows``(기록 행수)·``judged_by_reason``(사유별 성공 판정 수)·
        스킬별 라벨 분포·해당없음 개체 수(이번에 판정한 스킬 기준 — 커버리지 갭).

    Raises:
        ValueError: 쓰기 모드인데 ``persist_fn`` 이 없을 때 · 재료도 재료 함수도 없는 대상이 있을 때.
    """
    if not dry_run and persist_fn is None:
        raise ValueError("쓰기 모드인데 저장부(persist_fn)가 없다 — 조용한 0건 배치를 막는다")
    # 판정을 시작하기 전에 막는다 — 도중에 터지면 앞 개체들만 반쯤 판정·저장된 채 멈춘다.
    if material_fn is None and any("material" not in t for t in targets):
        raise ValueError("판정 재료가 없다 — 대상 행에 material 이 없으면 material_fn 이 필요하다")

    judge = judge_fn if judge_fn is not None else judge_asset_labels
    report: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "targets": len(targets),
        "judged": 0,
        "failed": 0,
        "rows": 0,
        "labeled_entities": 0,
        "unassigned_entities": 0,
        "failures": {},
        "by_skill": {},
        "judged_by_reason": {},
    }

    for target in targets:
        # 선별 때 지문을 뜬 재료를 그대로 쓴다 — 다시 조립하면 저장할 지문과 판정 재료가 어긋날 수 있다.
        if "material" in target:
            material = str(target["material"])
        elif material_fn is not None:
            material = material_fn(target)
        else:  # 위 사전 검사가 막으므로 오지 않는다 — 검사가 빠지면 조용히 지나가지 않게 둔다
            raise ValueError("판정 재료가 없다 — material 도 material_fn 도 없다")
        keywords = [str(target.get("name") or "")]
        # 개체마다 판정이 필요한 스킬만 돈다. 칸이 없으면(옛 호출) 전부 — 지문이 같은 스킬을 다시
        # 판정하면 LLM 만 쓰고 갈래가 흔들린다.
        wanted = target.get("skill_codes")
        reason_of: dict[str, str] = (
            dict(zip(wanted, target.get("reasons") or (), strict=False)) if wanted else {}
        )
        got_any = False
        for skill in skills:
            if wanted is not None and skill.skill_code not in wanted:
                continue
            slot = report["by_skill"].setdefault(
                skill.skill_code, {"labels": {}, "judged": 0, "unassigned": 0, "failed": 0}
            )
            try:
                judgement = judge(skill, material, keywords, client=client)
            except Exception as exc:  # 개체·스킬 하나의 실패를 배치 전체로 번지게 하지 않는다
                report["failed"] += 1
                slot["failed"] += 1
                _bump(report["failures"], f"exception:{type(exc).__name__}")
                continue
            if not getattr(judgement, "ok", False):
                report["failed"] += 1
                slot["failed"] += 1
                _bump(report["failures"], str(getattr(judgement, "failure", "unknown")))
                continue

            report["judged"] += 1
            slot["judged"] += 1
            if skill.skill_code in reason_of:
                _bump(report["judged_by_reason"], reason_of[skill.skill_code])
            names = tuple(getattr(judgement, "label_names", ()) or ())
            if names == (skill.policy.unassigned,):
                slot["unassigned"] += 1
            else:
                got_any = True
                for nm in names:
                    slot["labels"][nm] = slot["labels"].get(nm, 0) + 1

            if not dry_run and persist_fn is not None:
                try:
                    report["rows"] += persist_fn(skill, target, judgement)
                except Exception as exc:
                    report["failed"] += 1
                    slot["failed"] += 1
                    _bump(report["failures"], f"persist:{type(exc).__name__}")

        if got_any:
            report["labeled_entities"] += 1
        else:
            report["unassigned_entities"] += 1

    return report


def format_report(report: Mapping[str, Any]) -> str:
    """리포트를 사람이 읽는 여러 줄로 만든다.

    Args:
        report: ``run_entity_label`` 산출물.

    Returns:
        출력 문자열.
    """
    lines = [f"[개체 라벨] {'dry-run' if report['dry_run'] else 'apply'}"]
    if "visible" in report:  # 증분 선별을 거친 리포트(``run_label_pass``)일 때만 있는 칸
        lines.append(_selection_line(report, picked="이번 판정"))
    lines.append(
        f"  대상 {report['targets']}개 | 판정 {report['judged']} · 실패 {report['failed']}"
        f" | 기록 {report['rows']}행"
    )
    if report.get("judged_by_reason"):
        shown = " · ".join(f"{k} {v}" for k, v in sorted(report["judged_by_reason"].items()))
        lines.append(f"  사유별 판정: {shown}")
    tot = max(1, report["targets"])
    lines.append(
        f"  라벨 붙은 개체 {report['labeled_entities']} · 해당없음 "
        f"{report['unassigned_entities']} ({report['unassigned_entities'] * 100 // tot}%)"
        "   ← 해당없음 비율이 곧 **스킬 커버리지 갭**이다"
    )
    for code, slot in sorted(report["by_skill"].items()):
        top = sorted(slot["labels"].items(), key=lambda kv: (-kv[1], kv[0]))[:6]
        shown = " · ".join(f"{k} {v}" for k, v in top) or "(없음)"
        lines.append(f"  [{code}] 해당없음 {slot['unassigned']} · {shown}")
    if report["failures"]:
        lines.append(f"  ⚠️ 실패 사유: {dict(sorted(report['failures'].items()))}")
    lines.extend(f"  ⚠️ {w}" for w in report.get("warnings") or ())
    return "\n".join(lines)


def _selection_line(report: Mapping[str, Any], *, picked: str) -> str:
    """증분 선별 집계 한 줄 — 노출·건너뜀·판정 필요·이번 선택.

    Args:
        report: ``run_label_pass`` 리포트(``visible`` 등 선별 집계 칸이 있어야 한다).
        picked: 「이번 선택」 칸의 이름. 계획 보기는 「이번 선택」, 실제 판은 「이번 판정」이다.

    Returns:
        들여쓴 한 줄. 상한이 있으면 끝에 ``(상한 N)`` 을, 재료 조립에 실패한 개체가 있으면
        ``· 재료 실패 N`` 을 붙인다(그 개체는 노출 수에서 빠져 있다).
    """
    cap = report.get("limit")
    broken = int(report.get("material_failed") or 0)
    return (
        f"  노출 {report['visible']}개 | 건너뜀(재료 그대로) {report['skipped_unchanged']}"
        f" · 판정 필요 {report['need']} · {picked} {report['selected']}"
        + (f" (상한 {cap})" if cap is not None else "")
        + (f" · 재료 실패 {broken}" if broken else "")
    )


def format_plan(report: Mapping[str, Any], *, show: int = PLAN_SHOW) -> str:
    """``--plan`` 결과를 사람이 읽는 여러 줄로 만든다 — 집계와 이번에 판정할 개체 앞부분.

    Args:
        report: ``run_label_pass(plan_only=True)`` 산출물(``plan_items`` 포함).
        show: 목록에 보일 개체 수 상한. 나머지는 「외 N개」 한 줄로 줄인다(전체는 리포트에 있다).

    Returns:
        출력 문자열.
    """
    lines = ["[개체 갈래 계획] 판정·쓰기 0", _selection_line(report, picked="이번 선택")]
    # 사유 칸 이름은 코어 집계가 정한다(pairs_<사유>) — 사유 목록을 이 파일에 따로 두지 않는다.
    pairs = " · ".join(
        f"{k.removeprefix('pairs_')} {v}" for k, v in report.items() if k.startswith("pairs_")
    )
    lines.append(f"  사유별 (개체, 스킬) 짝(상한 전): {pairs}")
    items = list(report.get("plan_items") or ())
    if items:
        lines.append(f"  이번에 판정할 개체 (앞 {min(show, len(items))}개):")
    for it in items[:show]:
        why = " · ".join(f"{c}:{r}" for c, r in zip(it["skill_codes"], it["reasons"], strict=False))
        lines.append(f"    - {it['entity_type']}/{it['name']} · 구성원 {it['members']} · {why}")
    if len(items) > show:
        lines.append(f"    … 외 {len(items) - show}개")
    lines.extend(f"  ⚠️ {w}" for w in report.get("warnings") or ())
    return "\n".join(lines)


def _fetch_member_summaries(
    conn: Any, keys: Sequence[tuple[str, str]]
) -> dict[tuple[str, str], list[str]]:
    """노출 개체 전부의 구성 자산 요약을 **SQL 한 번**으로 읽는다(조회 전용 · 개체마다 앞 N건).

    개체마다 SQL 을 치면 노출 개체 수(수천)만큼 왕복하고, 그동안 읽기 트랜잭션이 표 잠금을 쥔 채
    길어진다. 결과는 옛 개체당 SQL 과 같다(``_MEMBER_SUMMARIES_SQL`` 주석의 근거 ①~③).

    Args:
        conn: DB 커넥션(호출부의 읽기 트랜잭션).
        keys: 요약을 읽을 개체의 ``(entity_type, entity_uid)`` 짝 목록. 비어 있으면 SQL 을 치지 않는다.

    Returns:
        ``{(entity_type, entity_uid): [요약, …]}`` — 개체마다 자산 id 순 앞 ``MEMBER_SUMMARIES`` 건.
        요약 있는 구성 자산이 없는 개체는 키가 없다. ``MEMBER_SUMMARIES`` 가 0 이하면 빈 dict.
    """
    if not keys or MEMBER_SUMMARIES <= 0:
        return {}
    params = {
        "types": [str(t) for t, _u in keys],
        "uids": [str(u) for _t, u in keys],
        "statuses": list(VISIBLE_STATUSES),
        "per_entity": MEMBER_SUMMARIES,
    }
    out: dict[tuple[str, str], list[str]] = {}
    with conn.cursor() as cur:
        cur.execute(_MEMBER_SUMMARIES_SQL, params)
        # SQL 이 개체 안에서 순위(rn) 순으로 내므로 받은 순서대로 붙이면 자산 id 순이 유지된다.
        for etype, euid, summary in cur.fetchall():
            out.setdefault((str(etype), str(euid)), []).append(str(summary))
    return out


def _read_skills_and_targets(
    conn: Any, *, min_members: int
) -> tuple[list[ClassificationSkill], list[dict[str, Any]], dict[tuple[str, str], list[str]]]:
    """활성 스킬·노출 개체 전부·그 구성 요약을 읽는다(짧은 읽기 트랜잭션 하나 · SQL 셋).

    Args:
        conn: DB 커넥션.
        min_members: 대상 최소 구성 자산 수(화면 노출 임계와 같은 값).

    Returns:
        ``(스킬 목록, 노출 개체 행 목록, 구성 요약 dict)``.
    """
    skills = [skill_from_row(dict(r)) for r in fetch_active_skills(conn)]
    # 상한 없이 전부 읽는다 — 상한은 걸러낸 뒤 선별 함수가 건다(SQL 에 걸면 굶주림).
    targets = fetch_label_targets(
        conn,
        min_members=min_members,
        statuses=VISIBLE_STATUSES,
        limit=None,
    )
    keys = [(str(t["entity_type"]), str(t["entity_uid"])) for t in targets]
    return skills, targets, _fetch_member_summaries(conn, keys)


def _build_candidates(
    targets: Sequence[Mapping[str, Any]],
    summaries: Mapping[tuple[str, str], Sequence[str]],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """노출 개체마다 판정 재료를 조립한다(순수 · 트랜잭션 밖) — 조립에 실패한 개체는 빼고 따로 돌려준다.

    개체 하나의 재료 오류(이름·타입이 비어 있음 등 ``EntityLabelError``)가 배치·``--plan`` 전체를
    멈추지 않게 그 개체만 후보에서 뺀다. 빠진 개체는 갈래 행을 받지 않으므로 데이터가 고쳐지면
    다음 판에서 「미판정」으로 다시 집힌다. 그 밖의 예외는 코드 결함이라 그대로 올린다.
    지문은 붙이지 않는다 — 코어 선별 함수가 재료에서 직접 계산한다.

    Args:
        targets: ``fetch_label_targets`` 행 목록.
        summaries: ``_fetch_member_summaries`` 결과. 키가 없는 개체는 요약 없이 조립한다.

    Returns:
        ``(재료가 붙은 후보 목록, 조립에 실패한 개체의 (타입, 표기) 목록)`` — 둘 다 입력 순서.
    """
    candidates: list[dict[str, Any]] = []
    failed: list[tuple[str, str]] = []
    for t in targets:
        key = (str(t["entity_type"]), str(t["entity_uid"]))
        try:
            material = build_entity_material(
                name=str(t["name"]),
                entity_type=key[0],
                description=t.get("description"),
                member_summaries=summaries.get(key, ()),
                max_member_summaries=MEMBER_SUMMARIES,
            )
        except EntityLabelError:
            failed.append(key)
            continue
        candidates.append({**t, "material": material})
    return candidates, failed


# 재료 실패 경고 줄에 늘어놓을 개체 수 — 나머지는 「외 N개」로 줄인다(경고는 판마다 1줄).
_FAILED_SHOW = 5


def _warn_material_failed(failed: Sequence[tuple[str, str]]) -> None:
    """재료 조립에 실패한 개체를 경고 **1줄**로 남긴다(없으면 아무 것도 하지 않는다).

    Args:
        failed: ``_build_candidates`` 가 돌려준 실패 개체 ``(entity_type, entity_uid)`` 목록.
    """
    if not failed:
        return
    shown = ", ".join(f"{t}/{u}" for t, u in failed[:_FAILED_SHOW])
    more = f" 외 {len(failed) - _FAILED_SHOW}개" if len(failed) > _FAILED_SHOW else ""
    _LOG.warning(
        "개체 갈래 재료 조립 실패 %d개 — 후보에서 뺐다(데이터를 고치면 다음 판에 다시 집힌다): %s%s",
        len(failed), shown, more,
    )


# 활성 스킬이 하나도 없을 때의 경고 — 리포트 ``warnings`` 와 로그에 같은 문장을 남긴다.
_NO_SKILL_WARNING = "활성 스킬이 0개다 — 개체 갈래를 판정하지 않았다(분류 스킬 등록·활성 상태를 확인)"


def run_label_pass(
    db: Any,
    *,
    min_members: int = DEFAULT_MIN_MEMBERS,
    limit: int | None = None,
    dry_run: bool = False,
    plan_only: bool = False,
    judge_fn: Callable[..., Any] | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """개체 갈래 한 판 — 판정이 필요한 (개체, 스킬)만 골라 판정하고 지문과 함께 저장한다(CLI·DAG 공용).

    읽기는 **짧은 트랜잭션 둘**이다 — ① 활성 스킬·노출 개체 전부·구성 요약(일괄 SQL 1회) ② 저장 상태.
    그 뒤 개체마다 재료를 조립하고(순수 · 트랜잭션 밖), 코어 ``select_label_work`` 가 재료에서 지문을
    떠 고른 개체만 판정해 개체 하나 × 스킬 하나 단위로 갈래 행을 교체한다(쓰기 트랜잭션은 그 단위마다).
    읽기를 나눠도 결과는 같다 — 기본 격리 수준(READ COMMITTED)에서는 문장마다 새 스냅숏이라 한
    트랜잭션으로 묶어 얻던 일관성이 원래 없었다. 묶으면 표 잠금만 오래 쥔다(v307 ALTER 가 뒤에서 기다린다).
    🔴 배선을 복제하지 않는다 — CLI 도 DAG 도 이 함수를 부른다(CLI 전용이던 때 DB 를 새로 적재한 뒤
    아무도 돌리지 않아 화면의 「갈래」 필터가 통째로 사라졌다).

    Args:
        db: 열려 있는 ``PostgresUtil``(호출부가 연다·닫는다).
        min_members: 대상 최소 구성 자산 수. **화면 노출 임계와 같아야 한다** — 다르면 화면에는
            보이는데 갈래가 없는 개체(또는 그 반대)가 생긴다.
        limit: 이번 판에서 **판정할** 개체 수 상한(재료가 그대로라 건너뛴 개체는 세지 않는다).
            ``None``(기본)이면 필요한 개체 전부 · 0 이하면 판정하지 않는다(CLI 는 음수를 받지 않는다).
        dry_run: 참이면 쓰지 않고 무엇이 붙을지만 집계한다(판정 LLM 호출은 한다).
        plan_only: 참이면 선별까지만 하고 판정·쓰기 0 으로 계획을 돌려준다. ``dry_run`` 보다 우선한다.
        judge_fn: 판정부 주입구(``run_entity_label`` 과 같은 계약). ``None`` 이면 코어 운영 판정.
        client: 판정에 쓸 LLM 클라이언트. ``None`` 이면 코어 seam 의 운영 클라이언트.

    Returns:
        ``run_entity_label`` 의 diff 리포트에 선별 집계를 합친 dict — ``targets`` 는 **이번에 판정한
        개체 수**(= ``selected``)이고, ``visible``(재료를 만든 노출 개체 수)·``skipped_unchanged``·
        ``need``·``selected``·``pairs_<사유>``·``material_failed``(재료 조립에 실패해 뺀 노출 개체 수)·
        ``warnings``(경고 문장 목록 — 예: 활성 스킬 0)·``limit``·``plan`` 칸이 붙는다. ``plan_only`` 면
        판정 칸은 0 이고 ``plan_items``(이번에 판정할 개체 · 선별 순서 · ``entity_type``·``entity_uid``·
        ``name``·``members``·``skill_codes``·``reasons``)가 붙는다.
    """
    skills, rows, summaries = db.execute_in_transaction(
        lambda conn: _read_skills_and_targets(conn, min_members=min_members),
        idempotent=True,
    )
    state: dict[tuple[str, str, str], LabelState] = db.execute_in_transaction(
        fetch_label_state, idempotent=True
    )

    candidates, material_failed = _build_candidates(rows, summaries)
    _warn_material_failed(material_failed)
    warnings: list[str] = []
    if not skills:  # 판정 0 이 「할 일 없음」으로 조용히 지나가지 않게 — 평시 0 과 구분된다
        warnings.append(_NO_SKILL_WARNING)
        _LOG.warning(_NO_SKILL_WARNING)

    works, stats = select_label_work(
        candidates, state, skills, prompt_version=PROMPT_VERSION, limit=limit
    )
    stats = {**stats, "material_failed": len(material_failed), "warnings": warnings}
    # 선별 결과에는 이름·설명이 없다 — 원래 대상 행에 판정할 스킬·사유를 얹어 판정부에 넘긴다.
    # 재료·지문은 선별 결과의 것을 쓴다(지문은 코어가 그 재료에서 계산했다).
    by_key = {(c["entity_type"], c["entity_uid"]): c for c in candidates}
    targets = [
        {
            **by_key[(w.entity_type, w.entity_uid)],
            "material": w.material,
            "material_hash": w.material_hash,
            "skill_codes": w.skill_codes,
            "reasons": w.reasons,
        }
        for w in works
    ]

    if plan_only:
        return {
            "plan": True,
            "dry_run": True,
            "limit": limit,
            "targets": len(targets),
            "judged": 0,
            "failed": 0,
            "rows": 0,
            **stats,
            "plan_items": [
                {k: t[k] for k in ("entity_type", "entity_uid", "name", "members",
                                   "skill_codes", "reasons")}
                for t in targets
            ],
        }

    def _persist(skill: ClassificationSkill, target: Mapping[str, Any], judgement: Any) -> int:
        """개체 하나 × 스킬 하나의 갈래 행을 지문과 함께 교체한다(단위마다 새 트랜잭션).

        Args:
            skill: 판정에 쓴 스킬.
            target: 대상 개체 행(선별 결과의 ``material_hash`` — 코어가 판정 재료에서 직접 계산한
                지문 — 를 싣고 있다).
            judgement: 판정 결과.

        Returns:
            기록 행수.
        """
        return db.execute_in_transaction(
            lambda conn: replace_entity_labels(
                conn,
                entity_type=str(target["entity_type"]),
                entity_uid=str(target["entity_uid"]),
                skill=skill,
                skill_version=skill.version,
                judgement=judgement,
                prompt_version=PROMPT_VERSION,
                material_hash=str(target["material_hash"]),
            ),
            idempotent=False,
        )

    report = run_entity_label(
        skills,
        targets,
        judge_fn=judge_fn,
        persist_fn=None if dry_run else _persist,
        dry_run=dry_run,
        client=client,
    )
    report.update(stats)
    report["limit"] = limit
    report["plan"] = False
    return report


def _build_parser() -> argparse.ArgumentParser:
    """명령행 파서.

    Returns:
        파서.
    """
    p = argparse.ArgumentParser(description="개체 라벨 배치 (묶음에 분류 스킬 갈래를 붙인다)")
    p.add_argument("--env", choices=("dev", "prod"), default="dev")
    p.add_argument("--dry-run", action="store_true", help="쓰기 0 — 무엇이 붙을지만 보고(판정은 한다)")
    p.add_argument(
        "--plan",
        action="store_true",
        help="판정·쓰기 0 — 무엇을 왜 판정할지 집계와 목록만 보인다(--dry-run 보다 우선)",
    )
    p.add_argument(
        "--min-members",
        type=int,
        default=DEFAULT_MIN_MEMBERS,
        help=f"대상 최소 구성 자산 수(기본 {DEFAULT_MIN_MEMBERS} — 화면 노출 임계와 같아야 한다)",
    )
    p.add_argument(
        "--limit",
        type=_non_negative_int,
        default=None,
        help="이번 실행에서 판정할 개체 수 상한 · 0 이상(재료가 그대로인 개체는 세지 않는다 · 기본 전부)",
    )
    return p


def _non_negative_int(text: str) -> int:
    """``--limit`` 값을 0 이상 정수로 읽는다 — 음수는 「판정 0」으로 조용히 접히므로 명령행에서 막는다.

    Args:
        text: 명령행에 적힌 값(예: ``"500"``). 0 은 「이번 판은 계획만 세고 판정하지 않음」이다.

    Returns:
        정수 값.

    Raises:
        argparse.ArgumentTypeError: 정수가 아니거나 음수일 때(argparse 가 사용법 오류로 바꾼다).
    """
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"정수가 아니다: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError(f"0 이상이어야 한다: {value}")
    return value


def main(argv: list[str] | None = None) -> int:
    """대상을 읽어 판정·저장하고 리포트를 출력한다.

    Args:
        argv: 명령행 인자. ``None`` 이면 실제 명령행.

    Returns:
        종료 코드 — 실패가 하나라도 있으면 1(배치 모니터가 실패를 놓치지 않게).
    """
    args = _build_parser().parse_args(argv)

    from pathlib import Path

    from dotenv import load_dotenv

    from src.config.settings import init_settings
    from src.database.postgres_util import PostgresUtil

    env_path = Path(__file__).resolve().parents[2] / f".env.{args.env}"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)
    init_settings(args.env)

    db = PostgresUtil()
    report = run_label_pass(db, min_members=args.min_members, limit=args.limit,
                            dry_run=args.dry_run, plan_only=args.plan)
    print(format_plan(report) if report.get("plan") else format_report(report))
    db.close()
    return 1 if report["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
