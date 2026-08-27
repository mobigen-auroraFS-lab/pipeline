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

실행:
    python -m processing.app.run_entity_label --env dev --dry-run   # 쓰기 0(판정은 함)
    python -m processing.app.run_entity_label --env dev             # 실제 저장
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from src.domain.status_vocab import GraphEdgeStatus
from src.mm_classify.judge import judge_asset_labels
from src.mm_classify.model import ClassificationSkill
from src.mm_classify.persist import fetch_active_skills, skill_from_row
from src.mm_meta.entity_label import (
    build_entity_material,
    fetch_label_targets,
    replace_entity_labels,
)

# 노출 임계 — 화면에 뜨는 묶음만 대상이다. 1건짜리 개체에 갈래를 붙여도 쓰이지 않고, 개체
# 전체(1,065개)를 판정하는 것은 시험 단계에서 과하다. 묶음이 커지면 다음 배치가 집는다.
#   ⚠️ 데모 화면의 `_MIN_BUNDLE_SIZE` 와 같은 값이어야 한다 — 갈리면 "라벨은 있는데 화면에 없는"
#      개체가 생긴다(또는 그 반대).
DEFAULT_MIN_MEMBERS = 3

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

# 판정 문안 판 — 개체 판정은 **재료 조립이 다르다**. 자산 판정과 같은 값으로 찍으면 두 판정을
# 구분할 수 없고 재선별 범위도 잡을 수 없다(084 pv 스탬프와 같은 규율).
PROMPT_VERSION = "entity_label.v1"

_MEMBER_SUMMARY_SQL = """
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
    material_fn: Callable[[Mapping[str, Any]], str],
    judge_fn: Callable[..., Any] | None = None,
    persist_fn: Callable[[ClassificationSkill, Mapping[str, Any], Any], int] | None = None,
    dry_run: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    """개체마다 스킬을 판정·저장하고 diff 리포트를 만든다.

    **이 함수는 DB·LLM 을 모른다** — 재료 조립도 판정도 저장도 주입된 함수가 한다. 그래서 실 DB·
    실 LLM 없이 순서·정책·집계를 통째로 단위 검증할 수 있다(`run_mm_classify` 와 같은 관례).

    개체 하나의 실패가 배치를 멈추지 않는다(개체 단위 ``try/except`` 격리). 실패한 개체는 행을
    남기지 않으므로 다음 배치가 다시 집는다.

    Args:
        skills: 돌릴 스킬 목록. 순서대로 처리한다(결정적 리포트).
        targets: 대상 개체 목록 — ``{entity_type, entity_uid, name, description, members}``.
        material_fn: ``(target) -> 판정 재료 문자열``. 구성 자산 요약 조회가 필요하므로 호출부가
            DB 를 물려 넣는다.
        judge_fn: ``(skill, summary, keywords, *, client)`` 판정부. ``None``(기본)이면 코어
            ``judge_asset_labels`` 를 **호출 시점에 이름으로** 찾는다 — def 기본값으로 묶으면
            ``mock.patch.object`` 가 안 먹어 배선 테스트가 네트워크를 탄다.
        persist_fn: ``(skill, target, judgement) -> 기록 행수`` 저장부. 미주입이면 ``dry_run`` 일
            때만 허용되고 쓰기 모드에서는 ``ValueError`` — 저장부 없는 쓰기 배치는 조용한 0건이 된다.
        dry_run: 참이면 **아무 것도 쓰지 않고** 무엇이 붙을지만 보고한다. 판정 LLM 호출은 한다.
        client: 판정에 쓸 LLM 클라이언트. ``None`` 이면 코어 seam 의 운영 클라이언트.

    Returns:
        diff 리포트 dict — 개체 수·판정/실패·기록 행수·스킬별 라벨 분포·해당없음 비율(커버리지 갭).

    Raises:
        ValueError: 쓰기 모드인데 ``persist_fn`` 이 없을 때.
    """
    if not dry_run and persist_fn is None:
        raise ValueError("쓰기 모드인데 저장부(persist_fn)가 없다 — 조용한 0건 배치를 막는다")

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
    }

    for target in targets:
        material = material_fn(target)
        keywords = [str(target.get("name") or "")]
        got_any = False
        for skill in skills:
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
    lines.append(
        f"  대상 {report['targets']}개 | 판정 {report['judged']} · 실패 {report['failed']}"
        f" | 기록 {report['rows']}행"
    )
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
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """명령행 파서.

    Returns:
        파서.
    """
    p = argparse.ArgumentParser(description="개체 라벨 배치 (묶음에 분류 스킬 갈래를 붙인다)")
    p.add_argument("--env", choices=("dev", "prod"), default="dev")
    p.add_argument("--dry-run", action="store_true", help="쓰기 0 — 무엇이 붙을지만 보고(판정은 한다)")
    p.add_argument(
        "--min-members",
        type=int,
        default=DEFAULT_MIN_MEMBERS,
        help=f"대상 최소 구성 자산 수(기본 {DEFAULT_MIN_MEMBERS} — 화면 노출 임계와 같아야 한다)",
    )
    p.add_argument("--limit", type=int, default=None, help="이번 실행에서 처리할 개체 수 상한")
    return p


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

    def _load(conn: Any) -> tuple[list[ClassificationSkill], list[dict[str, Any]]]:
        """스킬·대상을 **읽기 트랜잭션 한 번**에 읽는다.

        Args:
            conn: DB 커넥션.

        Returns:
            ``(스킬 목록, 대상 개체 목록)``.
        """
        skills = [skill_from_row(dict(r)) for r in fetch_active_skills(conn)]
        targets = fetch_label_targets(
            conn,
            min_members=args.min_members,
            statuses=VISIBLE_STATUSES,
            limit=args.limit,
        )
        return skills, targets

    skills, targets = db.execute_in_transaction(_load, idempotent=True)

    def _material(target: Mapping[str, Any]) -> str:
        """개체 하나의 판정 재료를 조립한다(구성 자산 요약을 DB 에서 읽어 붙인다).

        Args:
            target: 대상 개체 행.

        Returns:
            판정 재료 문자열.
        """

        def _q(conn: Any) -> list[str]:
            with conn.cursor() as cur:
                cur.execute(
                    _MEMBER_SUMMARY_SQL,
                    (
                        target["entity_type"],
                        target["entity_uid"],
                        list(VISIBLE_STATUSES),
                        MEMBER_SUMMARIES,
                    ),
                )
                return [str(r[0]) for r in cur.fetchall()]

        sums = db.execute_in_transaction(_q, idempotent=True) if MEMBER_SUMMARIES else []
        return build_entity_material(
            name=str(target["name"]),
            entity_type=str(target["entity_type"]),
            description=target.get("description"),
            member_summaries=sums,
            max_member_summaries=MEMBER_SUMMARIES,
        )

    def _persist(skill: ClassificationSkill, target: Mapping[str, Any], judgement: Any) -> int:
        """개체 하나 × 스킬 하나의 라벨 행을 교체한다(개체마다 fresh 트랜잭션).

        Args:
            skill: 판정에 쓴 스킬.
            target: 대상 개체 행.
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
            ),
            idempotent=False,
        )

    report = run_entity_label(
        skills,
        targets,
        material_fn=_material,
        persist_fn=None if args.dry_run else _persist,
        dry_run=args.dry_run,
    )
    print(format_report(report))
    db.close()
    return 1 if report["failed"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
