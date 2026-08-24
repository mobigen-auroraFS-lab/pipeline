"""085 분류 스킬 배치 CLI — 등록된 스킬(분류 체계)로 자산을 분류하고 패싯 색인을 맞춘다.

**흐름에서의 위치**: 084 소속 배치와 동형인 **별도 배치**다(spec §5). 기존 적재 경로는 한 줄도
바뀌지 않고, 이 배치는 이미 저장된 산출물(요약·키워드)만 읽어 판정한다.

한 자산 × 한 스킬의 처리 순서::

    대상 선별(행 없음 또는 skill_version < 현행) → 재료 조회 → 판정(LLM 1회) →
    replace_asset_labels(자산×스킬 행 전체 교체) → OS ``mm_skill_labels`` 부분 갱신

⚠️ **스킬이 여럿이면 스킬마다 따로 판정한다** — 한 프롬프트에 섞지 않는다(격리·버전 분리 ·
spec §1). 스킬 하나의 라벨들만 한 호출에 함께 실린다.

🔴 **색인 값은 그 자산의 '전 스킬' 행에서 만든다.** OpenSearch 는 ``mm_skill_labels`` 한 칸에 모든
스킬의 라벨을 담고 부분 갱신은 그 칸을 통째로 덮어쓴다 — 방금 판정한 스킬 라벨만 실어 보내면 같은
자산의 다른 스킬 라벨이 색인에서 사라진다. 그래서 저장 뒤 ``fetch_asset_label_rows`` 로 다시 읽는다
(코어가 그 SQL 을 소유한다 · spec 구현확정 G2 — 여기서 재구현하지 않는다).

⚠️ **판정 실패는 행을 남기지 않는다**(spec §2·§4). 행 부재 = 미판정이므로 다음 배치가 자연히 다시
집는다. "해당없음"(미부여)은 실패가 아니라 **판정 결과**이고 단독 1행으로 기록된다.

IO 경계를 나눠 뒀다: 조립부(``run_classify``)는 DB·LLM·OpenSearch 를 **아예 모르고** 주입된 함수만
부른다. 실제 커넥션·클라이언트는 ``main`` 만 만든다(``run_opensearch_resync``·084 배치 선례).

사용법::

    python -m processing.app.run_mm_classify --env dev --dry-run        # 미리보기(쓰기 0)
    python -m processing.app.run_mm_classify --env dev --limit 200      # 스킬당 앞 200건
    python -m processing.app.run_mm_classify --env dev food_content     # 특정 스킬만
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from src.mm_classify import (
    ClassificationSkill,
    fetch_active_skills,
    fetch_asset_label_rows,
    fetch_asset_materials,
    fetch_pending_asset_ids,
    judge_asset_labels,
    replace_asset_labels,
    skill_from_row,
)
from src.search.opensearch_sync import mm_skill_label_keys

_LOG = logging.getLogger("meta_extract.run_mm_classify")

# 리포트에 담는 라벨 예시 상한 — 콘솔·XCom 이 넘치지 않게 자른다.
_MAX_LABELS_IN_LINE = 6


def _failure_key(exc: Exception) -> str:
    """예외를 리포트 집계 키로 바꾼다 — **타입명만** 담는다.

    예외 메시지·파일 경로에는 식별 가능한 내용이 섞일 수 있다(헌법 10조 · ``run_relations`` 의
    큐 사유 규율과 같다).

    Args:
        exc: 잡은 예외.

    Returns:
        ``exception:<타입명>`` 형식의 집계 키.
    """
    return f"exception:{type(exc).__name__}"


def _bump(counter: dict[str, int], key: str) -> None:
    """집계 dict 의 항목 하나를 1 올린다.

    Args:
        counter: 집계 dict(사유·라벨 코드별 건수).
        key: 올릴 항목 이름.
    """
    counter[key] = counter.get(key, 0) + 1


def run_classify(
    skills: Sequence[ClassificationSkill],
    *,
    targets_fn: Callable[[ClassificationSkill], Sequence[Mapping[str, Any]]],
    judge_fn: Callable[..., Any] | None = None,
    persist_fn: Callable[[ClassificationSkill, str, Any], int] | None = None,
    labels_fn: Callable[[str], Sequence[Mapping[str, str]]] | None = None,
    index_fn: Callable[[str, list[str]], None] | None = None,
    dry_run: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    """활성 스킬마다 대상 자산을 판정·저장·색인하고 diff 리포트를 만든다.

    **이 함수는 DB·LLM·OpenSearch 를 모른다** — 대상 조회도 판정도 저장도 색인도 주입된 함수가
    한다. 그래서 실 DB·실 LLM 없이 순서·정책·집계를 통째로 단위 검증할 수 있다.

    자산 하나의 실패가 배치를 멈추지 않는다(자산 단위 ``try/except`` 격리). 색인 실패는 **판정
    실패와 따로** 센다 — DB 판정은 이미 성공했으므로 그 자산은 다시 선별되지 않고, 색인만 낡은
    상태로 남기 때문이다(그때는 재색인 도구로 맞춘다).

    Args:
        skills: 돌릴 스킬 목록(보통 ``load_active_skills`` 결과). 순서대로 처리한다.
        targets_fn: ``(skill) -> [{asset_id, summary, keywords}]`` 대상·재료 조회부.
        judge_fn: ``(skill, summary, keywords, *, client)`` 판정부. ``None``(기본)이면 코어
            ``judge_asset_labels``(LLM 단일 seam·temp=0)를 **호출 시점에 이름으로** 찾는다 —
            def 기본값으로 묶어 두면 ``mock.patch.object`` 가 안 먹어 배선 테스트가 네트워크를
            타게 된다(``run_relations._domain_fn`` 과 같은 관례).
        persist_fn: ``(skill, asset_id, judgement) -> 삽입 행수`` 저장부. 미주입이면 ``dry_run`` 일
            때만 허용되고 쓰기 모드에서는 ``ValueError`` — 저장부 없는 쓰기 배치는 "조용한 0건"이 된다.
        labels_fn: ``(asset_id) -> [{skill_code, label_code}]`` 판정 행 조회부. 두 곳에 쓴다 —
            판정 **전**(신규/갱신 구분)과 저장 **뒤**(색인 값). ``None``(기본)이면 신규/갱신을 세지
            않고 색인도 하지 않는다(색인 값은 전 스킬 행에서만 만들 수 있기 때문이다).
        index_fn: ``(asset_id, keys) -> None`` 색인 부분 갱신부. ``None``(기본)이면 OpenSearch 를
            건드리지 않는다(OS 없는 환경·``--no-index``).
        dry_run: 참이면 **아무 것도 쓰지 않고**(DB·OS 모두) 무엇이 붙을지만 보고한다. 판정 LLM
            호출은 한다 — 무엇이 저장될지 알려면 판정이 필요하다.
        client: 판정에 쓸 LLM 클라이언트. ``None``(기본)이면 코어 seam 이 운영 온프레미스
            클라이언트를 쓴다(temperature 는 seam 기본값 0 · 헌법 3조).

    Returns:
        diff 리포트 dict — 스킬별 라벨 분포·해당없음 비율(커버리지 갭)·신규/갱신/실패·색인 건수와
        전체 합계. 같은 입력이면 같은 리포트가 나온다(결정적 정렬).

    Raises:
        ValueError: 쓰기 모드인데 ``persist_fn`` 이 없을 때.
    """
    if not dry_run and persist_fn is None:
        raise ValueError("쓰기 모드인데 저장부(persist_fn)가 없다 — 조용한 0건 배치를 막는다")

    # seam 기본값 해소: None 이면 모듈 수준 이름을 **호출 시점에** 잡는다(def 기본값으로 묶으면
    # 테스트가 바꿔 끼운 이름이 무시된다 · run_relations 의 _domain_fn 과 같은 이유).
    judge = judge_fn if judge_fn is not None else judge_asset_labels
    report: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "targets": 0,
        "judged": 0,
        "failed": 0,
        "rows": 0,
        "indexed": 0,
        "index_failed": 0,
        "skills": [],
    }

    for skill in skills:
        materials = list(targets_fn(skill))
        labels: dict[str, int] = {}
        failures: dict[str, int] = {}
        summary_row: dict[str, Any] = {
            "skill_code": skill.skill_code,
            "name": skill.name,
            "version": skill.version,
            "targets": len(materials),
            "judged": 0,
            "failed": 0,
            "created": 0,
            "updated": 0,
            "rows": 0,
            "indexed": 0,
            "index_failed": 0,
            "unassigned": 0,
        }

        for item in materials:
            asset_id = str(item.get("asset_id"))
            try:
                # 판정 **전** 행을 읽어 신규/갱신을 가른다(스킬 개정 백필이 얼마나 도는지의 지표).
                prior = None
                if labels_fn is not None:
                    prior = [
                        row for row in labels_fn(asset_id)
                        if str(row.get("skill_code")) == skill.skill_code
                    ]

                judgement = judge(
                    skill, item.get("summary") or "", list(item.get("keywords") or []),
                    client=client,
                )
                if not judgement.ok:
                    # 실패는 행을 남기지 않는다 → 다음 배치가 이 자산을 다시 집는다(spec §2).
                    summary_row["failed"] += 1
                    _bump(failures, str(judgement.failure.value if judgement.failure else "unknown"))
                    continue

                if dry_run:
                    written = len(judgement.label_codes)
                else:
                    # 자산 × 스킬 행 전체 교체(한 트랜잭션) — 라벨이 같아도 skill_version 을 갱신한다.
                    written = int(persist_fn(skill, asset_id, judgement))  # type: ignore[misc]

                summary_row["judged"] += 1
                summary_row["rows"] += written
                for code in judgement.label_codes:
                    _bump(labels, code)
                if judgement.is_unassigned:
                    summary_row["unassigned"] += 1
                if prior is not None:
                    key = "updated" if prior else "created"
                    summary_row[key] += 1

                if not dry_run and index_fn is not None and labels_fn is not None:
                    try:
                        # 🔴 저장 **뒤** 전 스킬 행을 다시 읽는다 — 다른 스킬 라벨이 색인에서
                        #    증발하지 않게(코어가 SQL 을 소유한다).
                        keys = mm_skill_label_keys(labels_fn(asset_id))
                        index_fn(asset_id, keys)
                        summary_row["indexed"] += 1
                    except Exception as exc:  # noqa: BLE001 — 색인 실패는 판정 실패와 별개다
                        # DB 판정은 이미 성공했으므로 이 자산은 다시 선별되지 않는다 → 색인만
                        # 낡은 채 남는다. 배치를 세우는 대신 세어서 보고하고, 재색인은 복구 도구 몫.
                        summary_row["index_failed"] += 1
                        _bump(failures, f"index:{type(exc).__name__}")
                        _LOG.warning("mm_classify index failed %s/%s: %s", skill.skill_code,
                                     asset_id, type(exc).__name__)
            except Exception as exc:  # noqa: BLE001 — 자산 단위 격리
                summary_row["failed"] += 1
                _bump(failures, _failure_key(exc))
                _LOG.warning("mm_classify failed %s/%s: %s", skill.skill_code, asset_id,
                             type(exc).__name__)

        judged = summary_row["judged"]
        # 커버리지 갭 지표(파일럿 발견 2) — 라벨이 못 덮는 영역이 얼마나 되는지.
        summary_row["unassigned_ratio"] = (
            round(summary_row["unassigned"] / judged, 4) if judged else 0.0
        )
        summary_row["labels"] = dict(sorted(labels.items()))
        summary_row["failures"] = dict(sorted(failures.items()))
        report["skills"].append(summary_row)
        for key in ("targets", "judged", "failed", "rows", "indexed", "index_failed"):
            report[key] += summary_row[key]
        _LOG.info("mm_classify %s v%d: %s", skill.skill_code, skill.version, summary_row["labels"])

    return report


def load_active_skills(conn: Any) -> list[ClassificationSkill]:
    """활성 스킬 행을 읽어 검증된 스킬 객체로 되돌린다(조회 전용).

    정본은 **DB 등록 행**이다 — 배치는 파일을 읽지 않는다(파일은 등록 CLI 의 입력 수단일 뿐 ·
    spec §1). 행이 검증을 통과하지 못하면 **예외가 그대로 오른다**: 등록은 검증하는 CLI 로만
    이뤄지므로, 깨진 행은 손 SQL 이 남긴 것이고 조용히 건너뛰면 그 스킬이 영영 안 도는 이유를
    아무도 모른다.

    Args:
        conn: DB 커넥션.

    Returns:
        ``skill_code`` 오름차순 스킬 목록(코어 조회의 정렬을 그대로 잇는다). 활성 스킬이 없으면 빈 목록.
    """
    return [skill_from_row(row) for row in fetch_active_skills(conn)]


def fetch_skill_targets(
    conn: Any,
    skill: ClassificationSkill,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """스킬 하나의 대상 자산과 재료(요약·키워드)를 읽는다(조회 전용 · spec §5 대상 선별).

    대상은 **판정 행이 없는 자산**(신규 수집·이전 판정 실패)과 **구버전 판정 자산**(스킬 개정 후
    백필)이다. 두 질의 모두 코어 함수에 위임한다 — 파이프가 SQL 을 다시 쓰면 선별 규칙이 갈린다.

    Args:
        conn: DB 커넥션.
        skill: 대상 스킬. 재선별 기준 버전은 **이 객체의 버전**(= DB 카운터)이다.
        limit: 한 번에 가져올 상한. ``None``(기본)이면 전량(배치를 나눠 돌 때만 준다).

    Returns:
        ``[{asset_id, summary, keywords}]`` — asset_id 오름차순. 대상이 없으면 **빈 목록**이며
        재료 질의는 아예 돌지 않는다(빈 목록으로 질의하면 왕복만 낭비한다).
    """
    pending = fetch_pending_asset_ids(
        conn, skill_code=skill.skill_code, skill_version=skill.version, limit=limit
    )
    if not pending:
        return []
    return fetch_asset_materials(conn, asset_ids=pending)


def run_batch(
    db: Any,
    *,
    skill_codes: Sequence[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    no_index: bool = False,
) -> dict[str, Any]:
    """배치 한 판을 배선한다 — 활성 스킬 로드 → 스킬별 대상·판정·저장 → OS 부분 갱신.

    ``main``(명령행)과 Airflow DAG 가 **같은 이 함수**를 부른다. DAG 에 배선을 복사하면 한쪽만
    고쳐져 갈라진다.

    트랜잭션 경계가 이 함수의 책임이다:
        - 활성 스킬 로드·스킬별 대상 조회 = 짧은 **읽기** 트랜잭션.
        - 자산 × 스킬 저장 = **fresh 트랜잭션**(한 건 실패가 다른 자산을 롤백시키지 않는다).

    Args:
        db: 트랜잭션을 열 수 있는 DB 핸들(``PostgresUtil``). 풀 수명은 호출부가 관리한다.
        skill_codes: 돌릴 스킬 코드 목록. ``None``(기본)이면 활성 스킬 전부.
        limit: 스킬당 처리할 자산 수 상한. ``None``(기본)이면 전량.
        dry_run: 참이면 **아무 것도 쓰지 않는다**(DB·OS 모두 · OS 클라이언트도 만들지 않는다).
        no_index: 참이면 OpenSearch 부분 갱신을 건너뛴다(DB 판정만 — OS 가 없는 환경).

    Returns:
        ``run_classify`` 의 diff 리포트. 토글(``MM_CLASSIFY_ENABLED``)이 꺼져 있거나 활성 스킬이
        없으면 ``{"skipped": …}`` 를 함께 담아 돌려주고 질의도 LLM 호출도 하지 않는다.

    Raises:
        ValueError: ``skill_codes`` 에 활성 스킬이 아닌 코드가 있을 때(오타를 "0건 처리"로 흡수하면
            왜 안 도는지 추적하게 된다).
    """
    from src.config.settings import get_current_settings

    cfg = get_current_settings()
    empty: dict[str, Any] = {"dry_run": bool(dry_run), "targets": 0, "judged": 0, "failed": 0,
                             "rows": 0, "indexed": 0, "index_failed": 0, "skills": []}
    if not cfg.mm_classify.enabled:
        # 끄기 ≠ 삭제 — 이미 쌓인 판정 행·색인은 그대로 두고 **새 판정만** 멈춘다.
        _LOG.info("mm_classify disabled(MM_CLASSIFY_ENABLED=0)")
        return {**empty, "skipped": "disabled"}

    skills = db.execute_in_transaction(load_active_skills, idempotent=True)
    if skill_codes:
        wanted = set(skill_codes)
        missing = sorted(wanted - {s.skill_code for s in skills})
        if missing:
            raise ValueError(f"활성 스킬에 없는 코드: {missing}")
        skills = [s for s in skills if s.skill_code in wanted]
    if not skills:
        return {**empty, "skipped": "no_active_skills"}

    index_fn = None
    if not dry_run and not no_index:
        # 무거운 의존(opensearch-py)은 실제로 색인할 때만 끌어온다.
        from src.search.opensearch_sync import get_client, update_asset_mm_skill_labels

        os_client = get_client()
        os_index = cfg.opensearch.index

        def index_fn(asset_id: str, keys: list[str]) -> None:  # noqa: F811 — 조건부 배선
            """색인의 ``mm_skill_labels`` 칸만 갈아 끼운다(다른 필드·임베딩 불변).

            Args:
                asset_id: 갱신할 자산.
                keys: ``"스킬코드/라벨코드"`` 목록. **빈 목록이면 그 자산의 스킬 라벨을 전부
                    지운다**(강등·미부여 잔재 정리 — 필드 생략과 다르다).
            """
            update_asset_mm_skill_labels(os_client, os_index, asset_id, keys)

    def _targets(skill: ClassificationSkill) -> list[dict[str, Any]]:
        """스킬 하나의 대상·재료를 읽는다 — 짧은 읽기 트랜잭션."""
        return db.execute_in_transaction(
            lambda conn, _s=skill: fetch_skill_targets(conn, _s, limit=limit), idempotent=True
        )

    def _labels(asset_id: str) -> list[dict[str, str]]:
        """자산 하나의 활성 스킬 판정 행을 읽는다(신규/갱신 구분·색인 값의 원천)."""
        return db.execute_in_transaction(
            lambda conn, _a=asset_id: fetch_asset_label_rows(conn, _a), idempotent=True
        )

    def _persist(skill: ClassificationSkill, asset_id: str, judgement: Any) -> int:
        """자산 × 스킬 행을 **한 트랜잭션**으로 교체한다(DELETE→INSERT 전체 교체)."""
        return db.execute_in_transaction(
            lambda conn, _s=skill, _a=asset_id, _j=judgement: replace_asset_labels(
                conn, asset_id=_a, skill_code=_s.skill_code, skill_version=_s.version,
                judgement=_j,
            ),
            idempotent=False,
        )

    return run_classify(
        skills,
        targets_fn=_targets,
        persist_fn=None if dry_run else _persist,
        labels_fn=_labels,
        index_fn=index_fn,
        dry_run=dry_run,
    )


def format_report(report: Mapping[str, Any]) -> str:
    """배치 결과를 사람이 읽는 여러 줄로 만든다(순수 함수).

    Args:
        report: ``run_classify`` 가 돌려준 diff 리포트.

    Returns:
        요약 문자열. 라벨 분포는 **많이 붙은 순 앞 몇 개만** 덧붙인다 — 전부 찍으면 콘솔이 넘쳐
        정작 상태를 못 본다(자세한 목록은 반환 dict 에 있다).
    """
    dry = " (dry-run · 쓰기 0)" if report.get("dry_run") else ""
    lines = [
        f"[분류 스킬 배치]{dry} 스킬 {len(report.get('skills', []))}종 | "
        f"대상 {report.get('targets', 0)} · 판정 {report.get('judged', 0)} · "
        f"실패 {report.get('failed', 0)} · 색인 {report.get('indexed', 0)}"
    ]
    for row in report.get("skills", []):
        labels = sorted(row.get("labels", {}).items(), key=lambda kv: (-kv[1], kv[0]))
        joined = ", ".join(f"{code}={count}" for code, count in labels[:_MAX_LABELS_IN_LINE])
        pct = round(row.get("unassigned_ratio", 0.0) * 100, 1)
        lines.append(
            f"  · {row.get('name')}(v{row.get('version')}) 대상 {row.get('targets', 0)} → "
            f"판정 {row.get('judged', 0)}(신규 {row.get('created', 0)} · 갱신 {row.get('updated', 0)}) "
            f"· 실패 {row.get('failed', 0)} · 행 {row.get('rows', 0)}"
        )
        if joined:
            lines.append(f"      라벨 분포: {joined}")
        lines.append(f"      해당없음 {row.get('unassigned', 0)}건({pct}%) — 라벨이 못 덮는 영역")
        if row.get("failures"):
            lines.append(f"      ⚠️ 실패 사유: {row['failures']}")
    if report.get("index_failed"):
        lines.append(
            f"  ⚠️ 색인 실패 {report['index_failed']}건 — DB 판정은 저장됐다. "
            "run_opensearch_resync 로 색인을 맞춘다"
        )
    return "\n".join(lines)


def _build_parser():
    """명령행 옵션을 정의한다(환경·미리보기·상한·색인 생략·스킬 지정).

    Returns:
        구성된 ``argparse.ArgumentParser``.
    """
    import argparse

    p = argparse.ArgumentParser(
        description="분류 스킬 배치 (스킬별 판정 → asset_mm_skill_label → OS 부분 갱신)"
    )
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    p.add_argument(
        "--dry-run", dest="dry_run", action="store_true",
        help="쓰기 0 — 무엇이 붙을지만 보고한다(판정 LLM 호출은 한다)",
    )
    p.add_argument("--limit", type=int, default=None, help="스킬당 처리할 자산 수 상한")
    p.add_argument(
        "--no-index", dest="no_index", action="store_true",
        help="OpenSearch 부분 갱신을 건너뛴다(DB 판정만 · 색인은 나중에 재색인 도구로 맞춘다)",
    )
    p.add_argument(
        "skills", nargs="*", metavar="SKILL_CODE", default=[],
        help="돌릴 스킬 코드(미지정=활성 스킬 전부)",
    )
    return p


# ── 초기 설정(부트스트랩) 절차 ────────────────────────────────────────────────
# [런타임·main() 안·순서 중요]
#   1) bootstrap_env(env): load_dotenv + init_settings(필수 env 검증·frozen 설정)
#   2) PostgresUtil() + `with db:`: 연결 풀 + PG17 검증
#   3) run_batch: 토글 확인 → 활성 스킬 로드 → 스킬 루프(판정·저장·색인)
#   4) 결과 출력. 배선 자체는 run_batch 한 곳뿐이다 — DAG 도 같은 함수를 부른다.
def main() -> int:
    """분류 스킬 배치를 실행한다(명령행 진입점).

    Returns:
        0=성공, 1=판정 실패 또는 색인 실패가 있음, 2=지정한 스킬 코드가 활성 목록에 없음.
    """
    args = _build_parser().parse_args()

    from src.config.bootstrap import bootstrap_env
    from src.database.postgres_util import PostgresUtil

    bootstrap_env(args.env)

    db = PostgresUtil()
    with db:
        try:
            report = run_batch(
                db,
                skill_codes=list(args.skills) or None,
                limit=args.limit,
                dry_run=args.dry_run,
                no_index=args.no_index,
            )
        except ValueError as exc:
            # 스킬 코드 오타 — 배치를 돌리지 않고 사람이 읽는 한 줄로 끝낸다.
            print(f"[분류 스킬 배치] {exc}")
            return 2

    if report.get("skipped") == "disabled":
        print("[분류 스킬 배치] MM_CLASSIFY_ENABLED=0 — 아무 것도 하지 않는다")
        return 0
    if report.get("skipped") == "no_active_skills":
        print("[분류 스킬 배치] 활성 스킬 0종 — scripts/register_mm_skill.py 로 먼저 등록한다")
        return 0
    print(format_report(report))
    return 1 if (report["failed"] or report["index_failed"]) else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
