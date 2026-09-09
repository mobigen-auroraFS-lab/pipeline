"""F-1.2 모달리티 라우팅.

수집된 로컬 파일 경로마다 ``detect_file_kind`` 로 modality 를 판정하고,
처리 가능 여부(routable)와 사유를 담은 ``RouteResult`` 를 돌려준다.
- 없는 파일 → routable=False, reason='missing'
- 미지원/판별 불가(unknown) → routable=False, reason='unknown_modality'
- 장부 파일(수집 도구가 남긴 목록 파일 · ``LEDGER_FILE_NAMES``) → routable=False, reason='ledger_file'
- 그 외(txt/pdf/json/word/excel/powerpoint/image/video/audio) → routable=True

DB 를 직접 만지지 않는다(순수 판정). 적재·격리 등 후속 처리는 오케스트레이터(run_ingest)가
이 결과를 보고 결정한다. domain 은 기본 'general' 이며, 실제 도메인 분류(F-5.1)는
라우팅 이후 단계에서 채운다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.file.file_type_defs import MediaKind
from src.file.file_type_detector import detect_file_kind

REASON_MISSING = "missing"
REASON_UNKNOWN_MODALITY = "unknown_modality"
REASON_LEDGER_FILE = "ledger_file"

# 자산이 아닌 **장부 파일** — 수집 도구가 "어떤 파일을 모았나"를 적어 둔 목록(JSON). 내용이 코퍼스의
# 모든 주제를 훑기 때문에 자산으로 들어가면 거의 모든 검색에 「관련 있음」으로 걸린다(2026-09-09 실측:
# `manifest.json` 1건이 텍스트 자산으로 색인돼 오케스트라·양궁 등 무관한 질의에 반복 등장). 파일 **이름을
# 정확히** 대조한다(대소문자 무시) — 패턴으로 넓히면 정상 JSON 자료(`데이터_manifest.json` 같은 이름)까지
# 빠질 수 있어 닫힌 목록으로 둔다. 수집 도구가 남기는 이름 두 가지: `_manifest.json`(corpus_collect) · `manifest.json`.
LEDGER_FILE_NAMES: frozenset[str] = frozenset({"manifest.json", "_manifest.json"})


@dataclass(frozen=True)
class RouteResult:
    """라우팅 판정 결과."""

    file_path: str
    modality: str  # MediaKind/OfficeKind 값 또는 'unknown'
    domain: str  # 'general' | 'medical' | 'review' (분류는 후속 단계가 갱신)
    routable: bool  # True 면 처리 진행, False 면 skip
    reason: str = ""  # routable=False 사유


def route_file(file_path: str | Path, *, domain: str = "general") -> RouteResult:
    """파일 하나를 어떤 경로로 처리할지 판정한다(모달리티·도메인·처리 가능 여부).

    Args:
        file_path: 대상 파일.
        domain: 기본 도메인. 판정으로 바뀔 수 있다.

    Returns:
        라우팅 결과. **파일이 없거나 종류를 알 수 없어도 예외를 올리지 않고** 사유를 담아
        돌려준다 — 배치가 파일 하나 때문에 멈추지 않게 하려는 것이다.
    """
    p = Path(file_path)
    if not p.is_file():
        return RouteResult(str(file_path), MediaKind.UNKNOWN.value, domain, False, REASON_MISSING)
    if p.name.lower() in LEDGER_FILE_NAMES:
        # 장부 파일은 종류 판정 전에 걸러낸다 — 판정하면 'json' 텍스트로 통과해 자산이 된다.
        return RouteResult(str(file_path), MediaKind.UNKNOWN.value, domain, False, REASON_LEDGER_FILE)
    try:
        kind = detect_file_kind(p)
    except FileNotFoundError:
        # is_file 통과 후 사라진 경우(TOCTOU) 방어.
        return RouteResult(str(file_path), MediaKind.UNKNOWN.value, domain, False, REASON_MISSING)
    if kind == MediaKind.UNKNOWN.value:
        return RouteResult(str(file_path), kind, domain, False, REASON_UNKNOWN_MODALITY)
    return RouteResult(str(file_path), kind, domain, True, "")
