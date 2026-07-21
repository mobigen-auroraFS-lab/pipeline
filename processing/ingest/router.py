"""F-1.2 모달리티 라우팅.

수집된 로컬 파일 경로마다 ``detect_file_kind`` 로 modality 를 판정하고,
처리 가능 여부(routable)와 사유를 담은 ``RouteResult`` 를 돌려준다.
- 없는 파일 → routable=False, reason='missing'
- 미지원/판별 불가(unknown) → routable=False, reason='unknown_modality'
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


@dataclass(frozen=True)
class RouteResult:
    """라우팅 판정 결과."""

    file_path: str
    modality: str  # MediaKind/OfficeKind 값 또는 'unknown'
    domain: str  # 'general' | 'medical' | 'review' (분류는 후속 단계가 갱신)
    routable: bool  # True 면 처리 진행, False 면 skip
    reason: str = ""  # routable=False 사유


def route_file(file_path: str | Path, *, domain: str = "general") -> RouteResult:
    """단일 파일의 라우팅 판정."""
    p = Path(file_path)
    if not p.is_file():
        return RouteResult(str(file_path), MediaKind.UNKNOWN.value, domain, False, REASON_MISSING)
    try:
        kind = detect_file_kind(p)
    except FileNotFoundError:
        # is_file 통과 후 사라진 경우(TOCTOU) 방어.
        return RouteResult(str(file_path), MediaKind.UNKNOWN.value, domain, False, REASON_MISSING)
    if kind == MediaKind.UNKNOWN.value:
        return RouteResult(str(file_path), kind, domain, False, REASON_UNKNOWN_MODALITY)
    return RouteResult(str(file_path), kind, domain, True, "")
