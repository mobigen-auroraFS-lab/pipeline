"""「제목 참고」(spec 114) 판단 한 곳 — 스킬 4종 · 적재 단계 · 주제 판정이 함께 쓴다(순수).

요약기가 자료 제목(파일명)을 참고로 받게 하되, **설정을 켜기 전에는 지금과 똑같이** 동작해야 한다 — 코드가
먼저 배포돼도 운영 적재가 바뀌지 않게, 결과 판정 뒤 사람이 켠다. 켜는 판단 · 제목 계산 ·
「제목이 내용과 맞다고 확인됐나」를 여기 한 곳에 둬, 스킬마다 조건이 갈리지 않게 한다.
설계 배경: `specs/114-title-aware-summary`
"""

from __future__ import annotations

from typing import Any

from src.config.filename_util import summary_title


def hint_enabled(cfg: Any) -> bool:
    """설정 ``vlm.summary_title_hint`` 가 **정확히 True** 인가(순수).

    ``is True`` 로 보는 이유: 테스트 · 대역 설정(Mock)의 속성은 무엇이든 참처럼 보여, 단순 진릿값으로 보면
    켜지지 않은 설정이 켜진 것으로 새어 든다.

    Args:
        cfg: 설정 객체(``None`` · 속성이 없는 객체도 받는다 — 그때는 꺼짐).
    """
    return getattr(getattr(cfg, "vlm", None), "summary_title_hint", False) is True


def title_for_summary(file_path: Any, cfg: Any) -> str | None:
    """요약기에 넘길 제목 — 설정이 꺼져 있거나 뜻 없는 이름이면 ``None``(순수).

    Args:
        file_path: 자산 파일 경로(자산 id 접두가 붙어 있어도 된다).
        cfg: 설정 객체.
    """
    return summary_title(str(file_path)) if hint_enabled(cfg) and file_path else None


def confirmed_title(ext_meta: Any, file_path: Any, cfg: Any) -> str | None:
    """**내용과 맞다고 확인된** 제목 — 주제 판정이 자기 텍스트 앞에 붙일 값(순수).

    요약기가 같은 호출에서 ``title_match == "yes"`` 로 판단한 경우에만 돌려준다. ``no`` · ``unsure`` · 판단 없음
    (설정을 켜기 전 적재분)이면 ``None`` 이라 주제 판정이 지금과 같다.

    Args:
        ext_meta: 자산 확장 메타(``title_match`` 를 본다). ``None`` 도 받는다.
        file_path: 자산 파일 경로.
        cfg: 설정 객체.
    """
    if not hint_enabled(cfg) or not isinstance(ext_meta, dict) or ext_meta.get("title_match") != "yes":
        return None
    return title_for_summary(file_path, cfg)


def summarizer_kwargs(title: str | None) -> dict[str, str]:
    """요약기 호출에 덧붙일 인자 — 제목이 없으면 **빈 dict**(``title=None`` 조차 넘기지 않아 호출 모양이 지금과 같다).

    Args:
        title: ``title_for_summary`` 결과.
    """
    return {"title": title} if title else {}


# 요약기가 제목 참고 때 남기는 ext_meta 키 — 코어 마이그레이션 v308 이 등록 표에 올린다.
TITLE_EXT_KEYS = ("title_match", "title_only")


def missing_title_keys(conn: Any, domains: tuple[str, ...] = ("general", "medical")) -> list[str]:
    """제목 참고 키가 ext_meta 등록 표에 없는 도메인을 찾는다(읽기 전용).

    왜 필요한가: 쓰기 경로는 등록된 키만 받는다(``validate_ext_meta``). 제목 참고를 켠 채 v308 이 적용되지 않았으면
    그 도메인의 적재가 전부 「미등록 키」로 롤백된다 — 시작할 때 한 번 확인해 바로 멈추는 편이 낫다.
    등록 키가 하나도 없는 도메인은 원래 검증을 건너뛰므로 문제 삼지 않는다.

    Args:
        conn: DB 연결.
        domains: 확인할 도메인.

    Returns:
        ``["도메인:키", …]`` — 빠진 것이 없으면 빈 목록.
    """
    from src.registry.ext_meta_field_registry import fetch_allowed_ext_keys

    missing: list[str] = []
    for domain in domains:
        allowed = fetch_allowed_ext_keys(conn, domain)
        if allowed:
            missing += [f"{domain}:{k}" for k in TITLE_EXT_KEYS if k not in allowed]
    return missing

