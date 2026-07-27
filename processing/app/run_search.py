"""검색 CLI — 한국어 질의로 색인을 검색해 결과를 출력한다.

예) ``python -m processing.app.run_search --env dev --query "작년 워크숍 발표자료"``

**흐름에서의 위치**: 적재·관계 파이프라인 **밖**이다. 그래서 전략 레지스트리도 도메인 팩도
필요 없어 그 등록 부수효과를 일으키지 않는다.

검색은 반드시 코어 검색 함수를 거친다 — DB 를 직접 조회하지 않는다. 그래야 CLI 와 HTTP
백엔드가 **같은 결과**를 낸다.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from src.config.search_modalities import VALID_SEARCH_MODALITIES, parse_modalities_csv
from src.search.search_service import search_hybrid


def _resolve_modalities(raw: str | None) -> list[str] | None:
    """모달리티 옵션을 파싱하고 허용값인지 검증한다.

    파싱 규칙은 공용 파서 하나만 쓴다 — CLI 와 HTTP 가 다른 규칙을 갖게 되면 같은 문자열이
    다르게 해석된다.

    Args:
        raw: 콤마로 이은 문자열. ``None``·빈 값이면 **전체**를 뜻한다.

    Returns:
        검증된 목록, 또는 전체를 뜻하는 ``None``.

    Raises:
        ValueError: 허용 밖 모달리티가 섞였을 때. 호출부(``main``)가 이것을 CLI 사용법
            안내로 바꾼다 — 그냥 흘리면 사용자가 스택 트레이스를 보게 된다.
    """
    mods = parse_modalities_csv(raw)
    if mods is None:
        return None
    unknown = [m for m in mods if m not in VALID_SEARCH_MODALITIES]
    if unknown:
        raise ValueError(
            f"알 수 없는 modality: {unknown} (허용: {list(VALID_SEARCH_MODALITIES)})"
        )
    return mods


def _run(
    args: argparse.Namespace,
    *,
    search_fn: Callable[..., dict[str, Any]] = search_hybrid,
) -> dict[str, Any]:
    """파싱된 인자를 검색 호출로 옮긴다 — 전역 설정을 읽지 않는 순수 매핑이다.

    Args:
        args: 파싱된 명령행 인자.
        search_fn: 검색 함수. **바꿔 끼울 수 있게 열어 뒀다** — 검색 엔진 없이 인자 전달만
            단위로 검증할 수 있다.

    Returns:
        검색 결과 dict(그대로 출력용).
    """
    return search_fn(
        args.query,
        modalities=_resolve_modalities(args.modalities),
        limit_per_bucket=args.limit,
    )


def _build_parser() -> argparse.ArgumentParser:
    """명령행 옵션을 정의한다(환경·질의·모달리티·개수)."""
    parser = argparse.ArgumentParser(description="하이브리드 검색 (asset_* 인덱스)")
    parser.add_argument("--env", choices=["dev", "prod"], default="dev")
    parser.add_argument("--query", required=True, help="검색 질의(한국어)")
    parser.add_argument(
        "--modalities",
        default=None,
        help="콤마 구분 모달리티(text,audio,image,video). 미지정=전체",
    )
    parser.add_argument("--limit", type=int, default=20, help="버킷당 최대 결과 수")
    return parser


# 런타임 순서(run_ingest 와 동일): 1) load_dotenv(.env.{env}, override=False) →
# 2) init_settings(env)(필수 환경변수 검증) → 3) 검색 실행. LLM/임베딩 클라이언트는 첫 사용 시 지연 초기화.
def main() -> int:
    """명령행에서 검색을 실행해 결과를 출력한다.

    모달리티 오타는 **설정을 읽기 전에** 걸러낸다 — 불필요한 DB 초기화 없이 바로
    사용법을 보여주기 위해서다.

    Returns:
        0=성공.
    """

    from src.config.bootstrap import bootstrap_env

    parser = _build_parser()
    args = parser.parse_args()

    # 모달리티 검증을 부트스트랩(.env 로드·init_settings) **이전**에 수행한다 — 오타는 raw traceback·
    # DB 를 열기 **전에** 거부한다 — 인자가 틀렸는데 연결부터 맺으면 실패까지 오래 걸린다.
    # 아래 _run 이 동일 파서로 재해석하나 순수·저비용이라 무해하다(검증은 여기서 이미 통과 확정).
    try:
        _resolve_modalities(args.modalities)
    except ValueError as exc:
        parser.error(str(exc))

    bootstrap_env(args.env)

    result = _run(args)
    print(json.dumps(result, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
