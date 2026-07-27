"""F-4.3 하이브리드 검색 CLI 진입점 — 한국어 질의로 asset_* 인덱스를 검색해 결과를 출력한다.

예) python -m processing.app.run_search --env dev --query "작년 워크숍 발표자료" --modalities text,image

검색은 파이프라인(run_ingest/run_relations) 밖의 라이브러리 계층이라 레지스트리·도메인 팩
import 부수효과가 필요 없다. 질의 구조화 LLM 은 공통 seam(``src.llm.client``)을 쓰고, 검색은
OpenSearch 단일 백엔드(037·BM25+kNN)를 ``search_hybrid`` seam 경유로만 조회한다(PG 직조회 아님).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from typing import Any

from src.config.search_modalities import VALID_SEARCH_MODALITIES, parse_modalities_csv
from src.search.search_service import search_hybrid


def _resolve_modalities(raw: str | None) -> list[str] | None:
    """``--modalities`` 를 공유 파서로 파싱한 뒤 유효값을 검증한다(069 T301·D5).

    파싱(split/strip/소문자·미지정=None)은 ``parse_modalities_csv`` 단일 출처를 쓰고, 유효값 밖
    모달리티는 ``ValueError`` 로 거부한다(069 P3-12). 이 예외는 ``main`` 이 ``parser.error`` 로
    변환해 raw traceback 대신 명확한 CLI 에러+usage 로 안내한다(예전엔 search_hybrid 내부에서
    ValueError 가 터져 traceback 이 그대로 노출됐다).
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
    """파싱된 인자를 검색 서비스 호출로 매핑한다. ``search_fn`` 은 테스트 주입 seam.

    ``_run`` 은 settings 전역에 의존하지 않아 순수 매핑으로 단위 테스트된다.
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
    # 불필요한 DB 초기화 없이 즉시 명확한 에러+usage 로 거부(069 P3-12, parser.error → exit 2).
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
