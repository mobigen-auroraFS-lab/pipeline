"""F-1.1 로컬 파일 수집.

큐·이벤트 없이 로컬 디렉터리/파일 목록/명시 경로에서 처리 대상 파일 경로 리스트를 만든다.
오케스트레이터(``run_ingest``)가 이 리스트를 순회하며 라우팅→분류→추출→적재한다.
큐(Redis/Kafka)·FS 이벤트원은 추후 도입(이 모듈을 대체/감싸는 형태).

DB·LLM 불필요(순수 파일시스템). 기존 ``directory_paths.list_file_paths_under_directory`` 와
``run_ingest`` 진입점의 ``--input-dir``/``--file-list`` 수집 규칙을 재사용한다.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from src.file.directory_paths import list_file_paths_under_directory


def _read_file_list(file_list: str | Path) -> list[str]:
    """UTF-8 텍스트 목록(한 줄 하나). 공백 트리밍, 빈 줄 무시."""
    path = Path(file_list).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"--file-list 가 파일이 아닙니다: {path}")
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def collect_files(
    *,
    input_dir: str | Path | None = None,
    file_list: str | Path | None = None,
    paths: Iterable[str] | None = None,
    recursive: bool = True,
) -> list[str]:
    """로컬 파일 경로 리스트를 수집한다.

    소스(``file_list`` → ``input_dir`` → ``paths`` 순)를 합치고 **정확히 동일한 경로 문자열**은
    첫 등장 순서를 유지한 채 1개만 남긴다.

    Args:
        input_dir: 이 디렉터리 아래 파일 수집(``list_file_paths_under_directory``).
        file_list: UTF-8 경로 목록 파일(한 줄 하나).
        paths: 명시 파일 경로들.
        recursive: ``input_dir`` 하위 디렉터리까지 재귀 탐색할지.

    Returns:
        파일 경로 문자열 리스트(소스가 비어 있으면 빈 리스트).

    Raises:
        ValueError: 세 소스(``input_dir``/``file_list``/``paths``)가 모두 없을 때.
        FileNotFoundError: ``file_list`` 가 파일이 아닐 때.
        NotADirectoryError: ``input_dir`` 가 디렉터리가 아닐 때(하위 함수에서).
    """
    if input_dir is None and file_list is None and paths is None:
        raise ValueError("input_dir / file_list / paths 중 하나는 지정해야 합니다.")

    collected: list[str] = []
    if file_list is not None:
        collected.extend(_read_file_list(file_list))
    if input_dir is not None:
        collected.extend(
            list_file_paths_under_directory(input_dir, recursive=recursive)
        )
    if paths is not None:
        collected.extend(str(p) for p in paths)

    # 정확히 동일한 경로 문자열만 첫 등장 순서를 유지하며 제거.
    seen: set[str] = set()
    deduped: list[str] = []
    for p in collected:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped
