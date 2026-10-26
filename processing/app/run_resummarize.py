"""적재된 자산의 요약 · 키워드 · 주제 · about 을 **제목 참고 요약기로 다시 만드는** 재처리 배치(spec 114).

**흐름에서의 위치**: 정상 적재 경로가 아니다. 새 자산은 적재 때 스킬이 제목을 참고해 요약한다(설정
``SUMMARY_TITLE_HINT`` 를 켠 뒤). 이 배치는 **이미 적재된 자산**을 같은 규칙으로 다시 만드는 일회성 도구다.

무엇을 재사용하나(무거운 단계를 다시 돌리지 않는다):
  · 영상 — 저장된 장면 분석(``ext_meta.keyframes``)으로 영상 요약만 다시 만든다(키프레임 · 장면 VLM 재실행 없음).
  · 음성 — 저장된 STT(``ext_meta.stt``)로 요약만 다시 만든다(Whisper 재실행 없음).
  · 문서 — 원본 파일로 요약을 다시 만든다(청크 요약 + 종합).
  · 사진 — 원본으로 요약을 다시 만들고, **텍스트 임베딩도 다시 만든다**(요약 · 키워드 · 라벨이 곧 사진 임베딩의
    재료라 요약만 바꾸면 검색 글자와 벡터가 서로 다른 내용을 가리킨다). 라벨은 그대로 둔다(CLIP 이 그림 자체로
    매긴 값). 영상 · 음성 · 문서의 임베딩은 장면 설명 · STT · 본문 조각으로 만들어 요약과 무관하다.

자산마다: 새 요약 · about · (사진) 임베딩을 먼저 만든다(LLM · 트랜잭션 밖) → 한 트랜잭션에서 주제 재판정(확인된 제목만
붙임) · 요약 결과 키 덮어쓰기 · about · (사진) 임베딩 저장 → 검색 색인 갱신(따로 — 실패하면 ``index_failed`` 로 남기고 ``--index-only`` 로 다시).
개체 묶음은 배치가 끝난 뒤 기존 개체 배치를 대상 자산에만 돌린다(``run_mm_meta_binding --rejudge``).

건너뛰는 자산(실패 아님):
  · 뜻 있는 제목이 없는 자산 — 제목 없이 요약하면 지금과 같은 결과라 다시 만들 이유가 없다(조회 단계에서 뺀다).
  · 새 요약이 빈 자산(예: 장면 분석이 없는 영상) — 옛 요약을 빈 값으로 덮지 않는다.

🔴 **DB 에 쓴다.** DB 를 백업한 뒤에 돌린다. 먼저 ``--dry-run``
(생성만 · 기록 없음)으로 결과를 본다. 같은 자산을 다시 돌리면 덮어쓴다 — 기본은 이미 끝난 자산(``title_match``
가 있는 자산)을 건너뛰는 이어 하기다.

사용법
    SUMMARY_TITLE_HINT=1 python -m processing.app.run_resummarize --env dev --modality video,audio --dry-run --limit 20
    SUMMARY_TITLE_HINT=1 python -m processing.app.run_resummarize --env dev --workers 8 --out resummarize_all.jsonl
    # 색인만 실패한 자산 다시 색인(기록 파일 옆 *.index_failed.txt)
    SUMMARY_TITLE_HINT=1 python -m processing.app.run_resummarize --env dev --index-only \\
        --asset-ids-file resummarize_all.jsonl.index_failed.txt
설계 배경: `specs/114-title-aware-summary`
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import unicodedata
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from psycopg.rows import dict_row

from processing.classify.asset_topic import build_self_text
from processing.skills.title_context import (
    confirmed_title,
    hint_enabled,
    missing_title_keys,
    summarizer_kwargs,
    title_for_summary,
)
from src.config.embedding_constants import FIX_EMBEDDING_DIMENSION
from src.config.filename_util import display_file_name, summary_title

_LOG = logging.getLogger("meta_extract.run_resummarize")

# 요약기 결과 중 확장 메타에 쓰는 키 — 이 밖의 키(about · labels · keyframes · stt …)는 건드리지 않는다.
SUMMARY_KEYS = ("summary", "keywords", "objects", "title_match", "title_only")


class TitleHintOffError(RuntimeError):
    """제목 참고 설정이 꺼진 채 배치를 시작하려 했다 — 지금과 같은 요약을 다시 만들 뿐이라 시작하지 않는다."""


class SkipAsset(Exception):
    """이 자산은 다시 만들 것이 없어 건너뛴다(실패 아님) — 예: 새 요약이 비었다."""


_TARGET_SQL = """
SELECT a.asset_id, a.modality, a.fs_path, a.domain_label, m.ext_meta
FROM asset a
JOIN asset_metadata m ON m.asset_id = a.asset_id
WHERE a.status = 'registered'
  AND a.modality = ANY(%s)
"""


def fetch_targets(
    conn: Any,
    modalities: Sequence[str],
    *,
    exclude_names: Iterable[str] | None = None,
    asset_ids: Sequence[str] | None = None,
    resume: bool = True,
    titled_only: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """재처리 대상 자산을 읽는다(읽기 전용).

    Args:
        conn: DB 연결.
        modalities: 대상 종류(``video`` · ``audio`` · ``text`` · ``image``).
        exclude_names: 뺄 표시 파일명 집합(일부를 따로 돌릴 때). 자산 id 접두는 벗기고 비교한다.
        asset_ids: 이 자산들만(``None`` 이면 종류 전체).
        resume: 참이면 이미 재처리된 자산(``ext_meta`` 에 ``title_match`` 가 있는 자산)을 건너뛴다(이어 하기).
        titled_only: 참이면 뜻 있는 제목(``summary_title``)이 없는 자산을 뺀다 — 제목 없이 요약하면 지금과 같은
            결과라 LLM 만 다시 부르고, ``title_match`` 가 생기지 않아 이어 하기에서도 매번 다시 잡힌다.
        limit: 조회 상한(제외 이름 · 제목 없는 자산을 거르기 **전** 행 수 — 실제 대상은 더 적을 수 있다).

    Returns:
        ``[{asset_id, modality, fs_path, domain_label, ext_meta}]`` — asset_id 오름차순.
    """
    sql = _TARGET_SQL
    params: list[Any] = [list(modalities)]
    if asset_ids:
        sql += "  AND a.asset_id = ANY(%s::uuid[])\n"
        params.append([str(a) for a in asset_ids])
    if resume:
        sql += "  AND NOT (m.ext_meta ? 'title_match')\n"
    sql += "ORDER BY a.asset_id\n"
    if limit is not None:
        sql += "LIMIT %s\n"
        params.append(int(limit))
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, tuple(params))
        rows = cur.fetchall()
    # 제외 목록 · 경로 모두 NFC 로 모아 비교한다 — macOS 파일명(NFD)이면 같은 이름이 조용히 안 빠진다.
    excluded = {unicodedata.normalize("NFC", n) for n in (exclude_names or ())}
    return [
        {"asset_id": str(r["asset_id"]), "modality": r["modality"], "fs_path": r["fs_path"],
         "domain_label": r["domain_label"], "ext_meta": r["ext_meta"] or {}}
        for r in rows
        if unicodedata.normalize("NFC", display_file_name(r["fs_path"])) not in excluded
        and (not titled_only or summary_title(str(r["fs_path"] or "")) is not None)
    ]


def build_updates(asset: dict[str, Any], cfg: Any) -> dict[str, Any]:
    """자산 하나의 새 요약 결과(확장 메타에 덮어쓸 키만)를 만든다 — LLM 호출 · DB 무관.

    Args:
        asset: ``fetch_targets`` 의 한 행.
        cfg: 설정 — ``vlm.summary_title_hint`` 가 켜져 있어야 한다.

    Returns:
        ``SUMMARY_KEYS`` 중 요약기가 돌려준 키만 담은 dict.

    Raises:
        TitleHintOffError: 제목 참고 설정이 꺼져 있을 때.
        SkipAsset: 새 요약이 비었을 때(예: 장면 분석이 없는 영상) — 옛 요약을 빈 값으로 덮지 않는다.
        ValueError: 모르는 종류.
    """
    if not hint_enabled(cfg):
        raise TitleHintOffError("SUMMARY_TITLE_HINT=1 로 실행하세요 — 꺼진 채로는 지금과 같은 요약을 다시 만들 뿐입니다")
    kw = summarizer_kwargs(title_for_summary(asset["fs_path"], cfg))
    ext = asset.get("ext_meta") or {}
    mod = asset["modality"]
    if mod == "video":
        from src.llm.video_summarizer import summarize_video_from_scene_results
        result = summarize_video_from_scene_results(ext.get("keyframes") or [], **kw)
    elif mod == "audio":
        from src.llm.text_summarizer import summarize_and_extract_keywords_from_audio
        result = summarize_and_extract_keywords_from_audio(ext.get("stt") or "", **kw)
    elif mod == "text":
        from src.llm.text_summarizer import summarize_and_extract_keywords
        path = asset["fs_path"]
        result = summarize_and_extract_keywords(path, file_kind=Path(path).suffix.lstrip(".").lower(), **kw)
    elif mod == "image":
        from src.llm.image_summarizer import summarize_image_caption_keywords_objects
        result = dict(summarize_image_caption_keywords_objects(file_path=asset["fs_path"], **kw))
        # 적재와 같이 — 사진의 objects 는 CLIP 라벨 후보로만 쓰고 메타에 남기지 않는다(image_skill).
        result.pop("objects", None)
    else:
        raise ValueError(f"모르는 종류: {mod!r}")
    if not str(result.get("summary") or "").strip():
        raise SkipAsset("새 요약이 비었다 — 옛 요약을 그대로 둔다")
    return {k: result[k] for k in SUMMARY_KEYS if k in result}


def apply_updates(
    conn: Any,
    asset: dict[str, Any],
    updates: dict[str, Any],
    cfg: Any,
    *,
    about: list[str],
    validate_fn: Callable[..., Any],
    topic_fn: Callable[..., Any],
    persist_about_fn: Callable[..., Any],
    embedding: Any | None = None,
) -> None:
    """새 요약 · 주제 · about 을 저장한다(**DB 쓰기** · 한 트랜잭션 안에서 부른다 — 하나라도 실패하면 자산이 그대로).

    순서가 뜻이 있다: 주제 판정(조회 + LLM)을 **먼저** 하고 쓰기는 맨 뒤에 몰아 둔다. PostgreSQL 은 읽기에 행
    잠금을 잡지 않으므로, 이렇게 하면 LLM 을 기다리는 동안 이 자산의 행이 잠겨 있지 않다(적재 등 다른 쓰기를
    막지 않는다). about 은 호출부가 트랜잭션 밖에서 미리 뽑아 넘긴다.

    한계: 주제 판정은 조회 · LLM · 저장을 한 함수에서 하므로, LLM 을 기다리는 동안 이 트랜잭션의 **연결 하나**가
    열려 있다(행 잠금은 없음). 동시 처리 수는 DB 연결 풀 크기 이하로 둔다.

    Args:
        conn: DB 연결(호출부 트랜잭션).
        asset: ``fetch_targets`` 의 한 행.
        updates: ``build_updates`` 결과.
        cfg: 설정(주제 판정의 제목 확인에 쓴다).
        about: 새 요약에서 미리 뽑은 about 목록(``extract_about`` · 빈 목록 가능).
        validate_fn: 확장 메타 값 검사(``validate_ext_meta``) — **요약 결과 키만** 검사한다. 확장 메타 전체를 넘기면
            쓰기 경로 밖에서 붙는 키(about)가 「미등록 키」로 걸린다.
        topic_fn: 주제 판정 · 저장(``classify_asset_topic`` — 저장은 함수 끝에서 한 번).
        persist_about_fn: about 저장(``persist_about``).
        embedding: (사진) 새 요약으로 다시 만든 텍스트 임베딩 항목(``image_text_embedding``). ``None`` 이면 임베딩은
            건드리지 않는다(영상 · 음성 · 문서).
    """
    validate_fn(conn, asset["domain_label"], updates)
    labels = (asset.get("ext_meta") or {}).get("labels")
    self_text = build_self_text(updates.get("summary"), updates.get("keywords"), labels)
    title = confirmed_title(updates, asset["fs_path"], cfg)
    topic_fn(conn, asset["asset_id"], self_text=self_text, settings=cfg, client=None,
             **({"title": title} if title else {}))
    # 이번에 제목만 요약이 아니면 옛 표시(title_only)를 지운다 — 남겨 두면 새 요약이 제목에서 나왔다고 거짓말한다.
    remove = [] if updates.get("title_only") else ["title_only"]
    conn.execute(
        "UPDATE asset_metadata SET ext_meta = (ext_meta || %s::jsonb) - %s::text[] WHERE asset_id = %s",
        (json.dumps(updates, ensure_ascii=False), remove, asset["asset_id"]),
    )
    persist_about_fn(conn, asset["asset_id"], about)
    if embedding is not None:
        conn.execute(
            f"INSERT INTO asset_embedding (asset_id, channel, chunk_index, embedding, model_name, model_version) "
            f"VALUES (%s, %s, %s, %s::vector({FIX_EMBEDDING_DIMENSION}), %s, %s) "
            "ON CONFLICT (asset_id, channel, chunk_index) DO UPDATE SET embedding = EXCLUDED.embedding, "
            "model_name = EXCLUDED.model_name, model_version = EXCLUDED.model_version",
            (asset["asset_id"], embedding.channel, embedding.chunk_index, embedding.vector, embedding.model_name,
             embedding.model_version),
        )


def process_asset(
    asset: dict[str, Any],
    cfg: Any,
    *,
    db: Any,
    dry_run: bool,
    extract_about_fn: Callable[..., list[str]],
    validate_fn: Callable[..., Any],
    topic_fn: Callable[..., Any],
    persist_about_fn: Callable[..., Any],
    image_embed_fn: Callable[[dict[str, Any], Any], Any] | None = None,
    index_fn: Callable[[Any, str], Any] | None = None,
    write_line: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """자산 하나를 다시 만든다 — 요약 · about(LLM) → 저장(한 트랜잭션) → 색인(따로).

    색인을 저장과 다른 트랜잭션에 두는 이유: 색인은 검색 엔진 쓰기라 DB 와 한 덩어리로 되돌릴 수 없다. 색인만
    실패하면 DB 는 새 값인 채 ``index_error`` 로 돌려준다 — 이어 하기는 이 자산을 「끝남」으로 보므로, 호출부가
    따로 모아 ``--index-only`` 로 다시 색인한다.

    Args:
        asset: ``fetch_targets`` 의 한 행.
        cfg: 설정.
        db: ``transaction()`` 을 주는 DB 유틸.
        dry_run: 참이면 새 요약만 만들고 DB · 색인은 쓰지 않는다(about 도 뽑지 않는다).
        extract_about_fn: about 추출(``extract_about`` — LLM).
        validate_fn: ``apply_updates`` 참고.
        topic_fn: ``apply_updates`` 참고.
        persist_about_fn: ``apply_updates`` 참고.
        image_embed_fn: 사진 텍스트 임베딩을 만드는 함수(``image_text_embedding(meta, cfg)``). 사진을 기록할 때 필수
            — 없으면 요약만 바뀌고 벡터는 옛 요약을 가리키게 되므로 멈춘다.
        index_fn: ``(conn, asset_id)`` 를 받아 색인을 갱신하는 함수. ``None`` 이면 색인하지 않는다.
        write_line: 자산별 전후 기록을 남기는 함수(``None`` 이면 남기지 않는다).

    Returns:
        ``{"updates": build_updates 결과, "index_error": 색인 실패 사유 또는 None}``.

    Raises:
        SkipAsset · TitleHintOffError: ``build_updates`` 참고.
        RuntimeError: 사진을 기록하려는데 ``image_embed_fn`` 이 없을 때.
    """
    up = build_updates(asset, cfg)
    index_error: str | None = None
    if not dry_run:
        embedding = None
        if asset["modality"] == "image":
            if image_embed_fn is None:
                raise RuntimeError("사진은 임베딩 재계산 함수(image_embed_fn)가 있어야 기록한다")
            labels = (asset.get("ext_meta") or {}).get("labels")
            embedding = image_embed_fn({"summary": up.get("summary"), "keywords": up.get("keywords"), "labels": labels}, cfg)
        about = extract_about_fn(up.get("summary"), client=None)
        with db.transaction() as conn:
            apply_updates(conn, asset, up, cfg, about=about, validate_fn=validate_fn, topic_fn=topic_fn,
                          persist_about_fn=persist_about_fn, embedding=embedding)
        if index_fn is not None:
            try:
                with db.transaction() as conn:
                    index_fn(conn, asset["asset_id"])
            except Exception as exc:  # noqa: BLE001 — 색인 실패는 따로 모아 다시 색인한다(DB 는 이미 새 값)
                index_error = repr(exc)[:300]
                _LOG.warning("재요약 색인 실패 %s: %s", asset["asset_id"], type(exc).__name__)
    if write_line is not None:
        old = asset.get("ext_meta") or {}
        write_line({"asset_id": asset["asset_id"], "modality": asset["modality"],
                    "file": display_file_name(asset["fs_path"]),
                    "old": {"summary": old.get("summary"), "keywords": old.get("keywords")}, "new": up,
                    "index_error": index_error})
    return {"updates": up, "index_error": index_error}


def run_resummarize(
    targets: Sequence[dict[str, Any]],
    *,
    process_fn: Callable[[dict[str, Any]], dict[str, Any]],
    workers: int = 4,
) -> dict[str, Any]:
    """대상마다 ``process_fn`` 을 돌리고 결과를 모은다 — 한 자산의 실패가 나머지를 멈추지 않는다.

    Args:
        targets: ``fetch_targets`` 결과.
        process_fn: 자산 하나를 처리하고 ``process_asset`` 모양(``{"updates", "index_error"}``)을 돌려주는 함수
            (실패는 예외로 · 건너뜀은 ``SkipAsset``).
        workers: 동시 처리 수(LLM 서버 부하에 맞춘다 · DB 연결 풀 크기 이하 — 자산마다 주제 판정 동안 연결 하나를 쓴다).

    Returns:
        ``{targets, ok, skipped, failed, index_failed, title_match:{yes,no,unsure}, title_only, by_modality}`` —
        ``skipped`` · ``failed`` · ``index_failed`` 는 ``[{asset_id, error}]``(asset_id 순 — 실행마다 같은 보고).
    """
    lock = threading.Lock()
    tm: Counter[str] = Counter()
    by_mod: Counter[str] = Counter()
    report: dict[str, Any] = {"targets": len(targets), "ok": 0, "skipped": [], "failed": [], "index_failed": [],
                              "title_only": 0}

    def one(t: dict[str, Any]) -> None:
        """자산 하나를 처리하고 결과를 보고에 더한다(예외를 밖으로 내지 않는다).

        Args:
            t: 대상 자산 한 행.
        """
        try:
            res = process_fn(t)
        except SkipAsset as exc:
            with lock:
                report["skipped"].append({"asset_id": t["asset_id"], "error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001 — 자산 단위 격리(배치 전체를 멈추지 않는다)
            with lock:
                report["failed"].append({"asset_id": t["asset_id"], "error": repr(exc)[:300]})
            return
        up = res.get("updates") or {}
        with lock:
            report["ok"] += 1
            by_mod[t.get("modality", "")] += 1
            if up.get("title_match"):
                tm[up["title_match"]] += 1
            report["title_only"] += bool(up.get("title_only"))
            if res.get("index_error"):
                report["index_failed"].append({"asset_id": t["asset_id"], "error": res["index_error"]})

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        list(ex.map(one, targets))
    for key in ("skipped", "failed", "index_failed"):
        report[key].sort(key=lambda f: f["asset_id"])
    report["title_match"] = dict(tm)
    report["by_modality"] = dict(by_mod)
    return report


def _build_parser() -> argparse.ArgumentParser:
    """명령행 옵션을 정의한다."""
    p = argparse.ArgumentParser(description="제목 참고 요약 재처리 배치(spec 114 · DB 쓰기 — 백업 뒤에)")
    p.add_argument("--env", choices=["dev", "prod"], default="dev")
    p.add_argument("--modality", default="video,audio,text,image", help="쉼표 구분(video · audio · text · image)")
    p.add_argument("--exclude-names-file", default=None, help="뺄 표시 파일명 목록(한 줄에 하나 — 예: 국가유산 문서)")
    p.add_argument("--asset-ids-file", default=None, help="이 자산 id 들만(한 줄에 하나)")
    p.add_argument("--limit", type=int, default=None,
                   help="조회 상한(제외 이름 · 제목 없는 자산을 거르기 전 행 수 — 실제 대상은 더 적을 수 있다)")
    p.add_argument("--workers", type=int, default=4, help="동시 처리 수")
    p.add_argument("--no-resume", dest="resume", action="store_false", default=True,
                   help="이미 재처리된 자산도 다시(기본은 건너뜀)")
    p.add_argument("--dry-run", action="store_true", help="새 요약만 만들어 기록 파일에 남기고 DB · 색인은 쓰지 않는다")
    p.add_argument("--no-index", dest="index", action="store_false", default=True, help="검색 색인 갱신을 건너뛴다")
    p.add_argument("--index-only", action="store_true",
                   help="요약은 다시 만들지 않고 색인만 갱신한다(--asset-ids-file 필수 — 색인만 실패한 자산용)")
    p.add_argument("--out", default="resummarize.jsonl", help="자산별 전후 기록(JSONL)")
    return p


def _read_lines(path: str | None) -> list[str] | None:
    """한 줄에 하나씩 적힌 목록 파일을 읽는다(빈 줄 제외). 경로가 없으면 ``None``.

    Args:
        path: 파일 경로.
    """
    if not path:
        return None
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def invalid_ids(ids: Sequence[str] | None) -> list[str]:
    """UUID 모양이 아닌 자산 id 를 찾는다(순수) — 하나라도 섞이면 조회 SQL 전체가 실패하므로 미리 거른다.

    Args:
        ids: 자산 id 목록(``None`` 이면 빈 목록).
    """
    bad = []
    for x in ids or ():
        try:
            uuid.UUID(x)
        except ValueError:
            bad.append(x)
    return bad


def main() -> int:
    """재처리 배치 실행(IO). 0=모두 성공 · 1=일부 자산 실패 또는 색인 실패(목록을 확인한다) · 2=시작 조건 불충족."""
    args = _build_parser().parse_args()
    from processing.classify.aboutness import extract_about, persist_about
    from processing.classify.asset_topic import classify_asset_topic
    from processing.skills.image_skill import image_text_embedding
    from src.config.bootstrap import bootstrap_env
    from src.config.settings import get_current_settings
    from src.database.postgres_util import PostgresUtil
    from src.registry.ext_meta_field_registry import validate_ext_meta

    bootstrap_env(args.env)
    cfg = get_current_settings()
    if not hint_enabled(cfg):
        print("❌ SUMMARY_TITLE_HINT=1 로 실행하세요 — 꺼진 채로는 지금과 같은 요약을 다시 만들 뿐입니다.",
              file=sys.stderr)
        return 2
    ids = _read_lines(args.asset_ids_file)
    bad = invalid_ids(ids)
    if bad:
        print(f"❌ 자산 id 모양이 아니다({len(bad)}건): {bad[:5]}", file=sys.stderr)
        return 2
    if args.index_only and not ids:
        print("❌ --index-only 는 --asset-ids-file 과 함께 쓴다(색인만 실패한 자산 목록).", file=sys.stderr)
        return 2
    db = PostgresUtil()
    with db.transaction() as conn:
        missing = missing_title_keys(conn)
    if missing:
        print(f"❌ ext_meta 키가 등록되지 않았다: {missing} — 코어 마이그레이션 v308 을 먼저 적용하세요.",
              file=sys.stderr)
        return 2
    modalities = [m.strip() for m in args.modality.split(",") if m.strip()]
    exclude = set(_read_lines(args.exclude_names_file) or ())
    with db.transaction() as conn:
        # 색인만 다시 할 때는 이미 끝난 자산이 대상이다 — 이어 하기 · 제목 거르기를 끈다.
        targets = fetch_targets(conn, modalities, exclude_names=exclude, asset_ids=ids,
                                resume=args.resume and not args.index_only, titled_only=not args.index_only,
                                limit=args.limit)
    index_fn = None
    if (args.index or args.index_only) and not args.dry_run:
        from src.search.opensearch_sync import get_client, index_asset, resolve_channel
        client, index, channel = get_client(), cfg.opensearch.index, resolve_channel(None)

        def index_fn(conn: Any, asset_id: str) -> None:
            """한 자산의 검색 문서를 DB 의 현재 값으로 다시 쓴다.

            Args:
                conn: DB 연결.
                asset_id: 대상 자산.
            """
            index_asset(client, conn, asset_id, index=index, channel=channel)
    mode = "색인만" if args.index_only else ("dry-run" if args.dry_run else "재요약")
    print(f"[재요약] {mode} · 대상 {len(targets)} · 종류 {modalities} · 제외 이름 {len(exclude)} · "
          f"색인={'갱신' if index_fn else '안 함'} · 동시 {args.workers}")

    out_lock = threading.Lock()
    with open(args.out, "a", encoding="utf-8") as out:
        def write_line(line: dict[str, Any]) -> None:
            """자산별 전후 기록 한 줄을 남긴다(여러 스레드가 함께 쓴다).

            Args:
                line: 기록할 내용.
            """
            with out_lock:
                out.write(json.dumps(line, ensure_ascii=False) + "\n")
                out.flush()

        if args.index_only:
            def process(t: dict[str, Any]) -> dict[str, Any]:
                """색인만 다시 한다.

                Args:
                    t: 대상 자산 한 행.
                """
                if index_fn is None:
                    raise SkipAsset("dry-run — 색인하지 않는다")
                with db.transaction() as conn:
                    index_fn(conn, t["asset_id"])
                return {"updates": {}, "index_error": None}
        else:
            def process(t: dict[str, Any]) -> dict[str, Any]:
                """자산 하나를 다시 만든다(``process_asset``).

                Args:
                    t: 대상 자산 한 행.
                """
                return process_asset(t, cfg, db=db, dry_run=args.dry_run, extract_about_fn=extract_about,
                                     validate_fn=validate_ext_meta, topic_fn=classify_asset_topic,
                                     persist_about_fn=persist_about, image_embed_fn=image_text_embedding,
                                     index_fn=index_fn, write_line=write_line)

        report = run_resummarize(targets, process_fn=process, workers=args.workers)

    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}, ensure_ascii=False))
    for f in report["failed"][:20]:
        print(f"  실패 {f['asset_id']}: {f['error']}")
    if report["index_failed"]:
        path = f"{args.out}.index_failed.txt"
        Path(path).write_text("".join(f"{f['asset_id']}\n" for f in report["index_failed"]), encoding="utf-8")
        print(f"  색인 실패 {len(report['index_failed'])}건 — DB 는 새 값이다. 다시 색인: "
              f"SUMMARY_TITLE_HINT=1 python -m processing.app.run_resummarize --env {args.env} --index-only "
              f"--modality {args.modality} --asset-ids-file {path}")
    if not args.dry_run and not args.index_only:
        print(f"다음: 개체 재판정 — python -m processing.app.run_mm_meta_binding --env {args.env} --rejudge <자산 id …> "
              f"(기록 파일 {args.out} 의 asset_id — 성공한 자산만 기록된다 · 실패 · 건너뜀은 빠져 있다)")
    return 1 if report["failed"] or report["index_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
