"""021 G5 / 027 — OpenSearch 검색 read path 실OS e2e (게이트 ``RUN_OS_E2E=1``).

G1·G2 가 순수/가짜클라이언트로 검증한 쿼리 빌더·**클라이언트 융합**·검색 seam 이 **실제 OpenSearch
하이브리드(nori 한국어 BM25 + ``knn_vector`` kNN)**에서 의도대로 동작하는지 확정 검증한다. 027 전환:
서버 normalization-processor 검색 파이프라인이 **클라이언트 융합**(``fuse_hybrid``: min-max+가중평균,
모달리티당 [kNN + BM25] _msearch 1회)으로 대체돼, 등록할 서버 파이프라인이 없다. 검증 지점:
    - ``client.msearch`` 로 모달리티당 [kNN, BM25] 서브검색을 1회에 받아 클라이언트가 융합하는지,
    - 융합 점수 + ``(-similarity, id)`` 결정적 tiebreaker(FR-009)가 동작하는지,
    - 버킷 게이트(``gate_signal`` robust baseline → ``passes_cutoff``)·per-result 컷(``cut_rows``)이
      실 OS 의 원시 코사인 위에서 무관 질의를 빈 버킷으로 만드는지.

**자기완결 e2e — 실 PG 불요.** 테스트가 직접 작은 알려진 한국어 코퍼스(text/audio + 의료 1건 +
동점쌍 2건)를 임시 인덱스(``assets_021e2e``)에 ``asset_to_doc``(020 순수 문서 빌더)로 만들어
색인한다. 질의·문서 임베딩은 **실제 모델**(활성 채널, 게이트 안이라 무방)로 같은 벡터 공간에서
계산해 kNN 이 비교 가능하게 구성한다. 임시 인덱스는 ``tearDownClass`` 에서 삭제하고 변경한 env 도 복원한다.

기본(``RUN_OS_E2E`` 미설정) 회귀에서는 클래스 전체가 **auto-skip** 되어 모델 로드·OS 접속·env
변경이 일어나지 않는다(헌법 8조 회귀 0). 실 측정은 사람이 OpenSearch 기동 후 ``RUN_OS_E2E=1`` 로.

────────────────────────────────────────────────────────────────────────────
실행 런북 (사람)
────────────────────────────────────────────────────────────────────────────
1) OpenSearch(+ k-NN, analysis-nori 플러그인) 기동 — docker 예시:

    docker run -d --name os021 -p 9200:9200 \
      -e discovery.type=single-node -e plugins.security.disabled=true \
      -e OPENSEARCH_INITIAL_ADMIN_PASSWORD=Aurora!2026 \
      opensearchproject/opensearch:2.13.0
    # analysis-nori(한국어 형태소)·k-NN 은 배포 이미지에 기본 번들. nori 미포함 이미지면:
    #   docker exec os021 bin/opensearch-plugin install analysis-nori && docker restart os021

   ``.env.dev`` 의 ``OPENSEARCH_URL``(미설정 시 기본 http://localhost:9200)이 이 인스턴스를 가리키게 한다.
   (DEV 무인증 http 기준 — 보안 비활성.)

2) 본 e2e 실행 — 자기색인 코퍼스라 실 PG 불요, OpenSearch 만 있으면 된다:

    RUN_OS_E2E=1 conda run -n AuroraFS python -m unittest tests.test_opensearch_search_e2e -v

3) (참고) 실데이터로 검증하고 싶으면 — 020 동기화로 운영 인덱스(``assets``)를 채운 뒤,
   본 테스트 대신 ``SEARCH_BACKEND=opensearch`` 진입점(run_search 등)으로 육안 확인:

    OPENSEARCH_SYNC_ENABLED=true python -m processing.app.run_ingest --env dev <FILE>...   # 증분 색인
    python -m processing.app.run_opensearch_resync --env dev --recreate                    # 전체 재색인(서버 파이프라인 0)
    SEARCH_BACKEND=opensearch python -m processing.app.run_search --env dev "재무 보고서" --modalities text,audio

────────────────────────────────────────────────────────────────────────────
G4/G5 — KPI·골든·사용자 평가 (사람 측정, 본 파일 밖·하드 정지점)
────────────────────────────────────────────────────────────────────────────
골든 58질의 하니스(``scripts/measure_search_golden.py``)가 gate-off recall@20·**gate-on recall(027
신규·오컷 가시화)**·no-match 차단율을 산출한다. 실 OS·실데이터 인덱스가 필요하다:

    RUN_OS_E2E=1 SEARCH_BACKEND=opensearch conda run -n AuroraFS python scripts/measure_search_golden.py

   - **SC-001(품질)**: gate-off recall@20 ≥ 0.985 유지·gate-on recall 하락폭(오컷)이 작은지 확인.
   - 분포 실측·EPS/FLOOR/RESULT_FLOOR 재보정은 ``scripts/calibrate_search_cutoff.py``(robust baseline).
   - 결과는 ADR ``docs/decisions/2026-06-11-search-fusion-simplify.md`` 측정 기록에 남긴다.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

_RUN = os.getenv("RUN_OS_E2E") == "1"
_INDEX = "assets_021e2e"
_REPO_ROOT = Path(__file__).resolve().parents[1]

# 코퍼스 자산 id(결정적·테스트 스코프). 동점쌍은 asset_id 사전순(a<b)으로 tiebreaker(FR-009) 검증.
_ID_FIN = "021e2e-text-fin"     # 재무/매출 (text)
_ID_COOK = "021e2e-text-cook"   # 요리 (text)
_ID_TRIP = "021e2e-text-trip"   # 여행 (text)
_ID_MEET = "021e2e-audio-meet"  # 회의 녹취 (audio)
_ID_MUSIC = "021e2e-audio-music"  # 음악 팟캐스트 (audio)
_ID_MED = "021e2e-text-med"     # 의료(domain_label='medical', text) — 검색에서 배제돼야 함
_ID_TWIN_A = "021e2e-twin-a"    # 동점쌍 — 동일 내용/임베딩, asset_id 만 다름
_ID_TWIN_B = "021e2e-twin-b"

# ── 022 G3: image/video OS 회수 + 작은 k 비우세 모달리티 가드용 코퍼스 ────────────────
# knn pre-filter 회귀 가드의 핵심: **다수 text(강아지) + 소수 image(반려견/고양이)**. 작은 k(=버킷
# 한도)에서 전역 top-k 가 강아지 다수 text 로 채워지므로, knn 을 bool 사후필터로 감싸면(회귀) image
# 버킷이 0건이 된다 — knn native filter(pre-filter)로 모달리티 안에서 k 최근접을 뽑아야 image 가 회수된다.
# image 캡션은 질의(강아지)와 **어휘가 겹치지 않게**(반려견/고양이) 설계해, BM25(어휘) 가 image 를
# 끌어올려 회귀를 가리지 못하게 한다 — 오직 knn pre-filter 만이 image 를 회수하는 시나리오.
_ID_PET1 = "022e2e-text-pet1"   # 강아지 다수 text(전역 top-k 점유) — 사후필터 회귀 시 image 를 밀어냄
_ID_PET2 = "022e2e-text-pet2"
_ID_PET3 = "022e2e-text-pet3"
_ID_IMG_DOG = "022e2e-image-dog"   # image(general) — 캡션 '반려견'(질의 '강아지'와 어휘 무겹침·의미 인접)
_ID_IMG_CAT = "022e2e-image-cat"   # image(general)
_ID_VID_DOG = "022e2e-video-dog"   # video(general)
_ID_IMG_MED = "022e2e-image-med"   # 의료 image(domain_label='medical') — 결과 배제돼야 함(FR-004)
_ID_VID_MED = "022e2e-video-med"   # 의료 video(domain_label='medical') — 결과 배제돼야 함(FR-004)


def _os_reachable() -> bool:
    """``RUN_OS_E2E=1`` 이고 OpenSearch 가 응답하면 True(020 e2e 게이트 패턴 동형, 게이트만 RUN_OS_E2E)."""
    if not _RUN:
        return False
    try:
        import urllib.request

        from dotenv import load_dotenv

        load_dotenv(_REPO_ROOT / ".env.dev", override=False)
        url = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
        with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 — 로컬 DEV OS
            return r.status == 200
    except Exception:
        return False


# 작은 알려진 코퍼스. 각 질의의 키워드는 **목표 문서에만** 등장하도록 토픽을 분리해, nori BM25 가
# 결정적으로 목표를 상위로 올린다(recall sanity 의 안정성). (asset_id, modality, domain_label, summary, keywords, labels)
# 022 G3: image/video 자산을 text/audio 와 동일하게 asset_to_doc 로 색인한다(저장 modality 값 'image'/
# 'video', 한국어 캡션·KoSimCSE 캡션 임베딩 — CLIP 아님). image/video 검색이 020 동일 하이브리드로
# 회수되는지(FR-001)와 작은 k 비우세 모달리티 가드(knn pre-filter 회귀)를 검증한다.
_CORPUS: tuple[tuple[str, str, str | None, str, list[str], list[str]], ...] = (
    (_ID_FIN, "text", "general", "2024년 분기 재무 보고서 매출 영업이익 분석 자료", ["재무", "매출", "영업이익"], ["보고서"]),
    (_ID_COOK, "text", "general", "김치찌개 끓이는 요리법 재료 손질 방법 정리", ["요리", "김치찌개", "레시피"], ["가이드"]),
    (_ID_TRIP, "text", "general", "제주도 여행 일정 관광 명소 추천 코스 안내", ["여행", "제주도", "관광"], ["여행기"]),
    (_ID_MEET, "audio", "general", "프로젝트 주간 회의 녹취록 일정 공유 결정사항", ["회의", "녹취", "일정"], ["회의록"]),
    (_ID_MUSIC, "audio", "general", "인디 음악 팟캐스트 인터뷰 방송 에피소드", ["음악", "팟캐스트", "방송"], ["방송"]),
    (_ID_MED, "text", "medical", "환자 진료 기록 처방 내역 병원 검사 결과 소견", ["환자", "진료", "처방"], ["의무기록"]),
    (_ID_TWIN_A, "text", "general", "오늘의 날씨 기상 예보 강수 확률 안내", ["날씨", "기상", "예보"], ["기상"]),
    (_ID_TWIN_B, "text", "general", "오늘의 날씨 기상 예보 강수 확률 안내", ["날씨", "기상", "예보"], ["기상"]),
    # 022 — 강아지 다수 text(전역 top-k 점유) : 작은 k 가드의 'text 다수' 측. 질의 '강아지 키우기 사료'에
    # 어휘·의미로 강하게 매칭돼 전역 knn 최근접을 채운다(사후필터 회귀 시 image 를 전역에서 밀어냄).
    (_ID_PET1, "text", "general", "강아지 키우기 사료 추천 정보 글 모음", ["강아지", "사료", "키우기"], ["반려"]),
    (_ID_PET2, "text", "general", "강아지 키우기 사료 추천 정보 글 모음", ["강아지", "사료", "키우기"], ["반려"]),
    (_ID_PET3, "text", "general", "강아지 키우기 사료 추천 정보 글 모음", ["강아지", "사료", "키우기"], ["반려"]),
    # 022 — 소수 image(general). 캡션은 '반려견'/'고양이'로 질의 '강아지'와 **어휘 무겹침**(BM25 미회수)
    # 이나 의미 인접 → knn pre-filter 만이 회수한다(작은 k 가드가 잡는 정확한 시나리오).
    (_ID_IMG_DOG, "image", "general", "반려견 공원 야외 풍경 잔디밭 산책", ["반려견", "공원", "풍경"], ["사진"]),
    (_ID_IMG_CAT, "image", "general", "고양이 창가 햇살 낮잠 실내 모습", ["고양이", "낮잠", "실내"], ["사진"]),
    # 022 — 소수 video(general). FR-001 영상 회수·결정성 검증용(어휘 '영상' 매칭으로 명확히 회수).
    (_ID_VID_DOG, "video", "general", "강아지 산책 훈련 영상 클립 야외 장면", ["강아지", "훈련", "영상"], ["영상"]),
    # 022 — 의료 image/video(domain_label='medical') : 색인돼 있어도 결과에서 배제돼야 한다(FR-004).
    (_ID_IMG_MED, "image", "medical", "흉부 엑스레이 방사선 판독 의료 영상", ["엑스레이", "방사선", "판독"], ["의료영상"]),
    (_ID_VID_MED, "video", "medical", "초음파 검사 동영상 의료 진단 기록", ["초음파", "진단", "검사"], ["의료영상"]),
)


@unittest.skipUnless(_os_reachable(), "RUN_OS_E2E=1 + 실 OpenSearch 필요")
class TestOpenSearchSearchE2E(unittest.TestCase):
    """실 OS 하이브리드 검색 e2e — 자기색인 코퍼스로 recall·결정성·의료배제·LLM 0·정규화 융합 검증."""

    @classmethod
    def setUpClass(cls) -> None:
        # 임시 인덱스를 settings 에 주입해 search_hybrid(backend=opensearch) 도 같은 코퍼스를 보게 한다
        # (LLM 0 테스트가 진입점 경로를 그대로 탄다). .env.dev 엔 이 키가 없으므로 override=False 로드와
        # 무관하게 우리 값이 유지된다. 변경 전 값은 복원용으로 보관. 027: 서버 파이프라인 env 는 소멸.
        cls._saved_env = {k: os.environ.get(k) for k in ("OPENSEARCH_INDEX",)}
        os.environ["OPENSEARCH_INDEX"] = _INDEX

        from dotenv import load_dotenv

        load_dotenv(_REPO_ROOT / ".env.dev", override=False)
        from src.config.settings import active_embed_channel, init_settings

        init_settings("dev")
        from src.search.opensearch_search import embed_query
        from src.search.opensearch_sync import asset_to_doc, ensure_index, get_client

        cls.client = get_client()
        cls.channel = active_embed_channel()

        # 임시 인덱스(020 매핑·nori·knn_vector)만 만든다 — 027 클라이언트 융합이라 서버 검색 파이프라인
        # 등록은 없다(융합 수학이 클라이언트 fuse_hybrid 로 이동).
        ensure_index(cls.client, _INDEX, recreate=True)

        # 코퍼스를 실제 모델로 임베딩 → asset_to_doc(순수) → 색인. 문서·질의가 같은 임베더(채널 모델)를
        # 써 같은 벡터 공간에서 kNN 비교된다(FR-004 질의-문서 일치).
        for asset_id, modality, domain, summary, keywords, labels in _CORPUS:
            text = summary + " " + " ".join(keywords)
            row = {
                "asset_id": asset_id,
                "modality": modality,
                "domain_label": domain,
                "status": "registered",
                "fs_path": f"/data/{asset_id}.txt",
                "ext_meta": {"summary": summary, "keywords": keywords, "labels": labels},
                "emb": embed_query(text, channel=cls.channel),
                "chunk_count": 1,
            }
            doc = asset_to_doc(row, cls.channel)
            cls.client.index(index=_INDEX, id=doc["asset_id"], body=doc)
        cls.client.indices.refresh(index=_INDEX)

    @classmethod
    def tearDownClass(cls) -> None:
        # 임시 인덱스 삭제(best-effort) + env 복원. 027: 삭제할 서버 파이프라인 없음.
        try:
            if cls.client.indices.exists(index=_INDEX):
                cls.client.indices.delete(index=_INDEX)
        finally:
            for k, v in cls._saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def _search(
        self, query: str, modalities: list[str], *, k: int = 10
    ) -> dict[str, list[dict]]:
        """임시 인덱스로 클라이언트 융합 검색(seam 그대로). 버킷 {modality: [rows]} 반환.

        027: search_assets_os 는 (buckets, gate_meta) 튜플을 돌려준다 — 본 헬퍼는 버킷만 쓴다(게이트
        메커니즘 검증은 아래 023 섹션이 직접 gate_signal·search_assets_os 로 다룬다). 컷오프 기본 off
        (게이트·per-result 컷 미적용)로 회수 자체를 본다 — 게이트 분기 검증은 별도 테스트에서 켠다.
        """
        from src.search.opensearch_search import search_assets_os
        from src.search.search_tuning import SearchTuning

        buckets, _meta = search_assets_os(
            self.client,
            query,
            modalities=modalities,
            k=k,
            channel=self.channel,
            index=_INDEX,
            exclude_medical=True,
            tuning=SearchTuning(cutoff_enabled=False),
        )
        return buckets

    # ── 1) recall sanity (SC-002 방향) ──────────────────────────────────────
    def test_recall_sanity_top1(self) -> None:
        """특정 한국어 질의가 의도한 text/audio 문서를 상위로 회수한다(nori BM25 + kNN)."""
        text = self._search("재무 매출 보고서", ["text"])["text"]
        self.assertTrue(text, "text 버킷에 결과가 있어야 한다")
        self.assertEqual(text[0]["id"], _ID_FIN, "재무 질의의 top1 은 재무 문서여야 한다")
        self.assertEqual(text[0]["modality"], "text")  # 053: 저장 modality = canonical 'text'

        audio = self._search("회의 녹취 일정", ["audio"])["audio"]
        self.assertTrue(audio, "audio 버킷에 결과가 있어야 한다")
        self.assertEqual(audio[0]["id"], _ID_MEET, "회의 질의의 top1 은 회의 녹취 문서여야 한다")
        self.assertEqual(audio[0]["modality"], "audio")

    # ── 2) 결정성 (헌법 3조 · FR-009 tiebreaker) ─────────────────────────────
    def test_determinism_same_query_same_order(self) -> None:
        """같은 질의 2회 → 동일 top-k 순서. 동점쌍은 asset_id 오름차순(결정적 tiebreaker)."""
        run1 = [r["id"] for r in self._search("재무 매출 보고서", ["text"])["text"]]
        run2 = [r["id"] for r in self._search("재무 매출 보고서", ["text"])["text"]]
        self.assertEqual(run1, run2, "같은 질의 2회는 동일 순서여야 한다(결정성)")

        # 동점쌍(동일 내용·동일 임베딩 → 동일 정규화 점수)은 asset_id 오름차순으로 결정(FR-009).
        ids = [r["id"] for r in self._search("날씨 기상 예보", ["text"])["text"]]
        self.assertIn(_ID_TWIN_A, ids)
        self.assertIn(_ID_TWIN_B, ids)
        self.assertLess(
            ids.index(_ID_TWIN_A), ids.index(_ID_TWIN_B),
            "동점 시 asset_id 오름차순 tiebreaker — twin-a 가 twin-b 보다 앞이어야 한다",
        )

    # ── 3) 의료 배제 (FR-011 · SC-004 · 헌법 10조) ───────────────────────────
    def test_medical_excluded_from_results(self) -> None:
        """의료(domain_label='medical') 문서는 색인돼 있어도 검색 결과에서 제외된다(쿼리 단 실효)."""
        # 의료 문서가 인덱스엔 존재함(배제가 '없어서'가 아니라 '필터'임을 증명).
        src = self.client.get(index=_INDEX, id=_ID_MED)["_source"]
        self.assertEqual(src.get("domain_label"), "medical")

        # 의료 내용에 들어맞는 질의로도 의료 자산이 어느 버킷에도 안 나와야 한다.
        buckets = self._search("환자 진료 처방", ["text", "audio"])
        found = {r["id"] for rows in buckets.values() for r in rows}
        self.assertNotIn(_ID_MED, found, "의료 자산은 검색 결과에 없어야 한다(FR-011)")

    # ── 4) LLM 0 (SC-004 · FR-004) ──────────────────────────────────────────
    def test_search_hybrid_opensearch_calls_no_llm(self) -> None:
        """search_hybrid(backend='opensearch') text/audio 경로에서 LLM 질의 구조화 0회(seam 미접촉)."""
        from src.search import search_service

        # 069 US-C: 037 로 은퇴한 structure_user_query 死코드는 삭제됐다. 질의 구조화 LLM 은
        # complete_text 단일 seam 을 쓰던 유일 경로였으므로, complete_text 0회로 "검색 시점 질의
        # 구조화 LLM 미접촉"을 봉인한다(query-norm morph/off·llm_verify off 기본).
        with mock.patch("src.llm.client.complete_text") as m_llm:
            result = search_service.search_hybrid(
                "재무 매출 보고서",
                modalities=["text", "audio"],
                backend="opensearch",
            )
        self.assertEqual(m_llm.call_count, 0, "OS text/audio 경로는 LLM 질의 구조화 0(FR-004·SC-004)")
        # 응답 동형(SC-005): 표준 버킷 키.
        self.assertIn("text_documents", result["results"])
        self.assertIn("audio", result["results"])
        self.assertEqual(result["meta"].get("backend"), "opensearch")

    # ── 5) 클라이언트 융합 동작 (FR-002 — fuse_hybrid 실적용 최종 검증) ───────────
    def test_client_fusion_score_bounded(self) -> None:
        """클라이언트 융합(fuse_hybrid: min-max + 가중평균)이 실제 적용된다 — 결합 점수 ≤ 1.0 상한이 증거.

        min-max 정규화는 서브검색별 최고점을 1.0 으로 스케일하고, 가중평균(가중치 합 1.0)으로 결합하므로
        결합 similarity 의 이론 상한은 1.0(양 서브검색 동시 1등). 원시 BM25 라면 보통 1.0 을 초과한다 —
        이 상한이 ``fuse_hybrid`` 가 실제 적용됐다는 신호다(027: 서버 파이프라인이 아니라 클라이언트 융합).
        """
        rows = self._search("재무 매출 보고서", ["text"])["text"]
        self.assertTrue(rows, "결과가 있어야 융합 점수를 검증할 수 있다")
        top = rows[0]["similarity"]
        self.assertGreater(top, 0.0, "융합 점수는 양수여야 한다")
        self.assertLessEqual(top, 1.0 + 1e-6, "min-max + 가중평균 결합 점수 상한 1.0(fuse_hybrid 실적용)")
        # 점수 단조 비증가(정렬 일관성 — similarity desc).
        sims = [r["similarity"] for r in rows]
        self.assertEqual(sims, sorted(sims, reverse=True), "결과는 점수 내림차순이어야 한다")

    # ── 022 G3) image/video OS 회수 (FR-001 · 캡션 하이브리드) ────────────────
    def test_image_video_os_retrieval(self) -> None:
        """FR-001: image·video 버킷이 020 과 **동일 하이브리드**(캡션 nori + KoSimCSE kNN)로 회수된다(≥1).

        image/video 도 text/audio 처럼 ``asset_to_doc`` 로 색인(저장 modality 'image'/'video', 한국어
        캡션·임베딩)돼 ``search_assets_os`` 가 같은 경로로 검색한다 — CLIP 아님. knn pre-filter 라
        비우세 모달리티도 자기 모달리티의 k 최근접을 회수한다(사후필터 회귀 시 0건 → 실패).
        """
        buckets = self._search("반려동물 산책 사진 영상", ["image", "video"], k=5)
        self.assertTrue(buckets["image"], "image 버킷에 결과가 있어야 한다(FR-001)")
        self.assertTrue(buckets["video"], "video 버킷에 결과가 있어야 한다(FR-001)")
        # 저장 modality 값이 라벨과 동일('image'/'video')로 회수된다.
        self.assertTrue(all(r["modality"] == "image" for r in buckets["image"]))
        self.assertTrue(all(r["modality"] == "video" for r in buckets["video"]))
        # 의료 image/video 는 어느 버킷에도 없어야 한다(FR-004).
        found = {r["id"] for rows in buckets.values() for r in rows}
        self.assertNotIn(_ID_IMG_MED, found, "의료 image 는 결과에 없어야 한다(FR-004)")
        self.assertNotIn(_ID_VID_MED, found, "의료 video 는 결과에 없어야 한다(FR-004)")

    # ── 022 G3) 작은 k 비우세 모달리티 가드 (핵심 회귀 가드 — knn pre-filter) ────
    def test_small_k_nondominant_image_bucket_not_empty(self) -> None:
        """핵심 회귀 가드: 다수 text + 소수 image 코퍼스에서 작은 k(=버킷 한도)로도 image 버킷이 0건이 아니다.

        knn 을 **native filter(pre-filter)** 로 모달리티 안에서 k 최근접을 뽑으므로, 전역 top-k 가
        다수 강아지 text 로 채워져도 image 버킷은 자기 모달리티의 k 최근접(반려견/고양이)을 회수한다.
        (사후필터 회귀 시: knn 이 전역 k 최근접을 먼저 뽑고 modality 로 거르므로 — 강아지 다수 text 가
        전역 top-k 를 점유 → image 0건으로 **실패**한다. image 캡션은 질의 '강아지'와 어휘가 겹치지
        않아 BM25 가 image 를 끌어올려 회귀를 가리지도 못한다 → 오직 knn pre-filter 만이 image 를 회수.)
        """
        query = "강아지 키우기 사료 추천 정보"

        # 전제 증명: 같은 질의로 text 버킷을 보면 강아지 다수 text 가 상위를 점유한다(전역 top-k 를 text
        # 가 채운다는 회귀 시나리오의 전제 — 사후필터였다면 이 다수 text 가 image 의 전역 자리를 밀어냄).
        text_ids = [r["id"] for r in self._search(query, ["text"], k=3)["text"]]
        self.assertTrue(
            set(text_ids) & {_ID_PET1, _ID_PET2, _ID_PET3},
            "강아지 다수 text 가 상위로 회수돼야 한다(가드의 전제: text 다수 점유)",
        )

        # 핵심 단언: k=2(작은 버킷 한도)에서도 image 버킷이 0건이 아니다(knn pre-filter 가드).
        image = self._search(query, ["image"], k=2)["image"]
        self.assertTrue(
            image,
            "작은 k 에서도 비우세 모달리티(image) 버킷은 0건이 아니어야 한다 — knn pre-filter 가드"
            "(사후필터 회귀 시 전역 top-k 를 다수 text 가 점유해 image 0건이 되어 실패)",
        )
        for r in image:
            self.assertEqual(r["modality"], "image")
            self.assertNotIn(r["id"], {_ID_IMG_MED}, "의료 image 는 배제돼야 한다(FR-004)")

    # ── 022 G3) image/video 결정성 (헌법 3조 · SC-004) ──────────────────────
    def test_determinism_image_video(self) -> None:
        """같은 질의 2회 → image·video 버킷도 동일 top-k 순서(결정성·FR-006 tiebreaker)."""
        query = "반려동물 산책 사진 영상"
        run1 = self._search(query, ["image", "video"], k=5)
        run2 = self._search(query, ["image", "video"], k=5)
        self.assertEqual(
            [r["id"] for r in run1["image"]], [r["id"] for r in run2["image"]],
            "image 버킷은 2회 동일 순서여야 한다(결정성)",
        )
        self.assertEqual(
            [r["id"] for r in run1["video"]], [r["id"] for r in run2["video"]],
            "video 버킷은 2회 동일 순서여야 한다(결정성)",
        )

    # ── 022 G3) 의료 image/video 배제 (FR-004 · 헌법 10조) ───────────────────
    def test_medical_image_video_excluded(self) -> None:
        """의료(domain_label='medical') image/video 는 색인돼 있어도 검색 결과에서 제외된다(쿼리 단 실효)."""
        # 의료 image/video 가 인덱스엔 존재함(배제가 '없어서'가 아니라 '필터'임을 증명).
        for asset_id in (_ID_IMG_MED, _ID_VID_MED):
            src = self.client.get(index=_INDEX, id=asset_id)["_source"]
            self.assertEqual(src.get("domain_label"), "medical")

        # 의료 영상/사진에 들어맞는 질의로도 의료 자산이 어느 버킷에도 안 나와야 한다.
        buckets = self._search("엑스레이 초음파 의료 진단 검사", ["image", "video"], k=10)
        found = {r["id"] for rows in buckets.values() for r in rows}
        self.assertNotIn(_ID_IMG_MED, found, "의료 image 는 검색 결과에 없어야 한다(FR-004)")
        self.assertNotIn(_ID_VID_MED, found, "의료 video 는 검색 결과에 없어야 한다(FR-004)")

    # ── 022 G3) LLM 0 — image/video 경로 (SC-004 · FR-002) ──────────────────
    def test_search_hybrid_opensearch_image_video_no_llm(self) -> None:
        """search_hybrid(backend='opensearch') image/video 경로도 LLM 질의 구조화 0회(seam 미접촉)."""
        from src.search import search_service

        # 069 US-C: structure_user_query 死코드 삭제 후 complete_text 0회로 봉인(OS 경로는 미호출).
        with mock.patch("src.llm.client.complete_text") as m_llm:
            result = search_service.search_hybrid(
                "반려동물 산책 사진 영상",
                modalities=["image", "video"],
                backend="opensearch",
            )
        self.assertEqual(m_llm.call_count, 0, "OS image/video 경로는 LLM 질의 구조화 0(FR-002·SC-004)")
        # 응답 동형(SC-005): backend 무관 같은 버킷 키.
        self.assertIn("image", result["results"])
        self.assertIn("video", result["results"])
        self.assertEqual(result["meta"].get("backend"), "opensearch")

    # ── 023/027) 버킷 게이트 — 실 kNN 신호·empty/keep·결정성·의료배제 ──────────────
    # 주의: production 기본 임계(FLOOR·EPS)는 실 318자산 코퍼스로 보정된 값이라, 이 작은 자체색인
    # 코퍼스에는 그대로 전이하지 않는다(절대 코사인 스케일·baseline 이 다름). 따라서 e2e 는 **production
    # 임계를 재검증하지 않고**(그건 scripts/calibrate_search_cutoff.py 가 실코퍼스로 수행) 실 OS 경로의
    # 게이트 *메커니즘*(build_knn_body+client.search 의 실 kNN 코사인 → gate_signal → passes_cutoff →
    # 빈/유지)을 코퍼스에서 측정한 신호로 결정적으로 검증한다. 027: 별도 추가 검색(probe) 없이 같은 kNN
    # 본문으로 코사인을 모은다. 즉 매칭>무관 불변식과, 두 top 사이 floor 로 empty/keep 분기를 단언한다.

    def _knn_signal(
        self, query: str, label: str, *, exclude_medical: bool = True, k: int = 10
    ) -> tuple[float, float]:
        """실 kNN(build_knn_body+client.search)의 코사인에서 (top, robust baseline)을 잰다(임시 인덱스).

        023 probe 를 대체한다 — 융합이 클라이언트로 오면서 kNN 응답에 원시 코사인이 보존되므로 추가
        검색 없이 같은 본문에서 신호를 잰다. knn_score_to_cosine 으로 환산 후 gate_signal 로 신호화한다.
        """
        from src.search.opensearch_search import (
            _MODALITY_VALUES,
            build_knn_body,
            embed_query,
            gate_signal,
            knn_score_to_cosine,
        )

        vec = embed_query(query, channel=self.channel)
        values = _MODALITY_VALUES.get(label, frozenset({label}))
        body = build_knn_body(vec, modality_values=values, k=k, exclude_medical=exclude_medical)
        resp = self.client.search(index=_INDEX, body=body)
        hits = ((resp or {}).get("hits") or {}).get("hits") or []
        cosines = [knn_score_to_cosine(h.get("_score")) for h in hits]
        return gate_signal(cosines)

    def test_cutoff_signal_matching_above_foreign(self) -> None:
        """실 kNN: 코퍼스에 있는 토픽(강아지) top > 무관 토픽(양자역학) top — 매칭>무관 불변식(FR-003)."""
        present_top, _ = self._knn_signal("강아지 키우기 사료", "text")
        foreign_top, _ = self._knn_signal("양자역학 입자 가속기 실험", "text")
        self.assertGreater(
            present_top, foreign_top,
            f"매칭 질의 top({present_top:.3f})이 무관 질의 top({foreign_top:.3f})보다 커야 한다",
        )

    def test_cutoff_empties_foreign_keeps_present(self) -> None:
        """게이트 메커니즘(FR-001·003): 두 top 사이 floor 로 무관 질의는 빈 버킷·매칭 질의는 유지.

        eps=0.0 으로 상대신호를 끄고 floor 만으로 분기 — 실 kNN 이 잰 두 top 의 중앙을 floor 로 써
        production 기본값 의존 없이 메커니즘만 본다. result_floor=-1.0 으로 per-result 컷은 끄고(게이트
        분기만 격리). cutoff off 면 둘 다 비지 않음(게이트가 원인임을 격리).
        """
        from src.search.opensearch_search import search_assets_os
        from src.search.search_tuning import SearchTuning

        present_top, _ = self._knn_signal("강아지 키우기 사료", "text")
        foreign_top, _ = self._knn_signal("양자역학 입자 가속기 실험", "text")
        floor = (present_top + foreign_top) / 2.0  # 두 top 사이 → 매칭 유지·무관 차단

        def buckets(query: str, enabled: bool) -> dict:
            # bm25_operator='and': production(.env) 동일 — 'or'면 부분 토큰 어휘가 구제 규칙을 타서
            # 게이트 메커니즘 검증이 흐려진다(027 어휘 구제는 전 토큰 증거 기준).
            out, _meta = search_assets_os(
                self.client, query, modalities=["text"], k=10, channel=self.channel,
                index=_INDEX,
                tuning=SearchTuning(cutoff_enabled=enabled, cutoff_eps=0.0, cutoff_floor=floor,
                                    result_floor=-1.0, bm25_operator="and"),
            )
            return out

        # cutoff OFF: 무관 질의도 융합 랭킹으로 결과가 나온다(게이트 없음 → no-match 노이즈).
        self.assertTrue(buckets("양자역학 입자 가속기 실험", False)["text"], "게이트 off 면 무관 질의도 결과")
        # cutoff ON: 무관 질의 → 빈 버킷, 매칭 질의 → 유지.
        self.assertEqual(buckets("양자역학 입자 가속기 실험", True)["text"], [], "무관 질의는 빈 버킷")
        self.assertTrue(buckets("강아지 키우기 사료", True)["text"], "매칭 질의는 유지(과필터 0)")

    def test_cutoff_deterministic(self) -> None:
        """결정성(헌법 3조·SC-004): 같은 질의·게이트 2회 → 동일 버킷(빈/유지)."""
        from src.search.opensearch_search import search_assets_os
        from src.search.search_tuning import SearchTuning

        foreign_top, _ = self._knn_signal("양자역학 입자 가속기 실험", "text")
        present_top, _ = self._knn_signal("강아지 키우기 사료", "text")
        floor = (present_top + foreign_top) / 2.0

        def run(query: str) -> list:
            out, _meta = search_assets_os(
                self.client, query, modalities=["text"], k=10, channel=self.channel,
                index=_INDEX,
                tuning=SearchTuning(cutoff_enabled=True, cutoff_eps=0.0, cutoff_floor=floor,
                                    result_floor=-1.0, bm25_operator="and"),
            )
            return [r["id"] for r in out["text"]]

        self.assertEqual(run("양자역학 입자 가속기 실험"), run("양자역학 입자 가속기 실험"))
        self.assertEqual(run("강아지 키우기 사료"), run("강아지 키우기 사료"))

    def test_cutoff_signal_excludes_medical(self) -> None:
        """게이트 신호 의료배제(FR-008·헌법 10조): 의료 토픽만 매칭하는 질의는 kNN 후보에서 의료가 빠져 top↓.

        '환자 진료 처방' 은 코퍼스에서 의료 문서(_ID_MED)에만 매칭한다 — exclude_medical=True(기본)면
        그 후보가 빠져, 포함했을 때보다 top 코사인이 낮아야 한다(같은 후보 집단에서 의료 격리)."""
        top_excl, _ = self._knn_signal("환자 진료 처방 병원 검사", "text", exclude_medical=True)
        top_incl, _ = self._knn_signal("환자 진료 처방 병원 검사", "text", exclude_medical=False)
        self.assertLess(
            top_excl, top_incl,
            f"의료배제 top({top_excl:.3f})은 의료포함 top({top_incl:.3f})보다 낮아야 한다(의료 격리)",
        )


if __name__ == "__main__":
    unittest.main()
