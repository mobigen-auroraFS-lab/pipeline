"""065 자산 자기주제(aboutness) 정본화 — 분류 코어 단위 테스트 (mock, DB·LLM 불필요).

검증 의도 (FR-101/102·FR-201~204·068 G1)
    자산 스스로의 (topic, subtopic) 정본을 하이브리드(레지스트리 전체-27 후보 → LLM 닫힌 확정 → 058
    canonicalize)로 부여하는 seam. DB/LLM 없이 mock conn·mock client 로 분기·SQL 형상·결정성만 검증한다.
    - **결정성(헌법 3조)**: temp=0 + 닫힌 topic 후보 + 멱등 upsert → 같은 입력 같은 출력.
    - **LLM 단일 seam(헌법 6조)**: ``src.llm.client.complete_json``·``client=`` 주입.
    - **닫힌집합 검증(FR-203)**: LLM 이 후보 밖 topic 을 답하면 1회 재질의 후 실패 시 미부여(강제 매핑 금지).

mock 패턴은 ``tests/test_topic_canonicalize.py``(cursor mock·_mock_conn)·``tests/test_asset_topic_consumers.py`` 동형.
classify_asset_topic 은 헬퍼(topic_candidates_for_self_text·canonicalize_subtopic·_lookup_topic_en)를
**asset_topic 모듈 위치에서** patch 해 분기만 순수 검증한다.
"""
from __future__ import annotations

import os
import re
import unittest
from unittest.mock import MagicMock, patch

_MOD = "processing.classify.asset_topic"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))




import json  # noqa: E402  (테스트 헬퍼용 — 상단 import 블록 아래 배치)

_FIXTURE_PATH = os.path.join(
    _REPO_ROOT, "tests", "fixtures", "topics", "same_topic_groups_contract.json"
)
# 계약 스냅샷은 **실 코퍼스 asset_id** 를 담아 이 레포(공개)에 두지 않는다 — 비공개 문서 레포 소유.
_HAS_FIXTURE = os.path.isfile(_FIXTURE_PATH)
_NO_FIXTURE = f"계약 fixture 없음(비공개 문서 레포 소유): {_FIXTURE_PATH}"


def _mock_conn_seq(fetchone_val=None, fetchall_val=None):
    """``conn.cursor(...)`` 컨텍스트매니저 mock — fetchone/fetchall 을 각각 통제.

    ``__enter__`` 가 cur 를 돌려주고 fetchone/fetchall 이 주입값을 반환한다. 같은 cur 를 여러 query 가
    공유하므로, 서로 다른 접근자(fetchone vs fetchall)를 쓰는 2단 쿼리를 한 mock 으로 검증할 수 있다.
    """
    conn = MagicMock()
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.fetchone.return_value = fetchone_val
    cur.fetchall.return_value = fetchall_val if fetchall_val is not None else []
    conn.cursor.return_value = cur
    return conn, cur


def _client_once(content: str):
    """complete_json 이 호출하는 client.chat.completions.create 를 흉내(고정 응답·재호출도 동일)."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))]
    )
    return client


def _client_seq(*contents: str):
    """호출마다 다른 응답(재질의 시나리오용) — side_effect 순차 반환."""
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=c))]) for c in contents
    ]
    return client


class TestBuildSelfText(unittest.TestCase):
    """T201 — 자기 텍스트 결정적 구성(summary → keywords → labels)·빈/None 안전."""

    def test_deterministic_order_summary_keywords_labels(self) -> None:
        from processing.classify.asset_topic import build_self_text

        # 겹치지 않는 토큰으로 순서만 검증(부분문자열 충돌 방지).
        out = build_self_text(
            "요약본문",
            ["키워드하나", "키워드둘"],
            [{"label": "sport", "score": 0.9}, {"label": "ball", "score": 0.8}],
        )
        # summary 가 맨 앞, 그다음 keywords, 그다음 labels 순서(재실행 동일).
        self.assertTrue(out.startswith("요약본문"))
        self.assertLess(out.index("요약본문"), out.index("키워드하나"))
        self.assertLess(out.index("키워드하나"), out.index("sport"))
        self.assertLess(out.index("sport"), out.index("ball"))

    def test_labels_dict_takes_label_only(self) -> None:
        from processing.classify.asset_topic import build_self_text

        out = build_self_text(None, None, [{"label": "cat", "score": 0.5}])
        self.assertEqual(out, "cat")  # score 는 제외, label 만

    def test_all_empty_returns_empty_string(self) -> None:
        from processing.classify.asset_topic import build_self_text

        self.assertEqual(build_self_text(None, None, None), "")
        self.assertEqual(build_self_text("", [], []), "")
        self.assertEqual(build_self_text("  ", ["  ", None], [{}]), "")

    def test_none_keywords_and_labels_safe(self) -> None:
        from processing.classify.asset_topic import build_self_text

        self.assertEqual(build_self_text("요약만", None), "요약만")


class TestTopicCandidatesForSelfText(unittest.TestCase):
    """T101(068 G1) — topic 후보를 kNN 축소→레지스트리 전체-27 로 교체.

    닫힌 27집합에서 kNN top-k 축소가 정답 topic 을 후보에서 누락시켜(양궁·구형컴퓨터) LLM none →
    미부여·경계흡수를 낳았다(068 이슈1). 후보는 27개로 작아 전부 프롬프트에 담을 수 있으므로 임베딩 kNN
    대신 레지스트리 조회(parent NULL·taxonomy·미분류 배제·결정적 정렬)로 전체를 제공한다.
    """

    def test_empty_text_returns_empty_no_query(self) -> None:
        from processing.classify import asset_topic

        conn = MagicMock()
        out = asset_topic.topic_candidates_for_self_text(conn, "")
        self.assertEqual(out, [])
        conn.cursor.assert_not_called()  # 빈 텍스트면 DB 조회 자체를 건너뜀(비용 0)

    def test_returns_full_registry_topics_deterministic_order(self) -> None:
        from processing.classify import asset_topic

        # 레지스트리 전체-27(여기선 대표 행)을 topic_ko 오름차순으로 반환한다고 가정.
        rows = [{"topic_ko": "과학·기술"}, {"topic_ko": "스포츠·레저"}, {"topic_ko": "예술·문화"}]
        conn, cur = _mock_conn_seq(fetchall_val=rows)
        out = asset_topic.topic_candidates_for_self_text(conn, "농구 경기", k=5)
        self.assertEqual(out, ["과학·기술", "스포츠·레저", "예술·문화"])
        # SQL 형상: 닫힌 대분류 전체(parent NULL·taxonomy)·결정적 정렬. 임베딩 kNN(<=>)·LIMIT 아님.
        sql = " ".join(cur.execute.call_args[0][0].split()).lower()
        self.assertIn("from topic_registry", sql)
        self.assertIn("parent_topic is null", sql)
        self.assertIn("source = 'taxonomy'", sql)
        self.assertIn("order by topic_ko", sql)
        self.assertNotIn("<=>", sql)   # 임베딩 kNN 이 아님(레지스트리 직접 조회)
        self.assertNotIn("limit", sql)  # 전체 반환(top-k 축소 없음)

    def test_k_param_ignored_returns_full(self) -> None:
        from processing.classify import asset_topic

        # k 는 하위호환 위해 시그니처만 유지·무시(전체 반환) — top-k 축소 개념 자체가 없다.
        rows = [{"topic_ko": "A"}, {"topic_ko": "B"}, {"topic_ko": "C"}]
        conn, _ = _mock_conn_seq(fetchall_val=rows)
        out = asset_topic.topic_candidates_for_self_text(conn, "텍스트", k=1)
        self.assertEqual(out, ["A", "B", "C"])  # k=1 이어도 전체 반환

    def test_empty_registry_returns_empty(self) -> None:
        from processing.classify import asset_topic

        conn, _ = _mock_conn_seq(fetchall_val=[])  # 레지스트리 미시드 가드
        self.assertEqual(
            asset_topic.topic_candidates_for_self_text(conn, "미시드"), []
        )


_TOPIC_JSON = (
    '{"topic_ko": "스포츠·레저", "topic_en": "sports_leisure", '
    '"subtopic_ko": "농구", "subtopic_en": "basketball", "confidence": 0.9}'
)


class TestClassifyAssetTopic(unittest.TestCase):
    """T203/T302 — 하이브리드 판정: topic 후보 검증·재질의·미부여·결정성 + 068 G4 닫힌 subtopic 선택 배선."""

    def _patched(
        self,
        candidates,
        *,
        subcands=("농구",),
        picked_sub="농구",
        suben="basketball",
        en_return="sports_leisure",
    ):
        """068 G4 배선을 asset_topic 위치에서 patch 하는 컨텍스트 묶음.

        topic 후보(topic_candidates_for_self_text)·닫힌 subtopic 조회(fetch_closed_subtopics)·subtopic
        LLM 선택(_pick_subtopic_via_llm)·subtopic 정본 en(_lookup_subtopic_en)·topic 정본 en
        (_lookup_topic_en)을 함께 patch 해 분기만 순수 검증한다(058 canonicalize_subtopic 은 배선에서 제거됨).
        """
        from processing.classify import asset_topic

        p_knn = patch.object(
            asset_topic, "topic_candidates_for_self_text", return_value=candidates
        )
        p_subcands = patch.object(
            asset_topic, "fetch_closed_subtopics", return_value=list(subcands)
        )
        p_picksub = patch.object(
            asset_topic, "_pick_subtopic_via_llm", return_value=picked_sub
        )
        p_suben = patch.object(asset_topic, "_lookup_subtopic_en", return_value=suben)
        p_en = patch.object(asset_topic, "_lookup_topic_en", return_value=en_return)
        return p_knn, p_subcands, p_picksub, p_suben, p_en

    def test_candidate_topic_returns_dict(self) -> None:
        from processing.classify import asset_topic

        p_knn, p_subcands, p_picksub, p_suben, p_en = self._patched(["스포츠·레저", "예술"])
        client = _client_once(_TOPIC_JSON)
        with p_knn, p_subcands, p_picksub, p_suben, p_en:
            out = asset_topic.classify_asset_topic(
                MagicMock(), "A1", self_text="농구 경기 영상", client=client
            )
        self.assertEqual(out["topic_ko"], "스포츠·레저")
        self.assertEqual(out["topic_en"], "sports_leisure")
        self.assertEqual(out["subtopic_ko"], "농구")
        self.assertEqual(out["subtopic_en"], "basketball")  # 정본(registry) 조회 결과
        self.assertEqual(out["confidence"], 0.9)
        self.assertEqual(out["decided_by"], "hybrid")
        # 후보 내 topic → topic 재질의 없음(topic LLM 1회). subtopic 선택은 patch 로 LLM 미호출.
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_out_of_candidate_then_requery_success(self) -> None:
        from processing.classify import asset_topic

        p_knn, p_subcands, p_picksub, p_suben, p_en = self._patched(["스포츠·레저"])
        # 1차: 후보 밖('정치') → 2차 재질의: 후보 내('스포츠·레저').
        bad = '{"topic_ko": "정치", "subtopic_ko": "선거", "confidence": 0.7}'
        client = _client_seq(bad, _TOPIC_JSON)
        with p_knn, p_subcands, p_picksub, p_suben, p_en:
            out = asset_topic.classify_asset_topic(
                MagicMock(), "A2", self_text="농구", client=client
            )
        self.assertEqual(out["topic_ko"], "스포츠·레저")
        self.assertEqual(client.chat.completions.create.call_count, 2)  # 정확히 1회 재질의

    def test_two_failures_return_none(self) -> None:
        from processing.classify import asset_topic

        p_knn, p_subcands, p_picksub, p_suben, p_en = self._patched(["스포츠·레저"])
        bad = '{"topic_ko": "정치", "confidence": 0.7}'
        client = _client_seq(bad, bad)
        with p_knn, p_subcands, p_picksub, p_suben, p_en:
            out = asset_topic.classify_asset_topic(
                MagicMock(), "A3", self_text="농구", client=client
            )
            self.assertIsNone(out)
            # topic 미확정 → subtopic 조회·선택 자체를 하지 않는다(강제 매핑 금지).
            asset_topic.fetch_closed_subtopics.assert_not_called()
            asset_topic._pick_subtopic_via_llm.assert_not_called()
        self.assertEqual(client.chat.completions.create.call_count, 2)  # 2회 실패 후 중단

    def test_no_text_returns_none_no_llm(self) -> None:
        from processing.classify import asset_topic

        client = _client_once(_TOPIC_JSON)
        with patch.object(asset_topic, "topic_candidates_for_self_text") as m_knn:
            out = asset_topic.classify_asset_topic(
                MagicMock(), "A4", self_text="", client=client
            )
        self.assertIsNone(out)
        client.chat.completions.create.assert_not_called()  # 텍스트 없음 → LLM 미호출
        m_knn.assert_not_called()

    def test_empty_candidates_returns_none_no_llm(self) -> None:
        from processing.classify import asset_topic

        p_knn, p_subcands, p_picksub, p_suben, p_en = self._patched([])  # 레지스트리 미시드 가드
        client = _client_once(_TOPIC_JSON)
        with p_knn, p_subcands, p_picksub, p_suben, p_en:
            out = asset_topic.classify_asset_topic(
                MagicMock(), "A5", self_text="농구", client=client
            )
        self.assertIsNone(out)
        client.chat.completions.create.assert_not_called()

    def test_same_input_deterministic_output(self) -> None:
        from processing.classify import asset_topic

        p_knn, p_subcands, p_picksub, p_suben, p_en = self._patched(["스포츠·레저"])
        with p_knn, p_subcands, p_picksub, p_suben, p_en:
            out1 = asset_topic.classify_asset_topic(
                MagicMock(), "A6", self_text="농구", client=_client_once(_TOPIC_JSON)
            )
            out2 = asset_topic.classify_asset_topic(
                MagicMock(), "A6", self_text="농구", client=_client_once(_TOPIC_JSON)
            )
        self.assertEqual(out1, out2)  # temp=0·닫힌 후보 → 결정적

    def test_closed_subtopic_selection_wired(self) -> None:
        """068 G4/T302 — subtopic 은 부모 topic 의 닫힌 시드에서 LLM 이 선택(canonicalize 우회)."""
        from processing.classify import asset_topic

        p_knn = patch.object(
            asset_topic, "topic_candidates_for_self_text", return_value=["스포츠·레저"]
        )
        p_en = patch.object(asset_topic, "_lookup_topic_en", return_value="sports_leisure")
        client = _client_once(_TOPIC_JSON)
        conn = MagicMock()
        with p_knn, p_en, patch.object(
            asset_topic, "fetch_closed_subtopics", return_value=["농구", "축구"]
        ) as m_subcands, patch.object(
            asset_topic, "_pick_subtopic_via_llm", return_value="농구"
        ) as m_picksub, patch.object(
            asset_topic, "_lookup_subtopic_en", return_value="basketball"
        ) as m_suben:
            out = asset_topic.classify_asset_topic(conn, "A7", self_text="농구", client=client)
        # 닫힌 subtopic 조회는 (conn, 확정 topic_ko) 로 호출.
        m_subcands.assert_called_once_with(conn, "스포츠·레저")
        # LLM 선택은 (self_text, 확정 topic_ko, 닫힌 후보목록, client=client).
        args, kwargs = m_picksub.call_args
        self.assertEqual(args[0], "농구")  # self_text
        self.assertEqual(args[1], "스포츠·레저")  # 확정 topic
        self.assertEqual(args[2], ["농구", "축구"])  # 닫힌 후보(부모 스코프 시드)
        self.assertIs(kwargs.get("client"), client)
        # subtopic_en 은 registry 정본(부모 스코프) 조회로 얻는다 — LLM subtopic_en 아님.
        m_suben.assert_called_once_with(conn, "스포츠·레저", "농구")
        self.assertEqual(out["subtopic_ko"], "농구")
        self.assertEqual(out["subtopic_en"], "basketball")
        # 058 canonicalize_subtopic 은 classify 경로에서 더 이상 존재하지 않는다(import 제거·관계 경로 무영향).
        self.assertFalse(hasattr(asset_topic, "canonicalize_subtopic"))

    def test_unseeded_parent_yields_none_subtopic(self) -> None:
        """068 G4/T302 — 부모 subtopic 미시드(빈 후보)면 subtopic 미부여·LLM 선택/en 조회 안 함."""
        from processing.classify import asset_topic

        p_knn = patch.object(
            asset_topic, "topic_candidates_for_self_text", return_value=["스포츠·레저"]
        )
        p_en = patch.object(asset_topic, "_lookup_topic_en", return_value="sports_leisure")
        client = _client_once(_TOPIC_JSON)
        conn = MagicMock()
        with p_knn, p_en, patch.object(
            asset_topic, "fetch_closed_subtopics", return_value=[]
        ), patch.object(
            asset_topic, "_pick_subtopic_via_llm"
        ) as m_picksub, patch.object(
            asset_topic, "_lookup_subtopic_en"
        ) as m_suben:
            out = asset_topic.classify_asset_topic(conn, "U1", self_text="농구", client=client)
        # 시드 미존재 → subtopic 미부여(강제 생성 금지)·선택/en 조회 스킵.
        self.assertIsNone(out["subtopic_ko"])
        self.assertIsNone(out["subtopic_en"])
        m_picksub.assert_not_called()
        m_suben.assert_not_called()

    def test_upsert_sql_shape_on_conflict(self) -> None:
        from processing.classify import asset_topic

        p_knn, p_subcands, p_picksub, p_suben, p_en = self._patched(["스포츠·레저"])
        conn, cur = _mock_conn_seq()
        with p_knn, p_subcands, p_picksub, p_suben, p_en:
            asset_topic.classify_asset_topic(
                conn, "A8", self_text="농구", client=_client_once(_TOPIC_JSON)
            )
        sql = " ".join(cur.execute.call_args[0][0].split()).lower()
        self.assertIn("insert into asset_topic", sql)
        self.assertIn("on conflict (asset_id) do update", sql)
        self.assertIn("updated_at = now()", sql)
        # policy_version 이 파라미터로 기록되는지(FR-601).
        params = cur.execute.call_args[0][1]
        self.assertIn(asset_topic.POLICY_VERSION, params)


class TestUnclassifiedExclusion(unittest.TestCase):
    """T602 — 미분류 자기주제 배제(065-한정·FR-702·SC-07): 후보 필터 + none 도피처 → 미부여.

    taxonomy_seed.json·058 은 불변(관계 공유). 065 분류 경로에서만 '미분류'를 배제한다:
      ① ``topic_candidates_for_self_text`` 의 레지스트리 전체-27 조회가 SQL 단계에서 '미분류' 를 배제.
      ② 프롬프트에 "어느 후보도 안 맞으면 none" 도피처 + none/후보밖(미분류 포함) → None(미부여).
    강제 최근접 배정 금지(FR-203 계승): 억지로 27개 중 하나로 매핑하지 않는다.
    """

    def test_candidates_exclude_unclassified(self) -> None:
        from processing.classify import asset_topic

        # 레지스트리 전체-27 조회가 SQL/파라미터 단계에서 '미분류'(catch-all)를 배제한다(seed·058 불변).
        conn, cur = _mock_conn_seq(
            fetchall_val=[{"topic_ko": "스포츠·레저"}, {"topic_ko": "예술·문화"}]
        )
        out = asset_topic.topic_candidates_for_self_text(conn, "농구 경기", k=5)
        self.assertNotIn("미분류", out)
        # 배제가 조회 단계에서 이뤄지는지 — SQL 텍스트 또는 바인딩 파라미터에 '미분류' 가 등장.
        call = cur.execute.call_args[0]
        sql = " ".join(call[0].split())
        params = tuple(call[1]) if len(call) > 1 else ()
        self.assertTrue("미분류" in sql or "미분류" in params)

    def test_prompt_has_none_escape_hatch(self) -> None:
        from processing.classify import asset_topic

        # 프롬프트에 "none" 도피처 지시(어느 후보도 안 맞으면 none)가 있어야 억지 배정을 막는다.
        self.assertIn("none", asset_topic._CLASSIFY_PROMPT)
        # 재질의(후보 밖 응답) 시에도 none 도피처를 유지해 강제 최근접을 유발하지 않아야 한다.
        self.assertIn("none", asset_topic._RETRY_SUFFIX)

    def test_llm_none_response_returns_none_unassigned(self) -> None:
        from processing.classify import asset_topic

        p_knn = patch.object(
            asset_topic, "topic_candidates_for_self_text", return_value=["스포츠·레저"]
        )
        # LLM 이 "none"(어느 것도 안 맞음)으로 답 → 재질의 후에도 none → 미부여(강제 매핑 금지).
        client = _client_once('{"topic_ko": "none", "confidence": 0.1}')
        with p_knn, patch.object(asset_topic, "fetch_closed_subtopics") as m_subcands:
            out = asset_topic.classify_asset_topic(
                MagicMock(), "N1", self_text="정체불명 콘텐츠", client=client
            )
        self.assertIsNone(out)
        m_subcands.assert_not_called()  # 미부여 → 닫힌 subtopic 조회·선택·upsert 없음

    def test_llm_unclassified_response_treated_out_of_candidate(self) -> None:
        from processing.classify import asset_topic

        # 후보에서 '미분류' 가 제거됐으므로 LLM 이 '미분류' 로 답하면 후보 밖 → 재질의 → 미부여.
        p_knn = patch.object(
            asset_topic, "topic_candidates_for_self_text", return_value=["스포츠·레저"]
        )
        client = _client_once('{"topic_ko": "미분류", "confidence": 0.2}')
        with p_knn, patch.object(asset_topic, "fetch_closed_subtopics") as m_subcands:
            out = asset_topic.classify_asset_topic(
                MagicMock(), "N2", self_text="애매한 내용", client=client
            )
        self.assertIsNone(out)
        m_subcands.assert_not_called()


class TestSubtopicCoarsening(unittest.TestCase):
    """T603/T302 — 소분류 과코스닝 대응: 068 G4 는 subtopic 을 부모의 **닫힌 시드 선택**으로 바꾼다.

    종전 065 는 subtopic 을 058 ``canonicalize_subtopic`` 의 재사용-우선 경로(열린 어휘 생성 + 부모 스코프
    재사용)로 정했는데, 이게 과병합·과코스닝(여행>관광지 64%)을 낳았다. 068 은 topic 처럼 부모의 닫힌
    시드 목록에서 LLM 이 고르게 해 변별력을 회복한다 — topic 콜(_CLASSIFY_PROMPT)이 곁들여 만드는
    raw subtopic 은 무시하고, ``fetch_closed_subtopics`` 후보에서 ``_pick_subtopic_via_llm`` 이 고른 값을 쓴다.
    (_CLASSIFY_PROMPT 의 subtopic 생성 지시 제거는 후속으로 이연 — G4 는 배선만 교체.)
    """

    def test_prompt_still_generates_subtopic_but_ignored(self) -> None:
        from processing.classify import asset_topic

        # G4 는 _CLASSIFY_PROMPT(topic 콜)를 손대지 않는다(subtopic 생성 지시 제거는 후속 이연).
        # topic 콜은 여전히 subtopic_ko 를 함께 생성하나 classify 는 그 값을 subtopic 결정에 쓰지 않는다.
        prompt = asset_topic._CLASSIFY_PROMPT
        self.assertIn("subtopic_ko", prompt)  # topic 콜의 JSON 형식에 subtopic 잔존(무해·무시)
        self.assertIn("고유명사", prompt)  # 잔존 지시(후속 제거 대상)

    def test_raw_subtopic_from_topic_call_ignored_uses_closed_selection(self) -> None:
        """068 G4 — topic 콜 raw subtopic("도시여행")을 무시하고 닫힌 시드 선택 결과를 쓴다."""
        from processing.classify import asset_topic

        p_knn = patch.object(
            asset_topic, "topic_candidates_for_self_text", return_value=["여행·지역"]
        )
        p_en = patch.object(asset_topic, "_lookup_topic_en", return_value="travel")
        # topic 콜이 subtopic_ko="도시여행" 을 곁들여 생성 → 무시되어야 한다.
        client = _client_once(
            '{"topic_ko": "여행·지역", "topic_en": "travel", '
            '"subtopic_ko": "도시여행", "subtopic_en": "city_travel", "confidence": 0.8}'
        )
        conn = MagicMock()
        with p_knn, p_en, patch.object(
            asset_topic, "fetch_closed_subtopics", return_value=["국내여행·지역탐방", "해외여행"]
        ) as m_subcands, patch.object(
            asset_topic, "_pick_subtopic_via_llm", return_value="국내여행·지역탐방"
        ) as m_picksub, patch.object(
            asset_topic, "_lookup_subtopic_en", return_value="domestic_travel"
        ):
            out = asset_topic.classify_asset_topic(conn, "C1", self_text="파리 여행", client=client)
        # raw subtopic("도시여행") 이 아니라 닫힌 시드 선택 결과를 쓴다.
        self.assertEqual(out["subtopic_ko"], "국내여행·지역탐방")
        self.assertEqual(out["subtopic_en"], "domestic_travel")
        m_subcands.assert_called_once_with(conn, "여행·지역")
        # 닫힌 후보(부모 스코프 시드)가 그대로 선택 함수에 전달된다.
        self.assertEqual(m_picksub.call_args[0][2], ["국내여행·지역탐방", "해외여행"])


@unittest.skipUnless(_HAS_FIXTURE, _NO_FIXTURE)
class TestFetchAssetTopic(unittest.TestCase):
    """T204 — 정본 읽기(구 project_asset_topics 형상)·부재 []."""

    def _fixture_keys(self):
        with open(_FIXTURE_PATH, encoding="utf-8") as fh:
            fx = json.load(fh)
        return set(fx["project_asset_topics_shape"][0].keys())

    def test_row_present_returns_weight_one_shape(self) -> None:
        from src.topic.asset_topic_query import fetch_asset_topic

        row = {
            "topic_ko": "스포츠·레저",
            "subtopic_ko": "농구",
            "topic_en": "sports_leisure",
            "subtopic_en": "basketball",
        }
        conn, _ = _mock_conn_seq(fetchone_val=row)
        out = fetch_asset_topic(conn, "A1")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["weight"], 1)
        # 필드명 수준 fixture(구 투영 형상)와 동일해야 소비처 무변경 스왑 가능.
        self.assertEqual(set(out[0].keys()), self._fixture_keys())
        self.assertEqual(out[0]["topic_ko"], "스포츠·레저")

    def test_absent_returns_empty(self) -> None:
        from src.topic.asset_topic_query import fetch_asset_topic

        conn, _ = _mock_conn_seq(fetchone_val=None)
        self.assertEqual(fetch_asset_topic(conn, "missing"), [])


@unittest.skipUnless(_HAS_FIXTURE, _NO_FIXTURE)
class TestFindSameTopicGroups(unittest.TestCase):
    """T204 — 같은 (topic,subtopic) 자산 집계(구 find_topic_neighbor_groups 형상)."""

    def _fixture_group_keys(self):
        with open(_FIXTURE_PATH, encoding="utf-8") as fh:
            fx = json.load(fh)
        g = fx["find_topic_neighbor_groups_shape"][0]
        sub = g["subtopics"][0]
        asset = sub["assets"][0]
        return set(g.keys()), set(sub.keys()), set(asset.keys())

    def test_no_target_topic_returns_empty(self) -> None:
        from src.topic.asset_topic_query import find_same_topic_groups

        conn, _ = _mock_conn_seq(fetchone_val=None)  # 대상 자산 asset_topic 행 없음
        self.assertEqual(find_same_topic_groups(conn, "A"), [])

    def test_pair_match_shape_equals_contract(self) -> None:
        from src.topic.asset_topic_query import find_same_topic_groups

        target = {"topic_ko": "스포츠·레저", "subtopic_ko": "농구"}
        cand_rows = [
            {"asset_id": "019f-1", "topic_ko": "스포츠·레저", "subtopic_ko": "농구",
             "fs_path": "/d/019f-1__wikipedia_농구_5166.txt", "modality": "text",
             "already_linked": True},
            {"asset_id": "019f-2", "topic_ko": "스포츠·레저", "subtopic_ko": "농구",
             "fs_path": "/d/019f-2__Kim_Tae-sul_(농구).JPG", "modality": "image",
             "already_linked": False},
        ]
        conn, cur = _mock_conn_seq(fetchone_val=target, fetchall_val=cand_rows)
        out = find_same_topic_groups(conn, "TARGET")

        self.assertEqual(len(out), 1)
        gk, sk, ak = self._fixture_group_keys()
        self.assertEqual(set(out[0].keys()), gk)
        self.assertEqual(out[0]["topic_ko"], "스포츠·레저")
        self.assertEqual(out[0]["asset_count"], 2)
        sub = out[0]["subtopics"][0]
        self.assertEqual(set(sub.keys()), sk)
        self.assertEqual(sub["subtopic_ko"], "농구")
        self.assertEqual(sub["asset_count"], 2)
        asset0 = sub["assets"][0]
        self.assertEqual(set(asset0.keys()), ak)
        # file_name 은 fs_path basename(관례)·asset_id 는 str.
        self.assertTrue(asset0["file_name"].endswith(".txt") or asset0["file_name"].endswith(".JPG"))
        self.assertIsInstance(asset0["asset_id"], str)
        # subtopic 있는 대상 → 후보 쿼리에 subtopic 필터가 걸려야(같은 쌍 매칭).
        cand_sql = " ".join(cur.execute.call_args_list[1][0][0].split()).lower()
        self.assertIn("subtopic_ko", cand_sql)

    def test_topic_only_match_when_target_subtopic_none(self) -> None:
        from src.topic.asset_topic_query import find_same_topic_groups

        target = {"topic_ko": "음식·요리", "subtopic_ko": None}
        cand_rows = [
            {"asset_id": "b1", "topic_ko": "음식·요리", "subtopic_ko": "제빵",
             "fs_path": "/d/b1__bread.jpg", "modality": "image", "already_linked": False},
            {"asset_id": "b2", "topic_ko": "음식·요리", "subtopic_ko": None,
             "fs_path": "/d/b2__food.txt", "modality": "text", "already_linked": True},
        ]
        conn, cur = _mock_conn_seq(fetchone_val=target, fetchall_val=cand_rows)
        out = find_same_topic_groups(conn, "TARGET")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["topic_ko"], "음식·요리")
        # 두 하위주제 버킷(제빵·None) 모두 등장 — topic 단독 매칭.
        subs = {s["subtopic_ko"] for s in out[0]["subtopics"]}
        self.assertEqual(subs, {"제빵", None})
        # 대상 subtopic 이 None → 후보 쿼리에 subtopic 필터가 없어야(topic 단독).
        cand_sql = " ".join(cur.execute.call_args_list[1][0][0].split())
        self.assertNotIn("at.subtopic_ko = %s", cand_sql)


# ── 068 G4: 닫힌 subtopic 조회 + LLM 선택 (T301) ──────────────────────────────
# subtopic 도 topic 처럼 부모 topic 의 **닫힌 시드 목록**에서 LLM 이 고른다(058 열린 어휘 canonicalize
# 폐기·과병합/과코스닝 차단). 아래는 조회·선택·정본 en 조회 3개 신설 함수의 단위 검증(mock·DB/LLM 불요).
_SUB_JSON_OK = '{"subtopic_ko": "국내여행·지역탐방"}'


class TestFetchClosedSubtopics(unittest.TestCase):
    """T301(068 G4) — 부모 topic 의 닫힌 소분류(subtopic) 시드 목록 조회.

    subtopic 후보 공급 = 부모 topic 스코프의 taxonomy 시드(닫힌 소분류). 미시드면 [](→ classify 는
    subtopic 미부여). 결정적 정렬(topic_ko asc)로 프롬프트 후보 순서를 재실행마다 고정한다(헌법 3조).
    """

    def test_returns_closed_subtopics_deterministic_order(self) -> None:
        from processing.classify import asset_topic

        rows = [{"topic_ko": "국내여행·지역탐방"}, {"topic_ko": "해외여행"}]
        conn, cur = _mock_conn_seq(fetchall_val=rows)
        out = asset_topic.fetch_closed_subtopics(conn, "여행·지역")
        self.assertEqual(out, ["국내여행·지역탐방", "해외여행"])
        # SQL 형상: 부모 스코프(parent_topic=%s)·taxonomy 시드·결정적 정렬. 임베딩 kNN 아님.
        sql = " ".join(cur.execute.call_args[0][0].split()).lower()
        self.assertIn("from topic_registry", sql)
        self.assertIn("parent_topic = %s", sql)
        self.assertIn("source = 'taxonomy'", sql)
        self.assertIn("order by topic_ko", sql)
        self.assertNotIn("<=>", sql)  # 닫힌 시드 조회(임베딩 kNN 축소 아님)
        # 부모 topic 이 파라미터로 바인딩(부모 스코프 조회).
        self.assertEqual(cur.execute.call_args[0][1], ("여행·지역",))

    def test_unseeded_parent_returns_empty(self) -> None:
        from processing.classify import asset_topic

        conn, _ = _mock_conn_seq(fetchall_val=[])  # 부모 미시드 가드
        self.assertEqual(asset_topic.fetch_closed_subtopics(conn, "미시드"), [])


class TestPickSubtopicViaLlm(unittest.TestCase):
    """T301(068 G4) — 닫힌 subtopic 후보 중 LLM 선택(_pick_topic_via_llm 대칭).

    부모 topic 명시 + 후보 subtopic 목록 → 후보 내 정확히 1개 또는 none. 후보 밖이면 1회 재질의,
    재실패/none → None(FR-203/205·환각 차단·강제 매핑 금지). temp=0·단일 seam(complete_json).
    """

    def test_in_candidate_returns_subtopic_str(self) -> None:
        from processing.classify import asset_topic

        client = _client_once(_SUB_JSON_OK)
        out = asset_topic._pick_subtopic_via_llm(
            "부산 여행 영상", "여행·지역", ["국내여행·지역탐방", "해외여행"], client=client
        )
        self.assertEqual(out, "국내여행·지역탐방")
        self.assertEqual(client.chat.completions.create.call_count, 1)  # 후보 내 → 재질의 없음

    def test_out_of_candidate_then_requery_success(self) -> None:
        from processing.classify import asset_topic

        bad = '{"subtopic_ko": "우주여행"}'  # 후보 밖
        client = _client_seq(bad, _SUB_JSON_OK)
        out = asset_topic._pick_subtopic_via_llm(
            "부산 여행", "여행·지역", ["국내여행·지역탐방"], client=client
        )
        self.assertEqual(out, "국내여행·지역탐방")
        self.assertEqual(client.chat.completions.create.call_count, 2)  # 정확히 1회 재질의

    def test_two_failures_return_none(self) -> None:
        from processing.classify import asset_topic

        bad = '{"subtopic_ko": "우주여행"}'
        client = _client_seq(bad, bad)
        out = asset_topic._pick_subtopic_via_llm(
            "부산 여행", "여행·지역", ["국내여행·지역탐방"], client=client
        )
        self.assertIsNone(out)  # 2회 후보 밖 → 미부여(강제 매핑 금지)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_none_response_returns_none(self) -> None:
        from processing.classify import asset_topic

        # 어느 후보도 안 맞음 → LLM 이 "none" → 재질의 후에도 none 처리 → None.
        client = _client_once('{"subtopic_ko": "none"}')
        out = asset_topic._pick_subtopic_via_llm(
            "정체불명", "여행·지역", ["국내여행·지역탐방"], client=client
        )
        self.assertIsNone(out)

    def test_prompt_includes_parent_topic_and_candidates(self) -> None:
        from processing.classify import asset_topic

        client = _client_once(_SUB_JSON_OK)
        asset_topic._pick_subtopic_via_llm(
            "부산 여행", "여행·지역", ["국내여행·지역탐방", "해외여행"], client=client
        )
        # 프롬프트에 부모 topic 명시 + 후보 subtopic 제시 + none 도피처(억지 배정 금지)가 있어야 한다.
        content = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        self.assertIn("여행·지역", content)  # 부모 topic 명시
        self.assertIn("국내여행·지역탐방", content)  # 후보 subtopic
        self.assertIn("해외여행", content)
        self.assertIn("none", content)  # 닫힌집합 none 도피처


class TestLookupSubtopicEn(unittest.TestCase):
    """T301(068 G4) — 부모 스코프 subtopic 정본 영문 조회(닫힌 시드 en).

    subtopic 층은 (parent_topic, topic_ko) 부분 유니크라 부모 스코프로 조여 조회한다(동음이의 보존).
    """

    def test_present_returns_en(self) -> None:
        from processing.classify import asset_topic

        conn, cur = _mock_conn_seq(fetchone_val={"topic_en": "domestic_travel"})
        out = asset_topic._lookup_subtopic_en(conn, "여행·지역", "국내여행·지역탐방")
        self.assertEqual(out, "domestic_travel")
        sql = " ".join(cur.execute.call_args[0][0].split()).lower()
        self.assertIn("select topic_en from topic_registry", sql)
        self.assertIn("parent_topic = %s", sql)
        self.assertIn("topic_ko = %s", sql)
        # 부모 topic·subtopic 이 순서대로 바인딩(부모 스코프 정확 조회).
        self.assertEqual(cur.execute.call_args[0][1], ("여행·지역", "국내여행·지역탐방"))

    def test_absent_returns_none(self) -> None:
        from processing.classify import asset_topic

        conn, _ = _mock_conn_seq(fetchone_val=None)
        self.assertIsNone(asset_topic._lookup_subtopic_en(conn, "여행·지역", "없음"))


class TestPatchSeamExists(unittest.TestCase):
    """e2e 가 patch 하는 후보 seam 의 존재를 비-DB 로 봉인(2026-07-15 B1 재발 차단).

    068 G1 때 seam 이 knn_topic_candidates→topic_candidates_for_self_text 로 바뀌었는데 e2e 의
    patch 대상이 안 따라가 RUN_DB_E2E 에서만 AttributeError 가 나는 잠복 파손이 있었다. mock.patch
    는 진입 시 대상 속성 부재를 즉시 던지므로, 게이트 없는 이 테스트가 seam 개명을 항상 잡는다.
    """

    def test_topic_candidates_seam_patchable(self) -> None:
        with patch("processing.classify.asset_topic.topic_candidates_for_self_text", return_value=[]):
            pass  # 진입 성공 = seam 실존(반환값 사용 안 함)


if __name__ == "__main__":
    unittest.main()
