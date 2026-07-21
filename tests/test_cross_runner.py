"""제네릭 cross_asset 러너(G2) 단위 테스트.

검증 대상: ``src/pipeline/cross_runner.py`` 의 ``run_cross_asset``.

핵심(헌법 4조 — 이 작업의 본질):
    러너는 **순수 오케스트레이션**이다. 팩이 resolve 한 4슬롯 Callable
    (candidates / score / decide / persist_edges)을 ``contracts.py`` 계약 순서대로
    실행할 뿐, ``domain==`` / ``pack.name==`` 같은 도메인 코드 분기를 두지 않는다.
    따라서 러너는 어떤 도메인의 전략이 주입되든 동일하게 동작해야 한다.

테스트 전략:
    실 전략·실 DB·LLM 불필요. fake 전략 4종(스파이)을 resolved dict 로 주입해
        - 호출 순서(candidates→score→decide→persist_edges)
        - 각 단계 출력이 다음 단계 입력으로 **그대로** 전달됨
        - 반환값 = verdict=='match' 결정 수(= 적재 엣지 수)
        - persist_edges 가 decisions 를 받음
        - 빈 입력(candidates=[])에서도 0 반환·안전
    을 검증한다. conn 은 불투명 sentinel 로 두고, 러너가 conn 을 받는 슬롯에 그대로
    전달하는지(candidates/score/persist_edges) / decide 에는 전달하지 않는지(순수)도 본다.
"""
from __future__ import annotations

import unittest

from processing.pipeline.cross_types import Candidate, Decision, ScoredPair


# ── fake 전략 빌더(도메인 무관 스파이) ────────────────────────────────────────
class _Spy:
    """호출 순서·인자를 공유 로그에 기록하는 fake 슬롯 묶음.

    실제 전략 시그니처(contracts.py Protocol)를 그대로 만족하되, 입력을 다음 단계로
    넘기기 좋은 결정적 더미 출력을 만든다. 어떤 슬롯도 도메인 이름을 모른다 —
    러너의 제네릭성을 증명하기 위해 의도적으로 도메인-불가지하게 만든다.
    """

    def __init__(self, candidates: list[Candidate], match_targets: set[str]) -> None:
        # candidates 슬롯이 돌려줄 후보(러너가 그대로 score 로 넘겨야 한다).
        self._candidates_out = candidates
        # decide 가 'match' 로 판정할 타깃 집합(반환 엣지 수 검증용).
        self._match_targets = match_targets
        # (스테이지명, 받은 주요 인자)를 시간순으로 기록.
        self.calls: list[tuple[str, object]] = []
        # 단계 간 전달 검증용: 각 슬롯이 실제로 받은 입력 객체를 보관.
        self.received: dict[str, object] = {}

    def candidates(self, conn, source_asset_id):
        self.calls.append(("candidates", source_asset_id))
        self.received["candidates.conn"] = conn
        self.received["candidates.source_asset_id"] = source_asset_id
        return self._candidates_out

    def score(self, conn, pairs):
        self.calls.append(("score", pairs))
        self.received["score.conn"] = conn
        self.received["score.pairs"] = pairs
        # 후보를 ScoredPair 로 매핑(점수는 결정적 더미). 입력 보존.
        return [ScoredPair(candidate=c, score=0.9, evidence=[]) for c in pairs]

    def decide(self, scored):
        self.calls.append(("decide", scored))
        self.received["decide.scored"] = scored
        out: list[Decision] = []
        for sp in scored:
            verdict = "match" if sp.candidate.target_id in self._match_targets else "non_match"
            out.append(Decision(candidate=sp.candidate, verdict=verdict, score=sp.score))
        return out

    def persist_edges(self, conn, decisions):
        self.calls.append(("persist_edges", decisions))
        self.received["persist_edges.conn"] = conn
        self.received["persist_edges.decisions"] = decisions
        # EdgePersistStage 계약: None 반환(러너가 match 수를 직접 센다).
        return None

    def as_resolved(self) -> dict:
        return {
            "candidates": self.candidates,
            "score": self.score,
            "decide": self.decide,
            "persist_edges": self.persist_edges,
        }


def _cand(src: str, tid: str) -> Candidate:
    return Candidate(src, tid, block_key="embedding", method="fake")


class TestRunCrossAssetOrder(unittest.TestCase):
    _SRC = "018f0000-0000-7000-8000-000000000001"
    _T1 = "018f0000-0000-7000-8000-000000000007"
    _T2 = "018f0000-0000-7000-8000-000000000012"
    _CONN = object()  # 불투명 sentinel — 러너가 슬롯에 그대로 넘기는지만 본다.

    def _spy(self, match_targets=None) -> _Spy:
        cands = [_cand(self._SRC, self._T1), _cand(self._SRC, self._T2)]
        return _Spy(cands, match_targets if match_targets is not None else {self._T1})

    def test_calls_in_contract_order(self) -> None:
        # candidates→score→decide→persist_edges 순서로 정확히 한 번씩.
        from processing.pipeline.cross_runner import run_cross_asset
        spy = self._spy()
        run_cross_asset(spy.as_resolved(), self._CONN, self._SRC)
        order = [name for name, _ in spy.calls]
        self.assertEqual(order, ["candidates", "score", "decide", "persist_edges"])

    def test_candidates_output_flows_to_score(self) -> None:
        # candidates 의 반환이 score 의 pairs 인자로 그대로 들어가야 한다.
        from processing.pipeline.cross_runner import run_cross_asset
        spy = self._spy()
        run_cross_asset(spy.as_resolved(), self._CONN, self._SRC)
        self.assertIs(spy.received["score.pairs"], spy._candidates_out)

    def test_score_output_flows_to_decide(self) -> None:
        # score 의 반환이 decide 의 scored 인자로 그대로 들어가야 한다.
        from processing.pipeline.cross_runner import run_cross_asset
        spy = self._spy()
        run_cross_asset(spy.as_resolved(), self._CONN, self._SRC)
        # decide 가 받은 객체는 score 가 반환한 ScoredPair 리스트와 동일 내용.
        scored_in = spy.received["decide.scored"]
        self.assertEqual(len(scored_in), 2)
        self.assertTrue(all(isinstance(sp, ScoredPair) for sp in scored_in))
        self.assertEqual(
            {sp.candidate.target_id for sp in scored_in}, {self._T1, self._T2}
        )

    def test_decide_output_flows_to_persist(self) -> None:
        # decide 의 반환이 persist_edges 의 decisions 인자로 그대로 들어가야 한다.
        from processing.pipeline.cross_runner import run_cross_asset
        spy = self._spy(match_targets={self._T1, self._T2})
        run_cross_asset(spy.as_resolved(), self._CONN, self._SRC)
        decisions_in = spy.received["persist_edges.decisions"]
        self.assertEqual(len(decisions_in), 2)
        self.assertTrue(all(isinstance(d, Decision) for d in decisions_in))

    def test_source_asset_id_passed_to_candidates(self) -> None:
        from processing.pipeline.cross_runner import run_cross_asset
        spy = self._spy()
        run_cross_asset(spy.as_resolved(), self._CONN, self._SRC)
        self.assertEqual(spy.received["candidates.source_asset_id"], self._SRC)

    def test_conn_passed_to_conn_taking_slots(self) -> None:
        # conn 은 candidates/score/persist_edges 에 그대로 전달돼야 한다(같은 객체).
        from processing.pipeline.cross_runner import run_cross_asset
        spy = self._spy()
        run_cross_asset(spy.as_resolved(), self._CONN, self._SRC)
        self.assertIs(spy.received["candidates.conn"], self._CONN)
        self.assertIs(spy.received["score.conn"], self._CONN)
        self.assertIs(spy.received["persist_edges.conn"], self._CONN)


class TestRunCrossAssetReturn(unittest.TestCase):
    _SRC = "018f0000-0000-7000-8000-000000000001"
    _T1 = "018f0000-0000-7000-8000-000000000007"
    _T2 = "018f0000-0000-7000-8000-000000000012"
    _T3 = "018f0000-0000-7000-8000-000000000017"
    _CONN = object()

    def _spy(self, targets, match_targets) -> _Spy:
        cands = [_cand(self._SRC, t) for t in targets]
        return _Spy(cands, set(match_targets))

    def test_returns_match_count(self) -> None:
        # 반환 = verdict=='match' 결정 수(= 적재 엣지 수). 여기선 2건 match.
        from processing.pipeline.cross_runner import run_cross_asset
        spy = self._spy([self._T1, self._T2, self._T3], {self._T1, self._T2})
        n = run_cross_asset(spy.as_resolved(), self._CONN, self._SRC)
        self.assertEqual(n, 2)

    def test_returns_zero_when_no_match(self) -> None:
        from processing.pipeline.cross_runner import run_cross_asset
        spy = self._spy([self._T1, self._T2], set())
        n = run_cross_asset(spy.as_resolved(), self._CONN, self._SRC)
        self.assertEqual(n, 0)

    def test_returns_all_when_all_match(self) -> None:
        from processing.pipeline.cross_runner import run_cross_asset
        spy = self._spy([self._T1, self._T2, self._T3], {self._T1, self._T2, self._T3})
        n = run_cross_asset(spy.as_resolved(), self._CONN, self._SRC)
        self.assertEqual(n, 3)

    def test_return_is_int(self) -> None:
        from processing.pipeline.cross_runner import run_cross_asset
        spy = self._spy([self._T1], {self._T1})
        n = run_cross_asset(spy.as_resolved(), self._CONN, self._SRC)
        self.assertIsInstance(n, int)

    def test_deterministic_same_input_twice(self) -> None:
        from processing.pipeline.cross_runner import run_cross_asset
        spy1 = self._spy([self._T1, self._T2], {self._T1})
        spy2 = self._spy([self._T1, self._T2], {self._T1})
        n1 = run_cross_asset(spy1.as_resolved(), self._CONN, self._SRC)
        n2 = run_cross_asset(spy2.as_resolved(), self._CONN, self._SRC)
        self.assertEqual(n1, n2)


class TestRunCrossAssetEmpty(unittest.TestCase):
    _SRC = "018f0000-0000-7000-8000-000000000001"
    _CONN = object()

    def test_empty_candidates_returns_zero(self) -> None:
        # 빈 후보면 0 반환. 결정 0 → 적재 엣지 0.
        from processing.pipeline.cross_runner import run_cross_asset
        spy = _Spy([], set())
        n = run_cross_asset(spy.as_resolved(), self._CONN, self._SRC)
        self.assertEqual(n, 0)

    def test_empty_candidates_still_runs_full_chain(self) -> None:
        # 빈 입력이라도 체인을 일관되게 끝까지 돈다(persist 도 호출돼 빈 decisions 처리).
        # 빈 결정도 persist 가 안전하게 처리해야 한다는 계약 일관성.
        from processing.pipeline.cross_runner import run_cross_asset
        spy = _Spy([], set())
        run_cross_asset(spy.as_resolved(), self._CONN, self._SRC)
        order = [name for name, _ in spy.calls]
        self.assertEqual(order, ["candidates", "score", "decide", "persist_edges"])
        self.assertEqual(list(spy.received["persist_edges.decisions"]), [])


class TestRunCrossAssetDomainAgnostic(unittest.TestCase):
    """헌법 4조: 러너에 도메인 코드 분기가 없음을 증명한다."""

    _SRC = "018f0000-0000-7000-8000-000000000001"
    _T1 = "018f0000-0000-7000-8000-000000000007"
    _CONN = object()

    def test_works_for_arbitrary_domain_strategies(self) -> None:
        # 도메인 이름을 모르는 동일 fake 전략으로도, '다른 도메인'이라 라벨만 바꾼
        # 또 다른 fake 전략으로도 결과가 같아야 한다(러너는 전략에만 의존).
        from processing.pipeline.cross_runner import run_cross_asset
        spy_a = _Spy([_cand(self._SRC, self._T1)], {self._T1})
        spy_b = _Spy([_cand(self._SRC, self._T1)], {self._T1})
        n_a = run_cross_asset(spy_a.as_resolved(), self._CONN, self._SRC)
        n_b = run_cross_asset(spy_b.as_resolved(), self._CONN, self._SRC)
        self.assertEqual(n_a, n_b)

    def test_runner_source_has_no_domain_branch(self) -> None:
        # 정적 가드(헌법 4조): 러너 **실행 코드**에 도메인 분기가 없어야 한다.
        # docstring/주석의 설명 문구는 분기가 아니므로, 토큰을 떼어내고(문자열·주석 제거)
        # 남은 실행 소스만 검사한다 — '의도 설명'과 '실제 분기'를 구분하기 위함.
        import inspect
        import io
        import tokenize

        from processing.pipeline import cross_runner

        src = inspect.getsource(cross_runner)
        # 문자열 리터럴(docstring 포함)·주석 토큰을 제거해 실행 코드만 남긴다.
        executable_tokens = []
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                continue
            executable_tokens.append(tok.string)
        code_only = " ".join(executable_tokens)

        # 도메인 코드 분기의 흔적이 실행부에 없어야 한다.
        for forbidden in ("domain ==", "pack.name", "== 'sample'", '== "sample"',
                          "== 'general'", '== "general"', "domain in", "name =="):
            self.assertNotIn(
                forbidden, code_only,
                msg=f"러너 실행 코드에 도메인 분기 발견: {forbidden!r}",
            )
        # 보강: 러너 실행부에 'sample'/'general' 같은 도메인 리터럴 자체가 없어야 한다
        # (문자열 토큰을 이미 제거했으므로, 남아 있다면 식별자/속성명으로 쓰인 셈).
        self.assertNotIn("sample", code_only)
        self.assertNotIn("general", code_only)


if __name__ == "__main__":
    unittest.main()
