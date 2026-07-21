"""검색 평가 지표(이진 적합도, 순수 함수). ranked=id 순위 리스트, relevant=정답 id 집합."""
from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = sum(1 for x in ranked[:k] if x in relevant)
    return hit / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hit = sum(1 for x in ranked[:k] if x in relevant)
    return hit / k


def mrr(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, x in enumerate(ranked, start=1):
        if x in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    # 표준 이진 nDCG@k. IDCG는 '전체 정답 수'(min(len(relevant), k)) 기준이라
    # top-k 안에 못 들어온 정답도 분모에 반영된다 → 저조한 recall 을 패널티(관례적 정의).
    # 결정적 산술만 사용(헌법 3조).
    dcg = sum(1.0 / math.log2(i + 1) for i, x in enumerate(ranked[:k], start=1) if x in relevant)
    ideal_n = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_n + 1))
    return dcg / idcg if idcg else 0.0


# ── 무관(non-relevant) 노이즈 지표(019 G3) ──────────────────────────────────────
# 청크 집계 MAX 는 긴 무관 영상(많은 청크 중 운 좋은 한 청크)을 상위로 끌어올린다(spec 019 §무엇·왜).
# recall/MRR/nDCG 는 "정답이 잘 올라왔나"만 보므로, "무관 자산이 위로 오는 정도"를 직접 재려고
# 아래 두 보조 지표를 둔다. 둘 다 ranked=id 순위·relevant=정답 집합의 순수 함수(헌법 3조 결정적).


def nonrelevant_mean_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """정답이 아닌 자산의 평균 1-기반 순위. **높을수록** 무관 자산이 아래로 밀린 것(좋음).

    수식: mean({ i : ranked[i-1] ∉ relevant }), i 는 1-기반 순위. 경계 처리 —
      - 빈 랭킹 또는 전부 정답(무관 항목 없음): ``0.0``(측정 대상 없음).
      - 정답 0개: 전 항목이 무관 → 평균 = (n+1)/2.
    """
    ranks = [i for i, x in enumerate(ranked, start=1) if x not in relevant]
    if not ranks:
        return 0.0
    return sum(ranks) / len(ranks)


def nonrelevant_exposure_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """상위 k 무관 노출률 = (상위 k 중 정답 아님) / min(k, len(ranked)). **낮을수록** 좋음.

    분모를 ``min(k, len(ranked))`` 로 둬 "실제로 상위 k 에 노출된 항목" 기준의 비율을 낸다
    (랭킹이 k 보다 짧을 때 빈 슬롯을 무관으로 과대 계상하지 않음). 경계 처리 —
      - ``k <= 0`` 또는 빈 랭킹: ``0.0``(노출 슬롯 없음, 0 나눗셈 방지).
      - 정답 0개: 상위 k 전부 무관 → ``1.0``. 전부 정답: ``0.0``.
    """
    if k <= 0:
        return 0.0
    top = ranked[:k]
    if not top:
        return 0.0
    noise = sum(1 for x in top if x not in relevant)
    return noise / len(top)
