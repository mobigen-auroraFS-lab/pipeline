"""cross-asset 파이프라인 스테이지 간 타입(후보→점수→결정→증거).

배선 현황(두 표현이 공존)
    - ``Candidate``/``ScoredPair``/``Decision`` 은 contracts.py cross_asset Protocol 의 입출력
      타입이며, 샘플 도메인(spec 016)이 ``cross_runner`` + ``sample_strategies`` 로 처음 실제
      배선·실행한다. (단계 A에서 정의만 했고 016 에서 활성화됨.)
    - 일반·의료의 legacy relations 경로(``run_relations`` propose)는 아직 이 dataclass 가
      아니라 dict(``src/relations`` 의 ``EmbeddingCandidate`` TypedDict 등)를 쓴다.
    - ``Evidence`` 의 ``m_prob``/``u_prob``/``weight``(=log(m/u))는 레코드 링크
      (Fellegi-Sunter) 스코어링 필드로 아직 미배선(의료 ER, 단계 D=3년차 이연·2026-07-06 대기). 샘플 ``sample_score``
      는 ``similarity``/``weight`` 만 채워 부착하고 m/u 확률은 사용하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    source_id: str
    target_id: str
    block_key: str = ""        # 어떤 블로킹 키로 잡혔나(추적). embedding_topk 면 'embedding'.
    method: str = ""           # 'blocking' | 'embedding_topk'


@dataclass(frozen=True)
class Evidence:
    field: str                 # 비교 필드명
    comparator: str            # exact | jaro_winkler | embedding_cosine | llm_zs ...
    similarity: float | None = None
    m_prob: float | None = None
    u_prob: float | None = None
    weight: float = 0.0        # log(m/u)


@dataclass(frozen=True)
class ScoredPair:
    candidate: Candidate
    score: float
    evidence: list[Evidence] = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    candidate: Candidate
    verdict: str               # match | review | non_match
    score: float
    overrides: list[str] = field(default_factory=list)  # 예: ['negative_override:dob']
