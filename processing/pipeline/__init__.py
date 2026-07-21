"""v2 **조합(composition) 레이어** — 고정 뼈대 + 도메인 팩이 슬롯 전략을 고른다.

설계 의도
    도메인(일반·의료·샘플)을 core 코드의 if/else 로 분기하지 않는다. 파이프라인 뼈대는
    슬롯(classify/extract/embed/persist · candidates/score/decide/persist_edges) 순서만
    고정하고, "각 슬롯에 어떤 전략을 쓸지"는 도메인 팩이 선언한다. 도메인 차이는 *어떤
    전략이 resolve 됐는가*(팩 선택)로만 드러나며 core 실행 경로는 모든 도메인에 동일하다
    (헌법 4조). 신규 도메인 추가 = 전략 등록 + 팩 정의만으로 완결(core 수정 불요).

파일 맵
    - ``contracts.py``         : 슬롯 Protocol 계약(per-asset 4종 + cross-asset 4종). 정적
                                 타입·문서용 구조적 계약(런타임 isinstance 아님).
    - ``registry.py``          : 전략 레지스트리(슬롯·이름 → 구현 + capability 태그).
                                 ``DEFAULT_REGISTRY`` 프로세스 전역 인스턴스.
    - ``builtins.py``          : 기본 전략을 DEFAULT_REGISTRY 에 등록(**import 부수효과**).
    - ``packs.py``             : ``DomainPack``(슬롯별 전략 배선표 + 정책) + ``for_domain``.
    - ``policy.py``            : 정책 엔진 — 팩이 고른 전략의 태그를 컴포지션 시점에 검증
                                 (예: 의료 medical_strict = 외부 LLM 금지; 헌법 2조).
    - ``cross_types.py``       : cross-asset 스테이지 타입(Candidate→ScoredPair→Decision).
    - ``cross_runner.py``      : 제네릭 cross-asset 러너(resolve 된 4슬롯을 계약 순서 실행).
    - ``sample_strategies.py`` : 샘플 도메인 cross-asset 전략(결정적·무LLM 데모).

부트스트랩(중요)
    이 패키지는 **패키지 마커**다 — 레벨 재수출 없이 소비자가 서브모듈을 직접 import 한다.
    단, run_ingest/run_relations 는 ``from processing.pipeline import builtins`` 를 import 만 해도
    ``register_defaults(DEFAULT_REGISTRY)`` 가 실행되어(부수효과) 레지스트리가 채워진다.
    이 import 가 없으면 팩의 전략 이름이 resolve 시점에 KeyError 가 난다.
"""
